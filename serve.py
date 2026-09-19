"""
Local dashboard comparing frozen GPT-2 against its 1-CPA variant.

    python serve.py                 # picks a CUDA GPU if there is one
    python serve.py --device cpu --port 8080

One model is loaded, not two: alpha=0 reproduces GeLU bit-for-bit, so the control
and the variant are the same weights with alpha toggled between runs. Both runs
reset the same sampling seed, so any difference in the generated text comes from
the activation and not from the sampler.

Endpoints
    GET  /            the dashboard
    POST /generate    {prompt, tokens, temp, top_k, alpha, p0} -> both completions
    POST /metrics     {alpha, p0, layer} -> loss/ppl/top-1 for both, plus CPA
                      diagnostics and (c, h) scatter points for the p0 panel
    POST /hellaswag   {alpha, p0, limit} -> starts a background run
    GET  /progress    poll the running HellaSwag job
"""

import argparse
import json
import math
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from torch.nn import functional as F
from transformers import GPT2TokenizerFast

import hellaswag
from activation import CPAActivation, set_cpa
from model import GPT

SEED = 1337


def pick_device(requested):
	"""'auto' takes a CUDA GPU if present, else CPU.

	MPS is deliberately not auto-selected: measured on this workload it runs
	slower than CPU, because batch-1 generation is dominated by per-kernel
	launch overhead and CG adds 18 small matmuls per layer per token. Ask for it
	explicitly with --device mps if you want to check on your own machine.
	"""
	if requested != 'auto':
		return requested
	if torch.cuda.is_available():
		# Ampere and later: TF32 roughly doubles fp32 matmul throughput for a
		# precision loss CG tolerates easily. Full bf16 would be faster still,
		# but CG accumulates over iterations and bf16's 8-bit mantissa is a poor
		# fit for that -- left alone rather than silently risked.
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True
		return 'cuda'
	return 'cpu'

# Held-out prose for the fast metrics: five registers, none of it in the prompt
# box, so the numbers move for reasons other than the text being memorised.
EVAL_TEXT = [
	"The Roman Republic was founded in 509 BC after the overthrow of the monarchy. Its "
	"political institutions, including the Senate and the consulship, shaped European "
	"governance for centuries afterward.",
	"Photosynthesis converts light energy into chemical energy stored in glucose. In plants "
	"the process occurs in chloroplasts, where chlorophyll absorbs photons in the blue and "
	"red portions of the visible spectrum.",
	"She had not expected the letter to arrive so late in the afternoon, nor to find it "
	"unsigned. The handwriting was unfamiliar, sloping heavily to the left, and the paper "
	"carried a faint smell of tobacco.",
	"To install the package, first create a virtual environment and activate it. If "
	"compilation fails, check that the development headers for the linear algebra library "
	"are present on your system.",
	"Interest rates remained unchanged at the September meeting, though two members of the "
	"committee dissented in favor of a quarter-point increase. The statement noted that "
	"inflation had moderated.",
]


@torch.no_grad()
def generate(model, ids, n_tokens, temp, top_k):
	"""Sample n_tokens continuations. No KV cache, so cost grows with the prefix."""
	torch.manual_seed(SEED)
	for _ in range(n_tokens):
		logits, _ = model(ids[:, -model.config.block_size:])
		logits = logits[:, -1, :] / max(temp, 1e-6)
		if top_k:
			kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1:]
			logits = logits.masked_fill(logits < kth, float('-inf'))
		nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
		ids = torch.cat([ids, nxt], dim=1)
	return ids


@torch.no_grad()
def evaluate(model, batches):
	"""Cross-entropy, perplexity and next-token top-1 accuracy over EVAL_TEXT."""
	total, correct, n = 0.0, 0, 0
	for x, y in batches:
		logits, loss = model(x, y)
		total += loss.item() * y.numel()
		correct += int((logits.argmax(-1) == y).sum())
		n += y.numel()
	mean = total / n
	return dict(loss=round(mean, 4), ppl=round(math.exp(mean), 2), acc=round(correct / n, 4))


class State:
	"""Everything the request handlers share. One model, swept in place."""

	def __init__(self, device):
		self.device = device
		self.enc = GPT2TokenizerFast.from_pretrained('gpt2')
		self.model = GPT.from_pretrained('gpt2').eval().to(device)
		self.layers = self.model.config.n_layer
		self.cg_iters = self.model.transformer.h[0].mlp.activation.cg_iters
		self.batches = []
		for passage in EVAL_TEXT:
			ids = torch.tensor(self.enc(passage)['input_ids'], device=device).unsqueeze(0)
			self.batches.append((ids[:, :-1], ids[:, 1:]))
		self.lock = threading.Lock()
		self.job = None

	def tune(self, alpha, p0, record_layer=None):
		set_cpa(self.model, alpha=alpha, p0=p0, record=False)
		if record_layer is not None:
			self.model.transformer.h[record_layer].mlp.activation.record = True

	def run_generate(self, req):
		ids = torch.tensor(self.enc(req['prompt'] or '\n')['input_ids'],
		                   device=self.device).unsqueeze(0)
		out = {}
		for name, alpha in (('control', 0.0), ('cpa', float(req['alpha']))):
			self.tune(alpha, float(req['p0']))
			t0 = time.perf_counter()
			ids_out = generate(self.model, ids, int(req['tokens']),
			                   float(req['temp']), int(req['top_k']))
			out[name] = dict(
				text=self.enc.decode(ids_out[0].tolist()),
				completion=self.enc.decode(ids_out[0, ids.size(1):].tolist()),
				secs=round(time.perf_counter() - t0, 2))
		return out

	def run_metrics(self, req):
		layer = int(req.get('layer', self.layers // 2))
		out = {}
		for name, alpha in (('control', 0.0), ('cpa', float(req['alpha']))):
			self.tune(alpha, float(req['p0']), record_layer=layer if name == 'cpa' else None)
			out[name] = evaluate(self.model, self.batches)
		act = self.model.transformer.h[layer].mlp.activation
		stats = act.stats or {}
		act.record = False
		out['diag'] = {k: v for k, v in stats.items() if k != 'points'}
		out['points'] = stats.get('points', [])
		out['layer'] = layer
		return out

	def start_hellaswag(self, req):
		if self.job and not self.job['done']:
			return dict(error='a run is already in progress')
		self.job = dict(done=False, control=None, cpa=None, step=0,
		                total=int(req['limit']), which='control', acc=0.0, error=None)
		threading.Thread(target=self._hellaswag_worker,
		                 args=(float(req['alpha']), float(req['p0']), int(req['limit'])),
		                 daemon=True).start()
		return dict(started=True)

	def _hellaswag_worker(self, alpha, p0, limit):
		job = self.job
		try:
			for name, a in (('control', 0.0), ('cpa', alpha)):
				job['which'], job['step'] = name, 0
				self.tune(a, p0)
				def on_step(done, total, acc):
					job['step'], job['acc'] = done, round(acc, 4)
				job[name] = round(hellaswag.accuracy(
					self.model, self.enc, limit, self.device, on_step), 4)
		except Exception as exc:                       # network, parsing, anything
			job['error'] = f"{type(exc).__name__}: {exc}"
		job['done'] = True


class Handler(BaseHTTPRequestHandler):
	state = None

	def log_message(self, *args):
		pass                                            # keep the console quiet

	def _send(self, body, ctype='application/json'):
		body = body if isinstance(body, bytes) else body.encode()
		self.send_response(200)
		self.send_header('Content-Type', ctype)
		self.send_header('Content-Length', str(len(body)))
		self.end_headers()
		self.wfile.write(body)

	def do_GET(self):
		if self.path.startswith('/progress'):
			return self._send(json.dumps(self.state.job or dict(done=True)))
		if self.path in ('/', '/index.html'):
			with open('index.html', 'rb') as f:
				return self._send(f.read(), 'text/html; charset=utf-8')
		if self.path == '/config':
			return self._send(json.dumps(dict(
				device=self.state.device, layers=self.state.layers,
				cg_iters=self.state.cg_iters)))
		self.send_error(404)

	def do_POST(self):
		req = json.loads(self.rfile.read(int(self.headers['Content-Length'] or 0)) or b'{}')
		routes = dict(generate=self.state.run_generate, metrics=self.state.run_metrics,
		              hellaswag=self.state.start_hellaswag)
		fn = routes.get(self.path.strip('/'))
		if not fn:
			return self.send_error(404)
		with self.state.lock:                           # one inference at a time
			try:
				body = fn(req)
			except Exception as exc:
				body = dict(error=f"{type(exc).__name__}: {exc}")
		self._send(json.dumps(body))


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--device', default='auto', help='auto | cpu | cuda | mps')
	ap.add_argument('--port', type=int, default=8000)
	ap.add_argument('--no-browser', action='store_true')
	args = ap.parse_args()

	device = pick_device(args.device)
	name = torch.cuda.get_device_name(0) if device == 'cuda' else device
	print(f"loading gpt2 on {device} ({name}) ...")
	Handler.state = State(device)
	url = f"http://localhost:{args.port}"
	print(f"ready: {url}   (cg_iters={Handler.state.cg_iters})")
	if not args.no_browser:
		threading.Timer(0.5, lambda: webbrowser.open(url)).start()
	HTTPServer(('localhost', args.port), Handler).serve_forever()


if __name__ == '__main__':
	main()
