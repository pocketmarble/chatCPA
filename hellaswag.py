"""
Minimal HellaSwag evaluation.

Each example is a context plus four candidate endings, one of which is correct.
The model completes the context with each ending and the candidate with the
lowest mean cross-entropy over its own tokens wins. This is the protocol nanoGPT
uses, so scores are comparable to published GPT-2 figures: 124M sits around 0.29
accuracy, chance is 0.25.

Examples come from the HuggingFace datasets-server as plain JSON, which needs no
parquet reader, and are cached beside this file after the first fetch. (nanoGPT's
raw.githubusercontent URL for this dataset is dead -- the upstream repo moved.)
"""

import json
import os
import urllib.request

import torch
from torch.nn import functional as F

API = ("https://datasets-server.huggingface.co/rows"
       "?dataset=Rowan/hellaswag&config=default&split=validation&offset={off}&length={n}")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hellaswag_val.jsonl")
PAGE = 100  # the API's maximum rows per request


def download(limit):
	"""Ensure at least `limit` examples are cached locally, then return them."""
	have = []
	if os.path.exists(CACHE):
		with open(CACHE) as f:
			have = [json.loads(line) for line in f]
	while len(have) < limit:
		url = API.format(off=len(have), n=min(PAGE, limit - len(have)))
		with urllib.request.urlopen(url, timeout=60) as r:
			rows = json.load(r)["rows"]
		if not rows:
			break
		have += [row["row"] for row in rows]
		with open(CACHE, 'w') as f:
			for ex in have:
				f.write(json.dumps(ex) + "\n")
	return have[:limit]


def render(example, enc):
	"""One example -> (tokens (4,T), mask (4,T), label).

	The mask marks each ending's tokens, the only ones scored; the shared context
	is identical across the four rows and carries no signal.
	"""
	ctx = enc.encode(example["ctx"])
	rows = [ctx + enc.encode(" " + end) for end in example["endings"]]
	T = max(len(r) for r in rows)
	tokens = torch.zeros(4, T, dtype=torch.long)
	mask = torch.zeros(4, T, dtype=torch.long)
	for i, row in enumerate(rows):
		tokens[i, :len(row)] = torch.tensor(row)
		mask[i, len(ctx):len(row)] = 1
	return tokens, mask, int(example["label"])


@torch.no_grad()
def predict(model, tokens, mask):
	"""Index of the candidate with the lowest mean loss over its own ending."""
	logits, _ = model(tokens)
	loss = F.cross_entropy(
		logits[:, :-1].reshape(-1, logits.size(-1)),
		tokens[:, 1:].reshape(-1),
		reduction='none').view(tokens.size(0), -1)
	m = mask[:, 1:].float()
	return int(((loss * m).sum(1) / m.sum(1).clamp_min(1)).argmin())


@torch.no_grad()
def accuracy(model, enc, limit=50, device='cpu', on_step=None):
	"""Fraction correct over the first `limit` examples. on_step(done, total, acc)
	runs after each example so a caller can report progress."""
	correct = done = 0
	for example in download(limit):
		tokens, mask, label = render(example, enc)
		correct += predict(model, tokens.to(device), mask.to(device)) == label
		done += 1
		if on_step:
			on_step(done, limit, correct / done)
	return correct / max(1, done)
