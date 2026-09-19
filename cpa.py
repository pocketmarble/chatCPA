"""
Corrected Projections Algorithm (CPA) and its iterative form (iCPA).

Otazu & Leibold (2011), "A Corticothalamic Circuit Model for Sound Identification
in Complex Scenes", PLoS ONE 6(9): e24270.

Notation follows the paper:
  f            number of features of an observation
  n            number of dictionary elements (n >> f)
  B_i          dictionary element i, a UNIT vector in R^f      -> `dic`, shape (n, f)
  y(t)         observation at time t, in R^f                   -> `y`,   shape (..., T, f)
  Phi(t)       f-by-n matrix whose i-th column is (y.B_i) B_i
  yhat(t)      Phi(t) @ Theta(t-1), the internal estimate of y(t)
  Theta        n presence parameters, ~1 if element present, ~0 if absent
  P            n-by-n "uncertainty" matrix (memory of the estimator)
  K = P Phi^T  n-by-f sensitivity / gain matrix

Because Phi(t) = D diag(c(t)) with D = dic^T (f-by-n) and c(t) = dic @ y(t), the
whole thing is ordinary recursive least squares (RLS) for the linear model
y(t) ~= D (c(t) * Theta), i.e. CPA fits ONE Theta to ALL T observations.

Three solvers are provided:
  cpa_closed_form  non-iterative CPA: one n-by-n solve over the whole batch of T
  icpa_full        iCPA exactly as published: full n-by-n P, f-by-f inverse per step
  icpa_diag        iCPA with P constrained to its diagonal; exact when the
                   dictionary is orthonormal, O(n) state instead of O(n^2)

`lam` adds an optional exponential forgetting factor (lam=1.0 -> the published
algorithm). It is needed for long observation sequences: with lam=1 the gain
K -> 0 and Theta freezes once enough evidence has accumulated.
"""

import torch


def normalize_dictionary(dic, eps=1e-8):
	"""Scale each dictionary element to unit length, as CPA requires."""
	return dic / dic.norm(dim=-1, keepdim=True).clamp_min(eps)


def cpa_closed_form(y, dic, p0=None):
	"""Non-iterative CPA: least-squares Theta over all T observations at once.

	y   (..., T, f), dic (n, f) with unit-norm rows.
	Returns Theta (..., n).

	Minimizing sum_t ||y(t) - Phi(t) Theta||^2 gives Theta = M^-1 v with
	  M = sum_t Phi(t)^T Phi(t) = G * (sum_t c(t) c(t)^T),   G = dic dic^T
	  v = sum_t Phi(t)^T y(t)   = sum_t c(t)^2
	`p0`, if given, adds the ridge term (1/p0) I contributed by P(0) = p0 I.
	"""
	c = y @ dic.t()                                     # (..., T, n)
	G = dic @ dic.t()                                   # (n, n)
	M = G * (c.transpose(-2, -1) @ c)                   # (..., n, n)
	v = (c * c).sum(dim=-2)                             # (..., n)
	if p0 is not None:
		M = M + torch.eye(M.shape[-1], dtype=M.dtype, device=M.device) / p0
	return torch.linalg.solve(M, v.unsqueeze(-1)).squeeze(-1)


def icpa_full(y, dic, p0=1.0, theta0=0.0, lam=1.0, return_estimate=False):
	"""iCPA with the full n-by-n P matrix, exactly as in the paper's Methods.

	y (B, T, f), dic (n, f) unit-norm rows. Returns Theta (B, T, n): the running
	presence parameters, so Theta[:, t] uses observations 0..t only (causal).
	Memory is O(B n^2) -- fine for tests, prohibitive at GPT-2 width.
	"""
	Bsz, T, f = y.shape
	n = dic.shape[0]
	D = dic.t()                                         # (f, n)
	I_f = torch.eye(f, dtype=y.dtype, device=y.device)
	P = p0 * torch.eye(n, dtype=y.dtype, device=y.device).expand(Bsz, n, n).clone()
	theta = torch.full((Bsz, n), float(theta0), dtype=y.dtype, device=y.device)

	thetas, yhats = [], []
	for t in range(T):
		yt = y[:, t]                                    # (B, f)
		c = yt @ dic.t()                                # (B, n)
		Phi = D.unsqueeze(0) * c.unsqueeze(1)           # (B, f, n)
		yhat = torch.bmm(Phi, theta.unsqueeze(-1)).squeeze(-1)          # (B, f)
		PPhiT = torch.bmm(P, Phi.transpose(1, 2))                       # (B, n, f)
		S = lam * I_f + torch.bmm(Phi, PPhiT)                           # (B, f, f)
		P = (P - torch.bmm(PPhiT, torch.linalg.solve(S, torch.bmm(Phi, P)))) / lam
		K = torch.bmm(P, Phi.transpose(1, 2))                           # (B, n, f)
		theta = theta + torch.bmm(K, (yt - yhat).unsqueeze(-1)).squeeze(-1)
		thetas.append(theta)
		yhats.append(yhat)
	theta = torch.stack(thetas, dim=1)
	if return_estimate:
		return theta, torch.stack(yhats, dim=1)
	return theta


def icpa_diag(y, dic, p0=1.0, theta0=0.0, lam=1.0, c=None, return_estimate=False):
	"""iCPA with P constrained to a diagonal (exact iff the dictionary is orthonormal).

	With P = diag(p), Phi^T Phi = diag(c) G diag(c), so if G = I the recursion
	P^-1(T) = lam P^-1(T-1) + Phi^T Phi keeps P diagonal and collapses to
	  p_i(T) = p_i(T-1) / (lam + p_i(T-1) c_i(T)^2)
	and the gain to K = diag(p c) dic, i.e.
	  dTheta = p * c * (dic @ (y - yhat)).

	y (B, T, f), dic (n, f). `c` may be supplied precomputed as (B, T, n) when the
	caller already has the projections (e.g. a transformer's c_fc pre-activations).
	Returns Theta (B, T, n).
	"""
	Bsz, T, f = y.shape
	n = dic.shape[0]
	if c is None:
		c = y @ dic.t()                                 # (B, T, n)
	p = torch.full((Bsz, n), float(p0), dtype=y.dtype, device=y.device)
	theta = torch.full((Bsz, n), float(theta0), dtype=y.dtype, device=y.device)

	thetas, yhats = [], []
	for t in range(T):
		ct = c[:, t]                                    # (B, n)
		yhat = (theta * ct) @ dic                       # (B, f)   == Phi(t) @ theta
		p = p / (lam + p * ct * ct)
		theta = theta + p * ct * ((y[:, t] - yhat) @ dic.t())
		thetas.append(theta)
		yhats.append(yhat)
	theta = torch.stack(thetas, dim=1)
	if return_estimate:
		return theta, torch.stack(yhats, dim=1)
	return theta


# ---------------------------------------------------------------------------
# self-test: reproduce the paper's Figure 3 and cross-check the three solvers
# ---------------------------------------------------------------------------

def _scene(T=300, f=100, n=150, present=(4, 9), amps=(1.0, 1.0), seed=0):
	"""A paper-style auditory scene: a few dictionary elements, each amplitude
	modulated by an independent zero-mean unit-variance Gaussian, summed.

	The paper's Fig. 3 uses f=10, n=18; CPA's guarantee, however, only holds when
	the elements present are near-orthogonal, and in f=10 a random dictionary is
	nowhere near that (random pairs overlap by ~1/sqrt(f) = 0.32), so Theta comes
	out badly biased. We therefore run the same experiment at f=100, n=150, the
	size used for the paper's pulse-train simulation -- and the size regime that
	matters here, since GPT-2 has f = n_embd = 768.
	"""
	g = torch.Generator().manual_seed(seed)
	dic = torch.randn(n, f, generator=g, dtype=torch.float64)
	dic = normalize_dictionary(dic - dic.mean(dim=1, keepdim=True))
	A = torch.randn(T, len(present), generator=g, dtype=torch.float64)
	A = A * torch.tensor(amps, dtype=torch.float64)
	return (A @ dic[list(present)]).unsqueeze(0), dic


def _test():
	absent = lambda n, present: [i for i in range(n) if i not in present]

	# 1. Fig 3E: two equally loud sources -> Theta ~ 1 for both, ~0 elsewhere
	y, dic = _scene()
	th = icpa_full(y, dic, p0=1e3)[0, -1]
	print("fig3E  theta[present] =", [round(v, 3) for v in th[[4, 9]].tolist()],
	      " max|theta[absent]| =", round(th[absent(150, (4, 9))].abs().max().item(), 3))
	assert th[[4, 9]].sub(1.0).abs().max() < 0.05
	assert th[absent(150, (4, 9))].abs().max() < 0.5

	# 2. Fig 3H/3L: one source 10x quieter. CPA still returns Theta ~ 1 for it,
	#    whereas template matching on the RMS similarity buries it among absent
	#    elements -- the paper's central comparison.
	y, dic = _scene(amps=(1.0, 0.1))
	th = icpa_full(y, dic, p0=1e3)[0, -1]
	rms = (y[0] @ dic.t()).pow(2).mean(0).sqrt()
	print("fig3H  theta[quiet] =", round(th[9].item(), 3),
	      " rank of quiet element by theta =", int((th > th[9]).sum()),
	      " by RMS similarity =", int((rms > rms[9]).sum()))
	assert th[[4, 9]].sub(1.0).abs().max() < 0.05
	assert int((th > th[9]).sum()) <= 1           # CPA ranks the quiet source top-2
	assert int((rms > rms[9]).sum()) > 5          # template matching does not

	# 3. iCPA reproduces the non-iterative closed-form CPA solution
	th_cf = cpa_closed_form(y, dic, p0=1e3)[0]
	print("fig3   max|icpa_full - cpa_closed_form| =", "%.2e" % (th - th_cf).abs().max().item())
	assert torch.allclose(th, th_cf, atol=1e-8)

	# 4. diagonal-P iCPA is exact when the dictionary is orthonormal
	f, n, T = 12, 8, 40
	dic = torch.linalg.qr(torch.randn(f, f, dtype=torch.float64))[0][:n]
	y = torch.randn(2, T, f, dtype=torch.float64)
	a = icpa_full(y, dic, p0=0.7, theta0=0.05, lam=0.99)
	b = icpa_diag(y, dic, p0=0.7, theta0=0.05, lam=0.99)
	print("orth   max|icpa_full - icpa_diag| =", "%.2e" % (a - b).abs().max().item())
	assert torch.allclose(a, b, atol=1e-9)

	# 5. ...and a good approximation for a random overcomplete dictionary at the
	#    aspect ratio GPT-2 uses (n = 4f), which is what the transformer runs
	f, n = 128, 512
	dic = normalize_dictionary(torch.randn(n, f, dtype=torch.float64))
	y, _ = _scene(T=200, f=f, n=n, present=(4, 9))
	a = icpa_full(y, dic, p0=1.0, lam=0.999)[0, -1]
	b = icpa_diag(y, dic, p0=1.0, lam=0.999)[0, -1]
	corr = torch.corrcoef(torch.stack([a, b]))[0, 1]
	print("n=4f   corr(full, diag) =", round(corr.item(), 4),
	      " max|full-diag| =", round((a - b).abs().max().item(), 4))
	assert corr > 0.9

	print("all cpa tests passed")


if __name__ == '__main__':
	_test()
