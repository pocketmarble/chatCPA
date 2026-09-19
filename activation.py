"""
1-CPA as a drop-in replacement for the GeLU in a transformer MLP.

CPA reference: Otazu & Leibold (2011), "A Corticothalamic Circuit Model for Sound
Identification in Complex Scenes", PLoS ONE 6(9): e24270, with Text S2 (the
algorithm) and Text S5 (the iterative form).


WHAT THIS REPLACES
------------------
GeLU gates each of the 4C hidden units independently, on the size of its own
pre-activation. 1-CPA instead makes the 4C units compete: each is scaled by a
presence parameter theta_i, discounted to the extent that other units already
account for the same part of the MLP's input. A unit that looks active only
because its input direction overlaps an active neighbour gets suppressed.


THE ALGORITHM
-------------
The dictionary B is c_fc's own weight with rows scaled to unit length, so B_i is
the input direction of hidden unit i and c_i = y . B_i is its projection. CPA
models the MLP input as

    yhat = sum_i  theta_i * c_i * B_i                        (Text S2, eq. S2.1)

Every element enters already weighted by its own projection, so theta carries
only presence. Minimising ||y - yhat||^2 + (1/p0)||theta||^2 gives

    ( G * cc^T + (1/p0) I ) theta = c^2      G = B B^T,  * elementwise

  * The right-hand side is exactly c^2 -- the plain template-matching score --
    because c = By. So the MLP input y never appears, which is what lets this be
    a drop-in activation taking only the pre-activations.
  * G * cc^T is elementwise, not a matrix product: entry (i,j) is
    (B_i . B_j) * c_i * c_j. Two units compete only if their input directions
    overlap AND both are active at this token.

Solved matrix-free by conjugate gradient -- the matrix is never formed, since

    (G * cc^T) v  =  c * ( B ( B^T ( c * v ) ) )              two matmuls

CG suits it: the matrix is (1/p0)I plus something positive semidefinite, so its
condition number is at most 1 + p0*lambda_max, measured at 2.4-29 on GPT-2 and
needing 4-10 iterations. Forming the matrix instead would cost ~250x a whole
transformer layer, of which the factorisation is only 8% -- the assembly is what
makes the direct route unaffordable.

Text S5 closes with the shortcut (I+X)^-1 ~= I-X, which removes the solve
entirely. Measured on GPT-2 it is unusable: it needs ||X|| << 1, forcing
p0 < 0.01, and at that p0 theta is 99.9% identical to template matching. The
approximation is only valid where the algorithm does nothing, hence CG.


DEVIATIONS FROM THE PAPER
-------------------------
  * One observation per token, not T. The paper fits one theta across T samples;
    the competition matrix here has rank-one cc^T, so only geometric overlap
    discounts anything. The paper's temporal-decorrelation mechanism is absent.
  * The dictionary is trained MLP weights, not per-source templates. A c_fc row
    is generally a blend of features, so "one element, one source" does not hold.
  * Rows are unit-normalised but not mean-subtracted (the paper does both).
  * Level invariance is lost: finite p0 gives the gate an absolute scale, which
    is what makes it behave like GeLU's threshold.


PARAMETERS
----------
p0      Competition strength -- the prior variance on theta, and the paper's
        P(0) = p0*I initial condition seen through a single step. p0 -> 0 gives
        no competition (theta -> p0*c^2); large p0 gives strong explaining-away
        and is also what keeps the system non-singular. Useful range on
        pretrained GPT-2 is 0.1 to 3. Plays the role GeLU's threshold plays, so
        there is no value to inherit.

alpha   Blend: 0 reproduces GeLU exactly, 1 is pure 1-CPA.

Both are buffers, not parameters, so they can be swept at inference without
touching the checkpoint. A trained network could absorb any effective p0 by
rescaling c_fc's rows, so p0 is a frozen-weights artifact; cg_iters is a solver
setting and has no gradient at all.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class CPAActivation(nn.Module):
	"""Drop-in replacement for nn.GELU inside an MLP block.

	Needs a reference to the c_fc it follows, because the dictionary *is* c_fc's
	weight. forward() still takes only the pre-activations.

	    self.c_fc       = nn.Linear(C, 4*C)
	    self.activation = CPAActivation(self.c_fc)
	    self.c_proj     = nn.Linear(4*C, C)
	"""

	def __init__(self, c_fc, p0=1.0, alpha=0.0, cg_iters=9, match_scale=True):
		super().__init__()
		# held in a list so nn.Module does not register it and duplicate c_fc's
		# parameters in this module's state_dict
		self._c_fc = [c_fc]
		self.cg_iters = cg_iters
		self.match_scale = match_scale  # rescale the CPA branch to GeLU's per-token
		                                # RMS, so alpha interpolates shape and not size
		self.record = False             # stash diagnostics from the next forward
		self.stats = None
		self.register_buffer('p0', torch.tensor(float(p0)))
		self.register_buffer('alpha', torch.tensor(float(alpha)))

	def dictionary(self):
		"""B (n,f) with unit rows, and the row norms s (n,) divided out."""
		W = self._c_fc[0].weight               # (n,f); row i is unit i's input direction
		s = W.norm(dim=1).clamp_min(1e-12)
		return W / s.unsqueeze(1), s

	def solve_theta(self, c, B):
		"""Conjugate gradient on ( G * cc^T + (1/p0) I ) theta = c^2.

		Symmetric positive definite with smallest eigenvalue >= 1/p0, so CG is
		stable and needs no preconditioner at these p0.
		"""
		inv_p0 = 1.0 / self.p0
		rhs = c * c
		def A(v):                              # (G * cc^T) v + v/p0, matrix-free
			u = (c * v) @ B                    # (...,f)   B^T (c*v)
			return c * (u @ B.t()) + inv_p0 * v

		theta = torch.zeros_like(rhs)
		r = rhs.clone()
		p = r.clone()
		rs = (r * r).sum(-1, keepdim=True)
		for _ in range(self.cg_iters):
			Ap = A(p)
			step = rs / (p * Ap).sum(-1, keepdim=True).clamp_min(1e-30)
			theta = theta + step * p
			r = r - step * Ap
			rs_new = (r * r).sum(-1, keepdim=True)
			p = r + (rs_new / rs.clamp_min(1e-30)) * p
			rs = rs_new
		return theta

	def forward(self, c_pre):
		"""c_pre: (...,n) the output of c_fc. Returns (...,n) for c_proj."""
		gelu = F.gelu(c_pre, approximate='tanh')
		if float(self.alpha) == 0.0 and not self.record:
			return gelu

		B, s = self.dictionary()
		bias = self._c_fc[0].bias
		lin = c_pre - bias if bias is not None else c_pre
		c = lin / s                            # pure projections y . B_i
		theta = self.solve_theta(c, B)
		cpa = theta * lin                      # back to pre-activation scale
		if self.match_scale:
			rms = lambda t: t.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
			cpa = cpa * (rms(gelu) / rms(cpa))

		if self.record:
			self.stats = self._diagnose(c, theta, c_pre, cpa, gelu)
		a = self.alpha
		return (1.0 - a) * gelu + a * cpa

	def _diagnose(self, c, theta, c_pre, cpa, gelu, n_points=500):
		"""Numbers for the dashboard. cos(theta, c^2) is the key one: 1.0 means
		theta is pure template matching, i.e. no competition is happening."""
		t, c2 = theta.flatten(), (c * c).flatten()
		g = lambda x: x.flatten().abs().sort(descending=True).values
		frac = lambda v: (v[:max(1, v.numel() // 100)].sum() / v.sum().clamp_min(1e-12)).item()
		idx = torch.randperm(c_pre.numel(), device=c_pre.device)[:n_points]
		return dict(
			cos=float((t @ c2) / (t.norm() * c2.norm()).clamp_min(1e-12)),
			neg=float((theta < 0).float().mean()),
			theta_absmax=float(theta.abs().max()),
			top1_cpa=frac(g(cpa)), top1_gelu=frac(g(gelu)),
			points=[[round(float(v), 4) for v in p] for p in
			        zip(c_pre.flatten()[idx].tolist(), cpa.flatten()[idx].tolist())],
		)

	def extra_repr(self):
		return (f"p0={float(self.p0):g}, alpha={float(self.alpha):g}, "
		        f"cg_iters={self.cg_iters}, match_scale={self.match_scale}")


def set_cpa(model, alpha=None, p0=None, cg_iters=None, record=None):
	"""Retune every CPAActivation in a model in place, for sweeping at inference."""
	n = 0
	for m in model.modules():
		if isinstance(m, CPAActivation):
			if alpha is not None: m.alpha.fill_(float(alpha))
			if p0 is not None: m.p0.fill_(float(p0))
			if cg_iters is not None: m.cg_iters = int(cg_iters)
			if record is not None: m.record, m.stats = bool(record), None
			n += 1
	return n
