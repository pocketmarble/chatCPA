# chatCPA

Transplanting the **Corrected Projections Algorithm** (Otazu & Leibold 2011, *PLoS ONE*
6(9):e24270) into GPT-2, as a replacement for the GeLU in every MLP block.

Claude Code loads this file automatically. It is the project's memory: everything
below was measured or decided in earlier sessions, so **do not re-derive it.**

---

## 1. The idea in one paragraph

GeLU gates each of the 3072 MLP hidden units independently, on the size of its own
pre-activation. CPA instead makes the units **compete**: each is scaled by a presence
parameter `θ_i`, discounted to the extent that other units already explain the same
part of the MLP's input. A unit that looks active only because its input direction
overlaps an active neighbour gets suppressed. The dictionary CPA competes over is
`c_fc`'s own weight matrix, inherited frozen from OpenAI — so `c_fc` is doing double
duty as both the projection and the dictionary.

## 2. Status

Working: frozen GPT-2 124M, inference only, with a local dashboard that compares
stock GeLU against the CPA variant side by side. Nothing is trained. `α=0`
reproduces stock GPT-2 **bit-for-bit** (verified by `torch.equal`).

**Immediate task: get this running on an RTX 3090.** `--device` defaults to `auto`,
which selects CUDA when available and enables TF32. The CUDA path has never been
executed — it was written on a Mac with no NVIDIA device. All three device-sensitive
paths (forward-with-diagnostics, generation, HellaSwag) were verified on MPS, so
tensor placement is believed correct, but the first CUDA run is the real test. A
failure will almost certainly be a missing `.to(device)` and the traceback will
name it.

```bash
python serve.py                    # picks the GPU, opens http://localhost:8000
python serve.py --device cpu       # force
```

Expect generation to drop from ~18s (CPU) to roughly 1-3s. It will not scale with
raw FLOPS: at batch 1 this is kernel-launch bound, and CG adds 18 small matmuls per
layer per token.

## 3. The math, compactly

`B` = `c_fc.weight` with rows scaled to unit length, so `B_i` is unit `i`'s input
direction and `c_i = y · B_i` is its projection. CPA models the MLP input as

```
ŷ = Σ_i θ_i · c_i · B_i
```

Each element enters **already weighted by its own projection**, so `θ` carries only
presence. That second multiplier is the entire difference from earlier dictionary
methods, and it is why the least-squares problem has a unique solution with no
sparsity penalty bolted on. Minimising `‖y − ŷ‖² + (1/p₀)‖θ‖²` gives

```
( G ⊙ ccᵀ + (1/p₀)I ) θ = c²        G = BBᵀ,  ⊙ elementwise
```

Two facts that shape the implementation:

- **The right-hand side is exactly `c²`** — the plain template-matching score —
  because `c = By`. The MLP input `y` never appears, which is what lets this be a
  drop-in activation taking only the pre-activations.
- `G ⊙ ccᵀ` is elementwise, *not* a matrix product. Entry (i,j) is
  `(B_i·B_j)·c_i·c_j`: two units compete only if their directions overlap **and**
  both are active at this token.

Solved matrix-free by conjugate gradient — the matrix is never formed, since
`(G ⊙ ccᵀ)v = c ⊙ (B(Bᵀ(c ⊙ v)))`, two matmuls per iteration.

## 4. Files

| file | |
|---|---|
| `activation.py` | `CPAActivation` — the whole algorithm. Long docstring, read it first. |
| `model.py` | Karpathy-style GPT-2. Only the `MLP` class and `GPTConfig` differ from stock. |
| `serve.py` | stdlib `http.server` dashboard backend. Four endpoints. |
| `index.html` | the dashboard. Single page, no scrolling — keep it that way. |
| `hellaswag.py` | 4-way completion scoring. |
| `cpa.py` | standalone CPA/iCPA reference implementations + tests. Not imported by the model; it exists to check the algorithm against the paper. |
| `CPA_paper.pdf`, `cpa.doc` (Text S2), `iCPA.doc` (Text S5) | the source papers |

**One model is loaded, not two.** The control and the variant are the same weights
with `α` toggled between runs, which guarantees the control cannot drift. Both
generation runs reset the same seed, so text differences are the activation and not
the sampler. Do not "simplify" this into two model instances.

## 5. Measured facts — do not re-measure

On pretrained GPT-2 124M, real activations:

- `λ_max` of `Bᵀdiag(c²)B` is **12–21** across all 12 layers.
- **Text S5's `(I+X)⁻¹ ≈ I−X` shortcut is unusable.** It needs `‖X‖ ≪ 1`, forcing
  `p₀ < 0.01` — and at that `p₀`, `cos(θ, c²) ≥ 0.999`, i.e. θ is indistinguishable
  from template matching. The approximation is only valid where the algorithm does
  nothing. **Do not reinstate it.**
- CG needs `k ≈ √(1 + p₀·λ_max)`: measured **9 iterations at p₀=1** for 5e-3
  relative error. The system is `(1/p₀)I` plus something PSD, so its condition
  number is bounded by `1 + p₀λ_max` (measured 2.4–29) and CG needs no
  preconditioner.
- **Forming the f×f matrix, not inverting it, is the cost.** A direct solve is 278×
  the whole base model per token; the Cholesky is only 8% of that, the assembly is
  the rest. CG at k=9 is ~7× base in FLOPs, **4.8× measured wall clock**.
- Frozen swap degrades hard: dashboard baseline CE **3.355**; at α=0.3, p₀=1.0 it is
  **6.26**, top-1 37.0% → 13.8%, and generation collapses into degenerate repetition.
  It holds around α ≤ 0.1.
- **19–44% of θ comes out negative** (p₀=0.1 → p₀=1). Since `h = θ⊙c`, `c_proj`
  receives sign-flipped inputs it never saw in training. This is the main suspected
  cause of the degradation above — GeLU's output is never negative.
- **θ is not sparse.** Top 1% of entries hold 6.5–9.6% of the mass. CPA's "sparsity
  for free" does not materialise on GPT-2's dictionary, as expected if `c_fc` rows
  are in superposition.
- HellaSwag scorer validated: **0.31 on 200 examples at α=0**, against GPT-2 124M's
  published ~0.29–0.31.
- MPS is ~6× *slower* than CPU here (22.9s/16 tokens vs 14.4s/64). Auto-detect
  deliberately never selects it.

## 6. Deviations from the paper — know these before claiming a result

1. **T = 1.** The paper fits one θ across T observations; here each token is its own
   one-sample scene. The competition matrix's `Ψ = ccᵀ` is rank one, so only
   *geometric* overlap discounts anything. The paper's temporal-decorrelation
   mechanism — and its headline result, recovering a 10× quieter source — is absent.
   This was a deliberate choice: a θ shared across tokens makes `h = θ⊙c` a fixed
   diagonal map, which would collapse the MLP to a linear transform.
2. **The dictionary is trained MLP weights, not per-source templates.** The paper
   builds elements by averaging spectra of actual instruments — one element, one
   source. A `c_fc` row is generally a blend of features. CPA's core premise does not
   hold, and no implementation care fixes it.
3. Rows are unit-normalised but **not mean-subtracted** (the paper does both).
4. **Level invariance is lost.** Finite `p₀` gives the gate an absolute scale, which
   is exactly what makes it behave like GeLU's threshold.
5. The code solves the n-space form, not the paper's Kalman recursion. Verified
   identical to 2.4e-15, but `P`, `K`, `ŷ` and `e` appear nowhere in the code.

## 7. The two knobs are frozen-weights artifacts

- **`p₀`** — competition strength; the paper's `P(0) = p₀I` seen through a single
  step. `p₀ → 0` gives no competition (`θ → p₀c²`); large `p₀` gives strong
  explaining-away and also keeps the system non-singular. Useful range 0.1–3.
- **`α`** — blend weight, 0 = GeLU, 1 = pure CPA.

Both exist only because the weights are frozen. A trained network could absorb any
effective `p₀` by rescaling `c_fc`'s rows, and would be trained at `α=1`. `cg_iters`
is a solver setting and has no gradient at all. Neither is an `nn.Parameter` —
they're buffers, so they sweep at inference without touching the checkpoint.

## 8. Pitfalls

- `nanoGPT`'s `raw.githubusercontent.com/rowanz/hellaswag` URL is **dead** (repo
  moved). `hellaswag.py` fetches from HuggingFace's datasets-server as JSON, which
  also avoids needing a parquet reader. Don't "fix" it back.
- **Do not add bf16 naively.** CG accumulates a residual across iterations and
  bf16's 8-bit mantissa would let it stall or converge somewhere wrong *silently*.
  If wanted: bf16 for the model, fp32 inside `solve_theta`.
- `c_fc`'s bias must be stripped before treating `c` as a projection. With the bias
  left in, `c ≠ By`, the right-hand side is no longer `c²`, and the derivation breaks.
- `match_scale=True` rescales the CPA branch to GeLU's per-token RMS so `α`
  interpolates *shape* and not magnitude. It is an addition, not in the paper.

## 9. Open work

- Run on CUDA (§2).
- Training is the real experiment; the frozen swap is a smoke test. Budget ~13×
  base training cost. Use **implicit differentiation** for the backward pass — one
  more CG solve against the same symmetric matrix — never unroll the 9 iterations,
  which would store 9× the activations per layer.
- Scale ladder rather than one big run: ~7M params is ~35 min on a 3090, ~25M is
  ~8h, GPT-2 124M on 10B tokens is ~40 days (don't). Activation-function gains at
  small scale routinely vanish at scale, so measure whether the gap grows or closes.
- A fair comparison needs a matched-compute column: at matched *parameters* CPA gets
  13× the FLOPs.
