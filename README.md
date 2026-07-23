# P1: The Anatomy of the V-JEPA 2-AC Predictor

Mechanistic analysis of what the action-conditioned V-JEPA 2 predictor computes, with emphasis
on how action conditioning is routed. Pre-registration plus infrastructure. Encoder-side
analysis of JEPA is well studied; the predictor, which is the module that actually constitutes
the world model, is essentially uncharacterized. This is the predictor analogue.

## Phase 1 site verification + R7 gate (run this first)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grewalsk/p1-vjepa2-predictor/blob/main/P1_phase1_colab.ipynb)

`P1_phase1_colab.ipynb` clones `facebookresearch/vjepa2`, restores the real checkpoint URL,
loads `vjepa2_ac_vit_giant`, and discharges two gates:

- **R7 (real checkpoint):** the predictor `state_dict` populates the verified module tree
  (empty missing/unexpected keys) and `action_encoder.weight` std is well above the 0.02 init.
- **Frame-causal:** perturbing the action at frame `t` leaves every token in frames `< t`
  exactly unchanged (`per_frame_max_delta ~ [0, 0, >0, >0]`).

No GPU is needed for the gate. The 11.76 GB checkpoint is loaded with `mmap=True` and the
modules are built standalone, so RAM stays within free-tier limits and `xformers` is not
required. **R7's automated checks are an aid, not the reserved human attestation H1.**

## Phase 2 pilot (2a + 2b in one notebook)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grewalsk/p1-vjepa2-predictor/blob/main/P1_phase2_colab.ipynb)

`P1_phase2_colab.ipynb` hooks all 24 predictor blocks during an action sweep (Meta's own
example only reads the final output). It runs top to bottom on an L4 GPU and downloads the
checkpoint to `/content` once per session (free-tier Drive cannot hold 11.76 GB).

- **Phase 2a** (one real Franka scene, bundled with the repo): validates the internal-hook
  pipeline on real weights and sketches the depth/rank profile. Reports, per block, the
  residual variance the action drives on patch tokens, the action-vs-state-swap ratio, and the
  effective rank of the action-induced difference vectors.
- **Phase 2b** (a small DROID sample): the statistical pilot. Many real scenes, the empirical-
  distribution action null, a state-swap control, a conservative matched-norm null, bootstrap
  CIs, the frozen-gate readout, and the power calculation. The DROID loader prints its schema
  first so field-name mismatches are caught cleanly.

`P1_phase2a_colab.ipynb` is the standalone 2a notebook, kept for reference.

## Files

| file | what |
|---|---|
| `PRE_REGISTRATION.md` | Frozen Phase-0 record: mandate, theory-object analysis, prior art, gate criteria, confound table, verified site table, Arm A design amendments. |
| `p1_lib.py` | Infrastructure: site discovery, hooks, subspace ablation, geometry (participation ratio / effective rank), transition-operator spectrum, AC token layout, on-manifold nulls, CEM planning, occlusion stimulus, gate assertions. |
| `test_p1_lib.py` | 57 analytic unit tests (each recovers a known ground truth). Run: `python3 test_p1_lib.py`. |
| `P1_phase1_colab.ipynb` | The Colab gate notebook above (embeds `p1_lib.py` verbatim). |

## Key finding that shaped the design

Cui et al. (arXiv 2606.27014) claim low rank of the **transition operator** `M-bar(a)` (the
action-conditioned co-occurrence kernel over encoder embeddings), **not** of how the action
modulates the predictor's residual stream. These are different objects, so residual-stream rank
(Arm A) cannot confirm or refute the theory. The hypothesis is split accordingly:
`H1a-routing` (predictor residual stream) and `H1a-transition` (encoder-embedding `M-bar(a)`
spectrum, the only object that speaks to Cui). See `PRE_REGISTRATION.md` section 2.

## Style

No em dashes anywhere, per protocol.
