# P1 Pre-Registration (Phase 0, frozen 2026-07-23)

Mechanistic account of the V-JEPA 2-AC predictor, with emphasis on action routing.
Style: no em dashes. Every literature claim below was verified against the source
(arXiv IDs given), not paraphrased from the protocol.

---

## 1. Mandate (three sentences)

P1 tests what the action-conditioned V-JEPA 2 *predictor* computes and, specifically,
whether action conditioning is routed through a low-rank, spatially localized, and
action-separable structure, with a stretch test for an object-permanence mechanism across
occlusion. We target the predictor rather than the encoder because the predictor is the
module that actually constitutes the world model (it carries the dynamics used for
planning), yet every published JEPA analysis probes the encoder because that is what
downstream tasks consume, leaving the predictor essentially uncharacterized. The encoder
already has a known layerwise action-relevance curve (arXiv 2606.07687); our contribution
is the predictor analogue, which architecture predicts should have a different shape.

---

## 2. What object does the generalization theory claim is low-rank?

**Answer: the transition operator, not the residual-stream modulation.** Verified in
Cui et al., "A Generalization Theory for JEPA-Based World Models," arXiv **2606.27014**:

- **Eq. 5**: the action-conditioned co-occurrence matrix `M(a) := (w(x,x+,a))_{x,x+}`
  with entries `w(x,x+,a) = P(x, x+ | a)`, the joint probability of (current
  observation `x`, next observation `x+`) given action `a`. Rows/columns index
  observations; the action enters **as a conditioning index** (a separate matrix `M(a)`
  per action), not additively or multiplicatively.
- **Thm 3.1 / Eq. 6**: `R_JEPA(f,g,a) = || M-bar(a) - G(F,a)^T F ||^2 + const`. The JEPA
  objective is a low-rank factorization of the *normalized co-occurrence* `M-bar(a)`.
- **Thm 4.3**: `R_S-JEPA(f*,g*,a) = sum_{i>k} sigma_i^2(a)`, the tail singular-value sum
  of `M-bar(a)`. The low-rank claim is a **spectral property of the co-occurrence
  (transition) kernel**, where `k` is the **encoder latent dimension**.
- The paper never mentions residual streams, attention, tokens, transformers, participation
  ratio, or effective rank. It says nothing about predictor internals.

**Consequence for Arm A (this is the pivot of Phase 0).** H1a as operationalized measures
*action-induced residual-stream modulation* inside the width-1024 predictor. That is a
different object from `M-bar(a)`. Two facts make the distinction airtight:

1. Object mismatch: Cui's `k` bounds `rank(M-bar(a))` over the encoder state space; Arm A's
   effective rank lives in the predictor's residual stream across 24 blocks. The two ranks
   are not the same measurable and their numbers are not comparable.
2. Logical independence: even Cui's output-side constraint `B := G(F,a)^T F` has
   `rank <= k` does **not** imply a low-rank internal pathway; a full-rank nonlinear
   intermediate computation can realize a rank-`<=k` output map.

**Therefore a full-rank Arm A residual result cannot refute Cui, and a low-PR result cannot
confirm it.** To speak to the theory we split the hypothesis and add a second, separate
measurement:

- **H1a-routing (object b, network mechanism):** action-induced residual-stream modulation
  in the predictor is low-rank. This is Arm A. It stands on its own; it does NOT bear on Cui.
- **H1a-transition (object a, theory):** the empirical action-conditioned co-occurrence
  `M-bar(a)` built from **encoder embeddings** has fast singular-value tail decay at `k`.
  This is the only measurement that can populate outcome cell 5 (refute/confirm the theory).
  Implemented by `p1_lib.transition_operator_spectrum` on encoder embeddings, NOT on
  predictor activations.

This split is the guard against the "theory strawmanning" anti-pattern (Section 10) and the
explicit warning in outcome cell 5.

---

## 3. Prior art that reduces novelty, and what remains open

**Closest target-adjacent (reduces the "layerwise action structure" novelty):**
"What Makes Video World Model Latents Action-Relevant: Prediction over Reconstruction,"
arXiv **2606.07687**. It probes each of the 24 V-JEPA 2 blocks for action-relevant
structure and reports a frozen-encoder peak near block 14 that declines toward the final
layers, reversed by inverse-dynamics fine-tuning. **It is encoder-side**: it probes
mean-pooled frozen ViT-L features, discards the inverse-dynamics head at probe time, uses
per-layer linear probing (no activation patching), and never touches the predictor, the
action token, low-rank routing, or object permanence. **Open after it:** the entire
predictor analogue, i.e. what the AC predictor computes and how the action token is routed.

**Strongest method-level reducer (reduces methodological novelty):**
"Attention Gathers, MLPs Compose," arXiv **2603.11142**. It performs activation patching +
ablation to reverse-engineer an action-outcome circuit in a video transformer (attention
heads gather evidence, MLP blocks compose concepts, layers ~5-11). **But the model is a
supervised ViViT Kinetics-400 classifier** (google/vivit-b-16x2-kinetics400), not a JEPA
predictor: no action-as-separate-token injection, no world model, no low-rank / effective
rank / participation-ratio analysis, no object permanence. **Open after it:** the same
causal-circuit method applied to a JEPA action-conditioned *predictor*, plus the low-rank
action-routing lens and the persistence circuit. **P1 must cite 2603.11142 and differentiate
on architecture (JEPA AC predictor vs supervised classifier), target (predictor vs
encoder), and the low-rank routing analysis.** Effective rank as a world-model metric is
itself not novel (e.g. Sub-JEPA, arXiv 2605.09241), so P1's novelty must rest specifically
on the action-difference-vector covariance in the predictor residual stream and on the
persistence circuit.

**Other verified context:** V-JEPA 2 / 2-AC (arXiv 2506.09985; 24-layer, 16-head,
width-1024, ~300M predictor; action injected as a separate token via a linear input head;
block-causal chronological sequence; deterministic single next-latent per (state, action)).
Delta-JEPA (arXiv 2606.31232; reconstruction-free JEPA can collapse to action-insensitive
representations, so a genuinely low/insensitive action pathway is a live possibility that
Arm A must distinguish from artifact). Klindt/LeCun/Balestriero (arXiv 2605.26379; linear
identifiability iff isotropic Gaussian latents, which constrains what "the predictor
represents X" can mean). Causal-JEPA (arXiv 2602.11389; object-level latent interventions,
the closest existing intervention-based JEPA analysis). V-JEPA 2.1 (arXiv 2603.14482;
denser temporally consistent features, +20pt real-robot grasping over 2-AC).

---

## 4. Frozen gate criteria (Section 8)

```
p1.action_effect_over_baseline_min_ratio  : 3.0
p1.lowrank_max_effective_rank_frac        : 0.15     # of d = 1024  ->  <= 154 effective dims
p1.localization_blocks_for_80pct          : 12       # H1b: reach 80% within < 12 of 24 blocks
p1.separability_z_vs_random_rotation      : 2.0
p1.persistence_patch_recovery_min         : 0.50
p1.random_subspace_control_margin         : 2.0      # Arm B: action-subspace effect >= 2x random-subspace, non-overlapping 95% CIs
p1.min_seeds                              : 5
```

Justifications (one sentence each):

- **3.0**: a 3x separation of the action effect from the on-manifold null puts it well
  outside the null's bootstrap spread at 5 seeds and is the conventional bar for "clearly
  above baseline" in effect-vs-null probing.
- **0.15 (<= 154 of 1024)**: an order of magnitude below full width is the "clearly
  low-rank" regime while sitting comfortably above the input-head floor `d_a ~ 7`, so a pass
  means the pathway is genuinely compressed *after* mixing rather than trivially capped;
  primary metric is the participation-ratio effective count, corroborated by the estimator
  panel (`rank_report`).
- **12 (H1b < n/2)**: "localized" must mean a minority of the 24 blocks; 80% cumulative
  action effect within fewer than 12 blocks is the natural minority threshold, reported both
  as rank-ordered concentration and smallest contiguous window.
- **2.0 (separability z)**: a 2-sigma separation of the mean pairwise principal-angle spread
  from the random-rotation null is the standard "distinguishable from chance" bar and is
  exactly what Arm C's null exists to test against (never against zero).
- **0.50 (persistence recovery)**: patching the candidate component from an unoccluded into
  an occluded run must restore at least half of the occluded-vs-visible consistency gap to
  count as "the mechanism," a bar that is neither trivial (>0) nor demands full restoration.
- **2.0 (Arm B control margin)**: subspace ablation always hurts somewhat, so the action
  subspace must degrade the causal readout at least 2x more than a same-dimension random
  subspace with non-overlapping 95% bootstrap CIs, isolating the pathway from the generic
  cost of deleting directions.
- **5 seeds**: fixed by the protocol.

These are frozen. If one later looks wrong, add a parallel dated criterion and report both;
there is no retraction (Section 8).

---

## 5. Design amendments to Arm A (dated 2026-07-23, from the Phase-0 adversarial review)

The Section-5 Arm A spec, taken literally, largely measures the linear action input head
and the single fixed state rather than the action pathway. These amendments are frozen into
the operationalization before any run. The gate numbers above are unchanged; only the
*measurement procedure* is tightened. Rationale: the review (three independent skeptics)
returned a partial refutation of Arm A as written; the theory-object and novelty claims
survived unchanged.

1. **On-manifold null (replaces "random vector of matched norm").** A generic norm-matched
   R^1024 residual perturbation is off the action-embedding manifold, under-propagates
   through attention/MLP trained to read the thin action subspace, and thereby *inflates*
   real/null. The null is instead **random raw actions of matched norm passed through the
   real action head** (`random_action_null`) and, as the stronger variant, **permuted real
   actions** (`permuted_action_null`). Fixes confounds B2, B3, B5, B8.
2. **Fixed readout token set.** Read only patch/proprio tokens **causally downstream** of the
   action token, in **deep (post-mixing) blocks**, with the **action-token position
   excluded**, identical mask for real and null, and the mask verified empirically
   (perturbing the action token must leave past-frame tokens unchanged). Report per
   future-offset. Fixes B1, B4, and the A1 exposure at shallow sites.
3. **Multi-state pooling.** Repeat over a diverse bank of real latent states; report
   within-state vs across-state effective rank. Single-state rank is only a lower bound on
   the pathway rank because Cui's operator `G(F,a)` is state-varying. Fixes A5.
4. **Dense on-distribution action sampling (`dense_action_sample`, N >> 1024) over the full
   valid action box (translation + rotation + gripper), in addition to the designed basis.**
   The designed basis caps observable rank at its cardinality; report PR vs N and confirm it
   plateaus well below the sample cap. Fixes A4.
5. **Estimator panel, not PR alone (`rank_report`).** Report participation-ratio effective
   count, entropy effective rank, stable rank, numerical rank, and count above the
   on-manifold-null floor with bootstrap CIs and a permutation test. PR is low-biased (one
   dominant eigenvalue drives it to ~1). Fixes A7, B8.
6. **Centered and uncentered.** Report both the centered covariance and the uncentered
   second moment `E[dd^T]` (a rank-1 mean action effect is part of the pathway), and read
   the raw additive pre-norm residual, not post-LayerNorm. Fixes A3.
7. **Jacobian check (`jacobian_spectrum`).** Complement the finite-difference read with the
   autodiff Jacobian `d(residual)/d(action-embedding)` at several operating points; finite
   differences locally linearize GELU and can cancel multiplicative action*state directions.
   Fixes A6.
8. **`d_a` as the explicit null ceiling.** The pass condition for "not merely input-head
   capped" is that deep-token effective rank **exceeds `d_a`** while still meeting the 0.15
   fraction; a pathway pinned at `d_a` everywhere is low-rank but trivially so, and must be
   reported as such rather than as a discovered mechanism.
9. **Inferential scope.** Arm A speaks only to H1a-routing. H1a-transition (Cui) is tested
   separately via the encoder-embedding `M-bar(a)` spectrum. The attention-write vs MLP-write
   split (competing explanation from 2603.11142) is reported for any low-PR result.

---

## 6. Confound list (Section 13 step 5)

Each entry: how it could produce the predicted result without the hypothesis, and whether
the design ELIMINATES, MITIGATES, or ACKNOWLEDGES-ONLY it.

**Arm A / H1a-routing (apparent low rank without a genuinely low-rank pathway):**

- Input-head rank cap (A1): action lifted linearly from `d_a ~ 7` to 1024, so PR <= d_a at
  shallow/action-token sites by construction. **MITIGATED** by deep-token readout and the
  `d_a` ceiling test (amendments 2, 8).
- Pre-mixing readout (A2): block-causal attention means shallow patch tokens have ~0 action
  content, looking low-rank and low-magnitude. **MITIGATED** by reading at/after the
  empirically located mixing-onset depth (amendment 2).
- Centering / post-norm artifacts (A3): removes the mean action direction or the radial
  magnitude. **MITIGATED** by reporting centered and uncentered, pre-norm (amendment 6).
- Designed-basis ceiling (A4): k designed actions span <= k dims. **MITIGATED** by dense
  N >> 1024 sampling with a PR-vs-N plateau check (amendment 4).
- Fixed-state collapse (A5): single-state rank is a lower bound. **MITIGATED** by multi-state
  pooling (amendment 3).
- Finite-difference cancellation (A6): GELU linearization cancels multiplicative directions.
  **MITIGATED** by the Jacobian read (amendment 7).
- PR low bias (A7): one big eigenvalue forces PR ~ 1. **MITIGATED** by the estimator panel
  (amendment 5).
- MLP-composition alternative (2603.11142): a low PR could reflect MLP concept composition,
  not a dedicated action subspace. **MITIGATED** by the attention-write vs MLP-write split
  (amendment 9).
- Action-insensitive collapse (Delta-JEPA 2606.31232): the pathway may be genuinely
  low/insensitive for training reasons, muting Arm A independent of H1a. **ACKNOWLEDGED**;
  distinguished from artifact by the amendments but reported as a live alternative reading.

**Arm A (inflated real/null without a genuine effect):**

- Action-token readout (B1): trivially large variance at that position. **ELIMINATED** by
  excluding it (`exclude_positions`).
- Off-manifold null under-propagation (B2/B3/B5): deflates the null. **ELIMINATED** by the
  on-manifold null (amendment 1).
- Variance-vs-rank conflation (B7): "real/null ratio" is ambiguous. **ELIMINATED** by
  reporting the magnitude ratio and the rank ratio separately.
- Circular noise floor (B8): null used as floor while artificially suppressed. **MITIGATED**
  by on-manifold null + bootstrap CIs + permutation test.
- Encoder conditioning (2606.07687): action geometry is already shaped in the frozen encoder
  that feeds the predictor; the null controls action-token injection but not the pre-existing
  encoder geometry. **ACKNOWLEDGED-ONLY** as a stated limitation.

**Theory (H1a-transition / Cui):**

- Object substitution (theory strawmanning): testing residual rank and calling it Cui.
  **ELIMINATED** by the H1a-routing / H1a-transition split and by measuring `M-bar(a)` on
  encoder embeddings.

**Arm B (causal localization):**

- Generic ablation cost: any subspace ablation hurts. **ELIMINATED** by the matched-dimension
  random-subspace control and the 2x margin with non-overlapping CIs.
- Site drift: comparing block indices across runs that are not the same tensor.
  **MITIGATED** by the frozen site table and by hooking canonical named sites only.
- Off-distribution ablation inflation: zero ablation overstates importance. **ELIMINATED** by
  using subspace projection (`SubspaceAblator`), never zero ablation, as primary.

**Arm C (separability):**

- Comparison against zero rather than chance: random subspaces are never exactly aligned.
  **ELIMINATED** by the random-rotation null.
- Small-angle numerical noise: `arccos` is ill-conditioned near 0. **MITIGATED**; Arm C tests
  large-angle separability where `arccos` is well-conditioned, and near-0 cases are flagged.

**Arm D (persistence):**

- Motion-extrapolation vs persistence: a model could extrapolate linear motion without any
  identity-maintenance mechanism. **MITIGATED** by localizing via patching (a generic
  extrapolator has no single component whose patch restores occluded-object consistency).
- Task at ceiling/floor: no intervention can register an effect. **ACKNOWLEDGED**; kill
  criterion recalibrates difficulty before interpreting.
- Budget overrun: **ACKNOWLEDGED**; hard 30%-compute cap, then ship cell 2 or 3.

---

## 7. Outcome cells 5 and 7 confirmation (can we write them?)

- **Cell 5 (full rank, refutes the theory):** writable ONLY via H1a-transition (the encoder
  `M-bar(a)` spectrum), never via Arm A alone. If `sum_{i>k} sigma_i^2(a)` fails to decay at
  the encoder latent dimension across seeds and actions, that directly contradicts Cui's
  low-rank prediction. Before publishing, we will have verified we measured `M-bar(a)` and
  not the residual stream.
- **Cell 7 (no action effect above the on-manifold null after site verification):** writable.
  Trigger is a real/null magnitude ratio below 3.0 at every predictor block after confirming
  the action token, the readout mask, and the mixing-onset depth. If it survives
  verification, report to the checkpoint authors before publishing.

Prior to record for calibration: P(strong positive) ~ 0.30, P(informative negative) ~ 0.50,
P(pipeline/null) ~ 0.20.

Reserved human attestations H1/H2/H3 are intentionally left blank.
```

---

## 8. Verified site table and forced design updates (2026-07-23)

Source-verified against `facebookresearch/vjepa2` at HEAD (`src/models/ac_predictor.py`,
`src/models/utils/modules.py`, `src/hub/backbones.py`). Architecture, naming, shapes, token
layout, and causal reach are exact because the code fixes them. **Provenance caveat: this is
from the model DEFINITION, not from loaded weights. R7 (real checkpoint) is NOT discharged.**

**Loader reality.** `torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")`
returns a **tuple `(encoder, predictor)`**, two separate `nn.Module` trees; there is no
unified module. Weights are on `dl.fbaipublicfiles.com`, not HuggingFace. Blocker (not a
kill criterion): at HEAD `src/hub/backbones.py` sets `VJEPA_BASE_URL = "http://localhost:8300"`
with the real URL commented out; a plain `torch.hub.load` fails with a connection error
until that line is restored or a release tag is pinned.

**Verified predictor (`VisionTransformerPredictorAC`):** depth 24, width 1024, 16 heads;
block class `ACBlock`, attention `ACRoPEAttention` (RoPE), MLP `MLP`; frame-causal, RoPE on,
extrinsics off. Conditioning: `action_encoder = Linear(7->1024)`, `state_encoder =
Linear(7->1024)`, `extrinsics_encoder = Linear(6->1024)` (unused); `predictor_embed =
Linear(1408->1024)`, `predictor_proj = Linear(1024->1408)` (patch-only output).
**Encoder:** `vit_giant_xformers`, depth 40, width 1408. `discover_sites(predictor,'prd')` +
`discover_sites(encoder,'enc')` yields 78 sites (blocks 0..23 x {resid_post, attn_out,
mlp_out} plus the conditioning modules).

**discover_sites diagnosis corrected.** The Section-6 version returned zero sites due to
**path parsing**, not class naming (which was fine): the block regex needed a literal
`.blocks.` but the real path is `predictor_blocks.0`, and the family heuristic scanned the
path for `encoder`/`trunk` but the standalone encoder's paths are bare `blocks.0.attn`. Fixed
with `_BLOCK_RE` and an explicit `family` argument.

**Forced Arm A design updates (frozen; gate numbers unchanged):**

1. **Mechanism is attention-mediated, not additive.** The AC predictor interleaves the action
   as a dedicated token: `x = cat([action, state, patches], dim=2).flatten(1,2)`; per frame
   `[action, state, patch_0..patch_{HW-1}]`, stride `HW+K` (`K=2`; at 256px/patch16, HW=256,
   stride 258). The H1a-routing object is therefore the **action-token -> patch-token
   attention-mediated update**, read on **patch tokens only** (`patch_positions`), with the
   per-frame conditioning tokens excluded (`cond_positions`). The routing/transition split
   from Section 2 survives unchanged. For any low-rank finding, the attention-write vs
   MLP-write decomposition (cf. 2603.11142) is reported.
2. **Two conditioning streams, so a state-swap control is mandatory.** `state_encoder` is also
   `Linear(7->1024)` and enters as its own token. Arm A holds BOTH the patches AND the state
   token fixed while sweeping only the action token, and adds a **state-swap null (vary state,
   hold action)** as a second control; if action-attributable variance is not clearly above
   state-attributable variance, the effect is generic conditioning-token routing, not
   action-specific.
3. **Primary null is empirical-distribution action resampling** (`empirical_action_null`):
   resampled REAL actions embedded through the real `action_encoder`. Because the action
   enters through a learned 7->1024 encoder, the encoder-output manifold is what matters, so
   resampled-real actions are the only guaranteed on-manifold null. `permuted_action_null` is
   the paired variant; `random_action_null` / `matched_norm_null` are downgraded to
   sensitivity checks only.
4. **Full causal history + a hard Phase-1 gate.** Attention is frame-causal over the entire
   past (`local_window_time = T`; frame `t1` attends to all `t2 <= t1`). `assert_frame_causal`
   is added to the Phase-1 exit gate: perturbing the action at frame `t` must leave every
   token in frames `< t` EXACTLY unchanged (verified passing on the source model:
   per-frame max-delta `[0, 0, 4.69e-3, 4.75e-3]` for a frame-2 perturbation). This one
   forward pass catches hook misplacement, wrong token indexing, and mask misuse.

**R7 / kill-criterion status.** Structural verification passes; the AC predictor exists, is
separable from the encoder (they are literally separate trees), and carries the action
pathway, so the Section-9 kill criterion is NOT triggered. But no claim may be made until a
real `state_dict` is loaded and `assert_real_checkpoint(encoder, predictor)` passes; that
automated check is an aid, not a substitute for the reserved human attestation H1.
