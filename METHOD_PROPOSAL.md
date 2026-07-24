# Method proposal: world-model Jacobian-field spectroscopy (2026-07-23)

A proposed general method for characterizing and using learned world models, motivated by
project P1 (mechanistic analysis of the V-JEPA 2-AC predictor). Style: no em dashes.

## The motivating pattern
Every confound P1 hit was the same shape: grand-mean centering leaked spatial footprint; raw
variance leaked residual-norm growth; "residual-stream rank" was not Cui's k; "linear in a"
separated no dynamical formalism. In each case the fix was to stop measuring the coordinate and
measure the INVARIANT. That repetition indicates what the right method is.

## The method, in three steps

1. A world model's content is its DIFFERENTIAL, not its representation. It exists to predict
   change; change is the derivative. The Jacobian field z -> (F'(z), G(z)) with F'=dP/dz,
   G=dP/da fully characterizes the local dynamics. Planning (MPC/CEM/gradient-on-energy) is
   derivative-driven, so the Jacobian field is the object the model exists to get right.

2. The Jacobian-field functionals are simultaneously the things we care about AND the things
   that survive identifiability. LeJEPA/Klindt (2605.26379): the latent is recoverable only up
   to a linear/orthogonal Q, so representation-decodability is Q-dependent nuisance. But rank G,
   singular values sigma(G), subspace principal angles, Lie-algebra rank, and integrability are
   Q-invariant. They are the maximal class of measurements that are at once about the dynamics
   and properties of the WORLD, not the coordinatization. That coincidence is why the method
   works where representation interpretability stalls.
   Invariants: controllable dimension = rank G(z); control gains = sigma(G(z)); LTI vs
   state-dependent = dG/dz; stability / expressiveness = spectrum F'(z); planning conditioning
   = cond(G^T G); controllability = Lie algebra of {g_i, [F,g_i], ...}.

3. The geometry of the Jacobian field unifies the program and yields usable levers.
   - Planability vs expressiveness is a Jacobian-field statement: "simple to plan through,
     expressive to be accurate" = control structure G(z) simple (low-rank, smooth, low-D
     state-dependence) while drift F(z) carries the nonlinearity. Read off separately.
   - Persistence and objects fall out of the same object. The control distribution
     D(z)=range G(z); its annihilator intersected with the drift-invariant subspace is what no
     action can change and time preserves = conserved quantities = mechanistic object
     permanence. Integrability of D (Frobenius/Lie brackets) => the latent factorizes into
     controllable leaves = object-centric structure. So Arm D (persistence) and Arm C
     (separability) are the co-kernel and integrability of the same field, readable without
     occlusion stimuli.
   - Trust region: an action off range G(z) is the model leaving its learned dynamics.

Result: one differential object, a fixed menu of identifiability-invariant functionals, and
from them planability, controllability, disentanglement, persistence, and trust regions, all by
autodiff on any differentiable world model. P1's chosen G(z) state-geometry experiment
(PRE_REGISTRATION.md s12) is the first rung of this ladder.

## Honest limits
- Locality: single-step Jacobians miss horizon/compounding; need Jacobians along trajectories
  and the composed multi-step Jacobian; the drift carrying state into range(G) may dominate
  cond(G) for horizon-H success.
- Second-order cost/power: dG/dz and Lie brackets need second derivatives and many states; the
  integrability/persistence claims are least identifiable at ~12 states.
- Gaussian precondition: full spectral invariance needs near-Gaussian latents; else only rank
  is invariant.
- Restated control theory risk: the novelty must rest on (i) an unconstrained pretrained JEPA
  predictor, (ii) the identifiability-invariance framing, (iii) persistence-as-annihilator
  unification, not on applying Koopman/nonlinear-control machinery per se.

---

## Judgment (2026-07-23): PARTIALLY SOUND, leans FLAWED as pitched. Invert the hierarchy.

A LeCun-caliber review (verified against the literature) upheld the epistemics (study the
differential, report invariants) but refuted the specific object elevated (the pointwise
Jacobian SPECTRUM) and every single-step-as-horizon-proxy readout. Own the corrections.

**Transformation-law ledger (the piece the proposal omitted and needed).** Under z->Qz:
G->QG, F'->QF'Q^-1; under a->Ra: G->GR^-1.
- rank G: invariant under any GL (both sides). eig-spectrum F': invariant under any GL
  (similarity). Lie-algebra rank / involutivity: invariant under any DIFFEOMORPHISM. These are
  the robust readouts and are valid on V-JEPA 2 regardless of its (uncertified) nuisance group.
- sigma(G), cond(G^T G), principal angles: invariant under O(d) ONLY, and only with the LeJEPA
  precondition (whitening + near-Gaussian latents). V-JEPA 2 is NOT LeJEPA-trained; its latent
  carries no such guarantee, so these are plausibly pure coordinate nuisance. Listing sigma(G)
  alongside rank as "Q-invariant" is the proposal's central overclaim, and it is the SAME
  import-a-theorem-onto-a-model-that-does-not-satisfy-it error PRE_REG s2 was built to avoid.
- CRITICAL for our chosen experiment: "G constant in z" (the Koopman/LTI verdict, statistic C)
  is invariant only under LINEAR reparam. Under a nonlinear diffeomorphism phi, G~(w)=Dphi.G
  varies with w even if G is constant. So C~0 is NOT a coordinate-free Koopman verdict unless
  the latent is pinned up to a linear map. (Now unit-tested: p1_lib.reparam_invariance_report;
  C stays ~0 under orth/GL, blows up under a nonlinear reparam.)

**Persistence = annihilator (1b).** Correct as a control-affine LINEARIZATION; the review proved
sufficiency: a covector field with w^T G(z)=0 and w^T F'(z)=w^T at all reachable z annihilates
the whole accessibility algebra, a genuine conserved uncontrolled quantity. But it presupposes
control-affinity (which our own s12 experiment is testing), degrades to a reachable tangent CONE
(no linear annihilator) off the control-affine limit, and the field yields candidate conserved
subspaces whose SEMANTICS (object vs camera-calibration vs background) still need intervention.
Partial unification, not full; Arm D is not fully reducible to the Jacobian field.

**Planability / cond(G) (1c) -- weakest element, refuted as stated.** "controllable dim = rank G"
is wrong: rank G is the instantaneous input rank (a lower bound); the controllable dimension is
the accessibility-algebra dimension (Chow-Rashevskii: full reachability is possible with rank
G << d everywhere via brackets). Horizon-H planning conditioning is governed by the reachability
Gramian W_H = sum_t Phi_t G G^T Phi_t^T with Phi_t = prod F'(z_tau), NOT by single-state
cond(G^T G). The fix converts single-step spectroscopy into rollout-Gramian + bracket-rank
analysis (still pure autodiff, now multi-step).

**Prior art (2).** The FRAMEWORK is not novel: Koopman-with-control / DMDc (impose linear latent
dynamics), E2C/RCE/PCC/DVBF (locally-linear latent control via (A_t=F', B_t=G) for iLQR),
Riemannian pullback-metric spectroscopy (g(z)=J^T J SVD as coordinate-robust local descriptor),
geometric nonlinear control (accessibility algebra, involutivity, feedback linearization). The
ONLY open element is the forensic diagnostic: which normal form did an UNCONSTRAINED pretrained
JEPA spontaneously converge to (discovered vs imposed), with honest identifiability tiering.
Refs: 2210.07563, 1506.07365, 1710.05373, 1711.08014, 2506.10632, 2606.05045, 2607.19719.

**CORRECTED METHOD (invert the tiers).**
- Tier 1 (diffeomorphism-invariant, report unconditionally): accessibility-algebra rank (true
  controllable dim), involutivity / integrable-foliation dimension, feedback-linearizability,
  eig-spectrum F'. The coordinate-free Koopman/LTI question is NOT "is G constant" but "does the
  field admit a finite-dim Koopman-invariant subspace / is it feedback-linearizable" -- a
  Lie-algebraic, Diff-invariant test the proposal had the machinery for but pointed at the wrong
  target.
- Tier 2 (O(n)-invariant, ONLY after certification): sigma(G), cond, principal angles, gated
  behind a measured proof that the DROID-manifold latent is whitened / near-Gaussian, with the
  action metric declared as a convention.
- Semantics via intervention, not the field alone.

**DECISIVE NEXT TEST (replaces "re-run C"): the nuisance-orbit invariance audit + latent
certification.** On the 12 DROID states compute {rank G, sigma(G), cond, C, involutivity defect};
recompute all under random orthogonal Q, random GL Q, one nonlinear diffeomorphism, and a
non-orthogonal action rescale; report which survive (p1_lib.reparam_invariance_report). In
parallel measure whether the V-JEPA 2 latent on the DROID manifold is whitened + near-Gaussian
(covariance spectrum + normality statistic); if a second seed exists, whether two seeds relate by
orthogonal vs GL (gold standard for the true nuisance group). KILLS the spectral tier if
sigma/cond/C move under GL/nonlinear and normality fails (near-certain on an unwhitened V-JEPA2);
then only rank + Lie-rank survive. LICENSES it if the latent certifies near-Gaussian and readouts
are O(n)-stable.

**Sharpest insight the proposal missed:** invariance and diagnostic power TRADE OFF along the
reparam group, and V-JEPA 2 sits at the wrong end. Diff-invariant readouts (rank, Lie-rank,
involutivity, feedback-linearizability) are coordinate-free but coarse; O(n)-only readouts
(sigma, cond) are sharp but coordinate-fragile. The proposal led with the sharp-fragile ones on a
model whose frame is uncertified. Fix: ask the Diff-invariant feedback-linearizability question,
not the linear-only "is G constant" -- that single reframing rescues the discovered-vs-imposed
novelty from the identifiability objection that otherwise sinks it.

**Calibration:** grand "general method" version P ~ 0.10 (repackaged, fragile-invariant headline);
narrow forensic diagnostic (which normal form, discovered vs imposed) P ~ 0.4, conditional on the
Gaussianity certification landing favorably. Two load-bearing assumptions: (1) the operative
nuisance group is O(n) (latent whitened/near-Gaussian); (2) the predictor is approximately
control-affine AND the targets are single-step-local. Both are now things to MEASURE, not assume.
