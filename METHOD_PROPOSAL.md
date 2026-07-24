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
