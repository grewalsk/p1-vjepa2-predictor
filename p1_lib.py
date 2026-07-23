"""p1_lib.py -- infrastructure for P1 (mechanistic analysis of the V-JEPA 2-AC predictor).

Sections:
  [SPEC]  the Section 6 library, reproduced verbatim (site discovery, hooks, geometry,
          subspace ablation, CEM planning, occlusion stimulus).
  [P1ADD] Phase-1 additions used by Arms A/B and by the Cui-theory (transition-operator)
          measurement. Each addition has an analytic unit test in test_p1_lib.py.

Style: no em dashes anywhere (per protocol).
"""
import contextlib, re
import numpy as np, torch, torch.nn as nn

# =====================================================================================
# [SPEC] Section 6, reproduced verbatim
# =====================================================================================

# ---------- site discovery (AMENDED 2026-07-23 after source verification) ----------
# The Section-6 single-arg discover_sites(model) returns ZERO sites on the real AC model,
# for two verified path-parsing reasons (NOT class naming, which was fine):
#   (a) the block regex required a literal ".blocks." but the real path is
#       "predictor_blocks.0" (no dot before "blocks");
#   (b) the family heuristic scanned the path for "encoder"/"trunk", but the standalone
#       encoder tree's own paths are just "blocks.0.attn".
# Also: torch.hub.load("facebookresearch/vjepa2","vjepa2_ac_vit_giant") returns a TUPLE
# (encoder, predictor) -- two separate trees -- so family MUST be passed explicitly.
_BLOCK_RE = re.compile(r"(?:^|\.)(?:predictor_blocks|blocks|layers|layer)\.(\d+)(?:\.|$)")
_COND_MODULES = ("action_encoder", "state_encoder", "extrinsics_encoder",
                 "predictor_embed", "predictor_proj", "predictor_norm")

def discover_sites(module, family):
    """Verified for facebookresearch/vjepa2 AC (VisionTransformerPredictorAC) at HEAD.
    `family` is EXPLICIT ('enc'|'prd') because the loader hands back two separate trees and
    neither path string identifies itself. Returns {name: (path, family, blk, kind)} with
    canonical names {fam}.blk{j}.{kind} plus non-block conditioning modules {fam}.{name}
    (blk = -1, kind = 'cond'). Call as discover_sites(predictor,'prd') and
    discover_sites(encoder,'enc'), then merge."""
    assert family in ("enc", "prd")
    sites = {}
    for path, mod in module.named_modules():
        m = _BLOCK_RE.search(path)
        if m is None: continue
        blk, cls = int(m.group(1)), type(mod).__name__.lower()
        kind = ("attn_out" if ("attention" in cls or cls.endswith("attn")) else
                "mlp_out"  if ("mlp" in cls or "swiglu" in cls or "feedforward" in cls) else
                "resid_post" if ("block" in cls or "layer" in cls) else None)
        if kind is None: continue
        sites.setdefault(f"{family}.blk{blk}.{kind}", (path, family, blk, kind))
    for path, mod in module.named_modules():
        if path in _COND_MODULES:
            sites.setdefault(f"{family}.{path}", (path, family, -1, "cond"))
    if not sites:
        raise RuntimeError(f"no sites for family={family}; print(module) and update _BLOCK_RE")
    return sites

def n_blocks(sites, fam):
    b = {v[2] for v in sites.values() if v[1]==fam}
    return max(b)+1 if b else 0

def get_module(model, dotted):
    m = model
    for p in dotted.split("."):
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m

def find_action_pathway(model):
    return [p for p,_ in model.named_modules()
            if any(k in p.lower() for k in ("action","act_embed","cond"))]

# ---------- hooks ----------
def _tensor(o):
    if isinstance(o, torch.Tensor): return o
    if isinstance(o,(tuple,list)) and o: return _tensor(o[0])
    if hasattr(o,"last_hidden_state"): return o.last_hidden_state
    raise TypeError(type(o))

def _splice(o, new):
    if isinstance(o, torch.Tensor): return new
    if isinstance(o, tuple): return (new,)+tuple(o[1:])
    if isinstance(o, list):  return [new]+list(o[1:])
    o.last_hidden_state = new; return o

class Cache(contextlib.AbstractContextManager):
    def __init__(self, model, sites, names, to_cpu=True):
        self.model, self.sites, self.names, self.to_cpu = model, sites, names, to_cpu
        self.store, self._h = {}, []
    def __enter__(self):
        for n in self.names:
            def hk(_m,_i,o,n=n):
                t = _tensor(o).detach()
                self.store[n] = t.cpu() if self.to_cpu else t
                return o
            self._h.append(get_module(self.model, self.sites[n][0]).register_forward_hook(hk))
        return self
    def __exit__(self,*e):
        [h.remove() for h in self._h]; self._h.clear()
    def __getitem__(self,k): return self.store[k]

class SubspaceAblator(contextlib.AbstractContextManager):
    """Project OUT span(U) at each site: t <- t - U U^T t.
    Cleaner than whole-block ablation. NEVER use zero ablation as primary;
    it is off-distribution and reliably overstates importance."""
    def __init__(self, model, sites, bases):   # bases: {site: (d,k) orthonormal}
        self.model, self.sites, self.bases, self._h = model, sites, bases, []
    def __enter__(self):
        for n,U in self.bases.items():
            def hk(_m,_i,o,U=U):
                t = _tensor(o); Ud = U.to(t.device, t.dtype)
                return _splice(o, t - torch.einsum("...d,dk,ek->...e", t, Ud, Ud))
            self._h.append(get_module(self.model, self.sites[n][0]).register_forward_hook(hk))
        return self
    def __exit__(self,*e):
        [h.remove() for h in self._h]; self._h.clear()

# ---------- geometry ----------
def spectrum(x, center=True):
    a = x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    a = a.reshape(-1, a.shape[-1]).astype(np.float64)
    if center: a = a - a.mean(0, keepdims=True)
    s = np.linalg.svd(a, compute_uv=False)
    return np.clip(s**2 / max(a.shape[0]-1,1), 0, None)

def participation_ratio(lam):
    """Y2 = sum p_i^2, p_i = lam_i/sum lam. Y2->1 is condensation onto one
    direction; Y2 ~ 1/d is isotropic. Report 1/Y2 as an effective count."""
    lam = np.asarray(lam,float); t = lam.sum()
    return 1.0 if t<=0 else float(((lam/t)**2).sum())

def effective_rank(lam):
    lam = np.asarray(lam,float); t = lam.sum()
    if t<=0: return 1.0
    p = lam/t; p = p[p>0]
    return float(np.exp(-(p*np.log(p)).sum()))

def subspace_from_diffs(diffs, k=8):
    """Top-k principal directions of action-induced differences. (d,k) orthonormal."""
    A = diffs.reshape(-1, diffs.shape[-1]).float()
    A = A - A.mean(0, keepdim=True)
    _,_,Vh = torch.linalg.svd(A, full_matrices=False)
    return Vh[:k].T.contiguous()

def random_subspace(d, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q,_ = torch.linalg.qr(torch.randn(d,k,generator=g)); return Q

def principal_angles(U, V):
    return torch.arccos(torch.linalg.svdvals(U.T @ V).clamp(-1,1))

def bootstrap_ci(v, stat=np.mean, n=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(v,float); v = v[~np.isnan(v)]
    b = np.array([stat(rng.choice(v,v.size,replace=True)) for _ in range(n)])
    return float(stat(v)), float(np.percentile(b,100*alpha/2)), float(np.percentile(b,100*(1-alpha/2)))

# ---------- CEM planning (causal readout for Arm B) ----------
@torch.no_grad()
def cem_plan(predictor, z0, z_goal, horizon=8, n=512, elites=64, iters=6,
             action_dim=7, seed=0):
    g = torch.Generator(device=z0.device).manual_seed(seed)
    mu = torch.zeros(horizon, action_dim, device=z0.device)
    sd = torch.ones_like(mu)*0.5; trace=[]
    for _ in range(iters):
        a = (mu + sd*torch.randn((n,horizon,action_dim),generator=g,device=z0.device)).clamp(-1,1)
        z = z0.expand(n,*z0.shape[1:]).clone()
        for t in range(horizon): z = predictor(z, a[:,t])
        cost = (z - z_goal.expand_as(z)).pow(2).flatten(1).sum(1)
        idx = torch.topk(-cost, elites).indices
        w = torch.softmax(-cost[idx], 0).view(elites,1,1)
        mu = (w*a[idx]).sum(0)
        sd = ((w*(a[idx]-mu)**2).sum(0)).sqrt().clamp_min(1e-3)
        trace.append(float(cost.min()))
    return mu, trace

# ---------- occlusion stimulus generator (Arm D) ----------
class OccludedBounce(torch.utils.data.Dataset):
    """Object bounces with known dynamics; occluder hides it for occ_len frames.
    The object KEEPS MOVING behind the occluder, so a model with persistence
    should predict where it reappears."""
    def __init__(self, n=512, T=16, size=224, occ_start=5, occ_len=4, seed=0):
        self.n,self.T,self.S,self.o0,self.ol,self.seed = n,T,size,occ_start,occ_len,seed
    def __len__(self): return self.n
    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed*1000003+i); S=self.S
        yy,xx = np.mgrid[0:S,0:S]/S
        x,y = rng.uniform(.2,.8,2); vx,vy = rng.uniform(-.08,.08,2); r=.10
        frames, traj = [], []
        for t in range(self.T):
            img = np.full((3,S,S), .75, np.float32)
            m = ((np.sqrt((xx-x)**2+(yy-y)**2) < r)).astype(np.float32)
            img = img*(1-m) + np.array([.9,.2,.2],np.float32)[:,None,None]*m
            if self.o0 <= t < self.o0+self.ol:
                img[:, :, int(.35*S):int(.65*S)] = .15
            frames.append(img); traj.append((x,y))
            x,y = x+vx, y+vy
            if not r<x<1-r: vx=-vx; x=float(np.clip(x,r,1-r))
            if not r<y<1-r: vy=-vy; y=float(np.clip(y,r,1-r))
        return {"pixel_values": torch.tensor(np.stack(frames)),
                "traj": torch.tensor(traj, dtype=torch.float32),
                "occluded": list(range(self.o0, min(self.o0+self.ol,self.T))),
                "video_id": i}

# =====================================================================================
# [P1ADD] Phase-1 additions. Each has an analytic unit test.
#
# Design note (the pivot of Phase 0): H1a operationalizes ACTION-INDUCED RESIDUAL-STREAM
# MODULATION -- object (b). Cui et al. 2606.27014 Thm 3.1 claim low rank of the TRANSITION
# OPERATOR M-bar(a) -- object (a). These are different objects. To speak to the theory you
# must call transition_operator_spectrum (object a); to characterize routing you call the
# action_diffs / participation_ratio path (object b). Never report one as the other.
# =====================================================================================

# ---------- Arm A: designed action sweep + matched-norm null ----------
def action_basis(action_dim=7, magnitudes=(0.5, 1.0), sign_flips=True, include_zero=True):
    """Deterministic designed action set for Arm A: signed, scaled canonical basis
    directions. Returns (M, action_dim). Row 0 is the zero action (the reference) when
    include_zero. No randomness here: the randomized comparator is matched_norm_null."""
    rows = []
    if include_zero:
        rows.append(torch.zeros(action_dim))
    signs = (1.0, -1.0) if sign_flips else (1.0,)
    for i in range(action_dim):
        e = torch.zeros(action_dim); e[i] = 1.0
        for m in magnitudes:
            for s in signs:
                rows.append(s * m * e)
    return torch.stack(rows)

def matched_norm_null(actions, seed=0):
    """MANDATORY Arm A baseline. Random-direction vectors whose per-row L2 norm matches
    `actions`. Same shape. Reports are always real/null, never raw (Section 5, Section 10)."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(actions.shape, generator=g)
    rn = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return rn * actions.norm(dim=-1, keepdim=True)

def _keep_tokens(n_tok, exclude_positions):
    ex = set() if exclude_positions is None else set(int(t) for t in exclude_positions)
    return [t for t in range(n_tok) if t not in ex]

def action_induced_variance(acts, exclude_positions=None):
    """acts: (n_actions, n_tokens, d) residual-stream activations at ONE block, at FIXED
    state, one row per swept action. Returns trace of the across-action covariance averaged
    over kept token positions (the Arm A numerator). `exclude_positions` MUST include the
    action-token position(s): variance there is trivial and inflates the effect (confound).
    Report action_induced_variance(real) / action_induced_variance(null), never raw."""
    a = acts.detach().float()
    keep = _keep_tokens(a.shape[1], exclude_positions)
    a = a[:, keep, :]                                   # (n_act, n_keep, d)
    return float(a.var(dim=0, unbiased=True).sum(dim=-1).mean())

def effect_ratio(real, null, eps=1e-12):
    """Arm A / Arm B headline quantity: real over matched null."""
    return float(real / (null + eps))

def action_diffs(acts, ref_index=0, exclude_positions=None):
    """Stack the action-induced DIFFERENCE vectors (each swept action minus the reference
    action, pooled over kept tokens) into (n_act*n_keep, d). Feed to spectrum ->
    participation_ratio / effective_rank for the H1a routing-rank measurement (object b)."""
    a = acts.detach().float()
    keep = _keep_tokens(a.shape[1], exclude_positions)
    a = a[:, keep, :]
    d = a - a[ref_index:ref_index+1]
    return d.reshape(-1, a.shape[-1])

def effective_count(lam):
    """1/Y2, the participation-ratio effective count. Used for the H1a gate
    (effective count as a fraction of d)."""
    return 1.0 / max(participation_ratio(lam), 1e-12)

# ---------- Cui-theory object (a): transition-operator rank ----------
def transition_operator_spectrum(z_ctx, z_pred, center=True):
    """Singular values (descending) of the empirical cross-covariance between context
    embeddings z_ctx (..., d) and predicted-target embeddings z_pred (..., d') at FIXED
    action. This is the object Cui et al. 2606.27014 call low-rank (the transition kernel
    M-bar(a)). Its effective rank -- NOT the residual-stream rank -- is what a refutation of
    the low-rank theory (outcome cell 5) must target."""
    A = (z_ctx.detach().float().cpu().numpy() if torch.is_tensor(z_ctx) else np.asarray(z_ctx))
    B = (z_pred.detach().float().cpu().numpy() if torch.is_tensor(z_pred) else np.asarray(z_pred))
    A = A.reshape(-1, A.shape[-1]).astype(np.float64)
    B = B.reshape(-1, B.shape[-1]).astype(np.float64)
    if center:
        A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
    C = (A.T @ B) / max(A.shape[0]-1, 1)
    return np.linalg.svd(C, compute_uv=False)

# ---------- Arm B: localization ----------
def cumulative_localization(per_block_effect, frac=0.80):
    """per_block_effect: array of (real - null-corrected) action effect per predictor block,
    IN BLOCK ORDER. Returns (cum_blockorder, n_for_frac, contiguous_span) where
      cum_blockorder = cumulative fraction along block order (for the H1b plot),
      n_for_frac     = minimal number of blocks (highest-effect first) reaching `frac`
                       of total effect  --> the H1b gate quantity,
      contiguous_span= size of the smallest contiguous block window holding >= frac
                       (H1b's 'contiguous minority' reading)."""
    e = np.clip(np.asarray(per_block_effect, float), 0, None)
    tot = max(e.sum(), 1e-12)
    cum_blockorder = np.cumsum(e) / tot
    order = np.argsort(-e)
    n_for_frac = int(np.searchsorted(np.cumsum(e[order]) / tot, frac) + 1)
    # smallest contiguous window reaching frac
    nb = len(e); best = nb
    for i in range(nb):
        s = 0.0
        for j in range(i, nb):
            s += e[j]
            if s / tot >= frac:
                best = min(best, j - i + 1); break
    return cum_blockorder, n_for_frac, int(best)

# =====================================================================================
# [P1ADD-v2] Arm A hardening (mandated by the Phase-0 adversarial review).
#
# The Section-5 null ("random vector of matched norm") is OFF the action-embedding
# manifold: passed as a raw residual perturbation it under-propagates through attention/MLP
# trained to read the thin action subspace, which DEFLATES the null and INFLATES real/null.
# The correct nulls are on-manifold: random RAW actions embedded through the SAME action
# input head, or PERMUTED real actions. Both are provided. Also: PR alone is a low-biased
# rank estimator, so report a panel of estimators; and finite differences can cancel
# multiplicative action*state directions, so a Jacobian-SVD read is provided as a check.
# =====================================================================================

def permuted_action_null(actions, seed=0):
    """Strongest Arm A null: a permutation of the REAL swept actions. Preserves the exact
    action-embedding statistics (every row is a genuine action) while breaking the
    action<->state correspondence. Feed these through the real action head, same as `actions`.
    Returns a permuted copy (guaranteed != identity for n>1)."""
    n = actions.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    if n > 1:                                        # avoid the identity permutation
        tries = 0
        while bool((perm == torch.arange(n)).all()) and tries < 16:
            perm = torch.randperm(n, generator=g); tries += 1
    return actions[perm]

def random_action_null(action_dim, ref_norms, seed=0):
    """On-manifold null: random RAW action vectors (in R^{action_dim}) whose per-row norm
    matches `ref_norms`. MUST be embedded through the model's real action input head so it
    traverses the trained pathway (this is what makes it drive-matched, unlike a raw
    R^width residual perturbation). ref_norms: (n,) tensor of target action-space norms."""
    ref_norms = ref_norms.reshape(-1)
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(ref_norms.shape[0], action_dim, generator=g)
    rn = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return rn * ref_norms[:, None]

def dense_action_sample(action_dim, n, low=-1.0, high=1.0, seed=0):
    """Dense on-distribution action sampling for the PR-vs-N plateau check (fixes the
    designed-basis rank ceiling: k designed actions span <= k dims, so PR<=k trivially).
    Report PR vs N; if PR keeps rising with N you are sample-limited, not measuring the
    pathway. Uniform over the valid box [low,high]^action_dim by default."""
    g = torch.Generator().manual_seed(seed)
    return low + (high - low) * torch.rand(n, action_dim, generator=g)

# ---------- rank-estimator panel (report all; never PR alone) ----------
def stable_rank(lam):
    """sum(sigma_i^2)/sigma_max^2 = sum(lam)/max(lam) for eigenvalues lam=sigma^2.
    Less top-eigenvalue-dominated than PR."""
    lam = np.asarray(lam, float); lam = lam[lam > 0]
    return float(lam.sum() / lam.max()) if lam.size and lam.max() > 0 else 0.0

def numerical_rank(lam, rel_thresh=1e-2):
    """Count of eigenvalues above rel_thresh * max eigenvalue."""
    lam = np.asarray(lam, float)
    return int((lam > rel_thresh * (lam.max() if lam.size else 0)).sum())

def count_above_floor(lam, floor):
    """Count of eigenvalues above an absolute floor (e.g. the on-manifold null's spectrum).
    This is the on-manifold-null-referenced rank the low-rank claim should use."""
    return int((np.asarray(lam, float) > float(floor)).sum())

def rank_report(lam, null_floor=None, rel_thresh=1e-2):
    """Full panel for one spectrum. Report ALL of these, not PR alone (PR is low-biased:
    one dominant eigenvalue drives it to ~1)."""
    lam = np.asarray(lam, float)
    r = {
        "participation_effective_count": effective_count(lam),   # 1/Y2
        "entropy_effective_rank": effective_rank(lam),
        "stable_rank": stable_rank(lam),
        "numerical_rank": numerical_rank(lam, rel_thresh),
        "top_eigenvalue_frac": float(lam.max() / lam.sum()) if lam.sum() > 0 else 0.0,
    }
    if null_floor is not None:
        r["rank_above_null_floor"] = count_above_floor(lam, null_floor)
    return r

# ---------- analytic Jacobian read (local action-routing rank) ----------
def jacobian_spectrum(fn, a0, center=None):
    """Singular values of d fn / d a evaluated at a0, via autodiff. fn maps an action-space
    vector (shape (action_dim,)) to a flattened residual read (shape (m,)). Complements the
    finite-difference read: a two-forward-pass difference locally linearizes GELU and can
    cancel multiplicative action*state directions, understating rank; the Jacobian does not.
    Returns singular values (descending) of the (m x action_dim) Jacobian."""
    a0 = a0.detach().clone().requires_grad_(True)
    J = torch.autograd.functional.jacobian(fn, a0)           # (m, action_dim)
    J = J.reshape(-1, a0.numel()).double()
    return torch.linalg.svdvals(J).cpu().numpy()

# =====================================================================================
# [P1ADD-v3] AC token layout + gate assertions (from source verification 2026-07-23).
#
# The action is NOT an additive residual modulation. VisionTransformerPredictorAC interleaves
# it as a dedicated token:  x = cat([action, state, patches], dim=2).flatten(1,2). Per frame:
#   [action, state, patch_0 .. patch_{HW-1}], stride HW+K (K=2, or 3 with extrinsics).
# So Arm A reads the action effect on PATCH tokens only, excluding the per-frame cond tokens,
# and the routing object is the action-token -> patch-token ATTENTION-mediated update.
# There are TWO 7-d conditioning streams (action AND state); a state-swap control (vary state,
# hold action) is a mandatory second null, else action-attributable variance is confounded
# with proprio. Attention is frame-causal over the FULL history (frame t1 attends to all
# t2 <= t1); perturbing the action at frame t must leave every token in frames < t exactly
# unchanged -- a hard Phase-1 gate assertion.
# =====================================================================================

def token_index(T, H=16, W=16, K=2):
    """AC interleave map. Returns a dict of LongTensors: 'action', 'state', 'cond'
    (all conditioning tokens, sorted), 'patch'. For 256px/patch16: H=W=16, HW=256, K=2,
    stride=258 tokens per frame."""
    HW = H * W; stride = HW + K
    action = torch.tensor([t * stride + 0 for t in range(T)])
    state  = torch.tensor([t * stride + 1 for t in range(T)])
    cond   = torch.tensor([t * stride + c for t in range(T) for c in range(K)])
    patch  = torch.tensor([t * stride + K + i for t in range(T) for i in range(HW)])
    return {"action": action, "state": state, "cond": cond.sort().values, "patch": patch}

def cond_positions(T, H=16, W=16, K=2):
    """Per-frame conditioning tokens to EXCLUDE from Arm A readouts (their variance is
    trivially driven by the swept input)."""
    return token_index(T, H, W, K)["cond"]

def patch_positions(T, H=16, W=16, K=2):
    """The tokens Arm A reads (patch tokens; the action reaches them only through attention)."""
    return token_index(T, H, W, K)["patch"]

def empirical_action_null(action_pool, n, seed=0):
    """PRIMARY on-manifold null (site-verification finding 4). Resample n action VECTORS with
    replacement from the empirical action distribution `action_pool` (M, action_dim) of REAL
    actions, to be embedded through the real action_encoder. Because the action enters through
    a learned 7->1024 encoder, the encoder-OUTPUT manifold is what matters, and real (or
    resampled-real) actions are the only guaranteed-on-manifold null. Preferred over
    random_action_null / matched_norm_null."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(action_pool.shape[0], (n,), generator=g)
    return action_pool[idx]

@torch.no_grad()
def assert_frame_causal(predictor, B=2, T=4, HW=256, d_enc=1408,
                        action_dim=7, state_dim=7, perturb_frame=2):
    """Phase-1 structural gate. Perturbing the action at frame `perturb_frame` must produce
    EXACTLY zero change at every token in earlier frames, and a nonzero change at that frame.
    Catches hook misplacement, wrong token indexing, and mask misuse in one forward pass.
    Assumes predictor(x, actions, states) with x:(B,T*HW,d_enc), actions/states:(B,T,dim)."""
    x = torch.randn(B, T * HW, d_enc)
    a = torch.randn(B, T, action_dim); s = torch.randn(B, T, state_dim)
    out0 = predictor(x, a, s)
    a2 = a.clone(); a2[:, perturb_frame] += 5.0
    out1 = predictor(x, a2, s)
    d = (out1 - out0).abs().reshape(B, T, HW, -1).amax(dim=(0, 2, 3))
    for t in range(perturb_frame):
        assert float(d[t]) == 0, f"CAUSALITY VIOLATED at frame {t}: {float(d[t]):.3e}"
    assert float(d[perturb_frame]) > 0, "action had no forward effect; wiring is wrong"
    return {"per_frame_max_delta": [float(v) for v in d], "passed": True}

@torch.no_grad()
def assert_real_checkpoint(encoder, predictor,
                           predictor_depth=24, action_dim=7, init_std=0.02):
    """Protocol R7 aid (NOT a substitute for the reserved human attestation H1). Run ONCE
    against genuinely loaded weights before any claim. The std heuristic flags an
    action_encoder still at trunc_normal_(std=init_std) init, i.e. a state_dict that did not
    populate the conditioning pathway."""
    assert type(predictor).__name__ == "VisionTransformerPredictorAC", type(predictor).__name__
    assert len(predictor.predictor_blocks) == predictor_depth
    assert predictor.action_encoder.in_features == action_dim
    assert predictor.predictor_embed.in_features == encoder.embed_dim
    assert getattr(predictor, "is_frame_causal", True) and not getattr(predictor, "use_extrinsics", False)
    w = predictor.action_encoder.weight.detach()
    assert w.std().item() > init_std + 0.005, (
        f"action_encoder.weight std={w.std().item():.4f} looks like fresh init "
        f"(~{init_std}); state_dict did not populate the conditioning pathway")
    return {"loaded": True, "action_encoder_std": float(w.std())}
