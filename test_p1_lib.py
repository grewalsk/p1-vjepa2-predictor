"""Analytic ground-truth unit tests for p1_lib (Phase-1 exit gate).

Each test constructs a case with a KNOWN answer and asserts recovery. Run:
    python3 test_p1_lib.py
Exits non-zero on any failure. No pytest dependency required.
"""
import numpy as np, torch, torch.nn as nn
import p1_lib as P

torch.manual_seed(0)
FAILS = []
def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond: FAILS.append(name)

# ---------------------------------------------------------------------------
# geometry: spectrum / participation_ratio / effective_rank / effective_count
# ---------------------------------------------------------------------------
def test_spectrum_recovers_rank_and_eigs():
    # data lying exactly in a k-dim subspace with prescribed variances.
    # Orthonormalize the score matrix so cross-correlations vanish and the covariance
    # eigenvalues equal var_true EXACTLY (no finite-sample perturbation).
    n, d, k = 400, 32, 4
    U = torch.linalg.qr(torch.randn(d, k))[0]           # (d,k) orthonormal loadings
    var_true = np.array([9.0, 4.0, 1.0, 0.25])          # target eigenvalues
    G = torch.randn(n, k); G = G - G.mean(0, keepdim=True)
    G = torch.linalg.qr(G)[0]                            # orthonormal, mean-zero columns
    scores = G * float(np.sqrt(n - 1))                   # unbiased var 1, zero cross-corr
    X = (scores * torch.tensor(np.sqrt(var_true), dtype=torch.float32)) @ U.T
    lam = P.spectrum(X)                                  # length d
    nz = lam[lam > 1e-6]
    check("spectrum: exactly k nonzero eigenvalues", len(nz) == k, f"{len(nz)} vs {k}")
    check("spectrum: eigenvalues match prescribed variances",
          np.allclose(np.sort(nz)[::-1], var_true, rtol=1e-3, atol=1e-3),
          f"{np.sort(nz)[::-1].round(4)}")

def test_participation_ratio_uniform():
    d, k = 50, 5
    lam = np.array([1.0]*k + [0.0]*(d-k))
    check("participation_ratio: uniform over k gives 1/k",
          abs(P.participation_ratio(lam) - 1.0/k) < 1e-12, f"{P.participation_ratio(lam)}")
    check("effective_count: uniform over k gives k",
          abs(P.effective_count(lam) - k) < 1e-9, f"{P.effective_count(lam)}")
    check("effective_rank(entropy): uniform over k gives k",
          abs(P.effective_rank(lam) - k) < 1e-9, f"{P.effective_rank(lam)}")

def test_participation_ratio_single_mode():
    lam = np.array([7.0, 0, 0, 0])
    check("participation_ratio: single mode gives 1", abs(P.participation_ratio(lam) - 1.0) < 1e-12)
    check("effective_rank: single mode gives 1", abs(P.effective_rank(lam) - 1.0) < 1e-9)

# ---------------------------------------------------------------------------
# subspaces: subspace_from_diffs / random_subspace / principal_angles
# ---------------------------------------------------------------------------
def test_subspace_from_diffs_recovers_span():
    d, k = 40, 3
    Utrue = torch.linalg.qr(torch.randn(d, k))[0]
    coeffs = torch.randn(200, k)
    diffs = coeffs @ Utrue.T                              # exactly in span(Utrue)
    Uhat = P.subspace_from_diffs(diffs, k=k)
    ang = P.principal_angles(Utrue, Uhat)
    # arccos is ill-conditioned near 0; ~1e-3 rad is exact recovery to fp precision
    check("subspace_from_diffs: recovers the true k-dim span (angles ~ 0)",
          float(ang.max()) < 2e-3, f"max angle {float(ang.max()):.2e}")

def test_random_subspace_orthonormal():
    U = P.random_subspace(16, 4, seed=1)
    check("random_subspace: orthonormal columns",
          torch.allclose(U.T @ U, torch.eye(4), atol=1e-5))

def test_principal_angles_known():
    d = 10
    Q = torch.linalg.qr(torch.randn(d, d))[0]
    U = Q[:, :3]; V = Q[:, 3:6]                           # orthogonal subspaces
    ang = P.principal_angles(U, V)
    check("principal_angles: orthogonal subspaces give pi/2",
          torch.allclose(ang, torch.full((3,), np.pi/2), atol=1e-4), f"{ang.round(decimals=3)}")
    ang0 = P.principal_angles(U, U)
    # arccos(1-eps) ~ sqrt(2 eps): near-0 angles are numerically noisy by construction
    check("principal_angles: identical subspaces give ~0", float(ang0.max()) < 2e-3, f"{float(ang0.max()):.2e}")

# ---------------------------------------------------------------------------
# hooks: SubspaceAblator projects out span(U); Cache stores activations
# ---------------------------------------------------------------------------
class Passthrough(nn.Module):
    def forward(self, x): return x

class TinyModel(nn.Module):
    def __init__(self): super().__init__(); self.m = Passthrough()
    def forward(self, x): return self.m(x)

def test_subspace_ablator_projects_out():
    d, k = 20, 4
    model = TinyModel()
    sites = {"prd.blk0.resid_post": ("m", "prd", 0, "resid_post")}
    U = P.random_subspace(d, k, seed=2)
    x = torch.randn(5, 7, d)
    with P.SubspaceAblator(model, sites, {"prd.blk0.resid_post": U}):
        y = model(x)
    # residual must have zero projection onto U, and removed part must equal U U^T x
    proj_after = torch.einsum("...d,dk->...k", y, U)
    check("SubspaceAblator: output has zero component in span(U)",
          float(proj_after.abs().max()) < 1e-4, f"{float(proj_after.abs().max()):.2e}")
    expected = x - torch.einsum("...d,dk,ek->...e", x, U, U)
    check("SubspaceAblator: output equals x - U U^T x", torch.allclose(y, expected, atol=1e-5))

def test_cache_stores():
    d = 8
    model = TinyModel()
    sites = {"prd.blk0.resid_post": ("m", "prd", 0, "resid_post")}
    x = torch.randn(3, 4, d)
    with P.Cache(model, sites, ["prd.blk0.resid_post"]) as C:
        model(x)
        stored = C["prd.blk0.resid_post"]
    check("Cache: stores the exact activation", torch.allclose(stored, x, atol=1e-6))

class MLP(nn.Module):
    def forward(self, x): return x
class ACRoPEAttention(nn.Module):
    def forward(self, x): return x
class ACBlock(nn.Module):
    def __init__(self): super().__init__(); self.attn = ACRoPEAttention(); self.mlp = MLP()
    def forward(self, x): return x

def test_discover_sites_two_tree_ac_paths():
    # Reproduces the REAL failure mode: predictor path is "predictor_blocks.0" (no ".blocks."),
    # encoder path is "blocks.0.attn" (no "encoder" in the path). The corrected two-arg
    # discover_sites(module, family) must handle both and capture conditioning modules.
    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.predictor_embed = nn.Linear(1408, 1024)
            self.action_encoder = nn.Linear(7, 1024)
            self.state_encoder = nn.Linear(7, 1024)
            self.predictor_blocks = nn.ModuleList([ACBlock() for _ in range(24)])
            self.predictor_norm = nn.LayerNorm(1024)
            self.predictor_proj = nn.Linear(1024, 1408)
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__(); self.blocks = nn.ModuleList([ACBlock() for _ in range(40)])
    ps = P.discover_sites(Predictor(), "prd")
    es = P.discover_sites(Encoder(), "enc")
    check("discover_sites: 24 predictor blocks via predictor_blocks.N", P.n_blocks(ps, "prd") == 24, str(P.n_blocks(ps, "prd")))
    check("discover_sites: 40 encoder blocks via bare blocks.N", P.n_blocks(es, "enc") == 40, str(P.n_blocks(es, "enc")))
    check("discover_sites: all three kinds per predictor block",
          all(f"prd.blk0.{k}" in ps for k in ("resid_post", "attn_out", "mlp_out")))
    check("discover_sites: conditioning modules captured (not block-indexed)",
          "prd.action_encoder" in ps and "prd.state_encoder" in ps and "prd.predictor_proj" in ps)

# ---------------------------------------------------------------------------
# Arm A: action_basis / matched_norm_null / action_induced_variance / diffs
# ---------------------------------------------------------------------------
def test_action_basis_shape():
    B = P.action_basis(action_dim=7, magnitudes=(0.5, 1.0), sign_flips=True, include_zero=True)
    # 1 zero + 7 dims * 2 mags * 2 signs = 1 + 28
    check("action_basis: correct row count", B.shape == (29, 7), str(tuple(B.shape)))
    check("action_basis: row 0 is the zero action", float(B[0].abs().sum()) == 0.0)

def test_matched_norm_null_matches_norm():
    acts = P.action_basis(7)
    null = P.matched_norm_null(acts, seed=3)
    check("matched_norm_null: per-row norms match the actions",
          torch.allclose(null.norm(dim=-1), acts.norm(dim=-1), atol=1e-5))

def test_action_induced_variance_analytic():
    # activation at each token = base[token] + s_i * u ; across-action variance = var(s_i)*||u||^2
    n_act, n_tok, d = 6, 4, 12
    u = torch.randn(d); u = u / u.norm()
    s = torch.tensor([-2., -1., 0., 1., 2., 3.])
    base = torch.randn(n_tok, d)
    acts = base[None] + s[:, None, None] * u[None, None, :]   # (n_act, n_tok, d)
    var_s = float(s.var(unbiased=True))                       # ||u||=1 -> trace = var(s)
    got = P.action_induced_variance(acts)
    check("action_induced_variance: matches var(action-scalar)",
          abs(got - var_s) < 1e-4, f"{got:.4f} vs {var_s:.4f}")
    # excluding a trivial high-variance token must not change the (already token-averaged) value
    acts2 = torch.cat([acts, torch.randn(n_act, 1, d) * 50], dim=1)  # append a noisy 'action token'
    got_excl = P.action_induced_variance(acts2, exclude_positions=[n_tok])
    check("action_induced_variance: excluding the action-token position removes its inflation",
          abs(got_excl - var_s) < 1e-4, f"{got_excl:.4f} vs {var_s:.4f}")

def test_action_diffs_spectrum_rank():
    # diffs live in a rank-2 subspace -> participation-ratio effective count ~ 2
    n_act, n_tok, d, k = 9, 3, 16, 2
    U = torch.linalg.qr(torch.randn(d, k))[0]
    coeff = torch.randn(n_act, n_tok, k); coeff[0] = 0     # ref action has zero coeff
    acts = coeff @ U.T
    lam = P.spectrum(P.action_diffs(acts, ref_index=0))
    nz = (lam > 1e-6).sum()
    check("action_diffs: difference subspace has the injected rank", nz == k, f"{int(nz)} vs {k}")

# ---------------------------------------------------------------------------
# Cui-theory object (a): transition_operator_spectrum
# ---------------------------------------------------------------------------
def test_transition_operator_rank():
    # z_pred = z_ctx @ W with W of rank r -> cross-covariance has rank r
    n, d, r = 500, 24, 3
    W = torch.randn(d, r) @ torch.randn(r, d)             # rank-r map
    z_ctx = torch.randn(n, d)
    z_pred = z_ctx @ W
    sv = P.transition_operator_spectrum(z_ctx, z_pred)
    nz = int((sv > 1e-6 * sv.max()).sum())
    check("transition_operator_spectrum: recovers transition-operator rank r",
          nz == r, f"{nz} vs {r}")

# ---------------------------------------------------------------------------
# Arm B: cumulative_localization
# ---------------------------------------------------------------------------
def test_cumulative_localization_concentrated():
    # effect in 3 contiguous blocks, no 2-block window reaching 80% -> n_for_80 == span == 3
    e = np.zeros(24); e[[10, 11, 12]] = [50, 25, 25]   # 50/75/100 cumulative -> need 3 for 80%
    cum, n80, span = P.cumulative_localization(e, frac=0.80)
    check("cumulative_localization: n_for_80 counts the concentrated blocks", n80 == 3, str(n80))
    check("cumulative_localization: smallest contiguous 80% window", span == 3, str(span))
    check("cumulative_localization: cumulative curve ends at 1.0", abs(cum[-1] - 1.0) < 1e-9)
    # and the easy case: two blocks already reach 80% -> n_for_80 == 2
    e2 = np.zeros(24); e2[[3, 4, 5]] = [50, 30, 20]
    _, n80b, _ = P.cumulative_localization(e2, frac=0.80)
    check("cumulative_localization: two blocks reaching 80% -> n_for_80 == 2", n80b == 2, str(n80b))

# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------
def test_bootstrap_ci_constant():
    v = np.full(100, 3.5)
    m, lo, hi = P.bootstrap_ci(v, n=500, seed=0)
    check("bootstrap_ci: constant data collapses CI to the constant",
          abs(m-3.5) < 1e-9 and abs(lo-3.5) < 1e-9 and abs(hi-3.5) < 1e-9, f"{m},{lo},{hi}")

# ---------------------------------------------------------------------------
# CEM planning (behavioral sanity on an analytic linear predictor)
# ---------------------------------------------------------------------------
def test_cem_plan_linear_reaches_goal():
    # Behavioral sanity on a stochastic optimizer: the RETURNED plan must clearly beat the
    # do-nothing plan. (trace tracks best-of-n sample cost, which is noisy, so we evaluate mu.)
    dim = 5
    def pred(z, a): return z + a                          # z_{t+1} = z_t + a_t
    z0 = torch.zeros(1, dim)
    z_goal = torch.tensor([[0.3, -0.2, 0.1, 0.4, -0.1]])  # reachable within clamp [-1,1]
    mu, trace = P.cem_plan(pred, z0, z_goal, horizon=4, n=2048, elites=128, iters=10, action_dim=dim, seed=0)
    z = z0.clone()
    for t in range(mu.shape[0]): z = pred(z, mu[t:t+1])
    plan_cost = float((z - z_goal).pow(2).sum())
    zero_cost = float((z0 - z_goal).pow(2).sum())         # cost of the do-nothing plan
    check("cem_plan: returned plan clearly beats the do-nothing plan",
          plan_cost < 0.3 * zero_cost, f"plan {plan_cost:.4f} vs zero {zero_cost:.4f}")
    check("cem_plan: best sampled cost improved over the run",
          min(trace) < trace[0], f"min {min(trace):.4f} < first {trace[0]:.4f}")

# ---------------------------------------------------------------------------
# OccludedBounce ground-truth integrity (Arm D stimulus)
# ---------------------------------------------------------------------------
def test_occluded_bounce_groundtruth():
    ds = P.OccludedBounce(n=4, T=16, size=32, occ_start=5, occ_len=4, seed=0)
    item = ds[0]
    check("OccludedBounce: trajectory length == T", item["traj"].shape == (16, 2), str(tuple(item['traj'].shape)))
    check("OccludedBounce: occluded frame indices correct", item["occluded"] == [5, 6, 7, 8], str(item["occluded"]))
    check("OccludedBounce: pixel tensor shape (T,3,S,S)", tuple(item["pixel_values"].shape) == (16, 3, 32, 32),
          str(tuple(item["pixel_values"].shape)))
    # determinism
    check("OccludedBounce: deterministic per seed/index",
          torch.allclose(ds[0]["traj"], P.OccludedBounce(n=4,T=16,size=32,seed=0)[0]["traj"]))

# ---------------------------------------------------------------------------
# Arm A hardening (v2): on-manifold nulls, rank-estimator panel, Jacobian read
# ---------------------------------------------------------------------------
def test_permuted_action_null():
    acts = P.action_basis(7)
    null = P.permuted_action_null(acts, seed=1)
    check("permuted_action_null: preserves the exact set of action norms (multiset)",
          torch.allclose(torch.sort(null.norm(dim=-1))[0], torch.sort(acts.norm(dim=-1))[0], atol=1e-6))
    check("permuted_action_null: is not the identity permutation",
          not torch.allclose(null, acts))

def test_random_action_null_matches_norm():
    acts = P.action_basis(7)
    ref = acts.norm(dim=-1)
    null = P.random_action_null(7, ref, seed=2)
    check("random_action_null: per-row norms match reference and dim is action_dim",
          null.shape == acts.shape and torch.allclose(null.norm(dim=-1), ref, atol=1e-5))

def test_dense_action_sample_in_box():
    a = P.dense_action_sample(7, 2000, low=-1.0, high=1.0, seed=0)
    check("dense_action_sample: shape and box bounds",
          a.shape == (2000, 7) and float(a.min()) >= -1.0 and float(a.max()) <= 1.0)

def test_rank_estimator_panel_uniform():
    d, k = 60, 6
    lam = np.array([1.0]*k + [0.0]*(d-k))
    check("stable_rank: uniform over k gives k", abs(P.stable_rank(lam) - k) < 1e-9, f"{P.stable_rank(lam)}")
    check("numerical_rank: uniform over k gives k", P.numerical_rank(lam) == k, str(P.numerical_rank(lam)))
    rep = P.rank_report(lam, null_floor=0.5)
    check("rank_report: entropy effective rank == k", abs(rep["entropy_effective_rank"] - k) < 1e-9)
    check("rank_report: participation effective count == k", abs(rep["participation_effective_count"] - k) < 1e-9)
    check("rank_report: rank_above_null_floor counts modes above floor", rep["rank_above_null_floor"] == k)

def test_numerical_rank_threshold():
    lam = np.array([9.0, 4.0, 1.0, 0.25])
    # rel_thresh 0.1 -> floor 0.9 -> counts {9,4,1} = 3
    check("numerical_rank: relative threshold selects modes above 0.1*max",
          P.numerical_rank(lam, rel_thresh=0.1) == 3, str(P.numerical_rank(lam, rel_thresh=0.1)))

def test_jacobian_spectrum_recovers_linear_rank():
    # fn(a) = W a with W rank r -> Jacobian = W -> exactly r nonzero singular values
    m, da, r = 40, 7, 3
    W = torch.randn(m, r) @ torch.randn(r, da)
    def fn(a): return W @ a
    a0 = torch.zeros(da)
    sv = P.jacobian_spectrum(fn, a0)
    nz = int((sv > 1e-6 * sv.max()).sum())
    check("jacobian_spectrum: recovers the linear-map rank r", nz == r, f"{nz} vs {r}")

# ---------------------------------------------------------------------------
# AC token layout + gate assertions (v3)
# ---------------------------------------------------------------------------
def test_token_index_interleave():
    # T=2, H=W=2 -> HW=4, K=2, stride=6. Frame layout: [action, state, p0,p1,p2,p3]
    ti = P.token_index(T=2, H=2, W=2, K=2)
    check("token_index: action positions", ti["action"].tolist() == [0, 6], str(ti["action"].tolist()))
    check("token_index: state positions", ti["state"].tolist() == [1, 7], str(ti["state"].tolist()))
    check("token_index: cond = action+state sorted", ti["cond"].tolist() == [0, 1, 6, 7], str(ti["cond"].tolist()))
    check("token_index: patch positions", ti["patch"].tolist() == [2, 3, 4, 5, 8, 9, 10, 11], str(ti["patch"].tolist()))
    # disjoint and complete
    allpos = torch.cat([ti["cond"], ti["patch"]]).sort().values
    check("token_index: cond and patch partition the sequence", allpos.tolist() == list(range(12)))

def test_cond_positions_exclusion_is_full_layout():
    check("cond_positions == token_index cond",
          P.cond_positions(3, 2, 2, 2).tolist() == P.token_index(3, 2, 2, 2)["cond"].tolist())

def test_empirical_action_null_from_pool():
    pool = P.action_basis(7)                       # a pool of real actions
    null = P.empirical_action_null(pool, n=50, seed=0)
    check("empirical_action_null: shape (n, action_dim)", null.shape == (50, 7), str(tuple(null.shape)))
    # every drawn row must equal some pool row (resampled from the empirical distribution)
    matches = [(null[i] == pool).all(dim=1).any().item() for i in range(50)]
    check("empirical_action_null: every sample is a genuine pool action", all(matches))

class _MockAC(nn.Module):
    """Minimal frame-causal AC predictor: patch-only output where frame t depends only on
    actions in frames <= t (causal) or on all frames (non-causal, must trip the assertion)."""
    def __init__(self, d=8, action_dim=7, HW=256, causal=True):
        super().__init__(); self.enc = nn.Linear(action_dim, d); self.HW = HW; self.d = d; self.causal = causal
    def forward(self, x, a, s):
        B, N, d_enc = x.shape; T = a.shape[1]
        ae = self.enc(a)                                        # (B,T,d)
        acc = torch.cumsum(ae, 1) if self.causal else ae.sum(1, keepdim=True).expand(B, T, self.d)
        xf = x.reshape(B, T, self.HW, d_enc)[..., :self.d]
        return (xf + acc[:, :, None, :]).reshape(B, T * self.HW, self.d)

def test_assert_frame_causal_passes_and_catches():
    ok = P.assert_frame_causal(_MockAC(causal=True, HW=8), B=2, T=4, HW=8, d_enc=8, perturb_frame=2)
    check("assert_frame_causal: passes on a genuinely frame-causal model",
          ok["passed"] and ok["per_frame_max_delta"][0] == 0 and ok["per_frame_max_delta"][2] > 0,
          str([round(v, 4) for v in ok["per_frame_max_delta"]]))
    raised = False
    try:
        P.assert_frame_causal(_MockAC(causal=False, HW=8), B=2, T=4, HW=8, d_enc=8, perturb_frame=2)
    except AssertionError:
        raised = True
    check("assert_frame_causal: trips on a non-causal model", raised)

# ---------------------------------------------------------------------------
# v4: estimators from the Phase-2a adversarial review (constructive counterexamples
# from the review become the ground truth here)
# ---------------------------------------------------------------------------
def _token_diverse_linear_model(S=64, T=32, D=128, q=7, K=5, seed=4):
    """d_st = G_t (J a_s), G_t = sum_k C[t,k] H_k: a STRICTLY q-dim linear action code
    with token-diverse linear readouts. Pooled rank ~ q(K+1) > q; mode-s rank == q."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(S, q, generator=g)
    J = torch.randn(q, D, generator=g) / D ** 0.5
    H = torch.randn(K + 1, D, D, generator=g) / D ** 0.5
    C = torch.randn(T, K + 1, generator=g)
    G = torch.einsum('tk,kde->tde', C, H)
    d = torch.einsum('sd,tde->ste', A @ J, G)          # (S, T, D)
    return A, d

def test_mode_s_beats_pooled_on_linear_null():
    q, K = 7, 5
    A, d = _token_diverse_linear_model(q=q, K=K)
    lam_s = P.mode_s_spectrum(d)
    nr_s = P.numerical_rank(lam_s, rel_thresh=1e-6)
    lam_p = P.pooled_diff_spectrum(d)
    nr_p = P.numerical_rank(lam_p, rel_thresh=1e-6)
    check("mode-s: rank == q under token-diverse LINEAR readout (no false expansion)",
          nr_s == q, f"{nr_s} vs {q}")
    check("pooled: rank exceeds q under the SAME strictly-linear model (the trap)",
          nr_p > q, f"{nr_p} > {q}; linear null is q(K+1) = {q*(K+1)}")

def test_mode_s_detects_true_code_expansion():
    # add a quadratic code channel: mode-s rank must rise above q
    q, K = 7, 5
    g = torch.Generator().manual_seed(5)
    A, d = _token_diverse_linear_model(q=q, K=K, seed=5)
    D = d.shape[-1]
    v = torch.randn(d.shape[1], D, generator=g) / D ** 0.5   # token-varying quadratic readout
    quad = (A[:, 0] ** 2 - (A[:, 0] ** 2).mean())            # a genuine extra code dim
    d2 = d + 0.5 * torch.einsum('s,td->std', quad, v)
    nr = P.numerical_rank(P.mode_s_spectrum(d2), rel_thresh=1e-6)
    check("mode-s: detects a genuine extra (quadratic) code dimension", nr == q + 1, f"{nr} vs {q+1}")

def test_even_part_fraction_linear_zero_quadratic_positive():
    S, T, D, q = 32, 4, 64, 7
    g = torch.Generator().manual_seed(6)
    A = torch.randn(S, q, generator=g)
    Jl = torch.randn(q, D, generator=g); Jq = torch.randn(q, D, generator=g)
    lin_p = (A @ Jl).unsqueeze(1).repeat(1, T, 1); lin_m = (-A @ Jl).unsqueeze(1).repeat(1, T, 1)
    check("even_part_fraction: exactly 0 for a linear response",
          P.even_part_fraction(lin_p, lin_m) < 1e-10)
    ev = ((A ** 2) @ Jq).unsqueeze(1).repeat(1, T, 1)
    f = P.even_part_fraction(lin_p + ev, lin_m + ev)
    check("even_part_fraction: positive when an even component is present", f > 0.01, f"{f:.4f}")

def test_per_token_centering_removes_footprint():
    # rank-1 action signal + large token-dependent constant offsets (spatial footprint).
    S, T, D = 48, 24, 96
    g = torch.Generator().manual_seed(7)
    s = torch.randn(S, generator=g); v = torch.randn(D, generator=g); v = v / v.norm()
    U = torch.randn(T, D, generator=g) * 10.0            # big static footprint, no action dep
    d = torch.einsum('s,d->sd', s, v).unsqueeze(1) + U.unsqueeze(0)
    nr_tok = P.numerical_rank(P.pooled_diff_spectrum(d), rel_thresh=1e-6)
    lam_grand = P.spectrum(d.reshape(-1, D), center=True)   # grand-mean centering (old way)
    nr_grand = P.numerical_rank(lam_grand, rel_thresh=1e-6)
    check("per-token centering: footprint removed, true rank 1 recovered", nr_tok == 1, str(nr_tok))
    check("grand-mean centering: footprint contaminates the spectrum (old estimator)",
          nr_grand > 1, str(nr_grand))

def test_variance_fraction_scale_invariant():
    g = torch.Generator().manual_seed(8)
    a = torch.randn(16, 10, 32, generator=g)
    f1, f2 = P.variance_fraction(a), P.variance_fraction(10.0 * a)
    check("variance_fraction: scale-invariant (kills the norm-growth confound)",
          abs(f1 - f2) / max(f1, 1e-12) < 1e-5, f"{f1:.8f} vs {f2:.8f}")

def test_ln_normalized_unit_stats():
    a = torch.randn(5, 7, 64) * 50 + 3
    n = P.ln_normalized(a)
    check("ln_normalized: per-vector mean ~0 and std ~1",
          float(n.mean(-1).abs().max()) < 1e-4 and abs(float(n.std(-1).mean()) - 1) < 1e-2)

# ---------------------------------------------------------------------------
# v5: control-operator state-geometry (chosen avenue)
# ---------------------------------------------------------------------------
def test_input_jacobian_exact_linear():
    m, da = 20, 7
    W = torch.randn(m, da)
    def fn(a): return W @ a + torch.randn(m) * 0  # linear map
    G = P.input_jacobian(fn, torch.zeros(da))
    check("input_jacobian: recovers the exact linear map (m, d_a)",
          G.shape == (m, da) and torch.allclose(G.float(), W, atol=1e-5))

def test_across_state_variation_constant_vs_varying():
    m, da = 16, 7
    Gbase = torch.randn(m, da)
    Gs_const = Gbase[None].repeat(12, 1, 1)                      # Koopman: identical across states
    check("across_state_variation: ~0 for constant G (Koopman/LTI)",
          P.across_state_variation(Gs_const) < 1e-6, f"{P.across_state_variation(Gs_const):.2e}")
    Gs_vary = Gbase[None] + 0.5 * torch.randn(12, m, da)        # state-varying control operator
    C = P.across_state_variation(Gs_vary)
    check("across_state_variation: > 0 for state-varying G (control-affine)", C > 0.05, f"{C:.3f}")

def test_energy_hessian_is_2GtG():
    m, da = 12, 7
    G = torch.randn(m, da)
    H = P.energy_hessian_from_jacobian(G)
    check("energy_hessian: H == 2 G^T G, PSD, rank <= d_a",
          torch.allclose(H, 2*G.T@G, atol=1e-5) and int((torch.linalg.eigvalsh(H) > 1e-4).sum()) <= da)

def test_whiten_action_std_defuses_degenerate_axis():
    # one axis ~100x the others (the Euler-wraparound artifact)
    pool = torch.randn(500, 7) * torch.tensor([0.003,0.004,0.005,0.68,0.010,0.014,0.020])
    std, floored = P.whiten_action_std(pool)
    w = (pool / floored)   # whitened coordinates
    check("whiten_action_std: whitened per-dim std ~1 across all axes (artifact defused)",
          float((w.std(0) - 1).abs().max()) < 0.1, f"{[round(float(v),2) for v in w.std(0)]}")

# ---------------------------------------------------------------------------
# v6: nuisance-orbit audit encodes the judge's transformation-law ledger.
# The point: on a model whose identifiability group is uncertified, "G constant in z"
# (the Koopman/LTI verdict) is only a LINEAR-invariant, and sigma(G) is only ORTHOGONAL-
# invariant, so both can be pure coordinate nuisance. These tests VERIFY that ledger.
# ---------------------------------------------------------------------------
def test_reparam_ledger_constant_G():
    # constant-G stack (a Koopman/LTI verdict, C_baseline ~ 0)
    m, da = 20, 7
    Gbase = torch.randn(m, da)
    Gs = Gbase[None].repeat(12, 1, 1)
    rep = P.reparam_invariance_report(Gs, seed=1)
    b = rep['baseline']
    check("audit: rank invariant under ALL reparam",
          all(abs(rep[k]['mean_rank'] - b['mean_rank']) < 1e-6 for k in ('orthogonal','general_linear','nonlinear')))
    check("audit: 'G constant' (C~0) stays ~0 under LINEAR reparam (orth + GL)",
          rep['orthogonal']['C'] < 1e-4 and rep['general_linear']['C'] < 1e-4,
          f"orth {rep['orthogonal']['C']:.2e} gl {rep['general_linear']['C']:.2e}")
    check("audit: 'G constant' is DESTROYED by nonlinear reparam (Koopman verdict is linear-only)",
          rep['nonlinear']['C'] > 0.1, f"nonlinear C {rep['nonlinear']['C']:.3f}")

def test_reparam_ledger_sigma():
    m, da = 20, 7
    Gs = torch.randn(12, m, da)
    rep = P.reparam_invariance_report(Gs, seed=2)
    b = rep['baseline']['mean_sigma']
    check("audit: mean singular value invariant under ORTHOGONAL reparam",
          abs(rep['orthogonal']['mean_sigma'] - b) / b < 1e-4, f"{abs(rep['orthogonal']['mean_sigma']-b)/b:.2e}")
    check("audit: mean singular value CHANGES under general-linear reparam (sharp-but-fragile)",
          abs(rep['general_linear']['mean_sigma'] - b) / b > 0.05, f"{abs(rep['general_linear']['mean_sigma']-b)/b:.3f}")

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        try:
            t()
        except Exception as ex:
            check(t.__name__, False, f"EXCEPTION {type(ex).__name__}: {ex}")
    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    raise SystemExit(1 if FAILS else 0)
