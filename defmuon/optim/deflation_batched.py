"""Deflation as a preconditioner for Muon's polar factorization: a batched, gated
randomized-SVD front-end and the deflated Newton-Schulz arm.

Remove-then-restore ("rmfro").  For a momentum matrix A (n x m, n >= m):

    W   = A - U_k S_k V_k^T          remove the leading outlier directions
    X0  = W / (||W||_F + 1e-7)       Muon's Frobenius entry, on the remainder
    X0 += U_k (1/pad) V_k^T          restore the head at height 1/pad

msign depends on the singular vectors alone, so removing the head leaves the polar
factor of the remainder unchanged while taking the head mass out of the Frobenius
divisor -- otherwise loose by ~sqrt(stable rank) -- and the restored block re-enters
the polynomial already converged.  The entry is safe without estimating sigma_max: a
Frobenius divisor is >= the spectral one, so the remainder enters with sigma_max <= 1
and the head sits at 1/pad, well inside the convergence basin.  A gate declines
matrices whose leading window shows no outlier structure; those take exactly the
undeflated entry A / (||A||_F + 1e-7).

The front-end is iteration-agnostic and launch-bound rather than FLOP-bound, so
equal-shaped matrices run as one batch: every rSVD kernel batches (bmm, cholesky_ex,
solve_triangular, gesvda) and the gate is pure tensor arithmetic -- per-matrix k, the
removal mask and the divisor are all computed on device -- so the setup costs no host
sync beyond torch's svd convergence check.

Consumed by muon.py's `_muon_step_batched`: batched_rmfro_clip (the setup) and the
DeflatedNSPolar factorizer (pad + bf16 iteration).
"""

import math
import os

import torch

# ---------------------------------------------------------------------------
# deflation constants (all overridable by environment variable)
# ---------------------------------------------------------------------------
# The head is restored at height 1/DIVISOR_PAD = 0.990, just under the iteration's
# fixed point, so the restored directions are already converged on entry.  The
# remainder's divisor is its exact Frobenius norm (no estimate involved), so the
# entry stays inside NS5's basin by construction for any pad >= 1; the benchmark
# is insensitive to this value.
DIVISOR_PAD = float(os.environ.get("DEFMUON_DEFL_PAD", "1.01"))
# rSVD depth: power iterations, and CholeskyQR passes per orthonormalization.
# CholeskyQR passes are span-invariant, so one suffices at this operating point.
RSVD_NITER = int(os.environ.get("DEFMUON_DEFL_NITER", "1"))
CHOLQR_PASSES = int(os.environ.get("DEFMUON_DEFL_QR_PASSES", "1"))
GRAM_DTYPE = torch.float64
# Operating point, all widths proportional to m:
#     sketch   r  = ceil(SKETCH_FRAC*m) + ceil(OVERSAMPLE_FRAC*m)
#     gate     fire iff s[kw-1]/s[0] < GATE_RATIO,  kw = ceil(GATE_WINDOW*m)
#     clip     remove every s_i > DEFLATE_THRESH*s[0]
# SKETCH_FRAC defaults to GATE_WINDOW -- the sketch's target rank is exactly the depth
# the gate reads, so r >= kw always -- and DEFLATE_THRESH to GATE_RATIO, one threshold
# asked of the window by the gate and of every singular value by the clip.
GATE_WINDOW = float(os.environ.get("DEFMUON_DEFL_WINDOW", "0.025"))    # span window / m
SKETCH_FRAC = float(os.environ.get("DEFMUON_DEFL_SKETCH", str(GATE_WINDOW)))  # sketch target / m
OVERSAMPLE_FRAC = float(os.environ.get("DEFMUON_DEFL_OS", "0.025"))    # safety margin / m
GATE_RATIO = float(os.environ.get("DEFMUON_DEFL_RATIO", "0.10"))       # fire if s[kw-1]/s[0] < this
DEFLATE_THRESH = float(os.environ.get("DEFMUON_DEFL_THRESH", str(GATE_RATIO)))  # clip s_i > this * s[0]

# Gate statistics accumulate on device (int64[3]: total, fired, k_sum), ticked by
# muon._muon_step_batched with no host sync; polar_stats() reads them back outside the
# timed step.  _DEGENERATE and _PARACHUTE are always 0 and are kept only so the
# reported schema is stable across logs.
_GATE_ACC = [None]
_DEGENERATE = [0]
_PARACHUTE = [0]

# Optional per-call timing, clip vs iteration (DEFMUON_TIME_POLAR=1).  CUDA events
# rather than synchronize()+perf_counter, so the measurement does not stall the stream
# it measures; elapsed times read only after a sync, hence flush_timings().
_TIMING = {"clip_ms": 0.0, "iter_ms": 0.0, "calls": 0}
_PENDING = []
_TIMING_ON = [bool(int(os.environ.get("DEFMUON_TIME_POLAR", "0")))]


def enable_polar_timing(on=True):
    _TIMING_ON[0] = bool(on)


def flush_timings():
    """Drain recorded events into the accumulator.  Call only after a sync."""
    if not _PENDING:
        return
    for e0, e1, e2 in _PENDING:
        _TIMING["clip_ms"] += e0.elapsed_time(e1)
        _TIMING["iter_ms"] += e1.elapsed_time(e2)
        _TIMING["calls"] += 1
    _PENDING.clear()


def timing_stats(reset=True):
    """(clip_ms, iter_ms, calls) accumulated since the last reset."""
    flush_timings()
    out = dict(_TIMING)
    if reset:
        _TIMING.update(clip_ms=0.0, iter_ms=0.0, calls=0)
    return out


def polar_stats():
    """Gate firing rate and mean deflation width k.  The one host read of the
    device-side accumulator happens here, never inside a timed step."""
    if _GATE_ACC[0] is not None:
        total, fired, k_sum = _GATE_ACC[0].tolist()
    else:
        total = fired = k_sum = 0
    return dict(fired=fired, total=total,
                k_mean=(k_sum / max(fired, 1)),
                degenerate=_DEGENERATE[0], parachute=_PARACHUTE[0])


# ---------------------------------------------------------------------------
# batched randomized SVD: CholeskyQR + gesvda Rayleigh-Ritz
# ---------------------------------------------------------------------------

def _orth_b(X):
    """Batched CholeskyQR.  X (B, n, r) -> Q with orthonormal columns per batch.
    The Gram matrix and its Cholesky factor are computed in fp64, which keeps the
    squared conditioning of the Gram approach far from the potrf breakdown point on
    Gaussian-sketch images."""
    Q = X
    for _ in range(CHOLQR_PASSES):
        Xd = Q if Q.dtype == GRAM_DTYPE else Q.to(GRAM_DTYPE)
        G = Xd.mT @ Xd                                    # (B, r, r)
        L, _ = torch.linalg.cholesky_ex(G)
        Q = torch.linalg.solve_triangular(L.to(Q.dtype).mT, Q, upper=True, left=False)
    return Q


def _project_b(Q, A):
    """Batched Rayleigh-Ritz: thin SVD of Q^T A per batch.  cusolver's
    gesvdaStridedBatched (driver='gesvda') is the one truly batched SVD at this
    shape -- (B, r, m) wide, transposed to tall-skinny internally by torch; CPU
    tensors take LAPACK via the default driver."""
    Bp = Q.mT @ A                                         # (B, r, m)
    Ub, s, Vh = torch.linalg.svd(Bp, full_matrices=False,
                                 driver="gesvda" if Bp.is_cuda else None)
    return Q @ Ub, s, Vh.mT


def _rsvd_b(A, Omega, n_iter):
    """Batched subspace iteration."""
    Q = _orth_b(A @ Omega)
    for _ in range(n_iter):
        Z = _orth_b(A.mT @ Q)
        Q = _orth_b(A @ Z)
    return _project_b(Q, A)


# ---------------------------------------------------------------------------
# the deflation front-end (batched remove-then-restore), fp32
# ---------------------------------------------------------------------------

def batched_rmfro_clip(A, generator=None, Omega=None,
                       out_dtype=None, pad=DIVISOR_PAD):
    """Remove-then-restore deflation for a batch A (B, n, m) with n >= m; arithmetic
    in the module docstring.

    Everything after the sketch is tensor arithmetic, with no host sync.  `Omega` lets
    callers drive a fixed sketch; production draws fresh Gaussians from `generator`.
    `out_dtype` casts X0 on the way out, saving a full fp32 round-trip over the n x m
    array for bf16 consumers.  The gate is written as a span test, which for
    descending s is equivalent to a count test and stays a vectorised one-liner.

    Returns (X0, k (B,), d (B,)): the deflation entry, the per-matrix deflation width
    (0 where the gate declined), and the Frobenius divisor actually used.
    """
    Bn, n, m = A.shape
    r = min(int(math.ceil(SKETCH_FRAC * m)) + int(math.ceil(OVERSAMPLE_FRAC * m)), m)
    kw = min(max(1, int(math.ceil(GATE_WINDOW * m))), r)
    if Omega is None:
        Omega = torch.randn(Bn, m, r, device=A.device, dtype=A.dtype,
                            generator=generator)
    U, s, V = _rsvd_b(A, Omega, RSVD_NITER)               # s (B, r)

    kr = (s > DEFLATE_THRESH * s[:, :1]).sum(dim=1)       # (B,)
    fired = (s[:, kw - 1] < GATE_RATIO * s[:, 0]) & (kr >= 1) & (kr < s.shape[1])
    kk = kr.clamp(1, s.shape[1] - 1)                      # (B,)
    use = fired

    idx = torch.arange(s.shape[1], device=A.device)[None, :]
    head = (idx < kk[:, None]) * use[:, None]             # (B, r) top-block mask
    W = A - (U * (s * head)[:, None, :]) @ V.mT           # top block -> 0
    d = W.norm(dim=(-2, -1), keepdim=True) + 1e-7         # (B, 1, 1)
    X0 = W / d + (U * ((1.0 / pad) * head)[:, None, :]) @ V.mT
    if out_dtype is not None:
        X0 = X0.to(out_dtype)
    return X0, kk * use, d.reshape(Bn)


# ---------------------------------------------------------------------------
# the Muon-facing deflated arms: the setup above, plus a per-arm pad and iteration
# ---------------------------------------------------------------------------

def _horner_iterate(X, hs):
    """X <- aX + bX^3 + cX^5 per (a, b, c) in hs, in the tall (n >= m) convention:
    with G = X^T X (m x m), X(aI + bG + cG^2) is the same update the library's
    PolarExpress/zeropower apply on the transposed side."""
    for a, b, c in hs:
        Gm = X.mT @ X
        B = b * Gm + c * Gm @ Gm
        X = a * X + X @ B
    return X


# Compilable where the setup is not: hs holds Python floats that fold into the graph
# as constants, and the loop has no data-dependent control flow.  One specialization
# per (shape, dtype, schedule), pre-warmed at optimizer construction.
if os.environ.get("DEFMUON_COMPILE_DEFL_ITER", "1") == "1":
    _horner_iterate = torch.compile(_horner_iterate)


class _DeflatedPolarBase:
    """A deflated arm: the batched fp32 remove-then-restore setup, then a
    subclass-chosen iteration in bf16.  The front-end is identical across deflated
    arms -- same gate, removal, restore height and seeding -- so any difference
    between them comes from the iteration alone.  Subclasses supply `_iterate`.
    """
    PAD = DIVISOR_PAD          # restore height is 1/PAD; see batched_rmfro_clip
    BATCHED_CLIP = staticmethod(batched_rmfro_clip)

    def _iterate(self, X, steps):
        raise NotImplementedError


class DeflatedNSPolar(_DeflatedPolarBase):
    """The standard Newton-Schulz quintic, unmodified, behind the deflation."""
    NS_ABC = (3.4445, -4.7750, 2.0315)

    def _iterate(self, X, steps):
        return _horner_iterate(X, [self.NS_ABC] * steps)


class DeflatedPEPolar(_DeflatedPolarBase):
    """The Polar Express minimax schedule, unmodified, behind the deflation.

    Same front-end as the NS arm; two PE-specific choices:
      - the coefficient schedule is polar_express.coeffs_list (already carrying
        its 1.01 safety factor), truncated/extended to `steps` exactly as
        PolarExpress does;
      - PAD defaults to 1.1 rather than 1.01: PE's early polynomials have a much
        steeper slope at 1 than NS5's, so overshoot above the fixed point is
        amplified rather than damped -- the restored head enters at 1/1.1 = 0.909,
        comfortably inside the schedule's contraction region.  Override with
        DEFMUON_DEFL_PE_PAD.
    """
    PAD = float(os.environ.get("DEFMUON_DEFL_PE_PAD", "1.1"))

    def _iterate(self, X, steps):
        from itertools import repeat

        from defmuon.optim.polar_express import coeffs_list
        hs = list(coeffs_list[:steps]) + list(
            repeat(coeffs_list[-1], steps - len(coeffs_list)))
        return _horner_iterate(X, hs)
