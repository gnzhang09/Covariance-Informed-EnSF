"""First CI-EnSF experiment on the SQG turbulence model.

Model: jswhit/sqgturb, two bounding surfaces, doubly periodic, spectral.  The DA is done in
temperature units (K): the model state is PV in m/s and theta = pv * f*theta0/g.
Configuration follows sqgturb/sqg_lgetkf_cv.py: 6-hourly analyses, a random network of 10 %
of the grid points redrawn every cycle, sigma_obs = 1 K, K = 16 members, GC localization at
2000 km = 2 Rossby radii, and a climatological initial ensemble drawn from the nature run.
"""
import importlib.util, numpy as np, os, sys, time
from bfree import BFree
import gpu
xp = gpu.xp        # numpy, or cupy when SQG_GPU=1

# The model is vendored in vendor/sqgturb/ so that this repository runs on its own; set
# SQGTURB_PATH to the sqg.py of another checkout to override.
_SQGTURB = os.getenv("SQGTURB_PATH") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "sqgturb", "sqg.py")
if not os.path.exists(_SQGTURB):
    raise SystemExit("sqgturb model not found at %s; set SQGTURB_PATH" % _SQGTURB)
spec = importlib.util.spec_from_file_location("sqg", _SQGTURB)
_sqg = importlib.util.module_from_spec(spec); spec.loader.exec_module(_sqg)

# ---------------------------------------------------------------- configuration
N          = int(os.getenv("SQG_N", "32"))        # grid points per direction -> d = 2*N*N
DT         = float(os.getenv("SQG_DT", str({32: 1200., 64: 1200., 96: 900., 128: 600., 192: 300.}.get(N, 1200.))))
                                 # halve it at N=128: the CFL limit scales with the grid spacing
DIFF_EFOLD = float(os.getenv("SQG_DIFF", {32: 2*86400., 64: 86400., 96: 86400./2,
                                          128: 86400./3, 192: 86400./8}.get(N, 86400.)))
                                 # the table at the top of sqgturb/sqg_run.py: (N, dt, efold) =
                                 # (64,1200,86400) (96,900,86400/2) (128,600,86400/3)
                                 # (192,300,86400/8) (512,90,1800).  86400/2 at N=128 was mine
                                 # and is the N=96 value -- too little damping for that grid.
                                 # sqg_run.py: 86400 at N=64, larger at coarser resolution
NSQ, F, H, U, G0, THETA0 = 1e-4, 1e-4, 10.0e3, 20.0, 9.8, 300.0
LR         = np.sqrt(NSQ)*H/F    # Rossby radius, 1000 km
L_DOM      = 20*LR               # domain size, 20000 km
SCALE      = F*THETA0/G0         # PV (m/s) -> theta (K)

DT_OBS     = 6*3600.0            # analysis interval, s
NSTEP_OBS  = int(round(DT_OBS/DT))
SPINUP_D   = 100                 # nature-run spin-up, days
NCYCLE     = int(os.getenv("SQG_NCYC", "300"))
KENS       = int(os.getenv("SQG_K", "16"))
OBS_FRAC   = float(os.getenv("SQG_FRAC", "0.10"))
OBS_LEVEL  = os.getenv("SQG_OBS_LEVEL", "both")   # "both" | "0" | "1".  With "0" or "1" the
                                 # network is drawn from that level ONLY and OBS_FRAC is the
                                 # fraction of that level's N^2 points, not of d -- so the
                                 # other level is never observed and its error measures how
                                 # far information travels across the two surfaces.
ALPHA_V    = float(os.getenv("SQG_ALPHA", "1.0")) # cross-level taper strength; see BFree
RTPS       = float(os.getenv("SQG_RTPS", "0.3")) # LETKF relaxation-to-prior-spread.  This
                                 # is the LETKF's inflation and it was never tuned at this
                                 # resolution: at the default 0.3 the filter runs at
                                 # spread/RMSE 0.54, i.e. under-dispersive by about a factor
                                 # of two, which is what its occasional runaway seeds are.
INIT       = os.getenv("SQG_INIT", "clim")       # "clim" | "warm"
WARM_RATIO = float(os.getenv("SQG_WARM", "0.2")) # warm start: X_k = T_0 + ratio*(C_k - Cbar).
                                 # The perturbations are CLIMATOLOGICAL ANOMALIES, scaled --
                                 # not white noise, which would put energy at every scale and
                                 # is nothing like a forecast ensemble in a spectral model.
                                 # ratio is the initial spread as a fraction of the
                                 # climatological standard deviation.
SIGMA_OBS  = 1.0                 # K, for the linear operator.  For arctan it is rescaled
                                 # below to 1/s, so that the equivalent state-space error at
                                 # theta = 0 is the same 1 K: sigma_eff(theta) = sigma_obs/h' =
                                 # 1 + (theta/s)^2.  The best observations are then exactly as
                                 # informative as in the linear case and everything that differs
                                 # is attributable to the nonlinearity, not to a larger error.
OBS_KIND   = os.getenv("SQG_OBS", "linear")       # linear | arctan | arctan_k | sin
                                 # M_op/M_prime below.
ARCTAN_S   = float(os.getenv("SQG_ARCTAN_S", "6.637"))
                                 # arctan scale in K.  The dimensionless strength of the
                                 # nonlinearity is sigma_theta/s = how many transition widths
                                 # the state spans; at the default s = 6.637 (the
                                 # climatological std) that ratio is 1.05, i.e. only 2.2 % of
                                 # the state is saturated and no observation has an equivalent
                                 # error exceeding the climatological spread.  The L96 arctan
                                 # experiment runs at sigma/s = 3.67, 3.5x stronger.  Reduce s
                                 # to strengthen it.
                                 # h(x) = arctan(x/s) puts the transition at |theta| = s, so
                                 # 62 % of observations sit in the near-linear part and the
                                 # rest saturate: sigma_eff = 1 + (theta/s)^2 K spans 1 to 13.
                                 # s = 2 K would leave 27 % of the observations useless;
                                 # s = 13 K would make the operator effectively linear.
LLOC       = 2000e3              # GC localization SUPPORT in m -- the distance at which the
                                 # taper reaches zero.  sqgturb's gaspcohn(r) is normalized so
                                 # taper = 0 at r = 1, i.e. its hcovlocal_scale IS the support;
                                 # our rho_gc uses the half-support convention (zero at rr = 2),
                                 # so it must be called as rho_gc(dist/(LLOC/2)).  Getting this
                                 # wrong doubles the radius: measured, a single analysis reduces
                                 # the error by 21 % at a 2000 km support and only 11 % at 4000 km,
                                 # and cycling with 4000 km diverges (rmse 5 -> 39 K by cycle 60).
NT         = int(os.getenv("SQG_NT", "400"))       # reverse-ODE steps (a FLOOR when
                                 # SQG_ADAPT_NT=1; see steps_needed)
NT_MAX     = int(os.getenv("SQG_NT_MAX", "4000"))
ADAPT_NT   = os.getenv("SQG_ADAPT_NT", "1") == "1"
SAFETY_NT  = float(os.getenv("SQG_SAFETY_NT", "1.2"))
                                 # Measured lambda_max(B) (power iteration) and the N_t the
                                 # stability bound asks for at cycle 1 from a climatological
                                 # ensemble, i.e. the stiffest point of the whole run:
                                 #   N=32   lam 6.8    needs  82   amp at N_t=400 = 0.41
                                 #   N=64   lam 26.4   needs 327   amp at N_t=400 = 1.64
                                 #   N=128  lam 104.0  needs 1419  amp at N_t=400 = 7.09
                                 # lam_max grows as N^2 at fixed physical localization radius
                                 # (6.8/26.4/104.0 are in ratio 3.9, 3.9).  The bound is
                                 # accurate, not conservative: it predicts stable / marginal /
                                 # divergent and all three match what the runs did.  N=64 was
                                 # therefore running at 82 % of the stability limit all along.
                                 # With safety 1.2 the requirement stays under the 400 floor at
                                 # N=32 and N=64, so this is a strict no-op there and only
                                 # engages at N=128.
T_STOP     = 0.1
EPS_B      = 1e-6
STD_FLOOR  = 1e-8
D          = 2*N*N

SIN_S      = float(os.getenv("SQG_SIN_S", "2.1"))   # curvature scale of the "sin" operator
SIN_A      = float(os.getenv("SQG_SIN_A", "0.8"))   # amplitude; h' = 1 + a cos(.), so a < 1
                                 # keeps h strictly monotone.

if OBS_KIND == "sin":
    # h(x) = x + a s sin(x/s).  arctan ties its curvature scale and its saturation scale to
    # the same parameter, which is why it cannot be made to curve across the ENSEMBLE (spread
    # 2.1 K) without saturating across the STATE (sigma 6.7 K) and destroying the information:
    # the two requirements are s < 2.1 and s > 2.8.  This operator separates them.  Setting
    # s to the ensemble spread puts a full curvature oscillation inside the ensemble, where a
    # single statistical linearisation cannot represent it, while h' stays in [1-a, 1+a] so
    # no observation is ever uninformative.  h is in K like x, so sigma_obs keeps its 1 K
    # value and the configuration is directly comparable with the linear experiment.
    M_op     = lambda v: v + SIN_A*SIN_S*np.sin(v/SIN_S) if xp is np else v + SIN_A*SIN_S*xp.sin(v/SIN_S)
    M_prime  = lambda v: 1.0 + SIN_A*(np.cos(v/SIN_S) if xp is np else xp.cos(v/SIN_S))
    SIGMA_OBS = 1.0
elif OBS_KIND == "arctan_k":
    # The SAME operator as "arctan", rescaled by s so that h is in K like x.  Then h'(0) = 1
    # and h -> x as s -> inf, so sigma_obs is quoted in K and is directly comparable with the
    # linear experiment's.  ("arctan" is this operator at sigma_obs = 1 K; its sigma_obs = 1/s
    # convention is the s-rescaled version of exactly that.)
    #
    # The point of the rescaling is to make SQG_SIGMA_OBS meaningful.  For a nonlinear h the
    # noise lives in the units of h, so sigma_obs does not say how well the state is pinned
    # down; that is sigma_eff = sigma_obs/|h'| = sigma_obs (1 + (theta/s)^2), which VARIES
    # with the state.  At s = 2.3 K and sigma_obs = 1 K the median sigma_eff over the nature
    # trajectory is 6.41 K against the linear experiment's 1 K, and 48.8 % of observations
    # have sigma_eff above the climatological std of 6.68 K -- they say less about the state
    # than climatology does.  A linear-vs-arctan comparison at a common sigma_obs therefore
    # differs in TWO ways, curvature and information, and even an exact Bayesian filter would
    # lose ground on the second.  Setting sigma_obs = 0.1559 K brings the median sigma_eff
    # back to 1 K, which removes the information difference and leaves the curvature.  That
    # is a controlled diagnostic, not a more realistic instrument: a saturating instrument
    # really does lose information in the tails.  Both conventions are worth running.
    M_op     = lambda v: ARCTAN_S*np.arctan(v/ARCTAN_S)
    M_prime  = lambda v: 1.0/(1.0 + (v/ARCTAN_S)**2)
    SIGMA_OBS = 1.0
elif OBS_KIND == "arctan":
    M_op     = lambda v: np.arctan(v/ARCTAN_S)
    M_prime  = lambda v: (1.0/ARCTAN_S)/(1.0 + (v/ARCTAN_S)**2)
    SIGMA_OBS = 1.0/ARCTAN_S
else:
    M_op     = lambda v: v
    M_prime  = lambda v: np.ones_like(v)

# Explicit sigma_obs, in the units of whichever h was selected above.  Unset leaves each
# operator's own convention untouched, so this cannot change any existing experiment.
_SO = os.getenv("SQG_SIGMA_OBS")
if _SO is not None:
    SIGMA_OBS = float(_SO)

# adaptive layer (d > 100 -> one-sided, as established on L96)
GAMMA0, GAMMA_LO, GAMMA_HI = 0.1, 0.05, 0.5
P_GAIN, W_SMOOTH = 0.5, 0.3
TRIG, DECAY = 1.05, 0.97          # legacy rule only; the simple rule has neither

# Which gamma rule the analysis uses.  "simple" is the published one:
#
#     gamma <- clip(gamma * rho_s**P_GAIN, GAMMA_LO, GAMMA_HI)
#
# one multiplicative feedback with a fixed point exactly at rho_s = 1, which is the
# statistically meaningful condition.  P_GAIN = 1/2 because rho is a variance ratio and gamma
# a variance.  "legacy" keeps the dead band and the geometric decay it replaced:
#
#     rho_s > TRIG : gamma <- min(gamma*(rho_s/TRIG)**P_GAIN, GAMMA_HI)
#     otherwise    : gamma <- max(gamma*DECAY, GAMMA0)
#
# which has four constants and no interior fixed point -- it only ever rests on a bound or
# oscillates about the dead band.  Both are kept so the two can be compared on one run; see
# notebooks/gamma_controller.ipynb, where on Lorenz-96 with the arctan operator they are
# statistically indistinguishable over 40 seeds (paired +0.045 K, SE 0.035, 20/40 each way)
# and both survive 40/40 against the LETKF's 29/40.
GAMMA_RULE = os.getenv("SQG_GAMMA_RULE", "simple")


def update_gamma(gamma, rho_s):
    """One analysis cycle of the prior-scale controller."""
    if GAMMA_RULE == "simple":
        return float(min(max(gamma*rho_s**P_GAIN, GAMMA_LO), GAMMA_HI))
    if GAMMA_RULE == "legacy":
        if rho_s > TRIG: return min(gamma*(rho_s/TRIG)**P_GAIN, GAMMA_HI)
        return max(gamma*DECAY, GAMMA0)
    raise ValueError("SQG_GAMMA_RULE must be 'simple' or 'legacy', got %r" % GAMMA_RULE)

# ---------------------------------------------------------------- model wrapper
def _clamp_sinhmu(m):
    """Keep sinh(mu) finite and away from the float32 ceiling, for the GPU path.

    sqgturb computes sinh(mu) in float64 and casts to float32, where mu > 88.7 overflows to
    inf.  On the host that is harmless -- invert() only ever divides by sinhmu, and
    complex64/inf is 0 in numpy, which is the correct limit: large mu means the vertical
    penetration depth H/mu is negligible and the two surfaces are decoupled at that scale.

    cupy's complex-by-real division is NOT harmless there: it returns NaN for a divisor at or
    near the float32 ceiling (measured: fine up to 1e30, NaN at 1e38 and at inf).  At N = 512
    10.3 % of modes overflow and the whole field is NaN within ten steps, silently; at N = 256
    nothing overflows (mu_max = 56.9) and the GPU path is correct, which is why this went
    unnoticed.

    Clamping to 1e30 keeps the division well inside cupy's range.  The modes affected have a
    true 1/sinh(mu) below 7.2e-39 and contribute 4.8e-51 of the inversion, so 1e-30 stands in
    for zero to far beyond float32 precision.
    """
    import numpy as _np
    if hasattr(m, "sinhmu"):
        m.sinhmu = _np.minimum(_np.nan_to_num(m.sinhmu, posinf=1e30), _np.float32(1e30)).astype(m.sinhmu.dtype)
    return m

def make_model(theta):
    """theta (2,N,N) in K -> an SQG instance carrying it as PV."""
    return _clamp_sinhmu(_sqg.SQG(np.ascontiguousarray(gpu.asnumpy(theta)/SCALE, dtype=np.float32),
                    nsq=NSQ, f=F, U=U, H=H, r=0, tdiab=10*86400, dt=DT,
                    diff_order=8, diff_efold=DIFF_EFOLD, threads=1, precision="single"))

_MODEL = [None]          # one cached instance; the constructor is expensive and its
                         # precomputed spectral operators do not depend on the state

def _enable_batch():
    """Install the shape-generic model methods.  Idempotent; see sqg_batch.py."""
    import sqg_batch
    sqg_batch.enable(_sqg, xp)

def _model():
    """The cached model.  Under SQG_GPU=1 it is constructed on the HOST (the constructor
    allocates with host scalars) and then moved to the device once; the module-level FFTs
    and allocators are swapped at the same time and stay swapped."""
    if _MODEL[0] is None:
        m = make_model(np.zeros((2, N, N)))
        if gpu.ENABLED:
            m = gpu.model_to_gpu(m)
            gpu.patch_model_module(_sqg)
        if BATCH:
            _enable_batch()
        _MODEL[0] = m
    return _MODEL[0]

def advance(theta, nsteps):
    """Advance one state (2,N,N) in K.  threads=1 deliberately: pyfftw's threading is pure
    overhead at this size and costs a factor of 70.  The model instance is reused rather
    than rebuilt -- SQG.advance(pv=...) reseeds the spectral state, and the constructor is
    far more expensive than the integration at these sizes."""
    m = _model()
    return m.advance(timesteps=nsteps, pv=xp.ascontiguousarray(theta/SCALE, dtype=xp.float32))*SCALE

BATCH = os.getenv("SQG_BATCH", "1") == "1"

def advance_ens(E, nsteps):
    """Advance the ensemble.  E is (K, 2, N, N) in K.

    With SQG_BATCH=1 (the default) the whole ensemble goes through the model as one
    (K, 2, N, N) array.  The model was already on the GPU but was being driven one member at
    a time, so each FFT acted on a single 32768-point field -- about 600x off what the device
    can do, entirely kernel-launch bound.  Batching measured 6.5x faster end to end at N=128.

    It is NOT bit-identical to the per-member path, and the reason is worth recording so that
    a future mismatch is not misread as a bug.  cuFFT selects a different kernel for batches
    of 16 or more: a bare rfft2/irfft2 round-trip agrees exactly for K <= 8 and differs by
    2.3e-7 -- float32 epsilon -- at K >= 16.  One RK4 step amplifies that to 2.2e-4 and 36
    steps to 3.4e-3, because the Jacobian acts on grid-scale structure whose eddy turnover
    time is only ~2 h.  Both paths are legitimate float32 discretisations of the same
    equations; they are different random realisations of the same experiment, so results
    produced under the two cannot be paired seed by seed, only compared as statistics.
    Cached nature runs are unaffected: they advance a single state (K = 1).
    """
    if BATCH:
        m = _model()
        return m.advance(timesteps=nsteps,
                         pv=xp.ascontiguousarray(E/SCALE, dtype=xp.float32))*SCALE
    return xp.stack([advance(E[k], nsteps) for k in range(E.shape[0])])

# ---------------------------------------------------------------- localization
def rho_gc(rr):
    rr = np.abs(np.asarray(rr, float)); out = np.zeros_like(rr)
    n = rr <= 1; x = rr[n]; out[n] = 1-(5/3)*x**2+(5/8)*x**3+0.5*x**4-0.25*x**5
    f = (rr > 1) & (rr < 2); x = rr[f]
    out[f] = 4-5*x+(5/3)*x**2+(5/8)*x**3-0.5*x**4+x**5/12-2/(3*x)
    return out

def build_taper(N, L_dom, Lloc, alpha=1.0):
    """GC on the doubly periodic 2D distance, replicated over the two levels with no
    vertical localization (sqg_lgetkf_cv.py does the same: only two levels, strongly
    coupled through the spectral inversion)."""
    g = np.arange(N)*(L_dom/N)
    dx = np.abs(g[:, None]-g[None, :]); dx = np.minimum(dx, L_dom-dx)
    Dist = np.sqrt(dx[:, :, None, None]**2 + dx[None, None, :, :]**2)
    Dist = Dist.transpose(0, 2, 1, 3).reshape(N*N, N*N)
    C1 = rho_gc(Dist/(Lloc/2.0))          # Lloc is the SUPPORT; see the note at its definition
    C = np.empty((2*N*N, 2*N*N))
    C[:N*N, :N*N] = C1; C[N*N:, N*N:] = C1
    C[:N*N, N*N:] = alpha*C1; C[N*N:, :N*N] = alpha*C1
    return C

# The dense taper is only needed by the O(d^2)/O(d^3) reference path.  At N=128 it is
# 32768^2 doubles = 8.6 GB, so build it only when that path is actually selected.
BFREE = os.getenv("SQG_BFREE", "1") == "1"
BURN  = int(os.getenv("SQG_BURN", "0"))    # tests only; see _DenseB.draw
TAPER = None if BFREE else build_taper(N, L_DOM, LLOC, ALPHA_V)
BF    = BFree(N, L_DOM, LLOC, alpha=ALPHA_V) if BFREE else None
# (the taper eigenvalues are moved to the device lazily by BFree._dev, keyed on the
#  backend of the array being transformed -- no eager conversion here)
if BFREE and BF.mu_min < 0:
    raise RuntimeError("2D GC taper is not PSD on this grid: mu_min = %.3e" % BF.mu_min)

# ---------------------------------------------------------------- CI-EnSF
def build_B(U_):
    """Dense reference path: O(d^2) to form B, O(d^3) to factor it.  Kept for verification."""
    A = U_ - U_.mean(0)
    P = TAPER*((A.T @ A)/(KENS-1))
    Bm = (P + EPS_B*np.eye(D))/(1.0+EPS_B)
    lam, Q = np.linalg.eigh(0.5*(Bm+Bm.T))
    build_B.last = (float(lam.min()), float(lam.max()))
    lam = np.maximum(lam, 0.0)
    return (Q*lam) @ Q.T, Q*np.sqrt(lam)

class BOps:
    """The only two things the reverse ODE asks of B: apply it, and draw from N(0, B).

    B = (C o P + eps I)/(1+eps) with P = A^T A/(K-1).  Both operations avoid B entirely:

        B v = [ (1/(K-1)) sum_k a_k .* (C (a_k .* v)) + eps v ] / (1+eps)
        z   = [ (1/sqrt(K-1)) sum_k a_k .* (C^1/2 xi_k) + sqrt(eps) xi_0 ] / sqrt(1+eps)

    The ridge splits off exactly: N(0, P) + N(0, eps I) independent is N(0, P + eps I).
    Cost O(K d log d) per application instead of O(d^2), and no eigendecomposition at all,
    which is what actually dominated (80 s at N=64).  The eigenvalue clip the dense path
    applies is not needed: C is PSD (mu_min checked at import) and P is a Gram matrix, so
    C o P is PSD by the Schur product theorem -- the clip only ever removed roundoff."""
    __slots__ = ("A", "d")
    def __init__(self, U_):
        self.A = xp.ascontiguousarray(U_ - U_.mean(0)); self.d = U_.shape[1]
    def apply(self, V):
        return (BF.matvec(self.A, V) + EPS_B*V)/(1.0+EPS_B)
    def draw(self, n, rng):
        z = BF.draw(self.A, n, rng)
        if EPS_B > 0:
            z = z + np.sqrt(EPS_B)*gpu.asarray(rng.standard_normal((n, self.d)))
        return z/np.sqrt(1.0+EPS_B)

class _DenseB:
    """Same interface, dense path, so the two can be swapped and compared."""
    __slots__ = ("Bm", "Lm")
    def __init__(self, U_): self.Bm, self.Lm = build_B(U_)
    def apply(self, V): return V @ self.Bm
    def draw(self, n, rng):
        # SQG_BURN discards extra normals before the draw.  The B-free path consumes K*d
        # normals where the dense path consumes d, which shifts the WHOLE downstream stream
        # (observation networks, observation noise) for every later cycle.  Setting
        # SQG_BURN=K-1 reproduces exactly that stream consumption with the dense algorithm
        # unchanged, giving the correct null for "independent random stream, same algorithm".
        if BURN: rng.standard_normal((BURN, self.Bm.shape[0]))
        return rng.standard_normal((n, self.Bm.shape[0])) @ self.Lm.T

def lam_max(B, d, rng, nit=20):
    """Largest eigenvalue of B by power iteration.  Only needs matvecs, so it costs ~20 of
    the 400+ the reverse ODE already does (~5 %) and works at d = 32768 where eigh cannot."""
    v = gpu.asarray(rng.standard_normal((1, d))); v /= float(xp.linalg.norm(v)); lam = 1.0
    for _ in range(nit):
        w = B.apply(v); lam = float(xp.linalg.norm(w))
        if lam <= 0: return 1.0
        v = w/lam
    return lam

def steps_needed(lam, std_o, mprime, r_eff, safety=SAFETY_NT):
    """N_t the explicit-Euler reverse ODE needs to stay stable.

    Linearizing the likelihood term, d lik/d z_obs = -(std_o h')^2/r, and in the update
    (1/2) sigma^2(t) (1-t) = (1+t)/2 -> 1 as t -> 1, so the stiff mode is amplified by
    dt (std_o h')^2/r lambda_max(B) per step; explicit Euler needs that below 2.  Hence

        N_t > safety (1 - t_stop) <(std_o h')^2/r> lambda_max(B) / 2.

    The gain scales with the ensemble SPREAD, so the requirement peaks during spin-up from a
    climatological ensemble and falls by ~30x once the filter converges -- which is why this
    is adaptive rather than a fixed per-resolution N_t, and why annealing (which inflates r)
    relieves it directly.  Measured at N=128, d=32768: lambda_max = 107, <(std_o)^2/r> = 30.4
    at cycle 1, so N_t = 400 gave an amplification of 7.3 and the analysis overflowed."""
    g = float(xp.mean((std_o*mprime)**2/gpu.asarray(r_eff)))
    return int(np.ceil(safety*(1.0-T_STOP)*g*lam/2.0))

def cvt_analysis(Xf, y, obs, rng, gamma, r_eff, z0=None):
    shift = Xf.mean(0); std = xp.maximum(Xf.std(0, ddof=1), STD_FLOOR)
    U_ = (Xf - shift)/std
    B = BOps(U_) if BFREE else _DenseB(U_)
    off_o, std_o = shift[obs], std[obs]
    Q_ = obs.size
    yk = y + gpu.asarray(rng.normal(scale=np.sqrt(np.asarray(r_eff)), size=(KENS, Q_)))
    if ADAPT_NT:
        nt = int(np.clip(steps_needed(lam_max(B, D, rng), std_o,
                                      M_prime(Xf[:, obs].mean(0)), r_eff), NT, NT_MAX))
    else:
        nt = NT
    cvt_analysis.last_nt = nt
    dt = (1.0-T_STOP)/nt
    z = B.draw(KENS, rng) if z0 is None else z0.copy()   # z0: tests only, to
                                                         # share the draw across paths
    for n in range(nt):
        t = 1.0-(n+1)*dt
        c = (1.0-t)**2*gamma + t
        Bs = -(z - (1.0-t)*U_)/c
        xs = std_o*z[:, obs] + off_o                       # de-standardize for h
        lik = xp.zeros_like(z)
        lik[:, obs] = -(std_o*M_prime(xs))*(M_op(xs) - yk)/gpu.asarray(r_eff)
        Bs = Bs + (1.0-t)*B.apply(lik)
        z = z - dt*(-z/(1.0-t) - 0.5*((1.0+t)/(1.0-t))*Bs)
        if not bool(xp.isfinite(z).all()):
            raise FloatingPointError("reverse ODE diverged at step %d" % n)
    return std*z + shift

def innovation_ratio(Xf, y, obs, r):
    hx = M_op(Xf[:, obs]); d = y - hx.mean(0)
    return float(d @ d / xp.sum((1.0+1.0/KENS)*hx.var(0, ddof=1) + gpu.asarray(r)))

# ---------------------------------------------------------------- LETKF baseline
def rho_gc_xp(rr):
    """Gaspari-Cohn, backend-generic (numpy or cupy).  Same function as rho_gc, written
    without boolean fancy-indexing so it runs on the device."""
    rr = abs(rr)
    x = xp.minimum(rr, 2.0)
    near = 1-(5/3)*x**2+(5/8)*x**3+0.5*x**4-0.25*x**5
    xf = xp.maximum(x, 1e-12)
    far  = 4-5*xf+(5/3)*xf**2+(5/8)*xf**3-0.5*xf**4+xf**5/12-2/(3*xf)
    return xp.where(rr <= 1, near, xp.where(rr < 2, far, 0.0))


def letkf_loop(Xf, y, obs, Lloc, rtps=0.3):
    """Reference implementation: one K x K eigendecomposition per grid point, in Python.

    Kept because letkf_batched is verified against it.  At N = 128 it is ~32768 iterations
    per analysis and is far too slow to cycle with."""
    xbar = Xf.mean(0); Xb = Xf - xbar
    hx = M_op(Xf[:, obs])                       # h applied PER MEMBER, no linearization
    Yb = hx - hx.mean(0); d = y - hx.mean(0)
    g = np.arange(N)*(L_DOM/N)
    dx = np.abs(g[:, None]-g[None, :]); dx = np.minimum(dx, L_DOM-dx)
    Xa = np.empty_like(Xf)
    r = SIGMA_OBS**2
    oi = obs % (N*N); oy, ox = oi//N, oi % N
    for j in range(D):
        ji = j % (N*N); jy, jx = ji//N, ji % N
        dist = np.sqrt(dx[jy, oy]**2 + dx[jx, ox]**2)
        w = rho_gc(dist/Lloc)
        sel = w > 0
        if not sel.any(): Xa[:, j] = Xf[:, j]; continue
        Yl = Yb[:, sel]; inv = w[sel]/r
        A = (KENS-1)*np.eye(KENS) + (Yl*inv) @ Yl.T
        lam, Qm = np.linalg.eigh(0.5*(A+A.T)); lam = np.maximum(lam, 1e-12)
        Pa = (Qm/lam) @ Qm.T
        wa = Pa @ ((Yl*inv) @ d[sel])
        W = (Qm*np.sqrt((KENS-1)/lam)) @ Qm.T
        Xa[:, j] = xbar[j] + Xb[:, j] @ (wa[:, None] + W)
    sa = Xa.std(0, ddof=1); sf = Xf.std(0, ddof=1)
    fac = (1-rtps) + rtps*sf/np.maximum(sa, STD_FLOOR)
    return Xa.mean(0) + (Xa - Xa.mean(0))*fac


LETKF_CHUNK = int(os.getenv("SQG_LETKF_CHUNK", "2048"))

def letkf(Xf, y, obs, Lloc, rtps=0.3):
    """Batched LETKF.  Identical arithmetic to letkf_loop, reorganised two ways.

    (1) The localization uses only the HORIZONTAL distance (obs % N^2), so the two levels at
        the same horizontal point get the same weights and therefore the SAME K x K
        transform.  It is computed once per horizontal point, not once per state component:
        N^2 solves instead of 2N^2.  This is exact, not an approximation.

    (2) Observations outside the support enter with weight zero rather than being selected
        out.  A zero weight contributes nothing to (Y diag(w/r) Y^T), so the dense form is
        the same matrix the selected form builds, and it batches.  The no-observation case
        needs no special branch either: A = (K-1)I gives wa = 0 and W = I, i.e. Xa = Xf,
        which is what the loop's `continue` does.

    Chunked over horizontal points to bound the (chunk, n_obs) weight matrix."""
    nh = N*N
    xbar = Xf.mean(0); Xb = Xf - xbar
    hx = M_op(Xf[:, obs])
    Yb = hx - hx.mean(0); d = y - hx.mean(0)
    r = SIGMA_OBS**2

    g = xp.arange(N)*(L_DOM/N)
    dxm = xp.abs(g[:, None]-g[None, :]); dxm = xp.minimum(dxm, L_DOM-dxm)
    oi = gpu.asarray(np.asarray(obs) % nh)
    oy, ox = oi//N, oi % N

    G    = (Yb[:, None, :]*Yb[None, :, :]).reshape(KENS*KENS, -1)   # (K^2, nobs)
    Ybd  = Yb*d                                                     # (K, nobs)
    Xbr = Xb.reshape(KENS, 2, nh)
    Xa = xp.empty_like(Xbr)
    eyeK = xp.eye(KENS)
    for c0 in range(0, nh, LETKF_CHUNK):
        c1 = min(c0+LETKF_CHUNK, nh)
        jj = xp.arange(c0, c1); jy, jx = jj//N, jj % N
        dist = xp.sqrt(dxm[jy][:, oy]**2 + dxm[jx][:, ox]**2)     # (chunk, nobs)
        w = rho_gc_xp(dist/Lloc)/r
        # A_j = (K-1) I + Yb diag(w_j) Yb^T
        # A_j = (K-1)I + Yb diag(w_j) Yb^T for every j in the chunk, as ONE gemm.
        # einsum("kn,cn,ln->ckl") would materialise a (chunk, K, nobs) intermediate --
        # 1 GB at N=128 with chunk=2048 -- so the outer products are precomputed once into
        # G (K^2, nobs), 8 MB, and the chunk's weights contract against them.
        A = (KENS-1)*eyeK + (w @ G.T).reshape(c1-c0, KENS, KENS)
        A = 0.5*(A + xp.swapaxes(A, 1, 2))
        lam, Q = xp.linalg.eigh(A)
        lam = xp.maximum(lam, 1e-12)
        Pa = xp.einsum("ckm,cm,clm->ckl", Q, 1.0/lam, Q)
        rhs = w @ Ybd.T
        wa = xp.einsum("ckl,cl->ck", Pa, rhs)
        W = xp.einsum("ckm,cm,clm->ckl", Q, xp.sqrt((KENS-1)/lam), Q)
        T = wa[:, :, None] + W            # the loop's (wa[:, None] + W), per point
        Xa[:, :, c0:c1] = xp.einsum("klj,jkm->mlj", Xbr[:, :, c0:c1], T)
    Xa = Xa.reshape(KENS, D) + xbar
    sa = Xa.std(0, ddof=1); sf = Xf.std(0, ddof=1)
    fac = (1-rtps) + rtps*sf/xp.maximum(sa, STD_FLOOR)
    return Xa.mean(0) + (Xa - Xa.mean(0))*fac


# ---------------------------------------------------------------- experiment
def nature_run(seed):
    rs = np.random.RandomState(seed)
    pv = rs.normal(0, 100., size=(2, N, N)).astype(np.float32)
    ne = 20
    xx = np.arange(0, 2*np.pi, 2*np.pi/N); yy = np.arange(0, 2*np.pi, 2*np.pi/N)
    xx, yy = np.meshgrid(xx, yy)
    pv[1] += 2000.*(np.sin(xx/2)**(2*ne)*np.sin(yy)**ne)
    for k in range(2): pv[k] -= pv[k].mean()
    mod = _clamp_sinhmu(_sqg.SQG(pv, nsq=NSQ, f=F, U=U, H=H, r=0, tdiab=10*86400, dt=DT,
                   diff_order=8, diff_efold=DIFF_EFOLD, threads=1, precision="single"))
    # A nature run is a SINGLE member, so the ensemble batching does not apply, but the FFTs
    # still belong on the device: measured 153.9 ms/step on the host against 5.7 ms/step on
    # the device at N = 512, a factor of 27 that decides whether one run takes 7.8 h or 0.3 h.
    # The module patch is lazy and is normally tripped by _model(), which nature_run never
    # calls -- so every nature run in this project until now was integrated on the host.
    #
    # patch_model_module must come AFTER the constructor above (it swaps the module's
    # allocators), and it is undone before returning so that a process which generates a
    # nature run and then builds a filter model is not left with a patched module that the
    # SQG constructor cannot run under.
    # Default ON, after two validations.  (i) With the spin-up cut to 2 days the two
    # trajectories have not decorrelated and can be compared frame by frame: the relative
    # difference grows from 9.2e-07 to 3.9e-06 over five days -- float32 epsilon amplified by
    # chaos -- and the sampled std agrees to five decimals.  (ii) Three full device runs
    # against the three host ones differ by 1.3-1.6 pooled standard errors on std, the
    # 20-frame std and the 99th percentile, with the SIGN flipping between them.  An earlier
    # single device run that looked 7 % low on one of those statistics sits inside the device
    # runs' own spread; it was realisation scatter, not an implementation difference.
    # Set SQG_NATURE_GPU=0 to force the host path.
    on_gpu = gpu.ENABLED and os.getenv("SQG_NATURE_GPU", "1") == "1"
    if on_gpu:
        mod = gpu.model_to_gpu(mod)
        gpu.patch_model_module(_sqg)
    try:
        mod.advance(timesteps=int(SPINUP_D*86400/DT))
        T = np.empty((NCYCLE, 2, N, N), np.float32)
        NCLIM = int(os.getenv("SQG_CLIM", "200"))
        C = np.empty((NCLIM, 2, N, N), np.float32)   # climatology pool, for the IC
        for n in range(NCYCLE): T[n] = gpu.asnumpy(mod.advance(timesteps=NSTEP_OBS))*SCALE
        for n in range(NCLIM):  C[n] = gpu.asnumpy(mod.advance(timesteps=NSTEP_OBS*4))*SCALE
    finally:
        if on_gpu: gpu.unpatch_model_module(_sqg)
    return T, C

def run(seed, mode, kappa=0.0, nc=NCYCLE, verbose=True):
    """mode in {'static','adapt','letkf'}"""
    t_start = time.time()
    T, C = nature_run(seed)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(C), size=KENS, replace=False)
    # The ensemble lives on whichever backend is selected for the WHOLE cycle.  Before this
    # was made explicit, run() mixed host and device arrays: the analysis was only ever
    # exercised on the GPU by calling cvt_analysis directly, so the seams in this loop went
    # unnoticed.  cupy does not accept numpy operands, so every one of them is an error, not
    # a silent transfer.
    if INIT == "warm":
        A0 = C[idx].reshape(KENS, D).astype(np.float64)
        A0 = A0 - A0.mean(0)                       # climatological anomalies
        X0 = T.reshape(-1, D)[0].astype(np.float64) + WARM_RATIO*A0
    elif INIT == "clim":
        X0 = C[idx].reshape(KENS, D).astype(np.float64).copy()
    else:
        raise ValueError("SQG_INIT must be 'clim' or 'warm', got %r" % INIT)
    X  = gpu.asarray(X0)
    # T may hold more cycles than this call uses (it is cached per seed), so slice rather
    # than reshape to nc -- reshape(nc, D) silently assumed len(T) == nc.
    if len(T) < nc:
        raise ValueError("nature run has %d cycles, run() asked for %d" % (len(T), nc))
    Tf = T.reshape(len(T), D).astype(np.float64)[:nc]         # truth stays on the HOST
    Tg = gpu.asarray(Tf) if gpu.ENABLED else Tf               # ... and a device copy for y
    gamma, rho_s, hist = GAMMA0, 1.0, []
    # Optional hook: when run.field_sink is a list, the analysis ENSEMBLE MEAN of every cycle
    # is appended to it as a (2, N, N) float32 array.  Used by the animation script; costs
    # nothing when unset, and deliberately stores the mean rather than the full ensemble so
    # that 100 cycles are 13 MB rather than 200.
    sink = getattr(run, "field_sink", None)
    # Optional hook: when run.timing_sink is a list, (analysis_s, forecast_s) is appended for
    # every cycle.  Inert when unset, like field_sink.  The split matters because the two scale
    # differently with resolution -- the analysis as N^4 and the forecast as N^3 log N -- so a
    # single per-cycle number cannot be extrapolated.
    tsink = getattr(run, "timing_sink", None)
    # Optional hook: when run.ens_hook is callable it is given the ANALYSIS ENSEMBLE and the
    # cycle index every cycle, and whatever it wants to measure it measures in place.  The
    # ensemble is passed rather than stored because keeping it would be 2.1 GB at N = 128
    # over 100 cycles, while a diagnostic reduced on the fly costs nothing.  Inert when unset.
    ehook = getattr(run, "ens_hook", None)
    nh = N*N
    if OBS_LEVEL == "both":
        pool, nobs = np.arange(D), int(OBS_FRAC*D)
    else:
        lev  = int(OBS_LEVEL)
        pool = np.arange(nh) + lev*nh          # that level only
        nobs = int(OBS_FRAC*nh)                # fraction OF THAT LEVEL
    for n in range(nc):
        obs = np.sort(rng.choice(pool, size=nobs, replace=False))  # network redrawn each cycle
        # y is formed on the device because it meets device arrays in cvt_analysis; the
        # observation noise is still drawn with the HOST rng so that CPU and GPU runs
        # consume an identical random stream and remain directly comparable.
        y = M_op(Tg[n][obs]) + gpu.asarray(rng.normal(scale=SIGMA_OBS, size=nobs))
        r_eff = np.full(nobs, SIGMA_OBS**2)                   # host: steps_needed/anneal use it
        t_an = time.time()
        if mode == "letkf":
            X = letkf(X, y, obs, LLOC/2.0, rtps=RTPS)   # letkf() takes the rho_gc scale,
                                                        # not the support
        else:
            d_in = y - M_op(X[:, obs]).mean(0)
            rho_s = (1-W_SMOOTH)*rho_s + W_SMOOTH*innovation_ratio(X, y, obs, r_eff)
            if mode == "adapt":
                gamma = update_gamma(gamma, rho_s)
                if kappa > 0:
                    r_eff = np.maximum(r_eff, kappa*float(xp.mean(d_in**2)))
            try:
                X = cvt_analysis(X, y, obs, rng, gamma, r_eff)
            except FloatingPointError as e:
                print("  [%s] OVERFLOW at cycle %d: %s" % (mode, n+1, e), flush=True)
                return np.array(hist), n+1
        if tsink is not None and gpu.ENABLED: xp.cuda.runtime.deviceSynchronize()
        dt_an = time.time() - t_an
        if ehook is not None: ehook(X, n)
        e  = (gpu.asnumpy(X.mean(0)) - Tf[n]).reshape(2, nh)
        vr = gpu.asnumpy(X.var(0, ddof=1)).reshape(2, nh)
        rm = float(np.sqrt((e**2).mean())); sp = float(np.sqrt(vr.mean()))
        # per level, because the whole point of a single-level network is that the two
        # levels are NOT equivalent and a pooled RMSE is dominated by the observed one
        rm0 = float(np.sqrt((e[0]**2).mean())); rm1 = float(np.sqrt((e[1]**2).mean()))
        sp0 = float(np.sqrt(vr[0].mean()));     sp1 = float(np.sqrt(vr[1].mean()))
        ef  = e.reshape(-1)
        om  = np.zeros(D, bool); om[obs] = True
        rm_o = float(np.sqrt((ef[om]**2).mean()))
        rm_u = float(np.sqrt((ef[~om]**2).mean()))
        hist.append((rm, sp, sp/rm, gamma, float(np.sqrt(r_eff.mean())), rho_s,
                     rm0, rm1, sp0, sp1, rm_o, rm_u))
        if sink is not None:
            sink.append(gpu.asnumpy(X.mean(0)).reshape(2, N, N).astype(np.float32))
        if verbose and ((n+1) % 20 == 0 or n == 0):
            print("  [%-6s] c%-4d rmse %7.3f K  spread %6.3f  sp/rms %5.2f  gamma %.3f  "
                  "sig_eff %6.3f  rho_s %5.2f  (%.1f s)"
                  % (mode, n+1, rm, sp, sp/rm, gamma, np.sqrt(r_eff.mean()), rho_s,
                     time.time()-t_start), flush=True)
        t_fc = time.time()
        X = advance_ens(X.reshape(KENS, 2, N, N), NSTEP_OBS).reshape(KENS, D)
        if tsink is not None:
            if gpu.ENABLED: xp.cuda.runtime.deviceSynchronize()
            tsink.append((dt_an, time.time() - t_fc))
        if not bool(xp.isfinite(X).all()):
            print("  [%s] FORECAST overflow at cycle %d" % (mode, n+1), flush=True)
            return np.array(hist), n+1
    return np.array(hist), None

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print("SQG  N=%d  d=%d  dx=%.0f km  Lr=%.0f km  Lloc=%.0f km (%.1f Lr)"
          % (N, D, L_DOM/N/1e3, LR/1e3, LLOC/1e3, LLOC/LR))
    print("operator: %s%s" % (OBS_KIND,
          "  s=%.2f K  sigma_obs=%.4f (sigma_eff = 1 + (theta/s)^2 K)" % (ARCTAN_S, SIGMA_OBS)
          if OBS_KIND == "arctan" else "  sigma_obs=%.1f K" % SIGMA_OBS))
    _nh = N*N
    _nob = int(OBS_FRAC*D) if OBS_LEVEL == "both" else int(OBS_FRAC*_nh)
    print("K=%d  observations: %s, %.0f%% -> %d obs (%.1f%% of d), redrawn each cycle, "
          "sigma_obs=%.4f"
          % (KENS, "both levels" if OBS_LEVEL == "both" else "LEVEL %s ONLY" % OBS_LEVEL,
             100*OBS_FRAC, _nob, 100.0*_nob/D, SIGMA_OBS))
    print("cross-level taper strength alpha = %.2f%s"
          % (ALPHA_V, "  (block diagonal: no cross-level transfer)" if ALPHA_V == 0 else ""))
    print("analysis every %.0f h  N_t=%d  %d cycles  seed %d"
          % (DT_OBS/3600, NT, NCYCLE, seed))
    print("initial ensemble: %s" % ("CLIMATOLOGICAL (states 24 h apart from the nature run)"
          if INIT == "clim" else
          "WARM, truth + %.2f x climatological anomalies" % WARM_RATIO))
    print()
    T, _ = nature_run(seed)
    print("nature run: theta std %.2f K, range [%.1f, %.1f]"
          % (T.std(), T.min(), T.max()))
    print()
    out = {}
    for mode, kap in (("letkf", 0.0), ("static", 0.0), ("adapt", 0.0), ("adapt", 0.10)):
        lab = mode if kap == 0 else "%s k=%.2f" % (mode, kap)
        print("=== %s ===" % lab, flush=True)
        h, f = run(seed, mode, kappa=kap)
        out[lab] = (h, f)
        if len(h):
            i0 = len(h)//2
            print("  --> %s: %s, last-half rmse %.3f K, sp/rms %.2f\n"
                  % (lab, "survived" if f is None else "FAILED at c%d" % f,
                     h[i0:, 0].mean(), h[i0:, 2].mean()), flush=True)
