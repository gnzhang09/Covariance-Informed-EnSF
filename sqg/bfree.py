"""B-free CI-EnSF operators.  O(K d log d) per matvec and per draw, exact (no truncation).

The Schur product factors over ENSEMBLE MEMBERS, not over taper modes:

    B = C o P,   P = A^T A/(K-1) = sum_k a_k a_k^T /(K-1)
    C o (a a^T) = diag(a) C diag(a)
    =>  B = (1/(K-1)) sum_k diag(a_k) C diag(a_k)
    =>  B v = (1/(K-1)) sum_k a_k .* [ C (a_k .* v) ]                          (1)
    =>  B = (1/(K-1)) sum_k [diag(a_k) C^1/2][diag(a_k) C^1/2]^T
    =>  z = (1/sqrt(K-1)) sum_k a_k .* (C^1/2 xi_k),   xi_k ~ N(0,I) iid       (2)

so only K applications of C (or of its square root) are needed, not N^2.  On a doubly
periodic grid the taper depends only on (i-j) mod N in each direction, so C is
block-circulant with circulant blocks: C w and C^1/2 w are FFT convolutions, O(d log d),
with the eigenvalues mu = FFT2(first row) computed once.

An earlier version summed over the TAPER's Fourier modes instead.  That is also correct but
needs r ~ N^2 modes, because a compactly supported taper has a power-law Fourier spectrum:
measured at N=32, keeping 45 % of the modes captured only 92 % of the spectrum and left a
8 % error in C.  Summing over members instead removes the truncation entirely.

Measured cost, K=16, N_t=400, inside the real driver on an idle machine (the synthetic
benchmark on random anomalies is optimistic by ~1.9x because it times only matvec and
ignores the other ~6 (K,d) array operations the reverse ODE does per step):

    N=32   d= 2048   B-free  ~5 s      dense  ~4 s     -> 0.8x  (dense BLAS3 still wins)
    N=64   d= 8192   B-free  18.2 s    dense 78.5 s    -> 4.3x
    N=128  d=32768   B-free  ~87 s     dense infeasible (8.6 GB taper, O(d^3) eigh)

The crossover is between d=2048 and d=8192.  End-to-end, MEASURED per cycle at N=64 (not
inferred by subtraction, which is how an earlier and wrong claim of "~30 s/cycle of model
integration" arose -- the model is 0.86 s):
    matrix-free  analysis 18.6 s (96 %) + model 0.86 s (4 %) = 19.5 s/cycle -> 1.6 h / 300
    dense        analysis 78.5 s        + model 0.86 s       = 79.4 s/cycle -> 6.6 h / 300
so the end-to-end gain is 4.1x, essentially the analysis ratio, because the model is only
4 % of a cycle.  The overnight run's 109 s/cycle was measured under a different thread
setting and must not be compared against the idle-machine analysis timings.

Verified against the dense path (sqg_cvt.py keeps it as _DenseB):
  B*v relative error 2e-15; the whole 400-step reverse ODE agrees to 1.3e-15 when both are
  given the same terminal draw; the draw's sample covariance matches within Monte Carlo
  error.  The dense B's smallest eigenvalue is +4.9e-2, so the PSD clip the dense path
  applies only ever removed roundoff -- C o P is PSD by the Schur product theorem (C is
  PSD, checked via mu_min at construction; P is a Gram matrix), which is why dropping the
  eigendecomposition is safe rather than an approximation.

Cycling equivalence, 40 seeds x 40 cycles at N=32, paired (all percentages vs dense):
      B-free vs dense                          +1.16%  t=+1.11  sign 28/40 p=0.017
      NULL dense 8thr vs 6thr                  -0.07%  t=-0.09  sign 22/40 p=0.64
      NULL dense+burn vs dense                 +2.73%  t=+2.33  sign 22/40 p=0.64
      B-free vs dense+burn (stream fixed)      -1.52%  t=-1.21  sign 19/40 p=0.87
The dense+burn arm runs the IDENTICAL dense algorithm and only consumes K*d normals instead
of d (SQG_BURN), i.e. it is a pure null -- yet it shows a LARGER apparent effect than B-free
does.  So neither the t-test nor the sign test is calibrated at 40 seeds here, and the
apparent 28/40 is noise.  The last row holds stream consumption fixed so that only the
algorithm differs, and is null.  Run-to-run sd is 0.38 K on a 3.46 K mean (11 %), so
40 seeds resolves only about 5 %; a 1 % difference would need ~950 seeds.  The two paths
sample provably identical Gaussians, so there is nothing there to resolve.

The two levels carry the same taper with no vertical localization, i.e. C = J (x) C1 with
J = ones(2,2).  Then (C w)_l = C1 (w_0 + w_1) for each level l, and since J = 2 u u^T with
u = (1,1)/sqrt(2), C^1/2 acts as (C^1/2 xi)_l = C1^1/2 (xi_0 + xi_1)/sqrt(2).
"""
import sys
import os
import numpy as np

def _xp(a):
    """Array module of `a`: numpy, or cupy when the caller passes device arrays.  The
    operators below are pure FFT + elementwise work, so the same code runs on either
    backend; only the allocation of the accumulator has to follow the input."""
    return sys.modules.get("cupy") if type(a).__module__.startswith("cupy") else np

BATCH       = os.getenv("SQG_BFREE_BATCH", "0") == "1"
BATCH_BYTES = float(os.getenv("SQG_BFREE_BATCH_BYTES", "5.4e8"))   # work-array cap, ~512 MB


def rho_gc(rr):
    rr = np.abs(np.asarray(rr, float)); out = np.zeros_like(rr)
    n = rr <= 1; x = rr[n]; out[n] = 1-(5/3)*x**2+(5/8)*x**3+0.5*x**4-0.25*x**5
    f = (rr > 1) & (rr < 2); x = rr[f]
    out[f] = 4-5*x+(5/3)*x**2+(5/8)*x**3-0.5*x**4+x**5/12-2/(3*x)
    return out

class BFree:
    def __init__(self, N, L_dom, support, nlev=2, alpha=1.0):
        """alpha scales the CROSS-LEVEL block of the taper:

            C(alpha) = M(alpha) (x) C1,      M = (1-alpha) I + alpha J

        i.e. the within-level block is always C1 and only the coupling between the two
        levels is scaled.  alpha = 1 is the configuration this code has always used (no
        vertical localization at all, C = J (x) C1); alpha = 0 makes C block diagonal, which
        zeroes the likelihood's contribution to an unobserved level exactly.

        M has eigenvalues 1 +/- alpha, so C is PSD only for |alpha| <= 1, and alpha = 1 sits
        on that boundary: M is singular there (the antisymmetric vertical mode has
        eigenvalue 0).  The default is therefore an endpoint of the admissible range."""
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]; M(alpha) is not PSD outside it")
        self.alpha = float(alpha)
        self.N, self.nlev, self.d = N, nlev, nlev*N*N
        g = np.arange(N)*(L_dom/N)
        gy, gx = np.meshgrid(g, g, indexing="ij")
        dy = np.minimum(gy, L_dom-gy); dx = np.minimum(gx, L_dom-gx)
        row = rho_gc(np.sqrt(dy**2+dx**2)/(support/2.0))
        mu = np.fft.fft2(row).real                  # BCCB eigenvalues of the per-level taper
        self.mu_min = float(mu.min())               # diagnostic: must be >= 0 for C to be PSD
        self.mu = np.maximum(mu, 0.0)
        self.sqrt_mu = np.sqrt(self.mu)

    # ---- per-level circulant applications, O(d log d)
    def _C1(self, W):       # W (..., N, N)
        xp = _xp(W); mu = self._dev(self.mu, xp)
        return xp.fft.ifft2(xp.fft.fft2(W)*mu).real
    def _C1_half(self, W):
        xp = _xp(W); sq = self._dev(self.sqrt_mu, xp)
        return xp.fft.ifft2(xp.fft.fft2(W)*sq).real
    def _dev(self, arr, xp):
        """Cache the taper eigenvalues on whichever device the caller is using."""
        if xp is np: return arr
        key = "_dev_" + str(id(arr))
        if not hasattr(self, "_cache"): self._cache = {}
        if key not in self._cache: self._cache[key] = xp.asarray(arr)
        return self._cache[key]

    def _apply_C(self, V):
        """C @ V^T for V (n, d) with C = ones(2,2) (x) C1."""
        n = V.shape[0]; N = self.N
        W = V.reshape(n, self.nlev, N, N)
        s = self._C1(W.sum(axis=1))                 # C1 (w_0 + w_1)
        return np.repeat(s[:, None], self.nlev, axis=1).reshape(n, self.d)

    def _apply_C_half(self, V):
        n = V.shape[0]; N = self.N
        W = V.reshape(n, self.nlev, N, N)
        s = self._C1_half(W.sum(axis=1))/np.sqrt(self.nlev)
        return np.repeat(s[:, None], self.nlev, axis=1).reshape(n, self.d)

    # ---- the two operations the filter needs
    #
    # Both are written as an explicit loop over the K members with a (nlev,N,N) accumulator.
    # Batching the loop into one (K,n,d) FFT call was tried and is 1.5-5x SLOWER: the
    # temporary is 17 MB at N=64 and 67 MB at N=128 and falls out of cache, whereas the
    # loop's accumulator is 1 MB and stays hot.  The FFT is not the bottleneck; memory is.
    # For the same reason neither routine calls np.repeat: both levels of C w are the same
    # array, so they are broadcast into the accumulator instead of being materialized.

    def matvec(self, A, V):
        """B @ V^T for V (n, d); A is the (K, d) anomaly matrix.  Eq. (1)."""
        xp = _xp(A)
        K, N, nl = A.shape[0], self.N, self.nlev
        n = V.shape[0]
        if BATCH and self.alpha == 1.0 and xp is not np:
            return self._matvec_batched(A, V)
        out = xp.zeros((n, nl, N, N))
        Vr, Ar = V.reshape(n, nl, N, N), A.reshape(K, nl, N, N)
        al = self.alpha
        for k in range(K):
            a = Ar[k]
            Wk = Vr*a
            if al == 1.0:                       # J: one transform, the original path
                out += a*self._C1(Wk.sum(axis=1))[:, None]
            elif al == 0.0:                     # I: block diagonal, no cross-level term
                out += a*self._C1(Wk)
            else:                               # (1-al) I + al J
                out += a*((1.0-al)*self._C1(Wk) + al*self._C1(Wk.sum(axis=1))[:, None])
        return out.reshape(n, self.d)/(K-1)


    def _matvec_batched(self, A, V):
        """The alpha = 1 path with the K taper terms submitted as ONE transform batch.

        The loop above hands cuFFT a batch of only ``n`` 2-D transforms per term, and ``n`` is
        the ensemble size, 16.  Stacking the terms makes the batch ``kc*n``, which is where the
        GPU actually runs: the same covariance costs 0.070 ms per vector when the caller passes
        512 vectors at once and 0.343 ms when it passes 16.

        Off by default.  The sum over k is now a reduction rather than a running ``out +=``,
        so the last bits differ from the loop and a cycling run will diverge from a published
        one -- the same hazard as the forecast batching already documented in sqg_cvt.  Enable
        with SQG_BFREE_BATCH=1 and treat it as a separate numerical configuration.
        """
        xp = _xp(A)
        K, N, nl = A.shape[0], self.N, self.nlev
        n = V.shape[0]
        Vr, Ar = V.reshape(n, nl, N, N), A.reshape(K, nl, N, N)
        out = xp.zeros((n, nl, N, N))
        # bound the (kc, n, N, N) work array; 8 bytes per element
        kc = max(1, min(K, int(BATCH_BYTES/(max(n, 1)*N*N*8))))
        for lo in range(0, K, kc):
            Ak = Ar[lo:lo+kc]                                   # (kc, nl, N, N)
            Ws = xp.einsum("ilyx,klyx->kiyx", Vr, Ak)           # (kc, n, N, N)
            CW = self._C1(Ws)                                   # one batched transform
            out += xp.einsum("klyx,kiyx->ilyx", Ak, CW)
        return out.reshape(n, self.d)/(K-1)

    def draw(self, A, n, rng):
        """n draws from N(0, B).  Eq. (2)."""
        xp = _xp(A)
        K, N, nl = A.shape[0], self.N, self.nlev
        z = xp.zeros((n, nl, N, N))
        Ar = A.reshape(K, nl, N, N)
        al = self.alpha
        # M(alpha)^{1/2} in the symmetric/antisymmetric basis: the symmetric vertical mode
        # (xi_0 + xi_1)/sqrt(2) is scaled by sqrt(1+alpha) and the antisymmetric one by
        # sqrt(1-alpha).  At alpha = 1 the antisymmetric mode vanishes and this reduces to
        # the original single shared field; at alpha = 0 it gives two independent fields.
        c_sym, c_anti = np.sqrt(1.0+al), np.sqrt(1.0-al)
        for k in range(K):
            # the normals are drawn with the HOST rng and moved to the device, so that a
            # GPU run consumes exactly the same random stream as the CPU run and the two
            # can be compared directly.  The transfer is one (n, d) array per member per
            # analysis, which is negligible next to the 400-step ODE.
            xi0 = rng.standard_normal((n, nl, N, N))
            if xp is not np: xi0 = xp.asarray(xi0)
            if al == 1.0:
                xi = xi0.sum(axis=1)[:, None]/np.sqrt(nl)
            else:
                m = xi0.mean(axis=1)[:, None]            # symmetric part
                xi = c_sym*m + c_anti*(xi0 - m)
            z += Ar[k]*self._C1_half(xi)
        return z.reshape(n, self.d)/np.sqrt(K-1)
