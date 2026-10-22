"""GPU backend for the SQG CI-EnSF driver, enabled with SQG_GPU=1.

WHAT MOVES TO THE DEVICE
    - the whole reverse ODE (cvt_analysis), including the matrix-free B operators
    - the SQG model itself (it is spectral: rfft2/irfft2 plus elementwise work)
    - the standardization, observation operator, innovation diagnostics
so a cycle runs end to end in device memory and the host sees data only at the cycle
boundary, where the RMSE/spread diagnostics are computed.

WHAT DOES NOT
    - the LETKF baseline.  It loops over the d state components doing one K x K eigh each;
      at N = 256 that is 131072 tiny decompositions, which is the worst possible shape for
      a GPU.  It stays on the host and is simply not available under SQG_GPU=1.
    - the nature run, which is computed once and cached on the host.

ORDERING CONSTRAINT.  patch_model_module() swaps the sqgturb module's FFTs and allocators
GLOBALLY and leaves them swapped, so any SQG instance constructed afterwards would be built
against device allocators with host scalars and fail.  sqg_cvt.nature_run() builds its own
instance directly, so it must run BEFORE the first analysis -- which it does, since the
nature run is what the analysis assimilates.  Do not call nature_run() after an analysis
without calling unpatch_model_module() first.

PRECISION.  The model runs in float32 (precision="single", as on the host path); the
analysis runs in float64.  The interface between them converts, so the analysis keeps the
precision the near-null modes of B need.

NUMPY DISPATCH.  cupy implements __array_ufunc__, so np.sqrt/np.tanh/np.maximum and the
rest already dispatch to the device when handed device arrays.  Only the ALLOCATORS
(np.zeros/np.empty/np.ones) do not, because they take no array argument -- hence the small
shim below, which is the whole of what the sqgturb port needs.
"""
import os
import numpy as np

ENABLED = os.getenv("SQG_GPU", "0") == "1"
xp = np
cp = None

if ENABLED:
    import cupy as _cp
    cp = _cp
    xp = _cp


def asarray(a, dtype=None):
    return xp.asarray(a, dtype=dtype) if ENABLED else np.asarray(a, dtype=dtype)

def asnumpy(a):
    return cp.asnumpy(a) if (ENABLED and isinstance(a, cp.ndarray)) else np.asarray(a)

def sync():
    if ENABLED: cp.cuda.Stream.null.synchronize()


class _NPShim:
    """numpy namespace with device allocators, for patching into the sqgturb module."""
    def __getattr__(self, k):  return getattr(np, k)
    zeros = staticmethod(lambda *a, **k: cp.zeros(*a, **k))
    empty = staticmethod(lambda *a, **k: cp.empty(*a, **k))
    ones  = staticmethod(lambda *a, **k: cp.ones(*a, **k))


def patch_model_module(mod):
    """Swap the sqgturb module's FFTs and allocators over to the device.

    Must be called AFTER any SQG instance is constructed on the host, because the
    constructor itself uses numpy allocation with host scalars.  The driver's make_model
    therefore builds on the host and then calls model_to_gpu below.
    """
    if not ENABLED: return
    mod.rfft2  = lambda a, **kw: cp.fft.rfft2(a)
    mod.irfft2 = lambda a, **kw: cp.fft.irfft2(a)
    mod.np = _NPShim()

def unpatch_model_module(mod):
    if not ENABLED: return
    from pyfftw.interfaces import numpy_fft
    mod.rfft2, mod.irfft2, mod.np = numpy_fft.rfft2, numpy_fft.irfft2, np


def model_to_gpu(m):
    """Move a constructed SQG instance's precomputed arrays to the device."""
    if not ENABLED: return m
    for k, v in list(vars(m).items()):
        if isinstance(v, np.ndarray):
            setattr(m, k, cp.asarray(v))
    return m


def report():
    if not ENABLED:
        return "GPU: off (SQG_GPU != 1)"
    p = cp.cuda.runtime.getDeviceProperties(0)
    free, total = cp.cuda.runtime.memGetInfo()
    return ("GPU: %s, %.1f/%.1f GB free, cupy %s"
            % (p["name"].decode(), free/2**30, total/2**30, cp.__version__))
