"""Cache the N=128 nature run and climatology pool, once per seed.

Every arm of the cross-layer experiment needs the same truth and the same climatological
pool for a given seed, and the run costs minutes, so it is computed once and stored.  The
file also records the configuration it was produced under; a mismatch must be an error
rather than a silently reused wrong truth.
"""
import os as _os
_SQG = _os.path.dirname(_os.path.abspath(__file__))   # repo-relative: was an absolute path
import os, sys, time, numpy as np
sys.path.insert(0, _SQG)
import sqg_cvt as S, gpu

OUT = os.getenv("SQG_NATURE_DIR", _SQG + "/nature")
os.makedirs(OUT, exist_ok=True)

def path(seed):
    return os.path.join(OUT, "nature_N%d_c%d_seed%d.npz" % (S.N, S.NCYCLE, seed))

def meta():
    return dict(N=S.N, DT=S.DT, DIFF_EFOLD=S.DIFF_EFOLD, SPINUP_D=S.SPINUP_D,
                NCYCLE=S.NCYCLE, DT_OBS=S.DT_OBS, NCLIM=int(os.getenv("SQG_CLIM", "200")))

def load(seed):
    z = np.load(path(seed), allow_pickle=False)
    m = meta()
    for k, v in m.items():
        if abs(float(z["meta_" + k]) - float(v)) > 1e-9:
            raise RuntimeError("cached nature run was made with %s=%s, now %s" % (k, z["meta_"+k], v))
    return z["T"], z["C"]

def build(seed):
    t0 = time.time()
    T, C = S.nature_run(seed)
    kw = {"meta_" + k: np.array(float(v)) for k, v in meta().items()}
    np.savez(path(seed), T=np.asarray(T, np.float32), C=np.asarray(C, np.float32), **kw)
    print("seed %d: T %s  C %s  theta std %.2f K  [%.0f s] -> %s"
          % (seed, T.shape, C.shape, float(np.asarray(T).std()), time.time()-t0, path(seed)))
    return T, C

if __name__ == "__main__":
    for seed in [int(a) for a in sys.argv[1:]] or [0]:
        if os.path.exists(path(seed)):
            print("seed %d: cached already (%s)" % (seed, path(seed)))
        else:
            build(seed)
