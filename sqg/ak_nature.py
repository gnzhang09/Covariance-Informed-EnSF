"""Nature run for the arctan study at N = 256 or 512.

The efficiency study's nature runs cannot be reused: they sample every 6 h over 25 days, and
this study assimilates every 24 h for 150 cycles, i.e. 150 days.  Same model, different
sampling.  Settings match the N = 128 arctan runs (160 frames at 24 h, 24 climatology states
96 h apart, 100-day spin-up).
"""
import os as _os
_SQG = _os.path.dirname(_os.path.abspath(__file__))   # repo-relative: was an absolute path
import os, sys, time, numpy as np
NN, seed = int(sys.argv[1]), int(sys.argv[2])
DT   = {128: 600., 256: 200., 512: 90.}[NN]
DIFF = {128: 86400./3, 256: 6000., 512: 1800.}[NN]
out = _SQG + "/nature/ak24_N%d_seed%d.npz" % (NN, seed)
if os.path.exists(out): print("N=%d seed %d: cached" % (NN, seed), flush=True); sys.exit(0)
os.environ["SQG_N"]=str(NN); os.environ["SQG_GPU"]="1"
os.environ["SQG_DT"]=str(DT); os.environ["SQG_DIFF"]=str(DIFF)
os.environ["SQG_NCYC"]="160"; os.environ["SQG_CLIM"]="24"
sys.path.insert(0,_SQG)
import sqg_cvt as S
S.DT_OBS = 24*3600.0; S.NSTEP_OBS = int(round(S.DT_OBS/S.DT))
t0=time.time(); T,C = S.nature_run(seed); dt=time.time()-t0
np.savez_compressed(out, T=T, C=C, meta_N=NN, meta_DT=DT, meta_DIFF_EFOLD=DIFF,
                    meta_DT_OBS=S.DT_OBS, meta_NCYCLE=160, meta_NCLIM=24,
                    meta_SPINUP_D=S.SPINUP_D)
print("N=%-5d seed %-3d %6.0f s  std %.3f K  -> %s"
      % (NN, seed, dt, float(T.std()), os.path.basename(out)), flush=True)
