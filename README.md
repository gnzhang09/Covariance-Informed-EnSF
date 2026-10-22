# CI-EnSF — reproducibility repository

Code, data and notebooks for the covariance-informed ensemble score filter (CI-EnSF) paper.
Every table and figure in the manuscript is produced by one of the six notebooks in
`notebooks/`. Each is committed in an executed state, so the numbers can be read without
running anything.

CI-EnSF replaces the isotropic covariance of a standard ensemble score filter with a
member-conditioned background covariance

```
B = (1/(K-1)) * sum_k  diag(a_k) C diag(a_k)
```

applied matrix-free: `C` is a Gaspari–Cohn taper, block-circulant on the periodic grid, so
`C v` is an FFT convolution and one `B v` costs `O(K d log d)` with no `d x d` matrix ever
formed. The reverse-time probability-flow ODE is integrated with a step count set by
`lambda_max(B)`.

---

## Layout

```
notebooks/
  L96_linear/       Lorenz-96, linear observations        Table 1, Figure 1
  L96_nonlinear/    Lorenz-96, arctan observations        Table 2, Figure 3
  adaptivity/       prior scaling x likelihood tempering  Table 3, Figure 4
  SQG_linear/       SQG at N = 128, linear observations   Table 4, Figure 5
  SQG_nonlinear/    SQG at N = 128, arctan observations   Table 6, Figures 6 and 7
  transport_demo/   covariance transport, one analysis    Figure 2
sqg/                SQG driver (sqg_cvt.py), its matrix-free covariance operators, and the
                    stored results the SQG notebooks read
vendor/sqgturb/     the SQG model itself (jswhit/sqgturb), redistributed under its own licence
LICENSE   CITATION.cff   .zenodo.json   requirements.txt
```

Each notebook folder holds the notebook, the cache it reads and the figure it writes. There
is no separate `figures/` directory: the file you see is the file the notebook produced.

Table 5 of the manuscript is a hyperparameter sensitivity grid. It is not reproduced here;
its reduced output is committed under `sqg/` as the provenance of the selected settings.

---

## What reproduces what

| notebook | produces | as committed | from scratch |
|---|---|---|---|
| `L96_linear/gaussian_regime_24h.ipynb` | Table 1, Figure 1 | 5 s | 25 s, CPU |
| `L96_nonlinear/arctan_regime_24h.ipynb` | Table 2, Figure 3 | 10 s | 3 min, CPU |
| `adaptivity/adaptivity_2x2.ipynb` | Table 3, Figure 4 | 10 s | 7 min, CPU |
| `SQG_linear/sqg_n128_linear.ipynb` | Table 4, Figure 5 | 6 s | ~20 min/seed, GPU |
| `SQG_nonlinear/sqg_n128_arctan_matched.ipynb` | Table 6, Figures 6 and 7 | 7 s | ~24 min/seed, GPU |
| `transport_demo/mechanism_unobserved.ipynb` | Figure 2 | 141 s | 143 s, CPU |

**Every notebook is committed with `RUN_SIMULATION = False`**, which rebuilds its table and
figure from the data beside it. Setting the flag to `True` re-runs the experiment in the
notebook itself and rewrites those files; that is the "from scratch" column.
`transport_demo` costs the same either way because its kernel-density panels dominate and
are redrawn regardless.

For the two SQG notebooks `RUN_SIMULATION = True` needs a GPU **and** the nature runs, which
are not in this repository; see **Nature runs** below. Re-running the arctan experiment is
about 24 minutes per seed, so roughly four GPU-hours for the ten evaluation seeds.

Runtimes are wall clock for `jupyter nbconvert --execute` on the machine described under
Environment.

---

## Running it

```bash
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace L96_linear/gaussian_regime_24h.ipynb
jupyter nbconvert --to notebook --execute --inplace L96_nonlinear/arctan_regime_24h.ipynb
jupyter nbconvert --to notebook --execute --inplace adaptivity/adaptivity_2x2.ipynb
jupyter nbconvert --to notebook --execute --inplace SQG_linear/sqg_n128_linear.ipynb
jupyter nbconvert --to notebook --execute --inplace SQG_nonlinear/sqg_n128_arctan_matched.ipynb
jupyter nbconvert --to notebook --execute --inplace transport_demo/mechanism_unobserved.ipynb
```

or open them in Jupyter and run all. The SQG notebooks locate `sqg/` by walking up from the
working directory; set `CIENSF_SQG` to override.

The SQG model is `vendor/sqgturb/`, a copy of
[jswhit/sqgturb](https://github.com/jswhit/sqgturb) redistributed under its own permissive
licence so that the experiments run from this repository alone. `sqg/sqg_cvt.py` loads it
from there; set `SQGTURB_PATH` to point at a different checkout.

The scripts in `sqg/` resolve their own directory from `__file__` and take their parameters
from the environment, for example

```bash
SQG_N=128 SQG_GPU=1 python3 sqg/ak_spec_diag.py 0,1,2 ak_spec_diag.json
```

---

## Nature runs

Every SQG experiment integrates a two-surface SQG nature run per seed. These are too large
for this repository and are deposited alongside it as two archives:

| archive | size | used by |
|---|---|---|
| `nature-sqg-linear.tar` | 430 MB | `SQG_linear/` |
| `nature-sqg-arctan.tar` | 277 MB | `SQG_nonlinear/` |

Each holds thirteen seeds, the ten evaluation realizations and the three tuning ones. Unpack
into `sqg/nature/`, or point `SQG_NATURE_DIR` elsewhere. Each file records the configuration
it was produced under and `sqg/nature_cache.py` checks it on load, so a mismatched run raises
rather than being used silently. They can also be regenerated with `sqg/nature_cache.py`
(linear) and `sqg/ak_nature.py` (arctan), which is why `RUN_SIMULATION = True` needs a GPU.

The notebooks as committed do not need them.

---

## Environment

Produced with Python 3.12.3, numpy 2.5.2, scipy 1.18.0, matplotlib 3.11.1 and cupy 14.2.0 on
NVIDIA A100-SXM4-40GB GPUs. The Lorenz-96 notebooks need no GPU. `imageio-ffmpeg` supplies a
static ffmpeg binary, so no system ffmpeg is required.

---


## License and acknowledgment

MIT; see `LICENSE`. If you use this software or the data deposited with it, please cite both
the software record and the paper it accompanies (`CITATION.cff`).

This work has been authored by UT-Battelle, LLC, under contract DE-AC05-00OR22725 with
the US Department of Energy (DOE). The US government retains and the publisher, by accepting
the article for publication, acknowledges that the US government retains a nonexclusive,
paid-up, irrevocable, worldwide license to publish or reproduce the published form of this
manuscript, or allow others to do so, for US government purposes. DOE will provide public
access to these results of federally sponsored research in accordance with the DOE Public
Access Plan (https://www.energy.gov/doe-public-access-plan).
