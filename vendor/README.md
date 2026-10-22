# vendor/

`sqgturb/` is the two-surface surface quasi-geostrophic model of
[jswhit/sqgturb](https://github.com/jswhit/sqgturb), copyright 2016 Jeffrey Whitaker,
redistributed here under its own permissive licence (`sqgturb/COPYING`) so that the SQG
experiments are runnable from this repository alone. Only `sqg.py` is used; the other files
are kept as they ship.

`sqg/sqg_cvt.py` loads it from this directory. Set `SQGTURB_PATH` to the `sqg.py` of a
different checkout to override.
