#!/usr/bin/env bash
# Reproduce every result: simulate data, fit all model variants, compare.
# Assumes two virtual environments (see README): .venv-pymc and .venv-meridian
set -euo pipefail
PY=${PY_PYMC:-.venv-pymc/bin/python}
PM=${PY_MERIDIAN:-.venv-meridian/bin/python}
$PY src/simulate.py
$PY src/fit_pymc.py                  # Hill saturation
$PY src/fit_pymc.py --calibrate      # + Search lift test
$PY src/fit_pymc.py --logistic       # misspecified curve shape, for comparison
$PM src/fit_meridian.py
$PM src/fit_meridian.py --calibrate  # + Search go-dark ROI prior
$PY src/compare.py
