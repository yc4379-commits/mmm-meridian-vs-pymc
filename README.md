# Bayesian MMM: Google Meridian vs PyMC-Marketing

A side-by-side comparison of the two main open-source Bayesian Marketing Mix Modeling (MMM) packages:
[Google Meridian](https://github.com/google/meridian) and [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing).

Both are fit to the **same simulated dataset with a known ground truth**, so we can check what each one
recovers, how experiment calibration changes the answer, and how their workflows differ.

## The setup

A subscription dating app runs paid media in **10 US markets for 104 weeks** across four channels:
TV, Social, Search and TikTok. Revenue is generated as:

```
revenue[geo, week] = baseline (geo level x trend x yearly seasonality x Valentine's lift)
                   + sum over channels of  beta[geo, channel] * Hill(adstock(spend))
                   + noise
```

- **Adstock (carryover):** geometric, with known decay per channel (TV slowest, Search fastest)
- **Saturation (diminishing returns):** Hill curve
- **Geo heterogeneity:** each market's effect size varies around a channel mean
- **TV is flighted** (on/off bursts); digital channels are always-on with their own rhythms
- Media drives about 21% of revenue

The true parameters live in `src/simulate.py`. True ROI = total channel contribution / total spend:

| Channel | True ROI | True adstock decay |
|---|---|---|
| TV | 1.33 | 0.6 |
| Social | 1.11 | 0.3 |
| Search | 1.84 | 0.1 |
| TikTok | 0.46 | 0.2 |

We also simulate **two experiments** on Search for calibration:
- A **scale-up geo-lift test** (+30% Search spend in 3 markets), giving the *marginal* revenue lift (`data/lift_test_search.csv`)
- A **go-dark holdout test** giving the *average* Search ROI (≈1.8), used as a Meridian ROI prior

## How to run

The two packages need different ArviZ versions, so use **two virtual environments**.

```bash
# Data
python -m venv .venv-pymc && .venv-pymc/bin/pip install -r requirements-pymc.txt
.venv-pymc/bin/python src/simulate.py

# PyMC-Marketing (about 2 minutes each on a laptop CPU)
.venv-pymc/bin/python src/fit_pymc.py
.venv-pymc/bin/python src/fit_pymc.py --calibrate      # + lift-test calibration

# Meridian (about 10 minutes each)
python -m venv .venv-meridian && .venv-meridian/bin/pip install -r requirements-meridian.txt
.venv-meridian/bin/python src/fit_meridian.py
.venv-meridian/bin/python src/fit_meridian.py --calibrate   # + experiment ROI prior

# Compare
.venv-pymc/bin/python src/compare.py
```

Add `--fast` to any fit script for a quick smoke test with fewer draws.

## Model specifications

| | PyMC-Marketing (`src/fit_pymc.py`) | Meridian (`src/fit_meridian.py`) |
|---|---|---|
| Granularity | Geo x week, `dims=("geo",)` | Geo x week (native) |
| Adstock | Geometric, `l_max=8`, decay shared across geos | Geometric, `max_lag=8` |
| Saturation | Logistic | Hill |
| Geo hierarchy | Set up by hand: `beta[geo, channel] ~ LogNormal(mu[channel], sigma)` | Built in |
| Where priors go | On model parameters (beta, lambda, alpha) | **Directly on ROI** (`roi_m`) |
| Baseline | Intercept + Fourier seasonality + trend control | Geo intercepts + time-varying knots |
| Experiment calibration | Lift test as an **extra likelihood term** | Experiment result as an **ROI prior** |
| Sampler | NUTS (PyMC) | NUTS (TensorFlow Probability) |
| Budget optimizer | `mmm.budget_optimizer(...).allocate_budget(...)` | `BudgetOptimizer(mmm).optimize(...)` |

## Results

![ROI comparison](results/roi_comparison.png)

| Channel | True ROI | PyMC-Marketing | PyMC + lift test | Meridian | Meridian + ROI prior |
|---|---|---|---|---|---|
| TV | 1.33 | 1.63 [1.53, 1.73] | 1.62 [1.52, 1.72] | 1.80 [1.70, 1.90] | 1.80 [1.69, 1.90] |
| Social | 1.11 | 0.65 [0.49, 0.82] | 0.65 [0.49, 0.83] | 1.38 [1.02, 1.91] | 1.38 [1.02, 1.87] |
| Search | 1.84 | 0.98 [0.81, 1.21] | 0.84 [0.70, 1.01] | 1.52 [1.19, 2.09] | **1.67 [1.37, 2.03]** |
| TikTok | 0.46 | **0.50 [0.29, 0.84]** | 0.57 [0.29, 1.00] | 1.23 [0.55, 2.35] | 1.09 [0.53, 2.18] |

90% credible intervals in brackets. Adstock decay was recovered well by both packages
(e.g. TV ≈ 0.60, Search ≈ 0.05–0.08).

**Budget optimization** (same total budget, each channel within ±30%): both packages recommend
moving money **into TV and out of TikTok**. Meridian projects +4.2% incremental revenue from the reallocation.

## What this shows

1. **Neither model recovers every channel's ROI, even on clean simulated data.** Both overstate TV and
   disagree on Social and Search. With correlated spend and only ~2 years of data, MMM alone has an
   identification problem. This is the main argument for pairing MMM with experiments.

2. **The type of experiment matters, not just whether you calibrate.**
   - Meridian with a **go-dark ROI prior** moved Search from 1.52 to 1.67 (truth 1.84), because the
     experiment measures the same quantity the prior is on: average ROI.
   - PyMC with a **scale-up lift test** moved Search *down* (0.98 to 0.84). A +30% test measures the
     **marginal** return at current spend (≈0.5–0.8 here, because Search is near saturation), not the
     average. When the model's saturation curve is misspecified (logistic vs the true Hill), matching
     the marginal point does not fix the average ROI.
   - Takeaway: average ROI and marginal ROI are different quantities. Design the test to answer the
     decision you are making, and calibrate the matching quantity.

3. **Meridian's intervals were wider and covered the truth more often** (2 of 4 channels vs 1 of 4).
   PyMC's tighter intervals came from stronger default priors, not more information. Overconfident
   intervals are riskier than wide ones when the results drive budget.

4. **Omitted baseline drivers get absorbed by media.** In an early iteration without a trend control (and with more correlated digital spend),
   a slow-decaying channel soaked up the organic growth trend and its ROI was inflated several times.
   Baseline specification (trend, seasonality, holidays) is as important as media specification.

5. **Ranking can be more robust than levels.** Both packages agree TikTok is the weakest channel
   and the optimizer moves money the same direction, even though ROI levels differ.

## Meridian vs PyMC-Marketing: when to use which

- **Meridian:** strong defaults, geo hierarchy built in, ROI priors are intuitive for business
  stakeholders and easy to set from experiments, reach and frequency support for TV and YouTube.
  Good for a standardized, out-of-the-box setup.
- **PyMC-Marketing:** fully customizable model structure (custom priors, hierarchies, time-varying
  effects, DAGs), lift-test likelihood calibration, plus a separate CLV module. Good for teams
  building an in-house MMM system across several brands or business models.

## Limitations

- Short sampling runs (2 chains x 500 draws) to keep runtime low; a few parameters have R-hat up to
  ~1.09, so production use would need longer chains.
- Spend is used as the media exposure metric for Meridian (no impressions in the simulation).
- One simulated dataset; conclusions about which package is "more accurate" would need many seeds.

## Repo structure

```
src/simulate.py      simulate geo-level data, ground truth and lift tests
src/fit_pymc.py      PyMC-Marketing fit, ROI, lift-test calibration, budget optimization
src/fit_meridian.py  Meridian fit, ROI / mROI, ROI-prior calibration, budget optimization
src/compare.py       comparison table and chart
data/                simulated data and ground truth
results/             ROI tables, budget plans, chart
```
