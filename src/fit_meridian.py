"""Fit a geo-level Bayesian MMM with Google Meridian and recover channel ROI.

Steps: load data -> build InputData (impressions / GRPs + spend) -> set ROI priors -> sample posterior -> check R-hat -> ROI / mROI ->
optimize budget.

Flags
  --calibrate  replace the Search ROI prior with a (simulated) go-dark experiment
  --fast       quick smoke test with few draws
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow_probability as tfp
from meridian import constants
from meridian.analysis import analyzer, optimizer
from meridian.data.data_frame_input_data_builder import DataFrameInputDataBuilder
from meridian.model import model, prior_distribution, spec

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ["tv", "ooh", "social", "search", "tiktok"]
MEDIA_UNITS = ["tv_grp", "ooh_grp", "social_impressions", "search_impressions", "tiktok_impressions"]
# OOH could also enter as reach & frequency (DataFrameInputDataBuilder.with_reach), which
# lets Meridian estimate an optimal frequency; that mode exceeded this sandbox's 7 GB of RAM,
# so OOH enters as GRPs (= reach x frequency) here.
CONTROLS = ["holiday_peak", "holiday_dip", "promo", "unemployment", "competitor_pressure"]  # trend/season via knots
FAST = "--fast" in sys.argv
CALIBRATE = "--calibrate" in sys.argv
OPEN_SLOPE = "--tv-slope" in sys.argv   # estimate the TV Hill slope (S-shape); see README
tfd = tfp.distributions


def build_input_data(df: pd.DataFrame):
    df = df.copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    b = DataFrameInputDataBuilder(kpi_type=constants.REVENUE,
                                  default_geo_column="geo", default_time_column="date")
    b = (b.with_kpi(df, kpi_col="revenue")
          .with_population(df, population_col="population")
          .with_controls(df, control_cols=CONTROLS)
          .with_media(df, media_cols=MEDIA_UNITS, media_spend_cols=[f"{c}_spend" for c in MEDIA],
                      media_channels=MEDIA))
    return b.build()


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "mmm_geo_weekly.csv", parse_dates=["date"])
    data = build_input_data(df)

    # ROI priors: Meridian's default LogNormal(0.2, 0.9) for every channel
    mu = np.full(len(MEDIA), 0.2, dtype=np.float32)
    sd = np.full(len(MEDIA), 0.9, dtype=np.float32)
    if CALIBRATE:
        # A go-dark holdout (Search paused in test DMAs) measured Search ROI ~1.8
        # (90% CI ~1.4-2.3) -> tight LogNormal prior centered on the experiment
        i = MEDIA.index("search")
        mu[i], sd[i] = np.log(1.8), 0.15
    # Hill slope: Meridian fixes it at 1 (C-shape) by default for impression media.
    # TV needs repeated exposure, so let its slope be estimated (S-shape allowed);
    # keep the others pinned near 1.
    slope_mu = np.array([np.log(1.5), 0, 0, 0, 0], dtype=np.float32)
    slope_sd = np.array([0.3, 0.01, 0.01, 0.01, 0.01], dtype=np.float32)
    prior = prior_distribution.PriorDistribution(
        roi_m=tfd.LogNormal(mu, sd, name=constants.ROI_M),
        **({"slope_m": tfd.LogNormal(slope_mu, slope_sd, name=constants.SLOPE_M)} if OPEN_SLOPE else {}),
    )
    mmm = model.Meridian(input_data=data,
                         model_spec=spec.ModelSpec(prior=prior, max_lag=8, knots=8))

    n_keep, n_adapt, n_burn = (100, 200, 50) if FAST else (1000, 500, 300)
    mmm.sample_posterior(n_chains=2, n_adapt=n_adapt, n_burnin=n_burn, n_keep=n_keep, seed=1)

    a = analyzer.Analyzer(mmm)
    rhats = np.concatenate([np.ravel(v) for v in a.get_rhat().values()])
    print(f"Max R-hat across parameters: {np.nanmax(rhats[np.isfinite(rhats)]):.3f}")

    channels = MEDIA
    roi = np.asarray(a.roi(batch_size=20))
    mroi = np.asarray(a.marginal_roi(batch_size=20))
    post = mmm.inference_data.posterior
    alpha = post["alpha_m"].values.mean((0, 1))
    res = pd.DataFrame({
        "channel": channels,
        "roi_mean": roi.mean((0, 1)), "roi_lo": np.quantile(roi, 0.05, (0, 1)),
        "roi_hi": np.quantile(roi, 0.95, (0, 1)), "mroi_mean": mroi.mean((0, 1)),
        "adstock_alpha_mean": alpha,
    })
    if OPEN_SLOPE:
        print("TV Hill slope (posterior mean):", float(post["slope_m"].values.mean((0, 1))[0]))
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    tag = "_calibrated" if CALIBRATE else ""
    res.to_csv(out / f"roi_meridian{tag}.csv", index=False)
    print(res.round(3))

    # Budget optimization: same total budget, each channel within +/-30% of today
    opt = optimizer.BudgetOptimizer(mmm).optimize(
        fixed_budget=True, spend_constraint_lower=0.3, spend_constraint_upper=0.3)
    ns, os_ = opt.nonoptimized_data.spend.values, opt.optimized_data.spend.values
    plan = pd.DataFrame({"channel": channels, "current_spend": ns, "optimized_spend": os_,
                         "change_pct": (os_ / ns - 1) * 100})
    plan.to_csv(out / f"budget_meridian{tag}.csv", index=False)
    inc_now = float(opt.nonoptimized_data.total_incremental_outcome)
    inc_opt = float(opt.optimized_data.total_incremental_outcome)
    print(plan.round(1))
    print(f"Incremental revenue: {inc_now:,.0f} -> {inc_opt:,.0f} ({inc_opt / inc_now - 1:+.1%})")


if __name__ == "__main__":
    main()
