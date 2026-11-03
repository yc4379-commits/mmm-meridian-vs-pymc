"""Fit a geo-level Bayesian MMM with Google Meridian and recover channel ROI.

Steps: load data -> build InputData -> set ROI priors -> sample posterior ->
check R-hat -> read ROI / mROI -> optimize budget.

Run with --calibrate to replace the Search ROI prior with a (simulated)
geo-lift result, which is how Meridian encourages experiment calibration.
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
CHANNELS = ["tv", "social", "search", "tiktok"]
FAST = "--fast" in sys.argv
CALIBRATE = "--calibrate" in sys.argv


def build_input_data(df: pd.DataFrame):
    df = df.copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    b = DataFrameInputDataBuilder(kpi_type=constants.REVENUE,
                                  default_geo_column="geo", default_time_column="date")
    b = (b.with_kpi(df, kpi_col="revenue")
          .with_population(df, population_col="population")
          .with_controls(df, control_cols=["valentines"])
          # Meridian wants an exposure metric plus spend; we only have spend,
          # so spend doubles as the media metric (common in practice too).
          .with_media(df, media_cols=CHANNELS, media_spend_cols=CHANNELS, media_channels=CHANNELS))
    return b.build()


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "mmm_geo_weekly.csv", parse_dates=["date"])
    data = build_input_data(df)

    # Meridian puts the prior directly on ROI. Default: LogNormal(0.2, 0.9) per channel.
    mu = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    sd = np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32)
    if CALIBRATE:
        # Pretend a go-dark holdout test (Search paused in test geos) measured
        # Search ROI = 1.8 (90% CI ~1.4-2.3).
        # Convert to a tight LogNormal prior centered on the experiment.
        mu[2], sd[2] = np.log(1.8), 0.15
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(mu, sd, name=constants.ROI_M))

    mmm = model.Meridian(
        input_data=data,
        model_spec=spec.ModelSpec(prior=prior, max_lag=8, knots=8),  # knots = time-varying baseline
    )

    n_keep, n_adapt, n_burn = (100, 200, 50) if FAST else (500, 500, 200)
    mmm.sample_posterior(n_chains=2, n_adapt=n_adapt, n_burnin=n_burn, n_keep=n_keep, seed=1)

    a = analyzer.Analyzer(mmm)
    rhat = a.get_rhat()
    worst = max(float(np.max(v)) for v in rhat.values())
    print(f"Max R-hat across parameters: {worst:.3f}")

    roi = a.roi()                         # (chain, draw, channel), geos and weeks aggregated
    mroi = a.marginal_roi()
    alpha = mmm.inference_data.posterior["alpha_m"].values  # adstock decay
    res = pd.DataFrame({
        "channel": CHANNELS,
        "roi_mean": np.mean(roi, axis=(0, 1)),
        "roi_lo": np.quantile(roi, 0.05, axis=(0, 1)),
        "roi_hi": np.quantile(roi, 0.95, axis=(0, 1)),
        "mroi_mean": np.mean(mroi, axis=(0, 1)),
        "adstock_alpha_mean": alpha.mean(axis=(0, 1)),
    })
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    tag = "_calibrated" if CALIBRATE else ""
    res.to_csv(out / f"roi_meridian{tag}.csv", index=False)
    print(res.round(3))

    # Budget optimization: same total budget, each channel within +/-30% of today
    opt = optimizer.BudgetOptimizer(mmm).optimize(
        fixed_budget=True, spend_constraint_lower=0.3, spend_constraint_upper=0.3)
    ns = opt.nonoptimized_data.spend.values
    os_ = opt.optimized_data.spend.values
    plan = pd.DataFrame({"channel": CHANNELS, "current_spend": ns, "optimized_spend": os_,
                         "change_pct": (os_ / ns - 1) * 100})
    plan.to_csv(out / f"budget_meridian{tag}.csv", index=False)
    inc_now = float(opt.nonoptimized_data.total_incremental_outcome)
    inc_opt = float(opt.optimized_data.total_incremental_outcome)
    print(plan.round(1))
    print(f"Incremental revenue: {inc_now:,.0f} -> {inc_opt:,.0f} ({inc_opt / inc_now - 1:+.1%})")


if __name__ == "__main__":
    main()
