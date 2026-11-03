"""Fit a geo-level Bayesian MMM with PyMC-Marketing and recover channel ROI.

Steps: load data -> specify adstock + saturation + priors -> (optional) add
lift-test calibration -> fit with NUTS -> check convergence -> compute ROI
from posterior contributions -> optimize next quarter's budget.

Run with --calibrate to add the simulated Search geo-lift test as an extra
likelihood term, which is how PyMC-Marketing calibrates with experiments.
"""

import sys
from pathlib import Path

import arviz as az
import pandas as pd
import xarray as xr
from pymc_extras.prior import Prior
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ["tv", "social", "search", "tiktok"]
FAST = "--fast" in sys.argv          # quick smoke test
CALIBRATE = "--calibrate" in sys.argv


def build_mmm() -> MMM:
    return MMM(
        date_column="date",
        channel_columns=CHANNELS,
        target_column="revenue",
        control_columns=["valentines", "trend"],  # leaving trend out biases media; see README
        dims=("geo",),                            # one model, geo-level hierarchy
        # Carryover and curve shape are shared across geos (one per channel)
        adstock=GeometricAdstock(
            l_max=8, priors={"alpha": Prior("Beta", alpha=1, beta=3, dims="channel")}),
        # Effect size is HIERARCHICAL: each geo gets its own beta, partially
        # pooled toward a channel-level mean (same idea as a random slope).
        saturation=LogisticSaturation(priors={
            "lam": Prior("Gamma", alpha=3, beta=1, dims="channel"),
            "beta": Prior("LogNormal",
                          mu=Prior("Normal", mu=-1.5, sigma=1, dims="channel"),
                          sigma=Prior("HalfNormal", sigma=0.3),
                          dims=("geo", "channel")),
        }),
        yearly_seasonality=2,  # Fourier terms for seasonality
    )


def roi_table(mmm: MMM, df: pd.DataFrame) -> pd.DataFrame:
    """ROI = total channel contribution ($) / total spend ($), with 90% interval."""
    post = mmm.idata.posterior
    # Contributions are fitted on a scaled target; multiply back to dollars.
    target_scale = mmm.get_scales_as_xarray()["target_scale"]
    total = (post["channel_contribution"] * target_scale).sum(dim=["date", "geo"])
    roi_draws = total / df[CHANNELS].sum().values
    return pd.DataFrame({
        "channel": CHANNELS,
        "roi_mean": roi_draws.mean(("chain", "draw")).values,
        "roi_lo": roi_draws.quantile(0.05, ("chain", "draw")).values,
        "roi_hi": roi_draws.quantile(0.95, ("chain", "draw")).values,
        "adstock_alpha_mean": post["adstock_alpha"].mean(("chain", "draw")).values,
    })


def optimize_budget(mmm: MMM, df: pd.DataFrame) -> pd.DataFrame:
    """Reallocate the next 13 weeks' budget at the same total spend."""
    last = df["date"].max()
    start, end = last + pd.Timedelta(weeks=1), last + pd.Timedelta(weeks=13)
    # Controls must cover the window plus the adstock carry-over (8 weeks)
    future = pd.date_range(start, end + pd.Timedelta(weeks=8), freq="W-SUN")
    geos = sorted(df["geo"].unique())
    controls = pd.DataFrame(
        [(d, g, 0.0, 1 + (i + 1) / 104) for i, d in enumerate(future) for g in geos],
        columns=["date", "geo", "valentines", "trend"])

    # Current plan = average weekly spend per geo and channel over the last 13 weeks
    recent = df[df["date"] > last - pd.Timedelta(weeks=13)]
    current = recent.groupby("geo")[CHANNELS].mean()
    weekly_budget = float(current.values.sum())
    # Each geo x channel may move within +/-30% of its current weekly spend
    cur = xr.DataArray(current[CHANNELS].values, dims=("geo", "channel"),
                       coords={"geo": current.index.values, "channel": CHANNELS})
    bounds = xr.concat([0.7 * cur, 1.3 * cur], dim="bound").assign_coords(bound=["lower", "upper"])

    opt = mmm.budget_optimizer(start, end, control_data=controls)
    alloc, _ = opt.allocate_budget(total_budget=weekly_budget, budget_bounds=bounds)
    new = alloc.to_dataframe(name="spend").reset_index().groupby("channel")["spend"].sum()
    plan = pd.DataFrame({"channel": CHANNELS,
                         "current_spend": current.sum().reindex(CHANNELS).values,
                         "optimized_spend": new.reindex(CHANNELS).values})
    plan["change_pct"] = (plan.optimized_spend / plan.current_spend - 1) * 100
    return plan


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "mmm_geo_weekly.csv", parse_dates=["date"])
    X = df[["date", "geo", *CHANNELS, "valentines", "trend"]]
    y = df["revenue"]

    mmm = build_mmm()
    if CALIBRATE:
        # Experiment calibration: the model's predicted lift for a +30% Search
        # spend change must match what the geo-lift test observed.
        mmm.build_model(X, y)
        lift = pd.read_csv(ROOT / "data" / "lift_test_search.csv")
        mmm.add_lift_test_measurements(lift)

    draws, tune = (100, 100) if FAST else (500, 500)
    mmm.fit(X, y, draws=draws, tune=tune, chains=2, cores=2,
            target_accept=0.9, random_seed=1, progressbar=False)

    # Convergence diagnostics: want R-hat < 1.01 and zero divergences
    summ = az.summary(mmm.idata, var_names=["adstock_alpha", "saturation_lam", "saturation_beta_mu"])
    print(summ)
    print(f"Max R-hat: {summ['r_hat'].max():.3f} | "
          f"Divergences: {int(mmm.idata.sample_stats['diverging'].sum())}")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    tag = "_calibrated" if CALIBRATE else ""
    roi = roi_table(mmm, df)
    roi.to_csv(out / f"roi_pymc{tag}.csv", index=False)
    print(roi.round(3))

    plan = optimize_budget(mmm, df)
    plan.to_csv(out / f"budget_pymc{tag}.csv", index=False)
    print(plan.round(1))


if __name__ == "__main__":
    main()
