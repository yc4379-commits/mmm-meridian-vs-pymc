"""Fit a geo-level Bayesian MMM with PyMC-Marketing and recover channel ROI.

Steps: load data -> specify adstock + saturation + priors -> (optional) add
lift-test calibration -> fit with NUTS -> check convergence -> ROI from
posterior contributions -> weekly decomposition -> optimize next quarter's budget.

Flags
  --calibrate  add the simulated Search geo-lift test as an extra likelihood term
  --logistic   use Logistic saturation (always concave) instead of Hill, to show
               what a misspecified curve shape does
  --fast       quick smoke test with few draws
"""

import sys
from pathlib import Path

import arviz as az
import pandas as pd
import xarray as xr
from pymc_extras.prior import Prior
from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation, LogisticSaturation

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ["tv", "ooh", "youtube", "search", "tiktok"]
SPEND = [f"{c}_spend" for c in CHANNELS]
# Baseline controls: trend, holidays, promotions, macro, competition (seasonality via Fourier terms)
CONTROLS = ["trend", "holiday_peak", "holiday_dip", "promo", "unemployment", "competitor_pressure"]
FAST = "--fast" in sys.argv
CALIBRATE = "--calibrate" in sys.argv
LOGISTIC = "--logistic" in sys.argv

# Adstock retention priors per channel, from channel knowledge:
# TV slow decay, Search almost none, others in between (Beta mean = a / (a + b))
ALPHA_A = [3.0, 1.0, 1.0, 1.0, 1.0]   # tv, ooh, youtube, search, tiktok
ALPHA_B = [2.0, 3.0, 3.0, 6.0, 3.0]   # means: 0.60, 0.25, 0.25, 0.14, 0.25


def hierarchical_beta() -> Prior:
    """Each DMA gets its own effect size, partially pooled toward the channel mean."""
    return Prior("LogNormal", mu=Prior("Normal", mu=-1.5, sigma=1, dims="channel"),
                 sigma=Prior("HalfNormal", sigma=0.3), dims=("geo", "channel"))


def build_mmm() -> MMM:
    if LOGISTIC:
        saturation = LogisticSaturation(priors={
            "lam": Prior("Gamma", alpha=3, beta=1, dims="channel"),
            "beta": hierarchical_beta()})
    else:
        # Hill: kappa = half-saturation point, slope = shape (1 C-shape, >1 S-shape)
        saturation = HillSaturation(priors={
            "kappa": Prior("Gamma", alpha=3, beta=6, dims="channel"),   # mean 0.5 of scaled media
            "slope": Prior("Gamma", alpha=4, beta=3, dims="channel"),   # mean 1.33
            "beta": hierarchical_beta()})
    return MMM(
        date_column="date",
        channel_columns=SPEND,
        target_column="revenue",
        control_columns=CONTROLS,
        dims=("geo",),                       # one hierarchical model across DMAs
        adstock=GeometricAdstock(l_max=8, priors={
            "alpha": Prior("Beta", alpha=ALPHA_A, beta=ALPHA_B, dims="channel")}),
        saturation=saturation,
        yearly_seasonality=2,                # Fourier terms for seasonality
    )


def roi_table(mmm: MMM, df: pd.DataFrame) -> pd.DataFrame:
    """ROI = total channel contribution ($) / total spend ($), with 90% interval."""
    post = mmm.idata.posterior
    scale = mmm.get_scales_as_xarray()["target_scale"]
    total = (post["channel_contribution"] * scale).sum(dim=["date", "geo"])
    roi_draws = total / df[SPEND].sum().values
    return pd.DataFrame({
        "channel": CHANNELS,
        "roi_mean": roi_draws.mean(("chain", "draw")).values,
        "roi_lo": roi_draws.quantile(0.05, ("chain", "draw")).values,
        "roi_hi": roi_draws.quantile(0.95, ("chain", "draw")).values,
        "adstock_alpha_mean": post["adstock_alpha"].mean(("chain", "draw")).values,
    })


def optimize_budget(mmm: MMM, df: pd.DataFrame) -> pd.DataFrame:
    """Reallocate a typical week's budget for the next 13 weeks at the same total."""
    last = df["date"].max()
    start, end = last + pd.Timedelta(weeks=1), last + pd.Timedelta(weeks=13)
    future = pd.date_range(start, end + pd.Timedelta(weeks=8), freq="W-SUN")  # + carry-over
    geos = sorted(df["geo"].unique())
    last_unemp = df[df["date"] == last].set_index("geo")["unemployment"]
    controls = pd.DataFrame(
        [(d, g, 1 + (i + 1) / 104, 0.0, 0.0, 0.0, last_unemp[g], 0.0) for i, d in enumerate(future) for g in geos],
        columns=["date", "geo", *CONTROLS])
    # Typical week = average weekly spend over the last year (flighted channels included)
    current = df[df["date"] > last - pd.Timedelta(weeks=52)].groupby("geo")[SPEND].mean()
    cur = xr.DataArray(current[SPEND].values, dims=("geo", "channel"),
                       coords={"geo": current.index.values, "channel": SPEND})
    bounds = xr.concat([0.7 * cur, 1.3 * cur], dim="bound").assign_coords(bound=["lower", "upper"])
    opt = mmm.budget_optimizer(start, end, control_data=controls)
    alloc, _ = opt.allocate_budget(total_budget=float(current.values.sum()), budget_bounds=bounds)
    new = alloc.to_dataframe(name="spend").reset_index().groupby("channel")["spend"].sum()
    plan = pd.DataFrame({"channel": CHANNELS, "current_spend": current.sum().reindex(SPEND).values,
                         "optimized_spend": new.reindex(SPEND).values})
    plan["change_pct"] = (plan.optimized_spend / plan.current_spend - 1) * 100
    return plan


def export_decomposition(mmm: MMM, df: pd.DataFrame, path: Path) -> None:
    """Weekly national revenue split into base, seasonality, each control, each channel
    and residual (posterior means, in dollars). Feeds the dashboard decomposition chart."""
    post = mmm.idata.posterior
    scale = float(mmm.get_scales_as_xarray()["target_scale"])
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    out = pd.DataFrame(index=dates)

    def national(var: str) -> xr.DataArray:
        da = post[var].mean(("chain", "draw")) * scale
        if "date" not in da.dims:
            da = da.expand_dims(date=dates)
        return da.sum("geo") if "geo" in da.dims else da

    out["Base"] = national("intercept_contribution").values
    out["Seasonality"] = national("yearly_seasonality_contribution").values
    ctrl = national("control_contribution")
    for c in ctrl["control"].values:
        out[str(c)] = ctrl.sel(control=c).values
    ch = national("channel_contribution")
    for c in ch["channel"].values:
        out[str(c).removesuffix("_spend")] = ch.sel(channel=c).values
    actual = df.groupby("date")["revenue"].sum().reindex(dates).values
    out["Residual"] = actual - out.sum(axis=1).values
    out["Actual"] = actual
    out.index.name = "date"
    out.reset_index().to_csv(path, index=False)

    # Goodness of fit, at the national level and pooled across DMAs
    def by_geo(var: str) -> xr.DataArray:
        da = post[var].mean(("chain", "draw")) * scale
        if "date" not in da.dims:
            da = da.expand_dims(date=dates)
        for d in ("control", "channel"):
            if d in da.dims:
                da = da.sum(d)
        return da.transpose("date", "geo")

    fitted_geo = sum(by_geo(v) for v in ["intercept_contribution", "yearly_seasonality_contribution",
                                          "control_contribution", "channel_contribution"])
    act_geo = df.pivot(index="date", columns="geo", values="revenue").loc[dates, fitted_geo.geo.values].values
    rows = [{"level": "National", **fit_stats(actual, actual - out["Residual"].values)},
            {"level": "DMA (pooled)", **fit_stats(act_geo.ravel(order="F"), fitted_geo.values.ravel(order="F"),
                                                   groups=act_geo.shape[1])}]
    pd.DataFrame(rows).to_csv(path.with_name("fit_stats_pymc.csv"), index=False)
    print(pd.DataFrame(rows).round(3))


def fit_stats(actual, fitted, groups: int = 1) -> dict:
    """R-squared, MAPE and Durbin-Watson (DW ~ 2: no residual autocorrelation;
    well below 2: positive autocorrelation, often a missing driver or too little carryover)."""
    import numpy as np
    actual, fitted = np.asarray(actual, float), np.asarray(fitted, float)
    resid = actual - fitted
    r2 = 1 - np.sum(resid**2) / np.sum((actual - actual.mean()) ** 2)
    mape = np.mean(np.abs(resid / actual))
    # DW within each series (DMA), so differences never cross from one DMA to the next
    series = np.array_split(resid, groups)
    dw = sum(np.sum(np.diff(r) ** 2) for r in series) / np.sum(resid**2)
    return {"r2": r2, "mape": mape, "durbin_watson": dw}


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "mmm_geo_weekly.csv", parse_dates=["date"])
    X = df[["date", "geo", *SPEND, *CONTROLS]]
    y = df["revenue"]

    mmm = build_mmm()
    if CALIBRATE:
        # The model's predicted lift for a +30% Search spend change must match the experiment
        mmm.build_model(X, y)
        lift = pd.read_csv(ROOT / "data" / "lift_test_search.csv")
        lift["channel"] = lift["channel"] + "_spend"
        mmm.add_lift_test_measurements(lift)

    draws, tune = (100, 100) if FAST else (1000, 1000)
    mmm.fit(X, y, draws=draws, tune=tune, chains=2, cores=2,
            target_accept=0.9, random_seed=1, progressbar=False)

    sat_vars = ["saturation_lam"] if LOGISTIC else ["saturation_kappa", "saturation_slope"]
    summ = az.summary(mmm.idata, var_names=["adstock_alpha", *sat_vars, "saturation_beta_mu"])
    print(summ)
    print(f"Max R-hat: {summ['r_hat'].max():.3f} | "
          f"Divergences: {int(mmm.idata.sample_stats['diverging'].sum())}")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    tag = ("_logistic" if LOGISTIC else "") + ("_calibrated" if CALIBRATE else "")
    roi = roi_table(mmm, df)
    roi.to_csv(out / f"roi_pymc{tag}.csv", index=False)
    print(roi.round(3))
    summ.to_csv(out / f"params_pymc{tag}.csv")
    if not CALIBRATE and not LOGISTIC:
        export_decomposition(mmm, df, out / "decomposition_pymc.csv")
    plan = optimize_budget(mmm, df)
    plan.to_csv(out / f"budget_pymc{tag}.csv", index=False)
    print(plan.round(1))


if __name__ == "__main__":
    main()
