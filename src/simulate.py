"""Simulate geo-level weekly marketing data with a known ground truth.

A subscription dating app runs paid media in 10 US DMAs for two years:
TV and OOH (bought in GRPs, in concentrated flights) plus YouTube, Search and
TikTok (always-on digital, bought on CPM). Every channel has a known carryover
(geometric adstock) and a known diminishing-returns curve (Hill saturation), so
we can check how well each MMM package recovers the true ROI.

Output (data/):
  mmm_geo_weekly.csv     one row per DMA x week: spend, media units, controls, revenue
  ground_truth.csv       true ROI, contribution and curve parameters per channel
  lift_test_search.csv   a simulated +30% Search geo-lift experiment in 3 DMAs
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_GEOS = 10
N_WEEKS = 104
CHANNELS = ["tv", "ooh", "youtube", "search", "tiktok"]

# ---------------------------------------------------------------------------
# Ground-truth media parameters. Assumptions informed by:
# - TV has a long tail (slow decay) and needs repeated exposure (S-shaped);
# - OOH works mostly while it is up (short carryover);
# - Search captures existing demand: almost no carryover, saturates quickly (C-shaped);
# - YouTube / TikTok sit in between.
# alpha = weekly adstock retention (half-life = ln 0.5 / ln alpha),
# ec = Hill half-saturation point (share of the DMA's peak adstocked media),
# slope = Hill shape (1 = C-shape, >1 = S-shape), beta = effect size.
TRUTH = {
    "tv":     {"alpha": 0.60, "ec": 0.50, "slope": 2.0, "beta": 1.7},
    "ooh":    {"alpha": 0.20, "ec": 0.40, "slope": 1.5, "beta": 1.4},
    "youtube": {"alpha": 0.30, "ec": 0.40, "slope": 1.5, "beta": 0.7},
    "search": {"alpha": 0.10, "ec": 0.30, "slope": 1.0, "beta": 1.1},
    "tiktok": {"alpha": 0.25, "ec": 0.60, "slope": 1.0, "beta": 0.4},
}
# Average weekly spend per DMA (USD) for always-on digital channels
DIGITAL_SPEND = {"youtube": 25_000, "search": 20_000, "tiktok": 15_000}
CPM = {"youtube": 9.0, "search": 35.0, "tiktok": 7.0}  # cost per 1,000 impressions
# Offline buying: cost per rating point (scales with market size)
CPP = {"tv": 700.0, "ooh": 220.0}
# TV flights (start week, length in weeks): concentrated bursts around the
# January peak, Valentine's, spring and fall, repeated each year
TV_FLIGHTS = [(0, 5), (18, 4), (36, 6), (49, 3)]
OOH_FLIGHTS = [(12, 6), (30, 8)]  # spring and summer billboard campaigns
COMPETITOR_FLIGHTS = [(6, 5), (26, 6), (42, 5)]  # a rival app's media bursts
LIFT_TEST_GEOS = (0, 3, 6)  # DMAs where we "ran" a Search geo-lift test


def geometric_adstock(x: np.ndarray, alpha: float, l_max: int = 8) -> np.ndarray:
    """Carryover: effect_t = sum_{l=0..l_max-1} alpha^l * x_{t-l}."""
    out = np.zeros_like(x, dtype=float)
    for lag in range(l_max):
        out[lag:] += (alpha**lag) * x[: len(x) - lag]
    return out


def hill(x: np.ndarray, ec: float, slope: float) -> np.ndarray:
    """Diminishing returns, scaled to [0, 1)."""
    return x**slope / (x**slope + ec**slope)


def flight_mask(flights: list[tuple[int, int]]) -> np.ndarray:
    """1 in weeks on air (flights repeat every 52 weeks), with a taper inside each flight."""
    m = np.zeros(N_WEEKS)
    for year in range(N_WEEKS // 52 + 1):
        for start, length in flights:
            for i in range(length):
                w = year * 52 + start + i
                if w < N_WEEKS:
                    m[w] = 1 - 0.08 * i
    return m


def simulate(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-07", periods=N_WEEKS, freq="W-SUN")
    t = np.arange(N_WEEKS)

    population = rng.integers(500_000, 5_000_000, N_GEOS)
    size = population / population.mean()  # bigger markets spend and earn more
    tv_on, ooh_on = flight_mask(TV_FLIGHTS), flight_mask(OOH_FLIGHTS)
    comp_on = flight_mask(COMPETITOR_FLIGHTS)

    lift_rows, rows = [], []
    contrib_total = {c: 0.0 for c in CHANNELS}
    spend_total = {c: 0.0 for c in CHANNELS}
    for g in range(N_GEOS):
        geo = f"geo_{g:02d}"
        # ---- Baseline (non-media) drivers -------------------------------------
        # Each DMA has its own level; models handle this with a geo dimension
        # (one intercept per DMA), not with a "geo" regressor.
        base_level = 400_000 * size[g]
        trend = 1 + 0.002 * t                                   # organic growth
        season = 1 + 0.12 * np.cos(2 * np.pi * (t - 1) / 52)    # dating apps peak in January
        # Holidays: peaks (New Year "Dating Sunday", Valentine's) and dips (Thanksgiving, Christmas)
        holiday_peak = (((dates.month == 1) & (dates.day <= 14)) |
                        ((dates.month == 2) & (dates.day <= 14))).astype(float)
        holiday_dip = (((dates.month == 11) & (dates.day >= 22) & (dates.day <= 28)) |
                       ((dates.month == 12) & (dates.day >= 22))).astype(float)
        promo = (rng.random(N_WEEKS) < 0.10).astype(float)    # subscription discount weeks
        # Competition: a rival app's media pressure (index, 0 = dark, ~1 = typical burst),
        # national flights at different times from ours plus DMA-level variation
        competitor = comp_on * rng.lognormal(0, 0.15) * rng.lognormal(0, 0.10, N_WEEKS)
        # Macro: local unemployment rate (%), slow random walk around a DMA level
        unemployment = np.clip(rng.uniform(3.0, 6.0) + np.cumsum(rng.normal(0, 0.08, N_WEEKS)), 2.5, 9.0)
        baseline = (base_level * trend * season
                    * (1 + 0.08 * holiday_peak - 0.06 * holiday_dip + 0.06 * promo
                       - 0.05 * competitor)                     # rival bursts steal ~5% of revenue
                    * (1 - 0.02 * (unemployment - 4.5)))        # -2% revenue per +1pt unemployment

        row = {"geo": geo, "date": dates, "population": population[g],
               "trend": t / N_WEEKS, "holiday_peak": holiday_peak, "holiday_dip": holiday_dip,
               "promo": promo, "unemployment": unemployment, "competitor_pressure": competitor}

        # ---- Media buying -------------------------------------------------------
        # TV: national flights, DMA-level weight varies; bought in GRPs
        tv_grp = tv_on * 110 * rng.lognormal(0, 0.15) * rng.lognormal(0, 0.10, N_WEEKS)
        # OOH: billboard campaigns described by reach (% of DMA) x average frequency
        ooh_reach = ooh_on * np.clip(rng.normal(40, 5) * rng.lognormal(0, 0.08, N_WEEKS), 0, 85)
        ooh_freq = np.where(ooh_on > 0, np.clip(rng.normal(4.0, 0.6, N_WEEKS), 1.5, 8), 0.0)
        ooh_grp = ooh_reach * ooh_freq
        units = {"tv": tv_grp, "ooh": ooh_grp}
        spend = {"tv": tv_grp * CPP["tv"] * size[g], "ooh": ooh_grp * CPP["ooh"] * size[g]}
        for c, avg in DIGITAL_SPEND.items():  # always-on digital with its own rhythm
            phase = rng.uniform(0, 2 * np.pi)
            s = avg * size[g] * rng.lognormal(0, 0.35, N_WEEKS) * (1 + 0.3 * np.sin(2 * np.pi * t / 26 + phase))
            spend[c] = np.maximum(s, 0)
            units[c] = spend[c] / (CPM[c] * rng.lognormal(0, 0.05, N_WEEKS)) * 1000  # impressions

        # ---- Media effects -----------------------------------------------------
        media_effect = np.zeros(N_WEEKS)
        for c in CHANNELS:
            p = TRUTH[c]
            ad = geometric_adstock(units[c], p["alpha"])
            sat = hill(ad / ad.max(), p["ec"], p["slope"])
            beta_g = p["beta"] * rng.lognormal(0, 0.15)            # DMA heterogeneity
            effect = beta_g * sat * spend[c].mean() * 3.0          # scale so ROI lands near 0.5-2

            if c == "search" and g in LIFT_TEST_GEOS:
                # Geo-lift test: +30% weekly Search spend; delta_y = true steady-state weekly lift
                x = spend[c].mean()
                carry = sum(p["alpha"] ** lag for lag in range(8))
                to_units = units[c].mean() / x
                f = lambda v: beta_g * hill(v * to_units * carry / ad.max(), p["ec"], p["slope"]) * x * 3.0
                dy = f(1.3 * x) - f(x)
                lift_rows.append({"geo": geo, "channel": c, "x": x, "delta_x": 0.3 * x,
                                  "delta_y": dy, "sigma": 0.15 * dy})

            media_effect += effect
            contrib_total[c] += effect.sum()
            spend_total[c] += spend[c].sum()
            row[f"{c}_spend"] = spend[c]
        row["tv_grp"], row["ooh_grp"] = tv_grp, ooh_grp
        row["ooh_reach"], row["ooh_frequency"] = ooh_reach, ooh_freq
        for c in DIGITAL_SPEND:
            row[f"{c}_impressions"] = units[c]

        noise = rng.normal(0, 0.03 * base_level, N_WEEKS)
        row["revenue"] = baseline + media_effect + noise
        rows.append(pd.DataFrame(row))

    df = pd.concat(rows, ignore_index=True)
    truth = pd.DataFrame({
        "channel": CHANNELS,
        "true_roi": [contrib_total[c] / spend_total[c] for c in CHANNELS],
        "true_contribution": [contrib_total[c] for c in CHANNELS],
        "total_spend": [spend_total[c] for c in CHANNELS],
        "adstock_alpha": [TRUTH[c]["alpha"] for c in CHANNELS],
        "half_life_weeks": [np.log(0.5) / np.log(TRUTH[c]["alpha"]) for c in CHANNELS],
        "hill_ec": [TRUTH[c]["ec"] for c in CHANNELS],
        "hill_slope": [TRUTH[c]["slope"] for c in CHANNELS],
    })
    return df, truth, pd.DataFrame(lift_rows)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    df, truth, lift = simulate()
    df.to_csv(out / "mmm_geo_weekly.csv", index=False)
    truth.to_csv(out / "ground_truth.csv", index=False)
    lift.to_csv(out / "lift_test_search.csv", index=False)
    media_share = truth["true_contribution"].sum() / df["revenue"].sum()
    print(truth.round(3).to_string(index=False))
    print(f"Media share of revenue: {media_share:.1%}")
    print(df[[f"{c}_spend" for c in CHANNELS]].corr().round(2))
