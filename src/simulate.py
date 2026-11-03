"""Simulate geo-level weekly marketing data with a known ground truth.

The setup mimics a subscription dating app running paid media in 10 US markets
for two years. Every channel has a known carryover (geometric adstock) and a
known diminishing-returns curve (Hill saturation), so we can check how well
each MMM package recovers the true ROI.

Output: data/mmm_geo_weekly.csv, data/ground_truth.csv,
        data/lift_test_search.csv (a simulated geo-lift experiment on Search)
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_GEOS = 10
N_WEEKS = 104
CHANNELS = ["tv", "social", "search", "tiktok"]

# Ground-truth media parameters (shared across geos, with geo-level noise on beta)
TRUTH = {
    #          adstock decay, Hill half-saturation (as share of geo max), Hill slope, effect size
    "tv":     {"alpha": 0.6, "ec": 0.5, "slope": 2.0, "beta": 0.9},
    "social": {"alpha": 0.3, "ec": 0.4, "slope": 1.5, "beta": 0.7},
    "search": {"alpha": 0.1, "ec": 0.3, "slope": 1.0, "beta": 1.1},
    "tiktok": {"alpha": 0.2, "ec": 0.6, "slope": 1.5, "beta": 0.4},
}
# Average weekly spend per geo (USD) for each channel
LIFT_TEST_GEOS = (0, 3, 6)  # geos where we "ran" a Search geo-lift test
AVG_SPEND = {"tv": 40_000, "social": 25_000, "search": 20_000, "tiktok": 15_000}


def geometric_adstock(x: np.ndarray, alpha: float, l_max: int = 8) -> np.ndarray:
    """Carryover: today's effect = sum_{l=0..l_max-1} alpha^l * spend_{t-l}."""
    out = np.zeros_like(x, dtype=float)
    for lag in range(l_max):
        out[lag:] += (alpha**lag) * x[: len(x) - lag]
    return out


def hill(x: np.ndarray, ec: float, slope: float) -> np.ndarray:
    """Diminishing returns, scaled to [0, 1)."""
    return x**slope / (x**slope + ec**slope)


def simulate(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-07", periods=N_WEEKS, freq="W-SUN")
    t = np.arange(N_WEEKS)

    population = rng.integers(500_000, 5_000_000, N_GEOS)
    size = population / population.mean()  # bigger markets spend and earn more

    lift_rows = []
    rows, contrib_total, spend_total = [], {c: 0.0 for c in CHANNELS}, {c: 0.0 for c in CHANNELS}
    for g in range(N_GEOS):
        geo = f"geo_{g:02d}"
        # Baseline revenue: geo level + trend + yearly seasonality (Jan peak for dating apps)
        base_level = 400_000 * size[g]
        trend = 1 + 0.002 * t
        season = 1 + 0.12 * np.cos(2 * np.pi * (t - 1) / 52)
        holiday = ((dates.month == 2) & (dates.day <= 14)).astype(float)  # Valentine's run-up
        baseline = base_level * trend * season * (1 + 0.08 * holiday)

        media_effect = np.zeros(N_WEEKS)
        geo_rows = {"geo": geo, "date": dates, "population": population[g],
                    "valentines": holiday, "trend": t / N_WEEKS}
        for c in CHANNELS:
            # Spend: flighted (on/off bursts) for TV, always-on with noise for digital
            avg = AVG_SPEND[c] * size[g]
            if c == "tv":
                on = (rng.random(N_WEEKS) < 0.55).astype(float)
                spend = on * avg * rng.lognormal(0.3, 0.3, N_WEEKS)
            else:
                phase = rng.uniform(0, 2 * np.pi)  # each channel has its own flighting rhythm
                spend = avg * rng.lognormal(0, 0.35, N_WEEKS) * (1 + 0.3 * np.sin(2 * np.pi * t / 26 + phase))
            spend = np.maximum(spend, 0)

            p = TRUTH[c]
            ad = geometric_adstock(spend, p["alpha"])
            sat = hill(ad / ad.max(), p["ec"], p["slope"])
            beta_g = p["beta"] * rng.lognormal(0, 0.15)  # geo heterogeneity
            # Effect scaled so that beta ~ ROI-ish magnitude at average spend
            effect = beta_g * sat * avg * 3.0

            if c == "search" and g in LIFT_TEST_GEOS:
                # Simulated geo-lift test: raise weekly Search spend 30% in this geo.
                # delta_y is the true weekly revenue lift at steady state.
                x = spend.mean()
                carry = sum(p["alpha"] ** lag for lag in range(8))
                f = lambda v: beta_g * hill(v * carry / ad.max(), p["ec"], p["slope"]) * avg * 3.0
                dy = f(1.3 * x) - f(x)
                lift_rows.append({"geo": geo, "channel": c, "x": x, "delta_x": 0.3 * x,
                                  "delta_y": dy, "sigma": 0.15 * dy})

            media_effect += effect
            contrib_total[c] += effect.sum()
            spend_total[c] += spend.sum()
            geo_rows[c] = spend

        noise = rng.normal(0, 0.03 * base_level, N_WEEKS)
        geo_rows["revenue"] = baseline + media_effect + noise
        rows.append(pd.DataFrame(geo_rows))

    df = pd.concat(rows, ignore_index=True)
    truth = pd.DataFrame(
        {
            "channel": CHANNELS,
            "true_roi": [contrib_total[c] / spend_total[c] for c in CHANNELS],
            "true_contribution": [contrib_total[c] for c in CHANNELS],
            "total_spend": [spend_total[c] for c in CHANNELS],
            "adstock_alpha": [TRUTH[c]["alpha"] for c in CHANNELS],
        }
    )
    return df, truth, pd.DataFrame(lift_rows)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    df, truth, lift = simulate()
    lift.to_csv(out / "lift_test_search.csv", index=False)
    df.to_csv(out / "mmm_geo_weekly.csv", index=False)
    truth.to_csv(out / "ground_truth.csv", index=False)
    media_share = truth["true_contribution"].sum() / df["revenue"].sum()
    print(df.head())
    print(truth.round(3))
    print(f"Media share of revenue: {media_share:.1%}")
