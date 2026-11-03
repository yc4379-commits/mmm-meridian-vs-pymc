"""MMM dashboard: the stakeholder-facing layer on top of the model.

Tab 1  Curve setup    shape the two media transformations before modeling:
                      adstock (carryover) and Hill saturation (diminishing returns),
                      and see how a flighting plan turns into a response over time.
                      These shapes inform the priors of the Bayesian model.
Tab 2  Model results  weekly revenue decomposition from the fitted PyMC-Marketing model:
                      base, seasonality, trend, holidays, promotion, macro, each channel, residual.

Run:  streamlit run app/dashboard.py
Needs results/decomposition_pymc.csv (from src/fit_pymc.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "Base": "#C9CED6", "Seasonality": "#A7B4C6", "trend": "#8C9BB0",
    "holiday_peak": "#E9C46A", "holiday_dip": "#F4A261", "promo": "#E76F51", "unemployment": "#7F7F7F",
    "competitor_pressure": "#5E4B8B",
    "tv": "#4C78A8", "social": "#F58518", "search": "#54A24B", "tiktok": "#B279A2", "ooh": "#9D755D",
    "Residual": "#E45756",
}
LABELS = {"trend": "Trend", "holiday_peak": "Holiday (peak)", "holiday_dip": "Holiday (dip)",
          "promo": "Promotion", "unemployment": "Macro: unemployment",
          "competitor_pressure": "Competition", "tv": "TV", "social": "Social",
          "search": "Search", "tiktok": "TikTok", "ooh": "OOH"}


# ---------------------------------------------------------------- tab 1: curve setup
def geometric_adstock(x: np.ndarray, alpha: float) -> np.ndarray:
    """Carryover: each week keeps a share alpha of last week's adstocked media."""
    out = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        out[t] = x[t] + (alpha * out[t - 1] if t else 0.0)
    return out


def hill(x: np.ndarray, k: float, s: float, ceiling: float) -> np.ndarray:
    """Diminishing returns: ceiling * x^s / (x^s + K^s). s <= 1 C-shape, s > 1 S-shape."""
    x = np.asarray(x, dtype=float)
    return ceiling * x**s / (x**s + k**s)


def flighting_plan(n_weeks: int, bursts: list[tuple[int, int, float]]) -> np.ndarray:
    """Weekly GRPs: bursts of (start week, length, GRPs per week), tapering within a burst."""
    g = np.zeros(n_weeks)
    for start, length, level in bursts:
        for i in range(length):
            if start + i < n_weeks:
                g[start + i] = level * (1 - 0.12 * i)
    return g


# Default curve settings per channel (match the simulation's assumptions)
PRESETS = {
    "TV": dict(alpha=0.60, k=120, s=2.0, ceiling=90, unit="GRPs", flights=[(0, 5, 120), (18, 4, 90), (36, 6, 110), (49, 3, 100)]),
    "OOH": dict(alpha=0.20, k=150, s=1.5, ceiling=70, unit="GRPs", flights=[(12, 6, 160), (30, 8, 150)]),
    "Search": dict(alpha=0.10, k=40, s=1.0, ceiling=60, unit="impressions (k)", flights=None),
}


def curve_tab() -> None:
    st.markdown("#### How one channel turns media into response")
    channel = st.radio("Channel", list(PRESETS), horizontal=True, label_visibility="collapsed")
    p = PRESETS[channel]
    c1, c2, c3, c4 = st.columns(4)
    alpha = c1.slider("Carryover: decay rate α", 0.0, 0.95, p["alpha"], 0.05, key=f"a{channel}",
                      help="Share of last week's effect that is still working this week")
    k = c2.slider(f"Half-saturation K ({p['unit']})", 10, 400, p["k"], 10, key=f"k{channel}",
                  help="Weekly media level where the channel reaches half of its ceiling")
    shape = c3.slider("Shape s", 0.5, 4.0, p["s"], 0.1, key=f"s{channel}",
                      help="1 = C-shape (one exposure can act). Above 1 = S-shape: people need "
                           "several exposures before they act (an effective-frequency threshold)")
    ceiling = c4.slider("Ceiling (max response, %)", 10, 100, p["ceiling"], 5, key=f"c{channel}",
                        help="The most this channel can ever deliver, however much is spent")
    hl = np.log(0.5) / np.log(alpha) if 0 < alpha < 1 else 0.0
    cap = ceiling / 100

    left, right = st.columns(2)
    with left:  # carryover: what happens to a single week of media
        lags = np.arange(9)
        decay = 100 * alpha ** lags
        fig = go.Figure(go.Bar(x=lags, y=decay, marker_color=["#2A6F97"] + ["#A9CCE3"] * 8,
                               text=[f"{v:.0f}" for v in decay], textposition="outside"))
        fig.update_layout(title=f"Carryover of one week of media · half-life {hl:.1f} weeks",
                          height=330, xaxis_title="Weeks after airing", yaxis_title="Effect remaining (%)",
                          yaxis=dict(range=[0, 115]), margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with right:  # saturation: total response and the marginal return of the next unit
        x = np.linspace(0, k * 4, 300)
        y = hill(x, k, shape, cap)
        marginal = np.gradient(y, x)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y * 100, name="Response (%)", line=dict(color="#2A6F97", width=3)))
        fig.add_trace(go.Scatter(x=x, y=marginal / marginal.max() * 100, name="Marginal return (indexed)",
                                 line=dict(color="#C0504D", width=2, dash="dot"), yaxis="y2"))
        fig.add_vline(x=k, line_color="#999", line_dash="dash")
        fig.add_annotation(x=k, y=cap * 50, text="K", showarrow=True, ax=25, ay=-25)
        shape_txt = "C-shape: first exposures work best" if shape <= 1 else \
            "S-shape: needs repeated exposure before it works"
        fig.update_layout(title=f"Diminishing returns · {shape_txt}", height=330,
                          xaxis_title=f"Weekly adstocked {p['unit']}", yaxis_title="Response (%)",
                          yaxis=dict(range=[0, 105]),
                          yaxis2=dict(overlaying="y", side="right", range=[0, 105], showgrid=False,
                                      title="Marginal return"),
                          legend=dict(orientation="h", y=-0.3), margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")

    # Plan over time: always-on vs flighted
    weeks = pd.date_range("2025-01-05", periods=78, freq="W-SUN")
    if p["flights"]:
        media = flighting_plan(len(weeks), p["flights"] + [(f + 52, n, lv) for f, n, lv in p["flights"]])
    else:  # always-on digital
        rng = np.random.default_rng(0)
        media = k * 1.1 * rng.lognormal(0, 0.25, len(weeks))
    ad = geometric_adstock(media, alpha)
    resp = hill(ad, k, shape, cap) * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(x=weeks, y=media, name=f"Media ({p['unit']})", marker_color="#D9E4EC"))
    fig.add_trace(go.Scatter(x=weeks, y=ad, name="After carryover", line=dict(color="#2A6F97", width=2)))
    fig.add_trace(go.Scatter(x=weeks, y=resp, name="Response (%)", yaxis="y2",
                             line=dict(color="#E07A1F", width=2.5)))
    fig.update_layout(title="The plan over time: media in, response out", height=340,
                      yaxis=dict(title=p["unit"]), yaxis2=dict(overlaying="y", side="right",
                      range=[0, 105], tickvals=[0, 25, 50, 75, 100], showgrid=False, title="Response (%)"),
                      legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Carryover (adstock) spreads each week's media over the following weeks; half-life = "
               "ln 0.5 / ln α. The Hill curve ceiling · x^s / (x^s + K^s) turns that into response: K is "
               "where the channel reaches half its ceiling, and s > 1 creates a threshold, the "
               "media-planning idea of effective frequency. These shapes are checked here first, then "
               "used as priors in the Bayesian MMM.")


# ---------------------------------------------------------------- tab 2: model results
@st.cache_data
def load_decomposition() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "decomposition_pymc.csv", parse_dates=["date"])


def decomposition_tab() -> None:
    d = load_decomposition().copy()
    # Show continuous drivers relative to their average, so "Base" holds the average level
    # and the bars show each week's deviation (common decomposition practice)
    for col in ["trend", "unemployment"]:  # competition stays as a negative bar in rival flights
        if col in d:
            m = d[col].mean()
            d[col] -= m
            d["Base"] += m
    media = [c for c in ["tv", "ooh", "social", "search", "tiktok"] if c in d.columns]
    controls = [c for c in d.columns if c not in ["date", "Actual", "Residual", *media]]
    order = controls + media + ["Residual"]

    fit_path = ROOT / "results" / "fit_stats_pymc.csv"
    if fit_path.exists():
        fs = pd.read_csv(fit_path).set_index("level")
        st.markdown("**Model fit**")
        cols = st.columns(len(fs) * 3)
        i = 0
        for level, r in fs.iterrows():
            cols[i].metric(f"R² · {level}", f"{r['r2']:.3f}")
            cols[i + 1].metric(f"MAPE · {level}", f"{r['mape']:.1%}")
            cols[i + 2].metric(f"Durbin-Watson · {level}", f"{r['durbin_watson']:.2f}")
            i += 3
        st.caption("Durbin-Watson near 2 means residuals are not autocorrelated; well below 2 usually "
                   "points to a missing driver or too little carryover in the model.")

    fig = go.Figure()
    for col in order:
        fig.add_trace(go.Bar(x=d["date"], y=d[col], name=LABELS.get(col, col),
                             marker_color=COLORS.get(col, "#888"), marker_line_width=0))
    fig.add_trace(go.Scatter(x=d["date"], y=d["Actual"], name="Actual revenue", mode="lines",
                             line=dict(color="black", width=1.5)))
    fig.update_layout(barmode="relative", height=480, bargap=0.05,
                      yaxis_title="Weekly revenue ($)", margin=dict(l=10, r=10, t=10, b=60),
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    st.caption("Stacked weekly contributions (posterior means). Bars below zero are negative drivers "
               "(holiday dips, a rival app's ad flights, weeks with above-average unemployment). The black line is actual revenue.")

    total = d[order].sum()
    share = total / d["Actual"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Media-driven share of revenue", f"{share[media].sum():.1%}")
    c2.metric("Base + seasonality share", f"{share[['Base', 'Seasonality']].sum():.1%}")
    c3.metric("Residual share", f"{share['Residual']:.1%}")
    contrib = (total[media] / total[media].sum()).sort_values()
    fig = go.Figure(go.Bar(x=total[contrib.index] / 1e6, y=[LABELS.get(c, c) for c in contrib.index],
                           orientation="h", marker_color=[COLORS.get(c, "#888") for c in contrib.index],
                           text=[f"{v:.0%} of media" for v in contrib], textposition="auto"))
    fig.update_layout(title="Revenue driven by each channel (2 years, $M)", height=300,
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.set_page_config(page_title="MMM Dashboard", layout="wide")
    st.title("MMM Dashboard")
    st.caption("Simulated dating-app data · PyMC-Marketing model · all figures illustrative")
    t1, t2 = st.tabs(["1 · Curve design", "2 · Model results"])
    with t1:
        curve_tab()
    with t2:
        decomposition_tab()


if __name__ == "__main__":
    main()
