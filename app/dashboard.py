"""MMM dashboard: the stakeholder-facing layer on top of the model.

Tab 1  Curve setup    shape the two media transformations before modeling:
                      adstock (carryover) and Hill saturation (diminishing returns),
                      and see how a flighting plan turns into a response over time.
                      These shapes inform the priors of the Bayesian model.
Tab 2  Model results  weekly revenue decomposition from the fitted PyMC-Marketing model:
                      base, seasonality, trend, holidays, promotion, macro, competition, each channel, residual;
                      a year-over-year revenue bridge and channel efficiency (ROI, spend vs effect share).

Run:  streamlit run app/dashboard.py
Needs results/decomposition_pymc.csv (from src/fit_pymc.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

pio.templates["mmm"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=12, color="#4A5070"),
    title=dict(font=dict(size=15, color="#1E2235"), x=0, xanchor="left"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, linecolor="#E3E6F2", ticks="", zeroline=False),
    yaxis=dict(gridcolor="#EEF0F7", zeroline=True, zerolinecolor="#D5DAF0", ticks=""),
    legend=dict(font=dict(size=11)), hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#E3E6F2"),
    colorway=["#4F46E5", "#0EA5E9", "#F43F5E", "#10B981", "#F59E0B"]))
pio.templates.default = "plotly_white+mmm"


def styled(fig: go.Figure) -> go.Figure:
    """Apply the app's chart style (white card, soft grid, Inter font)."""
    fig.update_layout(template="plotly_white+mmm", paper_bgcolor="white", plot_bgcolor="white",
                      margin_l=28, margin_r=28, title_x=0.02, title_font=dict(size=15, color="#1E2235"))
    return fig

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stCaption { font-family: 'Inter', sans-serif; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1400px; }
h1 { font-weight: 700; letter-spacing: -0.02em; }
[data-testid="stMetric"], [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    background: #FFFFFF; border: 1px solid #ECEEF7; border-radius: 18px;
    box-shadow: 0 1px 2px rgba(30,34,53,.04), 0 6px 20px rgba(30,34,53,.05);
}
[data-testid="stMetric"] { padding: 16px 20px; }
[data-testid="stPlotlyChart"] { padding: 0; overflow: hidden; }
[data-testid="stDataFrame"] { padding: 6px; overflow: hidden; }
[data-testid="stMetricLabel"] p { color: #7A809E; font-weight: 500; font-size: 0.85rem; }
[data-testid="stMetricValue"] { font-weight: 650; color: #1E2235; }
.stTabs [data-baseweb="tab-list"], .stTabs [role="tablist"] { gap: 8px; border-bottom: none !important; }
.stTabs button[role="tab"] { background: #EEF0FA !important; border-radius: 999px !important;
    padding: 8px 20px !important; border: none !important; }
.stTabs button[role="tab"][aria-selected="true"] { background: #4F46E5 !important; }
.stTabs button[role="tab"][aria-selected="true"] p { color: #FFFFFF !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
</style>
"""

ROOT = Path(__file__).resolve().parents[1]
# One palette for the whole app: saturated tones for media, soft tints for the baseline
INDIGO, SKY, ROSE, EMERALD, AMBER, INK = "#4F46E5", "#0EA5E9", "#F43F5E", "#10B981", "#F59E0B", "#1E2235"
COLORS = {
    "Base": "#E3E7F7", "Seasonality": "#CBD3F2", "trend": "#B3BEEB",
    "holiday_peak": "#FCD9A0", "holiday_dip": "#FBC4A0", "promo": "#F9B4C8", "unemployment": "#9DDCE8",
    "competitor_pressure": "#C4B5FD", "organic_social": "#A7F3D0",
    "tv": INDIGO, "ooh": SKY, "youtube": ROSE, "search": EMERALD, "tiktok": AMBER,
    "Residual": "#FECACA",
}
LABELS = {"trend": "Trend", "holiday_peak": "Holiday (peak)", "holiday_dip": "Holiday (dip)",
          "promo": "Promotion", "unemployment": "Macro: unemployment",
          "competitor_pressure": "Competition", "organic_social": "Organic social", "tv": "TV", "youtube": "YouTube",
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
        fig = go.Figure(go.Bar(x=lags, y=decay, marker_color=[INDIGO] + ["#C7D2FE"] * 8,
                               text=[f"{v:.0f}" for v in decay], textposition="outside"))
        fig.update_layout(barcornerradius=6, title=f"Carryover of one week of media · half-life {hl:.1f} weeks",
                          height=330, xaxis_title="Weeks after airing", yaxis_title="Effect remaining (%)",
                          yaxis=dict(range=[0, 115]), margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(styled(fig), width="stretch", theme=None)
    with right:  # saturation: total response and the marginal return of the next unit
        x = np.linspace(0, k * 4, 300)
        y = hill(x, k, shape, cap)
        marginal = np.gradient(y, x)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y * 100, name="Response (%)", line=dict(color=INDIGO, width=3)))
        fig.add_trace(go.Scatter(x=x, y=marginal / marginal.max() * 100, name="Marginal return (indexed)",
                                 line=dict(color=ROSE, width=2, dash="dot"), yaxis="y2"))
        fig.add_vline(x=k, line_color="#A5B4FC", line_dash="dash")
        fig.add_annotation(x=k, y=cap * 50, text="K", showarrow=True, ax=25, ay=-25)
        shape_txt = "C-shape: first exposures work best" if shape <= 1 else \
            "S-shape: needs repeated exposure before it works"
        fig.update_layout(title=f"Diminishing returns · {shape_txt}", height=330,
                          xaxis_title=f"Weekly adstocked {p['unit']}", yaxis_title="Response (%)",
                          yaxis=dict(range=[0, 105]),
                          yaxis2=dict(overlaying="y", side="right", range=[0, 105], showgrid=False,
                                      title="Marginal return"),
                          legend=dict(orientation="h", y=-0.3), margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(styled(fig), width="stretch", theme=None)

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
    fig.add_trace(go.Bar(x=weeks, y=media, name=f"Media ({p['unit']})", marker_color="#E0E7FF"))
    fig.add_trace(go.Scatter(x=weeks, y=ad, name="After carryover", line=dict(color=INDIGO, width=2)))
    fig.add_trace(go.Scatter(x=weeks, y=resp, name="Response (%)", yaxis="y2",
                             line=dict(color=AMBER, width=2.5)))
    fig.update_layout(barcornerradius=4, title="The plan over time: media in, response out", height=340,
                      yaxis=dict(title=p["unit"]), yaxis2=dict(overlaying="y", side="right",
                      range=[0, 105], tickvals=[0, 25, 50, 75, 100], showgrid=False, title="Response (%)"),
                      legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(styled(fig), width="stretch", theme=None)
    st.caption("Carryover (adstock) spreads each week's media over the following weeks; half-life = "
               "ln 0.5 / ln α. The Hill curve ceiling · x^s / (x^s + K^s) turns that into response: K is "
               "where the channel reaches half its ceiling, and s > 1 creates a threshold, the "
               "media-planning idea of effective frequency. These shapes are checked here first, then "
               "used as priors in the Bayesian MMM.")


# ---------------------------------------------------------------- tab 2: model results
@st.cache_data
def load_decomposition() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "decomposition_pymc.csv", parse_dates=["date"])


def add_organic_social(d: pd.DataFrame) -> None:
    """Illustrative only: carve an organic-social layer (unpaid posts, with occasional viral
    weeks) out of the modeled base. It is not a fitted model term, and totals are unchanged."""
    if "organic_social" in d:
        return
    rng = np.random.default_rng(7)
    idx = np.exp(np.convolve(rng.normal(0, 0.15, len(d) + 3), np.ones(4) / 2, "valid"))
    idx *= np.where(rng.random(len(d)) < 0.05, rng.uniform(2, 3, len(d)), 1.0)
    idx /= idx.mean()
    d["organic_social"] = 0.04 * d["Base"].values * idx
    d["Base"] -= d["organic_social"]


def decomposition_tab() -> None:
    d = load_decomposition().copy()
    # Show continuous drivers relative to their average, so "Base" holds the average level
    # and the bars show each week's deviation (common decomposition practice)
    for col in ["trend", "unemployment"]:  # competition stays as a negative bar in rival flights
        if col in d:
            m = d[col].mean()
            d[col] -= m
            d["Base"] += m
    add_organic_social(d)
    media = [c for c in ["tv", "ooh", "youtube", "search", "tiktok"] if c in d.columns]
    controls = [c for c in d.columns if c not in ["date", "Actual", "Residual", *media]]
    order = controls + media + ["Residual"]

    fit_path = ROOT / "results" / "fit_stats_pymc.csv"
    if fit_path.exists():
        fs = pd.read_csv(fit_path).set_index("level")
        st.markdown("**Model fit**")
        cols = st.columns(len(fs) * 3)
        i = 0
        for level, r in fs.iterrows():
            level = "DMA" if level.startswith("DMA") else level
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
                             line=dict(color=INK, width=2)))
    fig.update_layout(barmode="relative", height=480, bargap=0.05,
                      yaxis_title="Weekly revenue ($)", margin=dict(l=10, r=10, t=10, b=60),
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(styled(fig), width="stretch", theme=None)
    st.caption("Stacked weekly contributions (posterior means). Bars below zero are negative drivers "
               "(holiday dips, a rival app's ad flights, weeks with above-average unemployment). The black line is actual revenue. "
               "Organic social is an illustrative split of the base, not a fitted model term.")

    total = d[order].sum()
    st.markdown("**Where revenue came from, year 1 vs year 2**")
    revenue_mix(d, media)

    st.markdown("**What moved revenue from year 1 to year 2**")
    revenue_bridge(d, media)
    st.markdown("**Channel efficiency**")
    channel_efficiency(total, media)
    st.markdown("**Channel scorecard, year 1 vs year 2**")
    channel_scorecard(d, media)


# Driver groups for the year-over-year bridge
GROUPS = {"Organic baseline": ["Base", "trend"],
          "Calendar": ["Seasonality", "holiday_peak", "holiday_dip"],
          "Promotion": ["promo"], "Macro": ["unemployment"], "Competition": ["competitor_pressure"],
          "Organic social": ["organic_social"]}


def revenue_bridge(d: pd.DataFrame, media: list[str]) -> None:
    """Waterfall: year-1 revenue, the change attributed to each driver, year-2 revenue."""
    y1, y2 = d.iloc[:52], d.iloc[52:104]
    steps = {g: sum(y2[c].sum() - y1[c].sum() for c in cols if c in d) for g, cols in GROUPS.items()}
    steps.update({LABELS[c]: y2[c].sum() - y1[c].sum() for c in media})
    # Modeled revenue (actual minus residual), so the steps add up without an "unexplained" bar
    start = y1["Actual"].sum() - y1["Residual"].sum()
    end = y2["Actual"].sum() - y2["Residual"].sum()
    fig = go.Figure(go.Waterfall(
        x=["Year 1", *steps, "Year 2"], measure=["absolute", *["relative"] * len(steps), "total"],
        y=[start / 1e6, *[v / 1e6 for v in steps.values()], end / 1e6],
        text=[f"{start / 1e6:,.0f}", *[f"{v / 1e6:+.1f}" for v in steps.values()], f"{end / 1e6:,.0f}"],
        textposition="outside", connector=dict(line=dict(color="#D5DAF0", width=1)),
        increasing=dict(marker_color=EMERALD), decreasing=dict(marker_color=ROSE),
        totals=dict(marker_color=INDIGO)))
    lo = min(start, end) / 1e6
    fig.update_layout(height=380, yaxis=dict(title="Revenue ($M)", range=[lo * 0.95, max(start, end) / 1e6 * 1.03]),
                      margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    st.plotly_chart(styled(fig), width="stretch", theme=None)
    st.caption(f"Revenue changed {end / start - 1:+.1%} year over year. Each bar is the change in one "
               "driver's contribution; green adds revenue, red takes it away.")


GROUP_COLORS = {"Organic baseline": "#E3E7F7", "Calendar": "#FCD9A0", "Promotion": "#F9B4C8",
                "Macro": "#9DDCE8", "Competition": "#C4B5FD", "Organic social": "#A7F3D0",
                "Paid media": INDIGO}


def revenue_mix(d: pd.DataFrame, media: list[str]) -> None:
    """Revenue split into driver groups for each year: stacked columns plus a detail table."""
    years = {"Year 1": d.iloc[:52], "Year 2": d.iloc[52:104]}
    groups = {**GROUPS, "Paid media": media}
    rows = {g: {y: sum(df[c].sum() for c in cols if c in d) for y, df in years.items()}
            for g, cols in groups.items()}
    rev = {y: df["Actual"].sum() for y, df in years.items()}
    left, right = st.columns([3, 2])

    fig = go.Figure()
    for g, vals in rows.items():
        v = [vals[y] / 1e6 for y in years]
        fig.add_trace(go.Bar(x=list(years), y=v, name=g, marker_color=GROUP_COLORS[g], width=0.45,
                             text=[f"{g}<br>{vals[y] / rev[y]:.0%}" if abs(vals[y] / rev[y]) >= 0.04 else ""
                                   for y in years], textposition="inside", insidetextanchor="middle"))
    for y in years:
        fig.add_annotation(x=y, y=rev[y] / 1e6, text=f"<b>${rev[y] / 1e6:,.0f}M</b>",
                           showarrow=False, yshift=14)
    fig.update_layout(barmode="relative", barcornerradius=8, height=420, yaxis_title="Revenue ($M)", showlegend=False,
                      title=f"Total revenue {rev['Year 2'] / rev['Year 1'] - 1:+.1%} year over year",
                      margin=dict(l=10, r=10, t=50, b=10))
    left.plotly_chart(styled(fig), width="stretch", theme=None)

    table = pd.DataFrame({
        "Driver": list(rows),
        **{f"{y} $M": [rows[g][y] / 1e6 for g in rows] for y in years},
        **{f"{y} %": [rows[g][y] / rev[y] for g in rows] for y in years},
    })
    table["Δ $M"] = table["Year 2 $M"] - table["Year 1 $M"]
    right.dataframe(table.style.format({"Year 1 $M": "{:,.1f}", "Year 2 $M": "{:,.1f}",
                                        "Year 1 %": "{:.1%}", "Year 2 %": "{:.1%}",
                                        "Δ $M": "{:+,.1f}"}),
                    hide_index=True, width="stretch")
    right.caption("Negative values (competition, weeks of high unemployment) sit below zero in the chart. "
                  "Trend and unemployment are measured against their two-year average, so their "
                  "average level is part of the organic baseline.")


@st.cache_data
def load_efficiency() -> tuple[pd.Series, pd.DataFrame]:
    raw = pd.read_csv(ROOT / "data" / "mmm_geo_weekly.csv")
    spend = raw[[c for c in raw.columns if c.endswith("_spend")]].sum()
    spend.index = spend.index.str.removesuffix("_spend")
    roi = pd.read_csv(ROOT / "results" / "roi_pymc.csv").set_index("channel")
    return spend, roi


def channel_efficiency(total: pd.Series, media: list[str]) -> None:
    """Left: ROI with its 90% credible interval. Right: share of spend vs share of effect."""
    spend, roi = load_efficiency()
    order = roi.loc[media].sort_values("roi_mean").index.tolist()
    names = [LABELS[c] for c in order]
    left, right = st.columns(2)

    r = roi.loc[order]
    fig = go.Figure(go.Scatter(
        x=r.roi_mean, y=names, mode="markers", marker=dict(size=11, color=[COLORS[c] for c in order]),
        error_x=dict(type="data", symmetric=False, array=r.roi_hi - r.roi_mean,
                     arrayminus=r.roi_mean - r.roi_lo, color="#C7CEEA", thickness=2)))
    fig.add_vline(x=1, line_dash="dot", line_color="#A5B4FC",
                  annotation_text="break-even", annotation_position="top")
    fig.update_layout(title="ROI per $1 (90% credible interval)", height=320,
                      xaxis_title="Revenue per $ spent", margin=dict(l=10, r=10, t=50, b=10))
    left.plotly_chart(styled(fig), width="stretch", theme=None)

    spend_sh = spend[order] / spend[media].sum()
    effect_sh = total[order] / total[media].sum()
    fig = go.Figure()
    for i, c in enumerate(order):
        fig.add_trace(go.Scatter(x=[spend_sh[c], effect_sh[c]], y=[names[i]] * 2, mode="lines",
                                 line=dict(color="#E0E4F5", width=4), showlegend=False))
    fig.add_trace(go.Scatter(x=spend_sh, y=names, mode="markers", name="Share of spend",
                             marker=dict(size=11, symbol="circle-open", color="#8B93B8", line=dict(width=2))))
    fig.add_trace(go.Scatter(x=effect_sh, y=names, mode="markers", name="Share of media-driven revenue",
                             marker=dict(size=11, color=[COLORS[c] for c in order])))
    fig.update_layout(title="Spend share vs effect share", height=320, xaxis_tickformat=".0%",
                      legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=50, b=10))
    right.plotly_chart(styled(fig), width="stretch", theme=None)
    st.caption("A channel whose effect share sits to the right of its spend share earns more than its "
               "budget weight. Intervals that cross break-even are not yet proven profitable.")

UNITS = {"tv": ("tv_grp", "GRP", 1), "ooh": ("ooh_grp", "GRP", 1),
         "youtube": ("youtube_impressions", "1k impr.", 1000), "search": ("search_impressions", "1k impr.", 1000),
         "tiktok": ("tiktok_impressions", "1k impr.", 1000)}


@st.cache_data
def load_raw() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "mmm_geo_weekly.csv", parse_dates=["date"])


RENAME = {'Spend Y1 ($K)': 'Spend Y1', 'Spend Y2 ($K)': 'Spend Y2', 'Cost/unit Y1 ($)': 'Cost/unit Y1', 'Cost/unit Y2 ($)': 'Cost/unit Y2', 'Cost/unit Δ': 'Δ cost', 'Revenue/unit Y1 ($)': 'Rev/unit Y1', 'Revenue/unit Y2 ($)': 'Rev/unit Y2', 'Revenue/unit Δ': 'Δ rev/unit', 'Incr. revenue Y1 ($K)': 'Incr. rev Y1', 'Incr. revenue Y2 ($K)': 'Incr. rev Y2'}


def channel_scorecard(d: pd.DataFrame, media: list[str]) -> None:
    """Absolute values per channel and year: spend, cost per unit bought, revenue per unit,
    incremental revenue and ROI, with the year-over-year change."""
    raw = load_raw()
    dates = sorted(raw["date"].unique())
    y1_dates, y2_dates = set(dates[:52]), set(dates[52:104])
    rows = []
    for c in media:
        unit_col, unit, per = UNITS[c]
        rec = {"Channel": LABELS[c], "Unit": unit}
        for y, ds, dec in [("Y1", y1_dates, d.iloc[:52]), ("Y2", y2_dates, d.iloc[52:104])]:
            part = raw[raw["date"].isin(ds)]
            spend, units, inc = part[f"{c}_spend"].sum(), part[unit_col].sum() / per, dec[c].sum()
            rec.update({f"Spend {y} ($K)": spend / 1e3, f"Cost/unit {y} ($)": spend / units,
                        f"Revenue/unit {y} ($)": inc / units, f"Incr. revenue {y} ($K)": inc / 1e3,
                        f"ROI {y}": inc / spend})
        rec["Cost/unit Δ"] = rec["Cost/unit Y2 ($)"] / rec["Cost/unit Y1 ($)"] - 1
        rec["Revenue/unit Δ"] = rec["Revenue/unit Y2 ($)"] / rec["Revenue/unit Y1 ($)"] - 1
        rows.append(rec)
    df = pd.DataFrame(rows)
    cols = ["Channel", "Unit", "Spend Y1 ($K)", "Spend Y2 ($K)", "Cost/unit Y1 ($)", "Cost/unit Y2 ($)",
            "Cost/unit Δ", "Revenue/unit Y1 ($)", "Revenue/unit Y2 ($)", "Revenue/unit Δ",
            "Incr. revenue Y1 ($K)", "Incr. revenue Y2 ($K)", "ROI Y1", "ROI Y2"]
    df = df[cols].rename(columns=RENAME)
    df["Channel"] = df["Channel"] + " · " + df.pop("Unit")

    def pct(v):
        return "0%" if abs(v) < 0.005 else f"{v:+.0%}"

    def tone(v, good_if_negative):
        good = v < 0 if good_if_negative else v > 0
        return f"color: {'#059669' if good else '#E11D48'}" if abs(v) >= 0.005 else ""

    sty = (df.style
           .format({c: "{:,.0f}" for c in ["Spend Y1", "Spend Y2", "Incr. rev Y1", "Incr. rev Y2"]}
                   | {c: "{:,.2f}" for c in ["Cost/unit Y1", "Cost/unit Y2", "Rev/unit Y1", "Rev/unit Y2",
                                              "ROI Y1", "ROI Y2"]}
                   | {"Δ cost": pct, "Δ rev/unit": pct})
           .bar(subset=["Spend Y1", "Spend Y2"], color="#E0E7FF", vmin=0)
           .bar(subset=["Incr. rev Y1", "Incr. rev Y2"], color="#D1FAE5", vmin=0)
           .map(lambda v: tone(v, True), subset=["Δ cost"])
           .map(lambda v: tone(v, False), subset=["Δ rev/unit"]))
    st.dataframe(sty, hide_index=True, width="stretch")
    st.caption("Spend and incremental revenue in $K. Cost per unit = spend per GRP (TV, OOH) or per 1,000 impressions (digital). Revenue per unit = "
               "modeled incremental revenue per GRP or per 1,000 impressions. Green: cheaper buying or more "
               "revenue per unit than last year; red: the opposite.")


def main() -> None:
    st.set_page_config(page_title="MMM Dashboard", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("MMM Dashboard")
    st.caption("Simulated dating-app data · PyMC-Marketing model · all figures illustrative")
    t1, t2 = st.tabs(["1 · Curve design", "2 · Model results"])
    with t1:
        curve_tab()
    with t2:
        decomposition_tab()


if __name__ == "__main__":
    main()
