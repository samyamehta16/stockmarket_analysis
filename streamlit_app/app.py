import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pandas_datareader import data as pdr
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

st.set_page_config(
    page_title="SP500 Forecasting",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

BLACK      = "#0A0A0A"
DARK       = "#111111"
CARD       = "#1A1A1A"
BORDER     = "#2A2A2A"
YELLOW     = "#F5C518"
YELLOW_DIM = "#C9A10E"
WHITE      = "#F0F0F0"
MUTED      = "#666666"
GREEN      = "#39FF14"
RED        = "#FF4444"
FONT       = "'Courier New', Courier, monospace"

st.markdown(f"""
<style>
  html, body, [class*="css"], .stApp {{
    background-color: {BLACK} !important;
    color: {WHITE} !important;
    font-family: {FONT} !important;
  }}
  section[data-testid="stSidebar"] {{
    background-color: {DARK} !important;
    border-right: 1px solid {BORDER} !important;
  }}
  section[data-testid="stSidebar"] * {{
    font-family: {FONT} !important;
    color: {WHITE} !important;
  }}
  section[data-testid="stSidebar"] .stButton > button {{
    background-color: {YELLOW} !important;
    color: {BLACK} !important;
    font-family: {FONT} !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 10px 0 !important;
    font-size: 12px !important;
    letter-spacing: 3px !important;
    width: 100% !important;
  }}
  section[data-testid="stSidebar"] .stButton > button:hover {{
    background-color: {YELLOW_DIM} !important;
  }}
  .stTabs [data-baseweb="tab-list"] {{
    background-color: {DARK} !important;
    border-bottom: 1px solid {BORDER} !important;
    gap: 0px !important;
  }}
  .stTabs [data-baseweb="tab"] {{
    background-color: transparent !important;
    color: {MUTED} !important;
    font-family: {FONT} !important;
    font-size: 11px !important;
    letter-spacing: 3px !important;
    padding: 14px 36px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
  }}
  .stTabs [aria-selected="true"] {{
    color: {YELLOW} !important;
    border-bottom: 2px solid {YELLOW} !important;
    background-color: transparent !important;
  }}
  .stTabs [data-baseweb="tab-panel"] {{
    background-color: {BLACK} !important;
    padding-top: 24px !important;
  }}
  .stDateInput input, .stNumberInput input, .stTextInput input, .stSelectbox div {{
    background-color: {CARD} !important;
    color: {WHITE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 2px !important;
    font-family: {FONT} !important;
    font-size: 12px !important;
  }}
  .stDownloadButton > button {{
    background-color: transparent !important;
    color: {YELLOW} !important;
    border: 1px solid {YELLOW} !important;
    border-radius: 2px !important;
    font-family: {FONT} !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    width: 100% !important;
  }}
  .stDownloadButton > button:hover {{
    background-color: {YELLOW} !important;
    color: {BLACK} !important;
  }}
  .stRadio label, .stRadio div {{
    font-family: {FONT} !important;
    color: {MUTED} !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
  }}
  .stAlert, .stSuccess, .stInfo {{
    background-color: {CARD} !important;
    border-left: 3px solid {YELLOW} !important;
    font-family: {FONT} !important;
    color: {WHITE} !important;
  }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding: 2rem 2.5rem !important; }}
  ::-webkit-scrollbar {{ width: 4px; background: {DARK}; }}
  ::-webkit-scrollbar-thumb {{ background: {BORDER}; }}
</style>
""", unsafe_allow_html=True)

def metric_card(label, value, delta=None, delta_good=True, accent=YELLOW):
    delta_html = ""
    if delta:
        c = GREEN if delta_good else RED
        delta_html = f"<div style='color:{c};font-size:11px;margin-top:6px;letter-spacing:1px;'>{delta}</div>"
    return f"""
    <div style='background:{CARD};border:1px solid {BORDER};border-top:2px solid {accent};
                padding:18px 16px;font-family:{FONT};'>
        <div style='color:{MUTED};font-size:9px;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px;'>{label}</div>
        <div style='color:{WHITE};font-size:20px;font-weight:700;'>{value}</div>
        {delta_html}
    </div>"""

def section_title(text):
    st.markdown(f"""
    <div style='margin:32px 0 14px 0;font-family:{FONT};'>
        <span style='color:{YELLOW};font-size:9px;letter-spacing:4px;'>// </span>
        <span style='color:{WHITE};font-size:11px;font-weight:700;letter-spacing:4px;text-transform:uppercase;'>{text}</span>
        <div style='border-bottom:1px solid {BORDER};margin-top:10px;'></div>
    </div>""", unsafe_allow_html=True)

def chart_style(fig, height=380):
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=DARK,
        font=dict(family=FONT, color=MUTED, size=10),
        height=height,
        margin=dict(l=12, r=12, t=16, b=12),
        legend=dict(orientation="h", y=-0.2, bgcolor="rgba(0,0,0,0)",
                    font=dict(family=FONT, color=MUTED, size=9)),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False,
                   tickfont=dict(family=FONT, color=MUTED, size=9)),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False,
                   tickfont=dict(family=FONT, color=MUTED, size=9), tickformat=",.0f"),
    )
    return fig

@st.cache_data(ttl=3600)
def fetch_live_data(start_date, end_date):
    api_key = os.getenv("FRED_API_KEY", None)
    df = pdr.DataReader("SP500", "fred", start=str(start_date),
                        end=str(end_date), api_key=api_key)
    df.columns = ["price"]
    df.index.name = "date"
    df = df.reset_index()
    df["date"]  = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(method="ffill")
    return df.dropna().sort_values("date").reset_index(drop=True)

@st.cache_data(ttl=3600)
def engineer_features(df):
    df = df.copy()
    df["daily_return"]  = df["price"].pct_change() * 100
    df["log_return"]    = np.log(df["price"] / df["price"].shift(1)) * 100
    df["ma_7"]          = df["price"].rolling(7,   min_periods=1).mean()
    df["ma_30"]         = df["price"].rolling(30,  min_periods=1).mean()
    df["ma_90"]         = df["price"].rolling(90,  min_periods=1).mean()
    df["ma_200"]        = df["price"].rolling(200, min_periods=1).mean()
    df["volatility_7"]  = df["daily_return"].rolling(7).std()
    df["volatility_30"] = df["daily_return"].rolling(30).std()
    df["volatility_90"] = df["daily_return"].rolling(90).std()
    df["bb_mid"]        = df["price"].rolling(20).mean()
    df["bb_std"]        = df["price"].rolling(20).std()
    df["bb_upper"]      = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"]      = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_width"]      = df["bb_upper"] - df["bb_lower"]
    df["golden_cross"]  = (df["ma_30"] > df["ma_200"]).astype(int)
    df["month_name"]    = df["date"].dt.strftime("%b")
    df["year"]          = df["date"].dt.year
    df["cumulative_return"] = (df["price"] / df["price"].iloc[0] - 1) * 100
    return df

@st.cache_data(ttl=3600)
def run_hybrid_model(df, forecast_days=30):
    FEATURES = ["ma_7","ma_30","ma_90","ma_200",
                "volatility_7","volatility_30","volatility_90",
                "bb_upper","bb_lower","bb_mid","bb_width",
                "daily_return","log_return","golden_cross"]
    dm      = df.dropna(subset=FEATURES+["price"]).reset_index(drop=True)
    n_test  = max(60, int(len(dm)*0.20))
    n_train = len(dm) - n_test
    train   = dm.iloc[:n_train].copy()

    pt = train[["date","price"]].rename(columns={"date":"ds","price":"y"})
    pf = dm[["date","price"]].rename(columns={"date":"ds","price":"y"})

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False, seasonality_mode="multiplicative",
                changepoint_prior_scale=0.05, interval_width=0.95)
    m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    m.fit(pt)

    pred = m.predict(pf[["ds"]])
    dm["prophet_pred"]  = pred["yhat"].values
    dm["prophet_lower"] = pred["yhat_lower"].values
    dm["prophet_upper"] = pred["yhat_upper"].values
    dm["residual"]      = dm["price"] - dm["prophet_pred"]

    train = dm.iloc[:n_train].copy()
    rf = RandomForestRegressor(n_estimators=300, max_depth=6,
                               min_samples_leaf=10, max_features="sqrt",
                               random_state=42, n_jobs=-1)
    rf.fit(train[FEATURES].values, train["residual"].values)
    dm["rf_correction"] = rf.predict(dm[FEATURES].values)
    dm["hybrid_pred"]   = dm["prophet_pred"] + dm["rf_correction"]

    test = dm.iloc[n_train:].copy()
    a, p = test["price"].values, test["hybrid_pred"].values
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(a, p))),
        "MAE" : float(mean_absolute_error(a, p)),
        "MAPE": float(np.mean(np.abs((a-p)/a))*100),
        "R2"  : float(1 - np.sum((a-p)**2)/np.sum((a-a.mean())**2))
    }

    last_date    = dm["date"].max()
    future_dates = pd.date_range(start=last_date+timedelta(days=1),
                                 periods=forecast_days, freq="B")
    fp  = m.predict(pd.DataFrame({"ds": future_dates}))
    lf  = dm[FEATURES].iloc[-1].values.reshape(1,-1)
    frf = rf.predict(np.repeat(lf, forecast_days, axis=0))
    future_df = pd.DataFrame({
        "date"    : future_dates,
        "forecast": fp["yhat"].values + frf,
        "lower_95": fp["yhat_lower"].values,
        "upper_95": fp["yhat_upper"].values,
    })
    feat_imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return dm, test, future_df, feat_imp, metrics, n_train

with st.sidebar:
    st.markdown(f"""
    <div style='padding:20px 0 8px 0;font-family:{FONT};'>
        <div style='color:{YELLOW};font-size:20px;font-weight:700;letter-spacing:4px;'>SP500</div>
        <div style='color:{MUTED};font-size:9px;letter-spacing:5px;margin-top:4px;'>FORECASTING SYSTEM</div>
        <div style='border-bottom:1px solid {BORDER};margin-top:16px;'></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<p style='color:{MUTED};font-size:9px;letter-spacing:3px;margin:14px 0 6px 0;'>START DATE</p>", unsafe_allow_html=True)
    start_date = st.date_input("s", value=datetime(2000,1,1), label_visibility="collapsed")
    st.markdown(f"<p style='color:{MUTED};font-size:9px;letter-spacing:3px;margin:10px 0 6px 0;'>END DATE</p>", unsafe_allow_html=True)
    end_date   = st.date_input("e", value=datetime.today(), label_visibility="collapsed")

    st.markdown(f"<div style='border-bottom:1px solid {BORDER};margin:14px 0;'></div>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED};font-size:9px;letter-spacing:3px;margin-bottom:6px;'>FORECAST HORIZON (BUSINESS DAYS)</p>", unsafe_allow_html=True)
    forecast_days = st.slider("fd", 10, 90, 30, label_visibility="collapsed")
    st.markdown(f"<div style='color:{YELLOW};font-size:11px;letter-spacing:2px;margin-top:4px;'>{forecast_days} DAYS AHEAD</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='border-bottom:1px solid {BORDER};margin:14px 0;'></div>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{MUTED};font-size:9px;letter-spacing:3px;margin-bottom:6px;'>FRED API KEY (OPTIONAL)</p>", unsafe_allow_html=True)
    api_key_input = st.text_input("k", type="password", placeholder="paste key here", label_visibility="collapsed")
    if api_key_input:
        os.environ["FRED_API_KEY"] = api_key_input

    st.markdown(f"<div style='border-bottom:1px solid {BORDER};margin:14px 0;'></div>", unsafe_allow_html=True)
    run_btn = st.button("RUN ANALYSIS", use_container_width=True)

    st.markdown(f"""
    <div style='margin-top:28px;font-family:{FONT};line-height:2;'>
        <div style='color:{BORDER};font-size:9px;letter-spacing:3px;'>MODEL ARCHITECTURE</div>
        <div style='color:{MUTED};font-size:10px;'>Prophet + Random Forest</div>
        <div style='color:{BORDER};font-size:9px;letter-spacing:3px;margin-top:10px;'>DATA SOURCE</div>
        <div style='color:{MUTED};font-size:10px;'>Federal Reserve / FRED</div>
        <div style='color:{BORDER};font-size:9px;letter-spacing:3px;margin-top:10px;'>CACHE TTL</div>
        <div style='color:{MUTED};font-size:10px;'>60 minutes</div>
    </div>""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
st.markdown(f"""
<div style='font-family:{FONT};border-bottom:1px solid {BORDER};padding-bottom:20px;margin-bottom:4px;'>
    <div style='color:{YELLOW};font-size:9px;letter-spacing:6px;text-transform:uppercase;margin-bottom:8px;'>
        FEDERAL RESERVE ECONOMIC DATA &nbsp;/&nbsp; S&P 500 INDEX
    </div>
    <div style='color:{WHITE};font-size:26px;font-weight:700;letter-spacing:3px;'>
        MARKET ANALYSIS &amp; FORECASTING
    </div>
    <div style='color:{MUTED};font-size:10px;letter-spacing:3px;margin-top:8px;'>
        HYBRID MODEL &nbsp;|&nbsp; PROPHET + RANDOM FOREST RESIDUAL CORRECTION &nbsp;|&nbsp; LIVE FRED DATA
    </div>
</div>""", unsafe_allow_html=True)

# =============================================================================
# LOAD DATA
# =============================================================================
if "df_full" not in st.session_state or run_btn:
    with st.spinner("CONNECTING TO FRED..."):
        raw = fetch_live_data(start_date, end_date)
    with st.spinner("COMPUTING TECHNICAL INDICATORS..."):
        df_full = engineer_features(raw)
    with st.spinner("TRAINING HYBRID MODEL — THIS MAY TAKE ~30 SECONDS..."):
        df_model, test_df, future_df, feat_imp, metrics, n_train = run_hybrid_model(df_full, forecast_days)
    st.session_state.update({
        "df_full":df_full,"df_model":df_model,"test_df":test_df,
        "future_df":future_df,"feat_imp":feat_imp,"metrics":metrics,"n_train":n_train
    })
else:
    df_full   = st.session_state["df_full"]
    df_model  = st.session_state["df_model"]
    test_df   = st.session_state["test_df"]
    future_df = st.session_state["future_df"]
    feat_imp  = st.session_state["feat_imp"]
    metrics   = st.session_state["metrics"]
    n_train   = st.session_state["n_train"]

# =============================================================================
# TABS
# =============================================================================
tab1, tab2 = st.tabs(["  MARKET OVERVIEW  ", "  ANALYST VIEW  "])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — GENERAL USER
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    latest = df_full["price"].iloc[-1]
    prev   = df_full["price"].iloc[-2]
    chg    = latest - prev
    pct    = (chg / prev) * 100
    ath    = df_full["price"].max()
    vol    = df_full["volatility_30"].iloc[-1]
    ytd    = df_full[df_full["year"]==datetime.today().year]["daily_return"].sum()
    ma30   = df_full["ma_30"].iloc[-1]
    ma200  = df_full["ma_200"].iloc[-1]

    section_title("LIVE MARKET SNAPSHOT")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("CURRENT PRICE", f"{latest:,.2f}",
            delta=f"{'+'if chg>=0 else ''}{pct:.2f}% TODAY", delta_good=(chg>=0)), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("ALL-TIME HIGH", f"{ath:,.2f}", accent=YELLOW_DIM), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("YTD RETURN", f"{ytd:+.1f}%",
            delta="YEAR TO DATE", delta_good=(ytd>=0)), unsafe_allow_html=True)
    with c4:
        vl = "LOW" if vol<0.8 else ("HIGH" if vol>1.5 else "MEDIUM")
        vc = GREEN if vol<0.8 else (RED if vol>1.5 else YELLOW)
        st.markdown(metric_card("30D VOLATILITY", vl, delta=f"{vol:.2f}% STD DEV",
            delta_good=(vol<0.8), accent=vc), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

    # Market mood bar
    if ma30 > ma200*1.02:
        mood, mc, md = "BULLISH", GREEN, "30-DAY MA IS ABOVE 200-DAY MA — UPWARD TREND CONFIRMED"
    elif ma30 < ma200*0.98:
        mood, mc, md = "BEARISH", RED, "30-DAY MA IS BELOW 200-DAY MA — DOWNWARD PRESSURE"
    else:
        mood, mc, md = "NEUTRAL", YELLOW, "30-DAY MA AND 200-DAY MA ARE CONVERGING — WATCH CLOSELY"

    st.markdown(f"""
    <div style='background:{CARD};border:1px solid {BORDER};border-left:4px solid {mc};
                padding:18px 22px;font-family:{FONT};display:flex;align-items:center;gap:28px;'>
        <div style='color:{mc};font-size:20px;font-weight:700;letter-spacing:5px;min-width:100px;'>{mood}</div>
        <div>
            <div style='color:{MUTED};font-size:9px;letter-spacing:3px;'>MARKET SIGNAL</div>
            <div style='color:{WHITE};font-size:11px;letter-spacing:1px;margin-top:5px;'>{md}</div>
            <div style='color:{MUTED};font-size:10px;margin-top:6px;'>
                MA30: {ma30:,.1f} &nbsp;/&nbsp; MA200: {ma200:,.1f}
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Price trend
    section_title("PRICE TREND")
    period = st.radio("", ["1M","3M","6M","1Y","5Y","ALL"], horizontal=True, index=3)
    dm     = {"1M":30,"3M":90,"6M":180,"1Y":365,"5Y":1825,"ALL":99999}
    dp     = df_full[df_full["date"] >= df_full["date"].max()-timedelta(days=dm[period])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dp["date"], y=dp["price"], mode="lines", name="PRICE",
        line=dict(color=YELLOW, width=1.5),
        fill="tozeroy", fillcolor="rgba(245,197,24,0.04)"))
    fig.add_trace(go.Scatter(x=dp["date"], y=dp["ma_30"], mode="lines", name="MA 30",
        line=dict(color=MUTED, width=1.0, dash="dot")))
    fig = chart_style(fig, 340)
    st.plotly_chart(fig, use_container_width=True)

    # Investment calculator
    section_title("INVESTMENT CALCULATOR")
    ca, cb = st.columns([1,2])
    with ca:
        inv_amt  = st.number_input("AMOUNT (RS)", min_value=1000, max_value=10000000,
                                    value=10000, step=1000)
        inv_year = st.selectbox("INVESTED IN YEAR", list(range(1960,2024))[::-1])
    with cb:
        inv_s = df_full[df_full["year"]==inv_year]
        if not inv_s.empty:
            sp = inv_s["price"].iloc[0]
            cw = inv_amt * (latest/sp)
            gr = (cw/inv_amt-1)*100
            wc = GREEN if cw>inv_amt else RED
            st.markdown(f"""
            <div style='background:{CARD};border:1px solid {BORDER};border-left:4px solid {wc};
                        padding:24px 22px;font-family:{FONT};margin-top:4px;'>
                <div style='color:{MUTED};font-size:9px;letter-spacing:3px;'>CURRENT VALUE</div>
                <div style='color:{wc};font-size:28px;font-weight:700;margin-top:6px;'>RS {cw:,.0f}</div>
                <div style='color:{MUTED};font-size:10px;margin-top:8px;'>
                    RS {inv_amt:,} &nbsp;IN&nbsp; {inv_year} &nbsp;|&nbsp;
                    <span style='color:{wc};font-weight:700;'>{gr:+.1f}% TOTAL RETURN</span>
                </div>
            </div>""", unsafe_allow_html=True)

    # Seasonality
    section_title("AVERAGE RETURN BY MONTH")
    mo = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ma = (df_full.dropna(subset=["daily_return"]).groupby("month_name")["daily_return"]
          .mean().reindex(mo).reset_index())
    ma.columns = ["month","avg_return"]
    fig2 = go.Figure(go.Bar(
        x=ma["month"], y=ma["avg_return"],
        marker_color=[GREEN if v>=0 else RED for v in ma["avg_return"]],
        text=[f"{v:.3f}%" for v in ma["avg_return"]], textposition="outside",
        textfont=dict(family=FONT, size=9, color=MUTED)
    ))
    fig2 = chart_style(fig2, 290)
    fig2.update_layout(showlegend=False)
    fig2.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig2, use_container_width=True)

    # Annual returns
    section_title("ANNUAL RETURNS")
    ar = df_full.dropna(subset=["daily_return"]).groupby("year")["daily_return"].sum().reset_index()
    ar.columns = ["year","ret"]
    fig3 = go.Figure(go.Bar(
        x=ar["year"].astype(str), y=ar["ret"],
        marker_color=[GREEN if v>=0 else RED for v in ar["ret"]],
        text=[f"{v:.0f}%" for v in ar["ret"]], textposition="outside",
        textfont=dict(family=FONT, size=8, color=MUTED)
    ))
    fig3 = chart_style(fig3, 290)
    fig3.update_layout(showlegend=False)
    fig3.update_xaxes(tickangle=45, tickfont=dict(size=8))
    fig3.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ANALYST VIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab2:

    section_title("MODEL PERFORMANCE METRICS")
    m1,m2,m3,m4 = st.columns(4)
    for col,lbl,val,fmt in zip(
        [m1,m2,m3,m4],
        ["RMSE","MAE","MAPE","R2 SCORE"],
        [metrics["RMSE"],metrics["MAE"],metrics["MAPE"],metrics["R2"]],
        ["{:,.2f}","{:,.2f}","{:.2f}%","{:.4f}"]
    ):
        with col:
            st.markdown(metric_card(lbl, fmt.format(val)), unsafe_allow_html=True)

    section_title("FORECAST VS ACTUAL — TEST SET")
    tp = df_model.iloc[:n_train]
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=tp["date"], y=tp["price"], mode="lines",
        name="TRAINING", line=dict(color=BORDER, width=0.7)))
    fig_fc.add_trace(go.Scatter(
        x=test_df["date"].tolist()+test_df["date"].tolist()[::-1],
        y=test_df["prophet_upper"].tolist()+test_df["prophet_lower"].tolist()[::-1],
        fill="toself", fillcolor="rgba(245,197,24,0.05)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
    fig_fc.add_trace(go.Scatter(x=test_df["date"], y=test_df["price"], mode="lines",
        name="ACTUAL", line=dict(color=WHITE, width=1.5)))
    fig_fc.add_trace(go.Scatter(x=test_df["date"], y=test_df["prophet_pred"], mode="lines",
        name="PROPHET ONLY", line=dict(color=MUTED, width=1.0, dash="dot")))
    fig_fc.add_trace(go.Scatter(x=test_df["date"], y=test_df["hybrid_pred"], mode="lines",
        name="HYBRID FORECAST", line=dict(color=YELLOW, width=2.0, dash="dash")))
    fig_fc = chart_style(fig_fc, 400)
    st.plotly_chart(fig_fc, use_container_width=True)

    section_title(f"{forecast_days}-DAY FORWARD FORECAST")
    rec = df_full[df_full["date"] >= df_full["date"].max()-timedelta(days=180)]
    fig_fut = go.Figure()
    fig_fut.add_trace(go.Scatter(
        x=future_df["date"].tolist()+future_df["date"].tolist()[::-1],
        y=future_df["upper_95"].tolist()+future_df["lower_95"].tolist()[::-1],
        fill="toself", fillcolor="rgba(245,197,24,0.06)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
    fig_fut.add_trace(go.Scatter(x=rec["date"], y=rec["price"], mode="lines",
        name="HISTORICAL", line=dict(color=WHITE, width=1.2)))
    fig_fut.add_trace(go.Scatter(x=future_df["date"], y=future_df["forecast"],
        mode="lines+markers", name="FORECAST",
        line=dict(color=YELLOW, width=2.0, dash="dash"),
        marker=dict(size=4, color=YELLOW)))
    vx = df_full["date"].max().strftime("%Y-%m-%d")
    fig_fut.add_shape(type="line", xref="x", yref="paper",
                      x0=vx, x1=vx, y0=0, y1=1,
                      line=dict(color=RED, width=1, dash="dot"))
    fig_fut.add_annotation(x=vx, yref="paper", y=1.05, text="FORECAST START",
                           showarrow=False, font=dict(family=FONT, color=RED, size=9))
    fig_fut = chart_style(fig_fut, 360)
    st.plotly_chart(fig_fut, use_container_width=True)

    cl, cr = st.columns(2)
    with cl:
        section_title("FEATURE IMPORTANCE")
        fig_imp = go.Figure(go.Bar(
            x=feat_imp.values, y=feat_imp.index, orientation="h",
            marker_color=[YELLOW if v>=feat_imp.median() else MUTED for v in feat_imp.values],
            text=[f"{v:.3f}" for v in feat_imp.values], textposition="outside",
            textfont=dict(family=FONT, size=9, color=MUTED)
        ))
        fig_imp = chart_style(fig_imp, 400)
        fig_imp.update_layout(showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    with cr:
        section_title("BOLLINGER BANDS — 6 MONTHS")
        bb = df_full[df_full["date"]>=df_full["date"].max()-timedelta(days=180)].dropna(subset=["bb_upper"])
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(
            x=bb["date"].tolist()+bb["date"].tolist()[::-1],
            y=bb["bb_upper"].tolist()+bb["bb_lower"].tolist()[::-1],
            fill="toself", fillcolor="rgba(245,197,24,0.05)",
            line=dict(color="rgba(0,0,0,0)"), name="BAND"))
        fig_bb.add_trace(go.Scatter(x=bb["date"], y=bb["price"], mode="lines",
            name="PRICE", line=dict(color=WHITE, width=1.2)))
        fig_bb.add_trace(go.Scatter(x=bb["date"], y=bb["bb_upper"], mode="lines",
            name="UPPER", line=dict(color=YELLOW, width=0.8)))
        fig_bb.add_trace(go.Scatter(x=bb["date"], y=bb["bb_lower"], mode="lines",
            name="LOWER", line=dict(color=YELLOW, width=0.8)))
        fig_bb.add_trace(go.Scatter(x=bb["date"], y=bb["bb_mid"], mode="lines",
            name="MID MA", line=dict(color=MUTED, width=1.0, dash="dot")))
        fig_bb = chart_style(fig_bb, 400)
        st.plotly_chart(fig_bb, use_container_width=True)

    section_title("30-DAY ROLLING VOLATILITY")
    vd = df_full.dropna(subset=["volatility_30"])
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=vd["date"], y=vd["volatility_30"], mode="lines",
        name="VOLATILITY", line=dict(color=YELLOW, width=0.8),
        fill="tozeroy", fillcolor="rgba(245,197,24,0.04)"))
    for s,e,lbl in [("2008-09-01","2009-06-01","2008 CRISIS"),
                     ("2020-02-01","2020-06-01","COVID-19"),
                     ("2001-09-01","2002-06-01","DOT-COM")]:
        fig_vol.add_vrect(x0=s, x1=e, fillcolor=RED, opacity=0.07,
                          layer="below", line_width=0,
                          annotation_text=lbl, annotation_position="top left",
                          annotation_font=dict(family=FONT, color=RED, size=9))
    fig_vol = chart_style(fig_vol, 280)
    fig_vol.update_layout(showlegend=False)
    fig_vol.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_vol, use_container_width=True)

    section_title("EXPORT DATA")
    d1,d2,d3 = st.columns(3)
    with d1:
        st.download_button("DOWNLOAD PROCESSED DATA",
            df_full.to_csv(index=False).encode(), "sp500_processed.csv", "text/csv")
    with d2:
        st.download_button("DOWNLOAD FORECAST RESULTS",
            test_df[["date","price","prophet_pred","hybrid_pred"]].to_csv(index=False).encode(),
            "forecast_results.csv", "text/csv")
    with d3:
        st.download_button("DOWNLOAD FUTURE FORECAST",
            future_df.to_csv(index=False).encode(), "future_forecast.csv", "text/csv")