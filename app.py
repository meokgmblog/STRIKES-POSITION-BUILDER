import gzip
import io
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import streamlit.components.v1 as components

# ================================================================
# CONFIGURATION & PAGE SETUP
# ================================================================
st.set_page_config(page_title="F&O Live Position Builder", layout="wide")

IST = ZoneInfo("Asia/Kolkata")
MARKET_START = "09:15"
MARKET_END = "15:30"
INTERVAL = 3

# Hardcoded Access Token (hidden from UI)
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI6M0FZSEUiLCJqdGkiOiI6YThkNTc1Y2Y4MTJmNjA0MzcxZDNlM2MiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc4NzY0NzgzNiwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzg3Njk1MjAwfQ.Z4zP9w3MecFeZEcX5sUt4YdhxS6skp25fbKOv8-_gPU"

@st.cache_data(ttl=3600)
# List of standard NSE F&O Indices
MAJOR_INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"]

@st.cache_data(ttl=3600)
def load_fno_symbols():
    """Reads symbol names from the FNO excel list file and adds major indices."""
    symbols = []
    try:
        fno_df = pd.read_excel("FNO all list.xlsx")
        excel_symbols = fno_df["SYMBOL"].dropna().astype(str).str.strip().str.upper().unique().tolist()
        symbols.extend(excel_symbols)
    except Exception:
        pass

    # Combine indices and stock symbols while removing duplicates
    all_symbols = sorted(list(set(MAJOR_INDICES + symbols)))
    return all_symbols

fno_symbol_list = load_fno_symbols()

# Sidebar Controls
st.sidebar.title("⚙️ Controls & Parameters")

# Searchable Dropdown for F&O Symbols (Includes Indices + All Stock Symbols from Excel)
default_index = fno_symbol_list.index("LAURUSLABS") if "LAURUSLABS" in fno_symbol_list else 0
SYMBOL_INPUT = st.sidebar.selectbox(
    "F&O Symbol",
    options=fno_symbol_list,
    index=default_index
).strip().upper()

NUM_STRIKES_BOUND = st.sidebar.slider("Strikes Range (± ATM)", min_value=2, max_value=12, value=2)

st.title(f"📈 {SYMBOL_INPUT} - Live 3-Minute Position Builder")

# ================================================================
# API HELPERS & MASTER FETCHERS
# ================================================================
def get_headers(token):
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {token.strip()}",
        "Cache-Control": "no-cache",
    }

def upstox_get(url, token, params=None):
    try:
        response = requests.get(url, headers=get_headers(token), params=params, timeout=10)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network Error: {str(e)}")

    if response.status_code != 200:
        raise RuntimeError(f"Upstox HTTP {response.status_code}: {response.text[:200]}")

    data = response.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Upstox API Error: {data}")

    return data

@st.cache_data(ttl=3600)
def fetch_upstox_master_instruments():
    """Downloads and caches the NSE instrument master file."""
    url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"
    try:
        res = requests.get(url, timeout=20)
        if res.status_code != 200:
            raise Exception(f"HTTP {res.status_code} while fetching master csv.")

        with gzip.open(io.BytesIO(res.content), "rt") as f:
            df = pd.read_csv(f)

        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        raise RuntimeError(f"Master file download error: {str(e)}")

def resolve_stock_instruments(master_df, symbol):
    """
    Dynamically resolves Spot key, Futures key, and Option chains for stocks and indices.
    Handles Upstox naming conventions (e.g., LAURUSLABS-EQ, NSE_EQ segment).
    """
    key_col = "instrument_key" if "instrument_key" in master_df.columns else "instrument_token"
    sym_col = "trading_symbol" if "trading_symbol" in master_df.columns else "tradingsymbol"
    type_col = "instrument_type" if "instrument_type" in master_df.columns else "segment"
    name_col = "name" if "name" in master_df.columns else ("asset_symbol" if "asset_symbol" in master_df.columns else sym_col)
    strike_col = "strike" if "strike" in master_df.columns else "strike_price"

    clean_symbol = symbol.strip().upper()

    # 1. Spot Key Resolution (Handles Stocks like LAURUSLABS-EQ and Indices like NIFTY 50)
    spot_mask = (
        (master_df[sym_col].astype(str).str.upper() == clean_symbol) |
        (master_df[sym_col].astype(str).str.upper() == f"{clean_symbol}-EQ") |
        (master_df[name_col].astype(str).str.upper() == clean_symbol)
    ) & (
        master_df[type_col].astype(str).str.upper().str.contains("EQ|EQUITY|INDEX|NSE_EQ", regex=True)
    )

    spot_rows = master_df[spot_mask]

    if spot_rows.empty:
        # Broader fallback search across trading symbol prefix
        spot_rows = master_df[
            master_df[sym_col].astype(str).str.upper().str.startswith(clean_symbol) &
            master_df[type_col].astype(str).str.upper().str.contains("EQ|EQUITY|INDEX|NSE_EQ", regex=True)
        ]

    if spot_rows.empty:
        raise RuntimeError(f"Could not find Equity Spot instrument for '{clean_symbol}'. Check symbol spelling or master mapping.")

    spot_key = spot_rows.iloc[0][key_col]

    # 2. Options Resolution
    opts_mask = (
        (master_df[name_col].astype(str).str.upper() == clean_symbol) |
        (master_df[sym_col].astype(str).str.upper().str.startswith(clean_symbol))
    ) & master_df[type_col].astype(str).str.upper().str.contains("OPTSTK|OPTIDX|CE|PE", regex=True)

    opts = master_df[opts_mask].copy()
    if opts.empty:
        raise RuntimeError(f"No active options contracts found for {clean_symbol}.")

    opts["expiry_dt"] = pd.to_datetime(opts["expiry"], errors="coerce")
    opts = opts.dropna(subset=["expiry_dt"])
    today = pd.Timestamp(datetime.now().date())

    active_opts = opts[opts["expiry_dt"].dt.date >= today.date()].sort_values("expiry_dt")
    if active_opts.empty:
        raise RuntimeError(f"No upcoming unexpired options contracts found for {clean_symbol}.")

    nearest_expiry = active_opts.iloc[0]["expiry_dt"]
    matching_opts = active_opts[active_opts["expiry_dt"] == nearest_expiry].copy()

    return spot_key, matching_opts, key_col, sym_col, strike_col

def get_intraday_candles(token, instrument_key):
    """Fetches intraday 3-minute candles for any instrument key."""
    if not instrument_key:
        return pd.DataFrame()

    encoded_key = quote(str(instrument_key), safe="")
    url = f"https://api.upstox.com/v3/historical-candle/intraday/{encoded_key}/minutes/{INTERVAL}"

    try:
        res = upstox_get(url, token)
        candles = res.get("data", {}).get("candles", [])
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(IST).dt.tz_localize(None)
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def filter_market_hours(df):
    if df.empty:
        return df
    df = df.copy()
    df["time"] = df["timestamp"].dt.time
    start = datetime.strptime(MARKET_START, "%H:%M").time()
    end = datetime.strptime(MARKET_END, "%H:%M").time()
    df = df[(df["time"] >= start) & (df["time"] <= end)].copy()
    return df.drop(columns=["time"]).reset_index(drop=True)

def fetch_option_data_parallel(token, option_rows, key_col):
    keys = [row[key_col] for _, row in option_rows.iterrows()]

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(
            executor.map(
                lambda key: filter_market_hours(get_intraday_candles(token, key)),
                keys,
            )
        )

    combined_df = None
    for opt_data in results:
        if not opt_data.empty:
            opt_sub = opt_data[["timestamp", "oi"]].copy()
            if combined_df is None:
                combined_df = opt_sub.rename(columns={"oi": "sum_oi"})
            else:
                combined_df = pd.merge(combined_df, opt_sub, on="timestamp", how="outer")
                combined_df["sum_oi"] = combined_df["sum_oi"].fillna(0) + combined_df["oi"].fillna(0)
                combined_df.drop(columns=["oi"], inplace=True)

    return combined_df

# ================================================================
# CALCULATIONS & POSITION BUILDER
# ================================================================
def calculate_position_builder(price_df, ce_df, pe_df):
    clean_price = price_df[["timestamp", "open", "high", "low", "close"]].copy()

    opts_merged = pd.merge(ce_df, pe_df, on="timestamp", how="inner").sort_values("timestamp")
    df = pd.merge(clean_price, opts_merged, on="timestamp", how="inner").sort_values("timestamp")

    if df.empty:
        raise RuntimeError("Timestamp alignment mismatch across spot and option market feeds.")

    df["ce_oi_diff"] = df["ce_oi"].diff(1).fillna(0)
    df["pe_oi_diff"] = df["pe_oi"].diff(1).fillna(0)

    df["net_oi_change"] = df["pe_oi_diff"] - df["ce_oi_diff"]

    max_val = max(abs(df["net_oi_change"].min()), abs(df["net_oi_change"].max()), 1)
    df["position_builder_scaled"] = (df["net_oi_change"] / max_val) * 100

    return df

# ================================================================
# PLOTLY CHART RENDERER
# ================================================================
def render_chart(df, symbol, expiry_str):
    last_price = df["close"].iloc[-1]
    last_time = df["timestamp"].iloc[-1].strftime("%H:%M:%S")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.68, 0.32],
        subplot_titles=(
            f"{symbol} Spot | 3m | Last: {last_price:.2f} | Updated: {last_time} IST",
            f"POSITION BUILDER HISTOGRAM ({expiry_str})",
        ),
    )

    # 1. Candlestick Trace
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol,
            increasing_fillcolor="#089981",
            increasing_line_color="#089981",
            decreasing_fillcolor="#f23645",
            decreasing_line_color="#f23645",
            whiskerwidth=0.4,
            hoverinfo="x+name",
        ),
        row=1,
        col=1,
    )

    # 2. Position Builder Histogram
    values = df["position_builder_scaled"].fillna(0)
    colors = ["#089981" if v >= 0 else "#f23645" for v in values]

    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=values,
            name="Net OI Scaled",
            marker_color=colors,
            marker_line_width=0,
            hovertemplate="OI Scaled: %{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Crosshair & Layout setup (Height scaled down to 420px)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        height=420,
        margin=dict(l=15, r=15, t=35, b=15),
        showlegend=False,
        hovermode="x unified",
        dragmode="pan",
        xaxis_rangeslider_visible=False,
    )

    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#89929e",
        spikethickness=1,
        spikedash="dash",
        gridcolor="#2a2e39",
        rangebreaks=[dict(bounds=["sat", "mon"])],
    )

    fig.update_yaxes(gridcolor="#2a2e39", zerolinecolor="#363a45", row=1, col=1)
    fig.update_yaxes(
        range=[-110, 110],
        gridcolor="#2a2e39",
        zerolinecolor="#363a45",
        row=2,
        col=1,
    )

    config = {
        "scrollZoom": True,
        "displayModeBar": True,
        "modeBarButtonsToAdd": ["pan2d"],
        "displaylogo": False,
    }

    st.plotly_chart(fig, use_container_width=True, config=config)

# ================================================================
# MAIN EXECUTION ENGINE
# ================================================================
try:
    with st.spinner("Downloading market metadata..."):
        master_df = fetch_upstox_master_instruments()

    # Step 1: Dynamically resolve spot instrument & nearest options chain
    spot_key, opts_df, key_col, sym_col, strike_col = resolve_stock_instruments(master_df, SYMBOL_INPUT)

    # Step 2: Get intraday spot candles
    spot_df = filter_market_hours(get_intraday_candles(ACCESS_TOKEN, spot_key))
    if spot_df.empty:
        st.error(f"No intraday candle data returned for {SYMBOL_INPUT} spot. Market may be closed or token expired.")
        st.stop()

    last_close = spot_df["close"].iloc[-1]

    # Step 3: Dynamic Strike & Step Detection
    opts_df["strike_num"] = pd.to_numeric(opts_df[strike_col], errors="coerce")
    unique_strikes = sorted(opts_df["strike_num"].dropna().unique())

    if len(unique_strikes) > 1:
        # Detect standard strike intervals dynamically
        strike_diffs = np.diff(unique_strikes)
        step_size = float(np.median(strike_diffs))
    else:
        step_size = 5.0

    # Auto-calculate ATM Strike
    atm_strike = round(last_close / step_size) * step_size
    min_stk = atm_strike - (NUM_STRIKES_BOUND * step_size)
    max_stk = atm_strike + (NUM_STRIKES_BOUND * step_size)

    # Filter ATM neighborhood options
    atm_opts = opts_df[(opts_df["strike_num"] >= min_stk) & (opts_df["strike_num"] <= max_stk)].copy()
    if atm_opts.empty:
        atm_opts = opts_df

    ce_opts = atm_opts[atm_opts[sym_col].astype(str).str.endswith("CE")]
    pe_opts = atm_opts[atm_opts[sym_col].astype(str).str.endswith("PE")]

    # Step 4: Fetch Call/Put Open Interest Data concurrently
    with st.spinner(f"Scouting {len(ce_opts) + len(pe_opts)} contracts around ATM ({atm_strike})..."):
        ce_df = fetch_option_data_parallel(ACCESS_TOKEN, ce_opts, key_col)
        pe_df = fetch_option_data_parallel(ACCESS_TOKEN, pe_opts, key_col)

    if ce_df is not None and pe_df is not None:
        ce_df = ce_df.rename(columns={"sum_oi": "ce_oi"}).sort_values("timestamp").ffill().dropna()
        pe_df = pe_df.rename(columns={"sum_oi": "pe_oi"}).sort_values("timestamp").ffill().dropna()

        # Build position histogram
        builder_df = calculate_position_builder(spot_df, ce_df, pe_df)
        exp_date_str = opts_df.iloc[0]["expiry_dt"].strftime("%b-%d")
        
        # Render dynamic chart
        render_chart(builder_df, SYMBOL_INPUT, f"Expiry: {exp_date_str}")
    else:
        st.error("Failed to fetch concurrent open interest data for strikes.")

except Exception as err:
    st.error(f"Execution Error: {str(err)}")

# ================================================================
# AUTO-REFRESH TRIGGER (SYNCED TO 3-MINUTE CANDLE BOUNDARIES)
# ================================================================
now = datetime.now()
seconds_past_3m = (now.minute % 3) * 60 + now.second
ms_until_candle_close = max((180 - seconds_past_3m + 2) * 1000, 3000)

components.html(
    f"""
    <script>
        setTimeout(function() {{
            window.parent.postMessage({{type: 'streamlit:render'}}, '*');
        }}, {ms_until_candle_close});
    </script>
    """,
    height=0,
)
