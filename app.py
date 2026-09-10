import gzip
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # <--- Added missing import
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

ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI6M0FZSEUiLCJqdGkiOiI6YThkNTc1Y2Y4MTJmZmQ0MzcxZDNlM2MiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc4NzY0NzgzNiwiaXNzIjoidWRhapI1ZXJ2aWNlIiwiZXhwIjoxNzg3Njk1MjAwfQ.Z4zP9w3MecFeZEcX5sUt4YdhxS6skp25fbKOv8-_gPU"

MAJOR_INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"]

@st.cache_data(ttl=3600)
def load_fno_symbols():
    symbols = []
    github_urls = [
        "https://raw.githubusercontent.com/meokgmblog/STRIKES-POSITION-BUILDER/main/FNO%20ALL%20LIST.txt",
        "https://raw.githubusercontent.com/meokgmblog/STRIKES-POSITION-BUILDER/main/FNO_ALL_LIST.txt",
        "https://raw.githubusercontent.com/meokgmblog/STRIKES-POSITION-BUILDER/main/FNO%20all%20list.txt"
    ]
    
    for url in github_urls:
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200 and res.text.strip():
                lines = res.text.splitlines()
                symbols = [line.strip().upper() for line in lines if line.strip()]
                if len(symbols) > 5:
                    break
        except Exception:
            continue

    if not symbols:
        possible_filenames = ["FNO ALL LIST.txt", "FNO_ALL_LIST.txt", "FNO all list.txt", "fno_all_list.txt"]
        for fname in possible_filenames:
            if os.path.exists(fname):
                try:
                    with open(fname, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        symbols = [line.strip().upper() for line in lines if line.strip()]
                        if len(symbols) > 5:
                            break
                except Exception:
                    continue

    return sorted(list(set(MAJOR_INDICES + symbols)))

fno_symbol_list = load_fno_symbols()

# Sidebar Controls
st.sidebar.title("⚙️ Controls & Parameters")

default_index = fno_symbol_list.index("NIFTY") if "NIFTY" in fno_symbol_list else 0
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
    key_col = "instrument_key" if "instrument_key" in master_df.columns else "instrument_token"
    sym_col = "trading_symbol" if "trading_symbol" in master_df.columns else "tradingsymbol"
    type_col = "instrument_type" if "instrument_type" in master_df.columns else "segment"
    name_col = "name" if "name" in master_df.columns else ("asset_symbol" if "asset_symbol" in master_df.columns else sym_col)
    strike_col = "strike" if "strike" in master_df.columns else "strike_price"

    clean_symbol = symbol.strip().upper()

    spot_mask = (
        (master_df[sym_col].astype(str).str.upper() == clean_symbol) |
        (master_df[sym_col].astype(str).str.upper() == f"{clean_symbol}-EQ") |
        (master_df[name_col].astype(str).str.upper() == clean_symbol)
    ) & (
        master_df[type_col].astype(str).str.upper().str.contains("EQ|EQUITY|INDEX|NSE_EQ", regex=True)
    )

    spot_rows = master_df[spot_mask]

    if spot_rows.empty:
        spot_rows = master_df[
            master_df[sym_col].astype(str).str.upper().str.startswith(clean_symbol) &
            master_df[type_col].astype(str).str.upper().str.contains("EQ|EQUITY|INDEX|NSE_EQ", regex=True)
        ]

    if spot_rows.empty:
        raise RuntimeError(f"Could not find Equity Spot instrument for '{clean_symbol}'.")

    spot_key = spot_rows.iloc[0][key_col]

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
# STACKED SUBPLOTS CHART RENDERER
# ================================================================
def render_chart(df, symbol, expiry_str):
    last_price = df["close"].iloc[-1]
    last_time = df["timestamp"].iloc[-1].strftime("%H:%M:%S")

    fig = go.Figure()

    # 1. Position Builder Histogram Trace (Assigned to Y2 Axis - Bottom Floor)
    values = df["position_builder_scaled"].fillna(0)
    colors = ["#089981" if v >= 0 else "#f23645" for v in values]
    formatted_times = df["timestamp"].dt.strftime("%B %d, %Y at %I:%M %p")

    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=values,
            customdata=formatted_times,
            name="Net OI Scaled",
            marker_color=colors,
            marker_line_width=0,
            opacity=0.8,
            yaxis="y2",
            hovertemplate="%{customdata}<extra></extra>",
        )
    )

    # 2. Candlestick Price Trace (Assigned to Y1 Axis - Top Section)
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
            yaxis="y1",
            hoverinfo="none",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{symbol} Spot</b> (3m) | Last: {last_price:.2f} | Updated: {last_time} IST | {expiry_str}",
            font=dict(size=14, color="#d1d4dc"),
            x=0.01,
            y=0.98,
        ),
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        height=620,
        margin=dict(l=20, r=20, t=45, b=20),
        showlegend=False,
        hovermode="x",
        dragmode="pan",
        # Shared X-Axis with Crosshair Line
        xaxis=dict(
            type="date",
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="#ffffff",
            spikethickness=1,
            spikedash="dash",
            gridcolor="#2a2e39",
            rangebreaks=[dict(bounds=["sat", "mon"])],
            rangeslider=dict(visible=False),
        ),
        # Primary Price Y-Axis (Top 75% Domain - Auto-scaling Enabled)
        yaxis=dict(
            title="Price",
            domain=[0.25, 1.0],  # Keeps candles in top 75%
            autorange=True,      # Enables smooth dynamic scaling on zoom
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="#ffffff",
            spikethickness=1,
            spikedash="dash",
            gridcolor="#2a2e39",
            side="right",
        ),
        # Secondary Histogram Y-Axis (Bottom 22% Domain)
        yaxis2=dict(
            title="",
            domain=[0.0, 0.22],  # Locks histogram to bottom 22%
            side="right",
            showgrid=False,
            showticklabels=False,
            zeroline=True,
            zerolinecolor="#363a45",
            zerolinewidth=1,
        ),
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

    spot_key, opts_df, key_col, sym_col, strike_col = resolve_stock_instruments(master_df, SYMBOL_INPUT)

    spot_df = filter_market_hours(get_intraday_candles(ACCESS_TOKEN, spot_key))
    if spot_df.empty:
        st.error(f"No intraday candle data returned for {SYMBOL_INPUT} spot.")
        st.stop()

    last_close = spot_df["close"].iloc[-1]

    opts_df["strike_num"] = pd.to_numeric(opts_df[strike_col], errors="coerce")
    unique_strikes = sorted(opts_df["strike_num"].dropna().unique())

    if len(unique_strikes) > 1:
        strike_diffs = np.diff(unique_strikes)
        step_size = float(np.median(strike_diffs))
    else:
        step_size = 5.0

    atm_strike = round(last_close / step_size) * step_size
    min_stk = atm_strike - (NUM_STRIKES_BOUND * step_size)
    max_stk = atm_strike + (NUM_STRIKES_BOUND * step_size)

    atm_opts = opts_df[(opts_df["strike_num"] >= min_stk) & (opts_df["strike_num"] <= max_stk)].copy()
    if atm_opts.empty:
        atm_opts = opts_df

    ce_opts = atm_opts[atm_opts[sym_col].astype(str).str.endswith("CE")]
    pe_opts = atm_opts[atm_opts[sym_col].astype(str).str.endswith("PE")]

    with st.spinner(f"Scouting {len(ce_opts) + len(pe_opts)} contracts around ATM ({atm_strike})..."):
        ce_df = fetch_option_data_parallel(ACCESS_TOKEN, ce_opts, key_col)
        pe_df = fetch_option_data_parallel(ACCESS_TOKEN, pe_opts, key_col)

    if ce_df is not None and pe_df is not None:
        ce_df = ce_df.rename(columns={"sum_oi": "ce_oi"}).sort_values("timestamp").ffill().dropna()
        pe_df = pe_df.rename(columns={"sum_oi": "pe_oi"}).sort_values("timestamp").ffill().dropna()

        builder_df = calculate_position_builder(spot_df, ce_df, pe_df)
        exp_date_str = opts_df.iloc[0]["expiry_dt"].strftime("%b-%d")
        
        render_chart(builder_df, SYMBOL_INPUT, f"Expiry: {exp_date_str}")
    else:
        st.error("Failed to fetch concurrent open interest data for strikes.")

except Exception as err:
    st.error(f"Execution Error: {str(err)}")

# ================================================================
# AUTO-REFRESH TRIGGER
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
