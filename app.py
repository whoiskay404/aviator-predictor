from __future__ import annotations

import os
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

import datastore as ds
import stats as ss

st.set_page_config(page_title="Aviator Predictor", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, .stButton, button, input,
select, textarea {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto,
                 sans-serif !important;
}

/* - hide streamlit chrome - */
#MainMenu, [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
[data-testid="stToolbarActionButton"], [data-testid^="stStatusWidget"],
footer {
    display: none !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* - one centered column - */
[data-testid="stMainBlockContainer"] {
    max-width: 720px !important;
    margin: 0 auto !important;
    padding-top: 40px !important;
    padding-bottom: 56px !important;
}

/* - typography - */
h1 {
    font-size: 22px !important;
    font-weight: 750 !important;
    color: #f4f6fa !important;
    letter-spacing: -0.01em;
    margin: 0 !important;
}
h2, h3 {
    font-size: 15px !important;
    font-weight: 650 !important;
    color: #dfe3ea !important;
    margin: 26px 0 6px !important;
}
[data-testid="stCaptionContainer"], [data-testid="stCaption"] {
    font-size: 12.5px !important;
    color: #7d8698 !important;
}
.stMarkdown p, .stMarkdown li {
    font-size: 13.5px !important;
    line-height: 1.6;
    color: #a7aebf;
}
.stMarkdown strong { color: #eef1f6; }
.chip-wrap { text-align: right; }
.chip {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 999px;
    background: #12151c;
    border: 1px solid #232833;
    color: #8b93a7;
    font-size: 12px;
    font-weight: 600;
    white-space: nowrap;
}
.app-note { margin: 0; font-size: 13px; color: #7d8698; }

/* - tabs: emerald accent - */
[data-testid="stTabs"] { border-bottom-color: #1d222c !important; }
[data-testid="stTabs"] button {
    color: #7b8496 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #10b981 !important;
    border-bottom-color: #10b981 !important;
}

/* - buttons: quiet neutral - */
[data-testid="stButton"] button {
    background: #161a22 !important;
    color: #c6ccd9 !important;
    border: 1px solid #2a303c !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    transition: all 0.15s ease !important;
}
[data-testid="stButton"] button:hover {
    background: #1c212b !important;
    border-color: #3d4557 !important;
    color: #eef1f6 !important;
}

/* predict circle */
.st-key-predict_btn,
.st-key-predict_btn [data-testid="stButton"] {
    display: flex !important;
    justify-content: center;
}
.st-key-predict_btn button, button.st-key-predict_btn {
    width: 240px !important;
    height: 240px !important;
    min-height: 240px !important;
    border-radius: 50% !important;
    font-size: 22px !important;
    font-weight: 850 !important;
    letter-spacing: 0.08em;
    color: #f0fdf4 !important;
    background: radial-gradient(circle at 32% 26%, #10b981 0%, #059669 45%,
                                #047857 100%) !important;
    border: 5px solid rgba(255, 255, 255, 0.22) !important;
    box-shadow: 0 14px 36px rgba(16, 185, 129, 0.45),
                inset 0 -8px 18px rgba(0, 0, 0, 0.35) !important;
}
.st-key-predict_btn button:hover, button.st-key-predict_btn:hover {
    transform: scale(1.04);
    box-shadow: 0 18px 48px rgba(16, 185, 129, 0.6),
                inset 0 -8px 18px rgba(0, 0, 0, 0.35) !important;
}
.st-key-predict_btn button:active, button.st-key-predict_btn:active {
    transform: scale(0.96);
}

/* reset ghost pill */
.st-key-reset_pred,
.st-key-reset_pred [data-testid="stButton"] {
    display: flex !important;
    justify-content: center;
}
.st-key-reset_pred button, button.st-key-reset_pred {
    background: transparent !important;
    border: 1px solid #2a303c !important;
    color: #98a0b2 !important;
    border-radius: 999px !important;
    padding: 6px 20px !important;
    height: auto !important;
    min-height: 0 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
.st-key-reset_pred button:hover, button.st-key-reset_pred:hover {
    background: #14171f !important;
    border-color: #3d4557 !important;
    color: #eef1f6 !important;
}

/* danger zone: red-outlined destructive buttons */
.st-key-clear_all button, .st-key-yes_clear button,
.st-key-restore_seed button, .st-key-yes_restore button,
button.st-key-clear_all, button.st-key-yes_clear,
button.st-key-restore_seed, button.st-key-yes_restore {
    background: #1a1417 !important;
    border-color: #7f1d1d !important;
    color: #f87171 !important;
}
.st-key-clear_all button:hover, .st-key-yes_clear button:hover,
.st-key-restore_seed button:hover, .st-key-yes_restore button:hover,
button.st-key-clear_all:hover, button.st-key-yes_clear:hover,
button.st-key-restore_seed:hover, button.st-key-yes_restore:hover {
    background: #24181b !important;
    border-color: #b91c1c !important;
    color: #fecaca !important;
}

/* - hero result - */
.hero-result { text-align: center; margin: 0 auto; }
#odd-circle {
    width: 240px;
    height: 240px;
    margin: 0 auto;
    border-radius: 50%;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    animation: odd-pop 0.35s ease;
}
@keyframes odd-pop {
    from { transform: scale(0.7); opacity: 0; }
    to   { transform: scale(1); opacity: 1; }
}
#odd-circle.golden {
    background: radial-gradient(circle at 32% 26%, #065f46 0%, #064e3b 70%);
    border: 5px solid #34d399;
    box-shadow: 0 0 44px rgba(52, 211, 153, 0.45),
                inset 0 -8px 18px rgba(0, 0, 0, 0.45);
    color: #ecfdf5;
}
#odd-circle.outside {
    background: radial-gradient(circle at 32% 26%, #1c1917 0%, #17130f 70%);
    border: 5px solid #fbbf24;
    box-shadow: 0 0 40px rgba(251, 191, 36, 0.3),
                inset 0 -8px 18px rgba(0, 0, 0, 0.45);
    color: #fef3c7;
}
#odd-circle.idle {
    background: #171b24;
    border: 5px dashed #3a4254;
    box-shadow: none;
    color: #8b93a7;
}
.odd-value {
    font-size: 56px;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -0.02em;
}
.odd-status {
    font-size: 12.5px;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 12px;
    color: inherit;
    opacity: 0.9;
}
.hero-meta {
    margin-top: 14px;
    font-size: 13px;
    color: #8b93a7;
}

/* - round selector + minute strip - */
.st-key-predict_minute {
    max-width: 320px !important;
    margin: 16px auto 0 !important;
}
.st-key-predict_minute label {
    text-align: center !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
    color: #7b8496 !important;
}
.minute-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 12px;
}
.strip-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 13px;
    border-radius: 999px;
    background: #11141b;
    border: 1px solid #222733;
    font-size: 12.5px;
    color: #8b93a7;
}
.strip-pill b {
    color: #e6e9f0;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}
.strip-pill .strip-odd { color: #c6ccd9; font-weight: 600; }
.strip-pill .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #4b5563;
}
.strip-pill.golden .dot {
    background: #10b981;
    box-shadow: 0 0 6px rgba(16, 185, 129, 0.8);
}
.strip-pill.next { background: #171b24; border-color: #333b4a; }

/* - alerts: quiet cards - */
[data-testid="stAlert"] {
    background: #10131a !important;
    border: 1px solid #222733 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    padding: 13px 16px !important;
}
[data-testid="stAlert"] strong, [data-testid="stAlert"] b {
    color: #eef1f6 !important;
}

/* - metrics + progress - */
[data-testid="stMetricValue"] {
    color: #eef1f6 !important;
    font-size: 19px !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #7b8496 !important;
    font-size: 11.5px !important;
    font-weight: 600 !important;
}
[data-testid="stProgressBarTrack"] {
    background: #1a1e27 !important;
    border-radius: 999px !important;
}
[data-testid="stProgressBarTrack"] > div { background: #556074 !important; }

/* - expanders: thin cards - */
[data-testid="stExpander"] {
    background: #10131a;
    border: 1px solid #1d222c;
    border-radius: 14px;
    margin-top: 8px;
}
[data-testid="stExpander"]:hover { border-color: #262c38; }
[data-testid="stExpander"] details {
    background: transparent !important;
    border: none !important;
}
[data-testid="stExpander"] summary {
    padding: 13px 16px !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
    color: #a7aebf !important;
}
[data-testid="stExpander"] summary:hover { color: #e6e9f0 !important; }
[data-testid="stExpanderDetails"] {
    padding: 0 16px 14px !important;
    border-top: none !important;
}

/* - inputs - */
[data-baseweb="base-input"] {
    background: #12151c !important;
    border-color: #2a303c !important;
    border-radius: 10px !important;
}
[data-baseweb="select"] > div {
    background: #12151c !important;
    border-color: #2a303c !important;
    border-radius: 10px !important;
}

/* - sidebar - */
[data-testid="stSidebar"] {
    background: #0d1016 !important;
    border-right: 1px solid #171b24 !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
    color: #697080 !important;
    font-size: 12px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: transparent !important;
    border: none !important;
    margin-top: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    padding: 6px 0 !important;
    color: #697080 !important;
    font-size: 12.5px !important;
}
[data-testid="stSidebar"] pre, [data-testid="stSidebar"] code {
    font-size: 11px !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
    color: #7d8698 !important;
}

/* - developer credit - */
.dev-credit {
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid #171b24;
}
.dev-head {
    font-size: 12.5px;
    color: #697080;
    margin-bottom: 8px;
}
.dev-head b {
    color: #c7cdda;
}
.dev-links {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 8px;
}
.dev-links a {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    border: 1px solid #1d222d;
    border-radius: 8px;
    color: #8b93a7;
    background: #12151c;
    transition: color .15s, border-color .15s;
}
.dev-links a:hover {
    color: #e6e9f0;
    border-color: #3a4254;
}
.dev-contacts {
    font-size: 11.5px;
    color: #697080;
    word-break: break-all;
}

/* - mobile - */
@media (max-width: 760px) {
    [data-testid="stMainBlockContainer"] {
        padding-left: 16px !important;
        padding-right: 16px !important;
    }
    .st-key-predict_btn button, button.st-key-predict_btn, #odd-circle {
        width: 200px !important;
        height: 200px !important;
        min-height: 200px !important;
    }
    .odd-value { font-size: 46px !important; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

NO_REFRESH = bool(os.environ.get("AVIATOR_NO_AUTOREFRESH"))
DATA_BACKED_N = ss.MIN_N_FOR_VERDICT


def get_df() -> pd.DataFrame:
    if "df" not in st.session_state:
        st.session_state["df"] = ds.load_rounds()
        st.session_state["df_version"] = 0
    return st.session_state["df"]


def set_df(df: pd.DataFrame, flash: str | None = None) -> None:
    st.session_state["df"] = ds.sort_rounds(ds.normalize_df(df))
    ds.save_rounds(st.session_state["df"])
    st.session_state["df_version"] = int(st.session_state.get("df_version", 0)) + 1
    if flash:
        st.session_state["flash"] = flash


def get_enrichment() -> pd.DataFrame:
    if "enrichment" not in st.session_state:
        frame, errors = ds.load_enrichment()
        st.session_state["enrichment"] = frame
        st.session_state["enrichment_errors"] = errors
    return st.session_state["enrichment"]


ODD_MIN = ss.ODD_MIN
ODD_SOFT_CAP = ss.ODD_SOFT_CAP
ODD_MAX = ss.ODD_MAX


def fnum(value, digits: int = 2, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if number != number:
        return "-"
    return f"{number:.{digits}f}{suffix}"


def pct(value, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if number != number:
        return "-"
    return f"{number * 100:.{digits}f}%"


def fmt_p(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if number != number:
        return "-"
    if number < 0.0001:
        return "<0.0001"
    return f"{number:.4f}"


def group_stats(values) -> dict:
    vals = [float(v) for v in values]
    n = len(vals)
    if not n:
        nan = float("nan")
        return {"n": 0, "low": nan, "band": nan, "median": nan, "mean": nan}
    return {
        "n": n,
        "low": sum(1 for v in vals if v < 1.10) / n,
        "band": sum(1 for v in vals if 1.10 <= v <= 2.50) / n,
        "median": ss.percentile(vals, 50),
        "mean": sum(vals) / n,
    }


def timed_frame(df: pd.DataFrame) -> pd.DataFrame:
    timed = df[~df["timestamp"].map(ds.is_untimed).astype(bool)].copy()
    if not len(timed):
        return timed
    timed["minute"] = timed["timestamp"].map(ds.minute_of)
    timed["hour"] = timed["timestamp"].map(ds.minute_of)
    timed["hour"] = timed["timestamp"].map(ds.hour_of)
    timed = timed[timed["minute"].notna() & timed["hour"].notna()].copy()
    timed["minute"] = timed["minute"].astype(int)
    timed["hour"] = timed["hour"].astype(int)
    return timed


def in_window(minute: int, windows) -> bool:
    for start, end in windows:
        lo, hi = min(start, end), max(start, end)
        if lo <= minute <= hi:
            return True
    return False


def windows_text(windows) -> str:
    if not windows:
        return "(none)"
    return ", ".join(str(s) if s == e else f"{s}-{e}" for s, e in windows)


def ensure_windows() -> list:
    if "windows" not in st.session_state:
        st.session_state["windows"] = [(int(s), int(e)) for s, e in ss.DEFAULT_WINDOWS]
    return st.session_state["windows"]


def predict_context() -> dict:
    """Everything the predictor needs, derived from the current data."""
    df = get_df()
    windows = ensure_windows()
    timed = timed_frame(df)
    window_values: list = []
    other_values: list = []
    minute_map: dict = {}
    favored_rows: list = []
    if len(timed):
        timed = timed.copy()
        timed["_in"] = timed["minute"].map(lambda m: in_window(m, windows)).astype(bool)
        window_values = [float(v) for v in timed[timed["_in"]]["multiplier"]]
        other_values = [float(v) for v in timed[~timed["_in"]]["multiplier"]]
        minute_map = {
            int(m): [float(v) for v in group["multiplier"]]
            for m, group in timed.groupby("minute")
        }
        favored_rows = ss.favored_minutes(minute_map)
    n_win = len(window_values)
    if n_win >= DATA_BACKED_N:
        mode = "DATA-BACKED"
    elif n_win >= ss.MIN_N_FOR_P:
        mode = "PROVISIONAL"
    else:
        mode = "THEORY"

    enrich = get_enrichment()
    enriched_values: list = []
    enriched_minute_map: dict = {}
    enrich_timed = 0
    if len(enrich):
        enriched_values = [float(v) for v in enrich["multiplier"]]
        et = enrich[~enrich["timestamp"].map(ds.is_untimed).astype(bool)].copy()
        et["minute"] = et["timestamp"].map(ds.minute_of)
        et = et[et["minute"].notna()]
        enrich_timed = len(et)
        enriched_minute_map = {
            int(m): [float(v) for v in group["multiplier"]]
            for m, group in et.groupby("minute")
        }

    return {
        "df": df,
        "windows": windows,
        "timed": timed,
        "window_values": window_values,
        "other_values": other_values,
        "minute_map": minute_map,
        "favored_rows": favored_rows,
        "n_timed": len(timed),
        "n_window": n_win,
        "mode": mode,
        "enrich_df": enrich,
        "enrich_n": len(enrich),
        "enrich_timed_n": enrich_timed,
        "enriched_values": enriched_values,
        "enriched_minute_map": enriched_minute_map,
    }


@st.fragment(run_every=None if NO_REFRESH else 5)
def sidebar_status() -> None:
    df = st.session_state.get("df")
    n = 0 if df is None else len(df)
    with st.sidebar.expander("Status"):
        st.caption(f"{n} rounds stored")
        st.code(ds.default_path(), language=None)
        st.caption(f"Refreshed {datetime.now(ds.GABORONE):%H:%M:%S}")


DEV_CREDIT_HTML = """
<div class="dev-credit">
  <div class="dev-head">Developed by <b>Karabo Kosi</b></div>
  <div class="dev-links">
    <a href="tel:+26778966834" title="+267 78966834" aria-label="Phone">
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none"
           stroke="currentColor" stroke-width="2" stroke-linecap="round"
           stroke-linejoin="round"><path d="M22 16.92v3a2 2 0 0 1-2.18 2
           19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0
           1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0
           0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2
           2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22
           16.92z"></path></svg>
    </a>
    <a href="mailto:karaboemma25@gmail.com" title="karaboemma25@gmail.com"
       aria-label="Gmail">
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none"
           stroke="currentColor" stroke-width="2" stroke-linecap="round"
           stroke-linejoin="round"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0
           1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
           <polyline points="22,6 12,13 2,6"></polyline></svg>
    </a>
    <a href="https://whoiskay.vercel.app/" title="whoiskay.vercel.app"
       target="_blank" rel="noopener noreferrer" aria-label="Website">
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none"
           stroke="currentColor" stroke-width="2" stroke-linecap="round"
           stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle>
           <line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3
           15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3
           0 0 1 4-10z"></path></svg>
    </a>
    <a href="https://www.linkedin.com/in/karabo-kosi-534501380"
       title="LinkedIn" target="_blank" rel="noopener noreferrer"
       aria-label="LinkedIn">
      <svg viewBox="0 0 24 24" width="16" height="16"
           fill="currentColor"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
    </a>
    <a href="https://www.youtube.com/@whoiskay404" title="YouTube"
       target="_blank" rel="noopener noreferrer" aria-label="YouTube">
      <svg viewBox="0 0 24 24" width="16" height="16"
           fill="currentColor"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>
    </a>
    <a href="https://www.instagram.com/kaysantanaax" title="Instagram"
       target="_blank" rel="noopener noreferrer" aria-label="Instagram">
      <svg viewBox="0 0 24 24" width="16" height="16"
           fill="currentColor"><path d="M7.0301.084c-1.2768.0602-2.1487.264-2.911.5634-.7888.3075-1.4575.72-2.1228 1.3877-.6652.6677-1.075 1.3368-1.3802 2.127-.2954.7638-.4956 1.6365-.552 2.914-.0564 1.2775-.0689 1.6882-.0626 4.947.0062 3.2586.0206 3.6671.0825 4.9473.061 1.2765.264 2.1482.5635 2.9107.308.7889.72 1.4573 1.388 2.1228.6679.6655 1.3365 1.0743 2.1285 1.38.7632.295 1.6361.4961 2.9134.552 1.2773.056 1.6884.069 4.9462.0627 3.2578-.0062 3.668-.0207 4.9478-.0814 1.28-.0607 2.147-.2652 2.9098-.5633.7889-.3086 1.4578-.72 2.1228-1.3881.665-.6682 1.0745-1.3378 1.3795-2.1284.2957-.7632.4966-1.636.552-2.9124.056-1.2809.0692-1.6898.063-4.948-.0063-3.2583-.021-3.6668-.0817-4.9465-.0607-1.2797-.264-2.1487-.5633-2.9117-.3084-.7889-.72-1.4568-1.3876-2.1228C21.2982 1.33 20.628.9208 19.8378.6165 19.074.321 18.2017.1197 16.9244.0645 15.6471.0093 15.236-.005 11.977.0014 8.718.0076 8.31.0215 7.0301.0839m.1402 21.6932c-1.17-.0509-1.8053-.2453-2.2287-.408-.5606-.216-.96-.4771-1.3819-.895-.422-.4178-.6811-.8186-.9-1.378-.1644-.4234-.3624-1.058-.4171-2.228-.0595-1.2645-.072-1.6442-.079-4.848-.007-3.2037.0053-3.583.0607-4.848.05-1.169.2456-1.805.408-2.2282.216-.5613.4762-.96.895-1.3816.4188-.4217.8184-.6814 1.3783-.9003.423-.1651 1.0575-.3614 2.227-.4171 1.2655-.06 1.6447-.072 4.848-.079 3.2033-.007 3.5835.005 4.8495.0608 1.169.0508 1.8053.2445 2.228.408.5608.216.96.4754 1.3816.895.4217.4194.6816.8176.9005 1.3787.1653.4217.3617 1.056.4169 2.2263.0602 1.2655.0739 1.645.0796 4.848.0058 3.203-.0055 3.5834-.061 4.848-.051 1.17-.245 1.8055-.408 2.2294-.216.5604-.4763.96-.8954 1.3814-.419.4215-.8181.6811-1.3783.9-.4224.1649-1.0577.3617-2.2262.4174-1.2656.0595-1.6448.072-4.8493.079-3.2045.007-3.5825-.006-4.848-.0608M16.953 5.5864A1.44 1.44 0 1 0 18.39 4.144a1.44 1.44 0 0 0-1.437 1.4424M5.8385 12.012c.0067 3.4032 2.7706 6.1557 6.173 6.1493 3.4026-.0065 6.157-2.7701 6.1506-6.1733-.0065-3.4032-2.771-6.1565-6.174-6.1498-3.403.0067-6.156 2.771-6.1496 6.1738M8 12.0077a4 4 0 1 1 4.008 3.9921A3.9996 3.9996 0 0 1 8 12.0077"/></svg>
    </a>
    <a href="https://www.tiktok.com/@karabo_kosi" title="TikTok"
       target="_blank" rel="noopener noreferrer" aria-label="TikTok">
      <svg viewBox="0 0 24 24" width="16" height="16"
           fill="currentColor"><path d="M12.525.02c1.31-.02 2.61-.01 3.91-.02.08 1.53.63 3.09 1.75 4.17 1.12 1.11 2.7 1.62 4.24 1.79v4.03c-1.44-.05-2.89-.35-4.2-.97-.57-.26-1.1-.59-1.62-.93-.01 2.92.01 5.84-.02 8.75-.08 1.4-.54 2.79-1.35 3.94-1.31 1.92-3.58 3.17-5.91 3.21-1.43.08-2.86-.31-4.08-1.03-2.02-1.19-3.44-3.37-3.65-5.71-.02-.5-.03-1-.01-1.49.18-1.9 1.12-3.72 2.58-4.96 1.66-1.44 3.98-2.13 6.15-1.72.02 1.48-.04 2.96-.04 4.44-.99-.32-2.15-.23-3.02.37-.63.41-1.11 1.04-1.36 1.75-.21.51-.15 1.07-.14 1.61.24 1.64 1.82 3.02 3.5 2.87 1.12-.01 2.19-.66 2.77-1.61.19-.33.4-.67.41-1.06.1-1.79.06-3.57.07-5.36.01-4.03-.01-8.05.02-12.07z"/></svg>
    </a>
  </div>
  <div class="dev-contacts">+267 78966834 · karaboemma25@gmail.com</div>
</div>
"""


def render_dev_credit() -> None:
    st.sidebar.markdown(DEV_CREDIT_HTML, unsafe_allow_html=True)


def sidebar_rules() -> None:
    with st.sidebar.expander("Rules & stake strategy"):
        st.markdown(
            """
**The rhythm - press, skip one, then enter**

1. Press **PREDICT**.
2. Watch the next round **without entering** - sit it out.
3. Enter the round after that, and keep entering only every second round:
   **1, 3, 5, 7 …**
4. Press PREDICT again for a fresh call - the rhythm restarts: predict, skip
   one round, then re-enter.
"""
        )
        st.markdown(
            """
**Risk ↔ target**

- **Risk low → target high.** Small stake, patient rhythm - you can afford
  to aim for a higher cash-out odd (a golden-window call = low risk →
  target high).
- **Risk high → target low.** Bigger stake, chasing losses, or a call
  outside your golden windows - drop the target and cash out early.
"""
        )
        st.markdown(
            """
**Stake strategy - the doubling ladder**

1. Start small: **P10**.
2. After a loss, raise the next round to **P20**.
3. After another loss, **double** the wager again.
4. Keep doubling after every loss until you win.
5. After a win, drop straight back to **P10**.
"""
        )
        st.caption(
            "The ladder only rearranges when wins and losses land - the -3% "
            "house edge per round never changes. Set a loss limit before you "
            "start, and stop for the session the moment you hit it."
        )


def on_quick_entry() -> None:
    raw = st.session_state.get("quick_value", "")
    try:
        multiplier = ds.parse_multiplier(raw)
    except ValueError as exc:
        st.session_state["flash_err"] = str(exc)
        return
    df = get_df()
    row = pd.DataFrame([{
        "timestamp": ds.now_iso(),
        "multiplier": multiplier,
        "source": "manual",
    }])
    set_df(pd.concat([df, row], ignore_index=True),
           flash=f"Logged {multiplier:.2f}x")
    st.session_state["quick_value"] = ""


def show_verdict(code: str, p_value, n_window: int, p_shown: bool = True) -> None:
    if code == "insufficient":
        st.warning(
            f"**Verdict: Not enough data.** Your windows contain {n_window} "
            f"timestamped round(s); a reliable verdict needs roughly "
            f"{DATA_BACKED_N} rounds inside them."
        )
    elif code == "hidden":
        st.warning(
            "**Verdict: Not enough data.** Need at least 30 rounds in both "
            "groups to compute a valid p-value."
        )
    elif code == "significant":
        st.success(
            f"**Verdict: your windows are significantly safer in this sample** "
            f"(p = {fmt_p(p_value)} < 0.05). Evidence, not a guarantee - "
            "confirm on fresh data before trusting it fully."
        )
    else:
        st.info(
            f"**Verdict: no significant difference yet** (p = {fmt_p(p_value)} "
            "≥ 0.05). Any gap vs other minutes is consistent with random "
            "variation - keep feeding the predictor data."
        )


# - predict

def nudge_odd(odd: float, history, promoted: bool) -> tuple[float, float | None]:
    """Return (odd, original-or-None) with odd stepped off any blocked value.

    Keeps the same suggested odd from showing twice in a row: the value moves
    one grid step (0.05) to the nearest alternative inside the same band
    (cautious 1.00-1.45x, or high 1.50-2.00x when promoted), preferring the
    lower, more conservative value when two are equally near.
    """
    blocked = {round(float(h), 2) for h in history}
    key = round(float(odd), 2)
    if key not in blocked:
        return float(odd), None
    lo = ODD_SOFT_CAP + 0.05 if promoted else ODD_MIN
    hi = ODD_MAX if promoted else ODD_SOFT_CAP
    grid = [round(lo + i * 0.05, 2)
            for i in range(int(round((hi - lo) / 0.05)) + 1)]
    cands = [v for v in grid if v not in blocked]
    if not cands:
        return float(odd), None
    best = min(cands, key=lambda v: (abs(v - key), 0 if v < key else 1))
    return best, float(odd)


def build_prediction(ctx: dict, minute: int, history=None) -> dict:
    pred = ss.predict_next(
        minute=minute,
        windows=ctx["windows"],
        window_values=ctx["window_values"],
        other_values=ctx["other_values"],
        minute_values=ctx["minute_map"].get(minute, []),
        favored=[row["minute"] for row in ctx["favored_rows"]],
    )
    pred["created"] = f"{datetime.now(ds.GABORONE):%H:%M:%S}"
    pred["window_n"] = ctx["n_window"]
    pred["other_n"] = len(ctx["other_values"])
    pred["golden"] = pred["matched_window"] is not None
    pred["windows_text"] = windows_text(ctx["windows"])

    per_min = ctx["enriched_minute_map"].get(minute)
    overall = ctx["enriched_values"]
    decision = pred["decision"]

    if decision == "go":
        group = ctx["window_values"]
        default = 1.25
    elif decision == "go_data":
        group = ctx["minute_map"].get(minute, [])
        default = 1.35
    else:
        group = per_min if per_min else overall
        default = 1.10

    odd, promoted = ss.suggest_odd(group, default=default)
    if group:
        if decision == "go":
            where = f"n={len(group)} rounds inside your golden windows"
        elif decision == "go_data":
            where = f"n={len(group)} rounds at minute {minute:02d}"
        elif per_min:
            where = f"n={len(group)} at minute {minute:02d}"
        else:
            where = f"n={len(group)} across all minutes"
        source_base = {"go": "golden-window data",
                       "go_data": "data-validated minute",
                       "skip": "enriched CSV odds"}[decision]
        odd_source = f"{source_base} ({where})"
    else:
        odd_source = {"go": "golden-window default (no window data yet)",
                      "go_data": "data-validated minute default",
                      "skip": "default (no enrichment CSV loaded)"}[decision]

    odd, adjusted_from = nudge_odd(odd, history or (), promoted)

    pred["odd"] = float(odd)
    pred["odd_adjusted_from"] = adjusted_from
    pred["odd_promoted"] = promoted
    pred["odd_high"] = odd > ODD_SOFT_CAP
    pred["odd_source"] = odd_source
    pred["odd_p"] = ss.theoretical_p(pred["odd"])

    pred["enrich_stats"] = None
    if decision == "skip" and (per_min or overall):
        group = per_min if per_min else overall
        pred["enrich_stats"] = {
            "n": len(group),
            "base": ss.percentile(group, 25),
            "median": ss.percentile(group, 50),
            "unsafe": sum(1 for v in group if v < 1.10) / len(group),
            "per_minute": bool(per_min),
        }
    return pred


def render_odd_display(pred: dict) -> None:
    status = {
        "go": "Golden window",
        "go_data": "Outside window · data",
        "skip": "Outside window",
    }[pred["decision"]]
    cls = "golden" if pred["decision"] == "go" else "outside"
    st.markdown(
        f"""
        <div class="hero-result">
            <div id="odd-circle" class="{cls}">
                <div class="odd-value">{pred["odd"]:.2f}x</div>
                <div class="odd-status">{status}</div>
            </div>
            <div class="hero-meta">P(reach) {ss.pct_text(pred["odd_p"])} ·
            pressed {pred["created"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(pred: dict) -> None:
    minute = pred["minute"]
    matched = pred["matched_window"]
    odd = pred["odd"]
    odd_p = ss.pct_text(pred["odd_p"])

    band_note = (
        f"promoted into the high band {ODD_SOFT_CAP:.2f}-{ODD_MAX:.2f}x - the "
        "data is very certain the plane will go far."
        if pred["odd_promoted"] else
        f"kept in the cautious {ODD_MIN:.2f}-{ODD_SOFT_CAP:.2f}x band - higher "
        "odds only appear when the data is very certain the plane goes far."
    )

    if pred["decision"] == "go":
        st.success(
            f"**GOLDEN WINDOW · ODD {odd:.2f}x**\n\n"
            f"Minute **{minute:02d}** falls inside golden window "
            f"**{matched[0]:02d}-{matched[1]:02d}**. Suggested cash-out odd "
            f"**{odd:.2f}x** ({band_note}) P(reach) {odd_p}. Chance of a safe "
            f"1.10-2.50x result: **{ss.pct_text(pred['headline_band'])}** "
            f"(basis: {pred['headline_source']})."
        )
    elif pred["decision"] == "go_data":
        st.success(
            f"**OUTSIDE GOLDEN WINDOW · DATA-VALIDATED · ODD {odd:.2f}x**\n\n"
            f"Minute **{minute:02d}** is not in a golden window, but your own "
            "logged data shows it stays safe unusually often. Suggested cash-out "
            f"odd **{odd:.2f}x** ({band_note} P(reach) {odd_p}.) Chance of a "
            f"safe 1.10-2.50x result: "
            f"**{ss.pct_text(pred['headline_band'])}** "
            f"(basis: {pred['headline_source']})."
        )
    else:
        st.warning(
            f"**OUTSIDE GOLDEN WINDOW · ODD {odd:.2f}x**\n\n"
            f"Minute **{minute:02d}** sits outside the golden windows "
            f"({pred['windows_text']}). Advanced odd **{odd:.2f}x** comes from "
            f"{pred['odd_source']} - {band_note} P(reach) {odd_p}. The unsafe "
            f"1.00-1.10x zone stays at its usual "
            f"~{ss.pct_text(pred['theory']['unsafe'])} here, and a safe "
            f"1.10-2.50x result lands ~{ss.pct_text(pred['theory']['band'])} of "
            f"the time (basis: {pred['headline_source']})."
        )
        es = pred.get("enrich_stats")
        if es:
            st.caption(
                f"Enrichment behind this odd: n = {es['n']} "
                f"{'(this minute)' if es['per_minute'] else '(all minutes)'}, "
                f"P25 {es['base']:.2f}x → odd, median {es['median']:.2f}x, "
                f"% under 1.10x {es['unsafe'] * 100:.1f}% - times are estimates "
                "(±1 min), odds read by eye from screenshots."
            )
    if pred["odd_promoted"]:
        st.caption(
            f"HIGH-ODD CALL: {odd:.2f}x sits above the {ODD_SOFT_CAP:.2f}x soft "
            f"cap because at least {int(ss.ODD_PROMOTE_RATE * 100)}% of a "
            f"{ss.ODD_PROMOTE_MIN_N}+ round sample reached it (Wilson lower "
            f"bound ≥ {int(ss.ODD_PROMOTE_LB * 100)}%) - a very-certain "
            "‘plane goes far’ call, not the usual suggestion."
        )
    if pred.get("odd_adjusted_from") is not None:
        st.caption(
            f"Repeat avoided: the sample suggested {pred['odd_adjusted_from']:.2f}x "
            f"again - the same odd as your previous call - so it was nudged one "
            f"grid step to {odd:.2f}x to keep back-to-back predictions different."
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Suggested odd", f"{odd:.2f}x")
    m2.metric(f"P(reach {odd:.2f}x)", odd_p)
    m3.metric("P(safe 1.10-2.50x)", ss.pct_text(pred["headline_band"]))
    m4.metric("Window sample", f"n = {pred['window_n']}")
    st.caption(
        f"{'Inside' if pred['golden'] else 'Outside'} your golden windows · "
        f"confidence: {pred['confidence']} · basis: {pred['headline_source']} · "
        f"odd source: {pred['odd_source']} · band: "
        f"{'high 1.45-2.00 (very certain)' if pred['odd_high'] else 'cautious 1.00-1.45'}"
        f" · pressed {pred['created']} Gaborone."
    )

    if pred["window"]["n"] >= ss.MIN_N_FOR_P:
        low, high = pred["window"]["band_ci"]
        st.caption(
            f"95% confidence interval on the window's safe-odds rate: "
            f"{ss.pct_text(low)} - {ss.pct_text(high)}."
        )

    comparison = pred["comparison"]
    if comparison:
        p_shown = (pred["window_n"] >= ss.MIN_N_FOR_P
                   and pred["other_n"] >= ss.MIN_N_FOR_P)
        st.dataframe(pd.DataFrame([
            ["Your safe windows", f"{pred['window_n']}",
             ss.pct_text(comparison["unsafe_window"])],
            ["All other minutes", f"{pred['other_n']}",
             ss.pct_text(comparison["unsafe_other"])],
            ["Fisher exact p-value",
             fmt_p(comparison["p"]) if p_shown else "hidden (n < 30)"],
        ], columns=["Group", "n", "% under 1.10x"]),
            hide_index=True, width="stretch")

    for note in pred["notes"]:
        st.markdown(f"- {note}")


def render_hour_plan(ctx: dict, next_minute: int) -> None:
    st.subheader("Your minute plan (every minute of the hour)")
    calls = {"go": "GOLDEN", "go_data": "DATA ✓", "skip": "OUTSIDE"}
    history = list(st.session_state.get("odd_history", []))
    plan_rows = []
    for minute in range(60):
        pred = build_prediction(ctx, minute, history=history)
        history = (history + [pred["odd"]])[-2:]
        marker = " ◀ next" if minute == next_minute else ""
        if pred["matched_window"]:
            why = (f"{pred['matched_window'][0]:02d}-"
                   f"{pred['matched_window'][1]:02d}")
        elif pred["decision"] == "go_data":
            why = "data-favored"
        elif pred.get("enrich_stats"):
            why = f"CSV n={pred['enrich_stats']['n']}"
        else:
            why = "-"
        plan_rows.append([
            f"{minute:02d}{marker}", calls[pred["decision"]],
            f"{pred['odd']:.2f}x", ss.pct_text(pred["odd_p"]),
            ss.pct_text(pred["headline_band"]), pred["confidence"], why,
        ])
    st.dataframe(pd.DataFrame(plan_rows, columns=[
        "Minute", "Call", "Suggested odd", "P(reach)",
        "P(safe 1.10-2.50x)", "Confidence", "Why",
    ]), hide_index=True, width="stretch")
    st.caption(
        "**GOLDEN** = inside your golden windows · "
        "**DATA ✓** = outside them but validated by your data · "
        "**OUTSIDE** = outside every golden window (small cautious odd). "
        f"Odds stay in the cautious {ODD_MIN:.2f}-{ODD_SOFT_CAP:.2f}x band; "
        f"{ODD_SOFT_CAP:.2f}-{ODD_MAX:.2f}x shows up only when a 30+ round "
        "sample very certainly reached it. Back-to-back rows are nudged one "
        "grid step so the same odd never appears twice in a row."
    )


def render_window_health(ctx: dict) -> None:
    windows = ctx["windows"]
    timed = ctx["timed"]
    st.subheader("Are your windows actually safer?")
    if not ctx["n_timed"]:
        st.info(
            "0 timestamped rounds - the window test needs timestamps. Timestamp "
            "your seed rows or log/paste rounds on the Data tab."
        )
        return

    rows = []
    for start, end in windows:
        lo, hi = min(start, end), max(start, end)
        values = [float(v) for v in timed[timed["minute"].map(lambda m: lo <= m <= hi)]
                  ["multiplier"]]
        stats = group_stats(values)
        rows.append([
            f"{lo:02d}-{hi:02d}" if lo != hi else f"{lo:02d}",
            stats["n"], pct(stats["low"]), pct(stats["band"]),
            fnum(stats["median"], suffix="x"),
        ])
    rows.append(["ALL WINDOWS", group_stats(ctx["window_values"])["n"],
                 pct(group_stats(ctx["window_values"])["low"]),
                 pct(group_stats(ctx["window_values"])["band"]),
                 fnum(group_stats(ctx["window_values"])["median"], suffix="x")])
    rows.append(["Other minutes", len(ctx["other_values"]),
                 pct(group_stats(ctx["other_values"])["low"]),
                 pct(group_stats(ctx["other_values"])["band"]),
                 fnum(group_stats(ctx["other_values"])["median"], suffix="x")])
    st.dataframe(pd.DataFrame(rows, columns=[
        "Window", "n", "% < 1.10x", "% 1.10-2.50x", "Median",
    ]), hide_index=True, width="stretch")

    win_values = ctx["window_values"]
    other_values = ctx["other_values"]
    n_win, n_other = len(win_values), len(other_values)
    if not n_win or not n_other:
        st.info("Need rounds both inside and outside your windows to compare.")
        return

    low_win = sum(1 for v in win_values if v < 1.10)
    low_other = sum(1 for v in other_values if v < 1.10)
    show_p = n_win >= ss.MIN_N_FOR_P and n_other >= ss.MIN_N_FOR_P
    fisher = ss.fisher_exact([[low_win, n_win - low_win],
                              [low_other, n_other - low_other]])
    mann = ss.mann_whitney_u(win_values, other_values)
    st.dataframe(pd.DataFrame([
        ["Unsafe rate difference (windows - others)",
         f"{(low_win / n_win - low_other / n_other) * 100:+.2f} pp"],
        ["Fisher exact p (outcome < 1.10x)",
         fmt_p(fisher["p"]) if show_p else "hidden (n < 30)"],
        ["Mann-Whitney U p (whole distribution)",
         fmt_p(mann["p"]) if show_p else "hidden (n < 30)"],
        ["Effect size (rank-biserial r)",
         f"{mann['r']:.3f}" if show_p else "-"],
    ], columns=["Measure", "Value"]), hide_index=True, width="stretch")
    show_verdict(ss.verdict(n_win, n_other, fisher["p"] if show_p else None),
                 fisher["p"] if show_p else None, n_win, p_shown=show_p)


def render_windows_editor() -> None:
    st.subheader("Your golden windows")
    st.caption(
        "These decide when you get the big odd vs the cautious one. Edit the "
        "start/end minutes of each hour (0-59) - defaults 03-04, 20-22, 29-31, "
        "40-42, 45-47, 50-52, 57-59."
    )
    windows = ensure_windows()
    windows_df = pd.DataFrame(windows, columns=["start minute", "end minute"])
    edited = st.data_editor(
        windows_df,
        num_rows="dynamic",
        hide_index=True,
        key=f"windows_editor_{st.session_state.get('windows_version', 0)}",
        column_config={
            "start minute": st.column_config.NumberColumn(
                "start minute", min_value=0, max_value=59, step=1),
            "end minute": st.column_config.NumberColumn(
                "end minute", min_value=0, max_value=59, step=1),
        },
    )
    w1, w2 = st.columns(2)
    if w1.button("Apply window edits", type="primary"):
        candidate, errors = [], []
        for i, row in edited.iterrows():
            start, end = row["start minute"], row["end minute"]
            if pd.isna(start) or pd.isna(end):
                errors.append(f"Window {i + 1}: start and end are required.")
                continue
            start, end = int(start), int(end)
            if not (0 <= start <= 59 and 0 <= end <= 59):
                errors.append(f"Window {i + 1}: minutes must be 0-59.")
                continue
            candidate.append((start, end))
        if errors:
            st.error("\n\n".join(errors))
        else:
            st.session_state["windows"] = candidate
            st.session_state["windows_version"] = \
                st.session_state.get("windows_version", 0) + 1
            st.rerun()
    if w2.button("Reset to defaults"):
        st.session_state["windows"] = [(int(s), int(e)) for s, e in ss.DEFAULT_WINDOWS]
        st.session_state["windows_version"] = \
            st.session_state.get("windows_version", 0) + 1
        st.rerun()


def render_favored(ctx: dict) -> None:
    favored_rows = ctx["favored_rows"]
    if not favored_rows:
        return
    st.subheader("Extra minutes your data supports")
    st.caption(
        "Minutes outside your windows whose unsafe-crash rate is significantly "
        "below baseline (Wilson interval must clear the overall rate - lucky "
        "minutes can't sneak in)."
    )
    st.dataframe(pd.DataFrame([
        [f"{row['minute']:02d}", row["n"], ss.pct_text(row["unsafe_rate"]),
         f"{ss.pct_text(row['unsafe_ci'][0])} - {ss.pct_text(row['unsafe_ci'][1])}"]
        for row in favored_rows[:10]
    ], columns=["Minute", "n", "% < 1.10x", "95% CI"]),
        hide_index=True, width="stretch")


def render_targets() -> None:
    st.subheader("Cash-out targets inside the safe band")
    targets = [1.10, 1.30, 1.50, 2.00, 2.50]
    st.dataframe(pd.DataFrame({
        "Target": [f"{t:.2f}x" for t in targets],
        "P(reach), theory": [ss.pct_text(ss.theoretical_p(t)) for t in targets],
        "EV per P100 staked": [f"-P{abs(100 * (ss.theoretical_p(t) * t - 1)):,.2f}"
                              for t in targets],
    }), hide_index=True, width="stretch")
    st.caption(
        "Every target carries the same -3% house edge (97% RTP): lower targets "
        "hit more often, higher targets pay more. Nothing here changes the "
        "expected loss."
    )


def render_minute_strip(ctx: dict, next_minute: int) -> None:
    pills = []
    shown = st.session_state.get("prediction")
    history = [shown["odd"]] if shown else []
    for offset in range(4):
        minute = (next_minute + offset) % 60
        call = build_prediction(ctx, minute, history=history)
        history = (history + [call["odd"]])[-2:]
        if call["decision"] == "go":
            cls, label = "golden", "GOLDEN window"
        elif call["decision"] == "go_data":
            cls, label = "data", "data-validated minute"
        else:
            cls, label = "outside", "outside the golden windows"
        next_cls = " next" if offset == 0 else ""
        pills.append(
            f'<span class="strip-pill {cls}{next_cls}" title="{label}">'
            f'<span class="dot"></span><b>{minute:02d}</b>'
            f'<span class="strip-odd">{call["odd"]:.2f}x</span></span>'
        )
    st.markdown('<div class="minute-strip">' + "".join(pills) + "</div>",
                unsafe_allow_html=True)


def render_predict() -> None:
    ctx = predict_context()
    now = datetime.now(ds.GABORONE)
    next_minute = (now + timedelta(seconds=30)).minute

    left, right = st.columns(2, gap="small")
    with left:
        if st.button("PREDICT", type="primary", key="predict_btn"):
            choice = st.session_state.get(
                "predict_minute", "Auto (from the clock)")
            if str(choice).startswith("Auto"):
                minute = (datetime.now(ds.GABORONE)
                          + timedelta(seconds=30)).minute
            else:
                minute = int(choice)
            history = list(st.session_state.get("odd_history", []))
            pred = build_prediction(ctx, minute, history=history)
            st.session_state["prediction"] = pred
            st.session_state["odd_history"] = (history + [pred["odd"]])[-2:]
            st.rerun()
    with right:
        pred = st.session_state.get("prediction")
        if pred:
            render_odd_display(pred)
            if st.button("Reset", key="reset_pred"):
                st.session_state.pop("prediction", None)
                st.rerun()
        else:
            st.markdown(
                """
                <div class="hero-result">
                    <div id="odd-circle" class="idle">
                        <div class="odd-value">?</div>
                        <div class="odd-status">Waiting</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.selectbox(
        "Round to predict",
        ["Auto (from the clock)"] + [f"{m:02d}" for m in range(60)],
        key="predict_minute",
    )
    render_minute_strip(ctx, next_minute)

    with st.expander("Details"):
        st.caption(
            f"Now {now:%H:%M:%S} Gaborone - the next round should land on "
            f"minute {next_minute:02d} of this hour. Golden windows: "
            f"{windows_text(ctx['windows'])}."
        )
        pred = st.session_state.get("prediction")
        if pred:
            render_prediction_card(pred)
        else:
            st.caption(
                "Press PREDICT for the call, the suggested odd and the full "
                "reasoning here."
            )

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Timestamped rounds", ctx["n_timed"])
        q2.metric("Inside your windows", ctx["n_window"])
        q3.metric("Needed for data-backed",
                  f"n {DATA_BACKED_N - ctx['n_window']}"
                  if ctx["n_window"] < DATA_BACKED_N else "reached ✓")
        q4.metric("Prediction mode", ctx["mode"])
        st.progress(min(ctx["n_window"] / DATA_BACKED_N, 1.0),
                    text=f"Progress to data-backed predictions: "
                         f"{min(ctx['n_window'], DATA_BACKED_N)}/{DATA_BACKED_N} "
                         "timestamped rounds inside your windows")
        if ctx["mode"] != "DATA-BACKED":
            st.caption(
                "While the sample is thin, headline probabilities come from "
                "the verified 97% RTP theory and say so; your window data is "
                "shown alongside and takes over once the bar fills."
            )

    with st.expander("Minute plan & windows"):
        render_hour_plan(ctx, next_minute)
        render_window_health(ctx)
        render_favored(ctx)
        render_targets()
        render_windows_editor()

    with st.expander("Sources & safety"):
        st.markdown(
            """
- **Spribe's official Aviator page lists RTP 97%** (spribe.co/games/aviator).
- The published *provably fair* formula bakes that 0.97 in directly:
  `multiplier = floor(0.97 × 2^52 / (2^52 - h) × 100) / 100`, where `h` comes
  from SHA-512 of the server + 3 client seeds - so
  **P(reach x) = 0.97 / x** exactly (reach 1.10x = 88.2%, 1.50x = 64.7%,
  2.50x = 38.8% → safe band 1.10-2.50x = 49.4%, unsafe <1.10x = 11.8%).
- Independently verified round-by-round by FairPlay Audit (fairplayaudit.com,
  2026): they recomputed real demo rounds from the seeds and reproduced the
  formula and RTP exactly.
- Consequence: the seed hash hides the future multiplier (SHA-512 is one-way),
  so nobody - this app included - can read the next result. Predictions here
  are *probability calls from your samples + verified theory*, and they say so
  honestly when the sample is too thin.
- **Want better predictions?** Feed the predictor: timestamped rounds beat
  raw rounds, 300+ inside your windows flips the mode to DATA-BACKED, and new
  data can promote extra minutes into your plan.
"""
        )
        st.markdown(
            """
- **Set a stop-loss before you play**, not after - house edge means the
  longer the session, the more certain the loss. Take profit and stop on time.
- No time window or pattern changes the next round's distribution; only your
  discipline changes what a session costs you.
- **Botswana:** free confidential counselling & self-exclusion via the
  Botswana Gambling Authority - [selfhelp.gamblingauthority.co.bw](https://selfhelp.gamblingauthority.co.bw/)
  · 24/7 toll-free **1144** (Orange), **71119604** (Mascom), **0800600644** (BTC)
  · gamblingauthority@tip-offs.com · +267 395 7672.
- Outside Botswana: Gambling Therapy (gamblingtherapy.org), GamCare (UK).
"""
        )


# - data

def render_data_quality(ctx: dict) -> None:
    df = ctx["df"]
    untimed = int(df["timestamp"].map(ds.is_untimed).sum())
    st.subheader("Feeding the predictor")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Total rounds", len(df))
    q2.metric("Timestamped (usable for time predictions)", ctx["n_timed"])
    q3.metric("Inside your windows", ctx["n_window"])
    q4.metric("Prediction mode", ctx["mode"])
    if untimed:
        st.warning(
            f"**{untimed} round(s) have no timestamp** (seed data). They cannot "
            "help time-based predictions until timestamped - use "
            "“Timestamp untimed rows” below."
        )
    if ctx["n_window"] < DATA_BACKED_N:
        st.caption(
            f"Predictions become DATA-BACKED at {DATA_BACKED_N} timestamped "
            f"rounds inside your windows - you have {ctx['n_window']}. Log "
            "rounds live, paste a list, or import a CSV below."
        )
    st.caption(
        f"Enrichment CSV: {ctx['enrich_n']} round(s) loaded "
        f"({ctx['enrich_timed_n']} with estimated time) - these set the "
        f"OUTSIDE golden-window odds (1.00-{ODD_SOFT_CAP:.2f}x typically; "
        "higher only when very certain)."
    )


def render_enrichment(ctx: dict) -> None:
    st.subheader("Enrichment CSV - odds + estimated timestamps")
    st.caption(
        "Optional extra odds file that drives the OUTSIDE golden-window "
        "predictions. Odds default to the cautious 1.00-1.45x band (25th "
        "percentile of the file, capped); 1.45-2.00x only appears when a "
        "30+ round sample very often reached it. Default: bundled "
        "`csv_data_odds/aviator_rounds.csv` - odds read by eye from "
        "screenshots with estimated times (~22 s per round, ±1 min error, "
        "gaps between segments, in-progress rounds excluded)."
    )
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Enrichment rounds", ctx["enrich_n"])
    e2.metric("With estimated time", ctx["enrich_timed_n"])
    e3.metric("Minutes covered", f"{len(ctx['enriched_minute_map'])}/60")
    e4.metric("Typical outside odd", f"{ODD_MIN:.2f}-{ODD_SOFT_CAP:.2f}x")

    b1, b2 = st.columns(2)
    if b1.button("Reload bundled enrichment CSV"):
        frame, errors = ds.load_enrichment()
        st.session_state["enrichment"] = frame
        st.session_state["enrichment_errors"] = errors
        st.session_state.pop("prediction", None)
        st.rerun()
    uploaded = b2.file_uploader("Upload odds CSV", type=["csv"],
                                key="enrich_uploader")
    if uploaded is not None:
        identity = (uploaded.name, uploaded.size)
        if st.session_state.get("pending_enrich") != identity:
            text = uploaded.getvalue().decode("utf-8-sig", errors="replace")
            frame, errors = ds.read_enrichment_csv(text)
            if len(frame):
                st.session_state["enrichment"] = frame
                st.session_state["enrichment_errors"] = errors
                st.session_state["pending_enrich"] = identity
                st.session_state.pop("prediction", None)
                st.session_state["flash"] = (
                    f"Enrichment loaded: {len(frame)} round(s) from "
                    f"{uploaded.name}. Outside-window odds updated."
                )
                st.rerun()
            else:
                st.error("; ".join(errors[:5]) or "CSV had no usable rows.")
    for message in st.session_state.get("enrichment_errors", [])[:5]:
        st.caption(message)


def render_data() -> None:
    df = get_df()
    ctx = predict_context()
    untimed = int(df["timestamp"].map(ds.is_untimed).sum())

    render_data_quality(ctx)

    with st.expander("Add rounds", expanded=True):
        st.text_input(
            "Multiplier of the round that just finished (press Enter)",
            key="quick_value",
            on_change=on_quick_entry,
            placeholder="e.g. 1.50",
        )
        st.caption("The timestamp fills in automatically: Africa/Gaborone, "
                   "UTC+02:00.")

        with st.form("batch_form", clear_on_submit=True):
            st.text_area(
                "Separate multipliers with comma, space or new line",
                key="batch_list",
                placeholder="1.24 2.10 1.05, 3.40\n1.12",
            )
            c1, c2, c3 = st.columns(3)
            start_date = c1.date_input("Start date", value=date.today())
            start_time = c2.time_input(
                "Start time", value=datetime.now(ds.GABORONE).time().replace(second=0)
            )
            interval = c3.number_input("Seconds per round", 1, 3600, 30)
            b1, b2 = st.columns(2)
            add_clicked = b1.form_submit_button("Add rounds")
            stamp_clicked = b2.form_submit_button("Timestamp untimed rows")

        if add_clicked or stamp_clicked:
            values, bad = ds.parse_number_list(st.session_state.get("batch_list", ""))
            good = [v for v in values if v >= 1.0]
            bad += [str(v) for v in values if v < 1.0]
            if not good and add_clicked:
                st.error("No valid multipliers found (must be ≥ 1.00).")
            else:
                start = datetime.combine(start_date, start_time).replace(
                    tzinfo=ds.GABORONE)
                if add_clicked:
                    incoming = ds.batch_rounds(good, start, float(interval))
                    merged, added, dupes = ds.merge_rounds(get_df(), incoming)
                    message = (f"Added {added} round(s) starting "
                               f"{start.isoformat(timespec='seconds')}, every "
                               f"{interval}s.")
                    if dupes:
                        message += f" Skipped {dupes} duplicate(s)."
                    if bad:
                        message += " Rejected: " + ", ".join(bad[:8]) + \
                                   ("…" if len(bad) > 8 else "")
                    set_df(merged, flash=message)
                    st.rerun()
                else:
                    stamped, count = ds.stamp_untimed(get_df(), start, float(interval))
                    if count:
                        set_df(stamped, flash=f"Stamped {count} untimed row(s) "
                                              f"starting "
                                              f"{start.isoformat(timespec='seconds')}.")
                        st.rerun()
                    else:
                        st.info("No untimed rows to timestamp.")

        st.caption("Aviator rounds typically land every ~25-35 seconds - "
                   "adjust the interval.")
        if untimed:
            st.caption(f"{untimed} round(s) still untimed (excluded from time "
                       "predictions).")

    with st.expander("Import / Export"):
        d1, d2 = st.columns(2)
        d1.download_button(
            "Export CSV",
            ds.df_to_csv(df),
            file_name="aviator_rounds.csv",
            mime="text/csv",
            type="primary",
        )
        d2.download_button(
            "Download template",
            ds.template_csv(),
            file_name="aviator_template.csv",
            mime="text/csv",
        )
        st.caption("Import merges with existing data; duplicates (same "
                   "timestamp + multiplier) are skipped.")

        uploaded = st.file_uploader("Choose a CSV file", type=["csv"],
                                    key="csv_uploader")
        if uploaded is not None:
            identity = (uploaded.name, uploaded.size)
            if st.session_state.get("pending_file") != identity:
                text = uploaded.getvalue().decode("utf-8-sig", errors="replace")
                incoming, errors = ds.read_csv_text(text)
                st.session_state["pending_import"] = {
                    "df": incoming, "errors": errors, "name": uploaded.name,
                }
                st.session_state["pending_file"] = identity
        pending = st.session_state.get("pending_import")
        if pending:
            st.caption(f"Ready to merge: {len(pending['df'])} valid row(s) from "
                       f"{pending['name']}.")
            for message in pending["errors"][:5]:
                st.caption(message)
            if len(pending["errors"]) > 5:
                st.caption(f"… and {len(pending['errors']) - 5} more problem(s).")
            if st.button("Merge imported rows", type="primary"):
                merged, added, dupes = ds.merge_rounds(get_df(), pending["df"])
                st.session_state.pop("pending_import", None)
                st.session_state.pop("pending_file", None)
                note = f"Imported: {added} new round(s), {dupes} duplicate(s) skipped."
                if pending["errors"]:
                    note += f" {len(pending['errors'])} row(s) rejected."
                set_df(merged, flash=note)
                st.rerun()

        render_enrichment(ctx)

    with st.expander("Danger zone"):
        st.caption("Irreversible: restore swaps everything for the 60-round "
                   "seed sample, clear deletes all logged rounds.")
        if not st.session_state.get("confirm_restore", False):
            if st.button("Restore seed data", key="restore_seed"):
                st.session_state["confirm_restore"] = True
                st.rerun()
        else:
            st.error("Replace ALL logged rounds with the 60-round seed sample?")
            y1, y2 = st.columns(2)
            if y1.button("Yes, restore seed data", key="yes_restore"):
                set_df(ds.seed_df(), flash="Restored the 60-round seed sample.")
                st.session_state["confirm_restore"] = False
                st.rerun()
            if y2.button("Cancel restore", key="cancel_restore"):
                st.session_state["confirm_restore"] = False
                st.rerun()

        if not st.session_state.get("confirm_clear", False):
            if st.button("Clear all data", key="clear_all"):
                st.session_state["confirm_clear"] = True
                st.rerun()
        else:
            st.error("Delete ALL logged rounds? This cannot be undone.")
            a1, a2 = st.columns(2)
            if a1.button("Yes, clear everything", key="yes_clear"):
                set_df(ds.empty_df(), flash="All rounds deleted.")
                st.session_state["confirm_clear"] = False
                st.rerun()
            if a2.button("Cancel", key="cancel_clear"):
                st.session_state["confirm_clear"] = False
                st.rerun()

    st.subheader(f"Logged rounds ({len(df)} total, {len(df) - untimed} timestamped)")
    if not len(df):
        st.info("No rounds yet. Type a multiplier above, paste a list, or "
                "import a CSV.")
        return

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        hide_index=True,
        key=f"rounds_editor_{st.session_state['df_version']}",
        column_config={
            "timestamp": st.column_config.TextColumn("timestamp (ISO local)"),
            "multiplier": st.column_config.NumberColumn("multiplier",
                                                        min_value=1.0,
                                                        format="%.2f"),
            "source": st.column_config.SelectboxColumn(
                "source", options=list(ds.VALID_SOURCES)),
        },
    )
    e1, e2 = st.columns(2)
    if e1.button("Apply table edits", type="primary"):
        errors = ds.validate_df(edited)
        if errors:
            st.error("\n\n".join(errors[:10]))
        else:
            set_df(edited, flash="Table edits applied.")
            st.rerun()
    if e2.button("Discard table edits"):
        st.session_state["df_version"] = int(st.session_state.get("df_version", 0)) + 1
        st.rerun()

    recent = df.tail(15).iloc[::-1]
    st.dataframe(recent, hide_index=True, width="stretch")
    if len(df) > 15:
        st.caption(f"Showing newest 15 of {len(df)}.")


def main() -> None:
    get_df()
    sidebar_status()
    render_dev_credit()
    sidebar_rules()
    ctx = predict_context()

    h1, h2 = st.columns([3, 2], vertical_alignment="center")
    with h1:
        st.title("Aviator Predictor")
    with h2:
        st.markdown(
            f'<div class="chip-wrap"><span class="chip">{len(ctx["df"])}'
            f' rounds · {ctx["mode"]} mode</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<p class="app-note">Estimate only. Rounds are independent - '
        'no prediction is guaranteed.</p>',
        unsafe_allow_html=True,
    )

    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)
    flash_err = st.session_state.pop("flash_err", None)
    if flash_err:
        st.error(flash_err)

    tab_predict, tab_data = st.tabs(["Predict", "Data"])
    with tab_predict:
        render_predict()
    with tab_data:
        render_data()


main()
