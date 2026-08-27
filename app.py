import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import yfinance as yf
import warnings
import json
from pathlib import Path
import subprocess
import os

warnings.filterwarnings('ignore')

# 🛠️ 視覺設定：設定寬螢幕與現代暗色金融終端風格
st.set_page_config(layout="wide", page_title="全球跨資產優化與市場風險監控終端", page_icon="🛡️")

st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; }
    h1, h2, h3, h4 { color: #58a6ff; font-family: 'Helvetica Neue', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.title("🎛️ 全球跨資產優化與市場風險監控終端")
st.caption("2026 終極合流版：整合馬可維茲資產優化模型、CDS 違約風險懲罰、與 GitHub 矩陣代數台股/日圓平倉風險即時診斷演算法")

# =========================================================================
# 初始化全域記憶緩存
# =========================================================================
if 'custom_funds' not in st.session_state:
    st.session_state.custom_funds = [
        {"名稱/代號": "SPY (SPDR 標普 500 ETF)", "Ticker": "SPY", "預期年化報酬率 (%)": 10.5, "預期年化波動度 (%)": 15.0},
        {"名稱/代號": "AGG (iShares 綜合債券 ETF)", "Ticker": "AGG", "預期年化報酬率 (%)": 4.5, "預期年化波動度 (%)": 4.5},
        {"名稱/代號": "QQQ (Invesco 納指 100 ETF)", "Ticker": "QQQ", "預期年化報酬率 (%)": 12.5, "預期年化波動度 (%)": 18.0}
    ]
if 'cached_asset_details' not in st.session_state: st.session_state.cached_asset_details = {}

FUND_SAMPLE_SITE = "https://d47x6npujduusep99gqp2a.streamlit.app/"
FUND_SAMPLE_ROOT = Path(
    os.environ.get(
        "FUND_SAMPLE_ROOT",
        str(Path.home() / "Desktop" / "grafana-dashboard-main"),
    )
)


def load_fund_screener_samples():
    """讀取基金篩選器同源樣本，並由本機淨值歷史估算年化報酬與波動度。"""
    config_path = FUND_SAMPLE_ROOT / "config" / "funds.json"
    if not config_path.exists():
        return [], "找不到基金篩選器的本機資料庫。"

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], f"基金樣本資料讀取失敗：{exc}"

    category_names = {
        item.get("id", ""): item.get("name", item.get("id", "未分類"))
        for item in payload.get("categories", [])
    }
    samples = []
    seen_codes = set()

    for item in payload.get("funds", []):
        code = str(item.get("twelve_data_symbol") or item.get("moneydj_id") or "").strip().upper()
        name = str(item.get("name") or code).strip()
        if not code or code in seen_codes:
            continue
        seen_codes.add(code)

        expected_return = 8.0
        expected_volatility = 12.0
        history_path = FUND_SAMPLE_ROOT / "data" / "nav_history" / f"{item.get('moneydj_id', '')}.csv"
        if item.get("moneydj_id") and history_path.exists():
            try:
                history = pd.read_csv(history_path, encoding="utf-8-sig")
                nav = pd.to_numeric(history.get("nav"), errors="coerce").dropna()
                daily_returns = nav.pct_change().dropna()
                if len(nav) >= 20 and not daily_returns.empty:
                    elapsed_years = max(len(daily_returns) / 252.0, 1 / 252.0)
                    expected_return = ((nav.iloc[-1] / nav.iloc[0]) ** (1 / elapsed_years) - 1) * 100
                    expected_volatility = daily_returns.std(ddof=1) * np.sqrt(252) * 100
            except (OSError, ValueError, TypeError):
                pass
        elif item.get("seed_return_1y") is not None:
            expected_return = float(item["seed_return_1y"]) * 100

        samples.append({
            "名稱/代號": f"{name} ({code})",
            "Ticker": code,
            "預期年化報酬率 (%)": round(float(np.clip(expected_return, -10, 40)), 2),
            "預期年化波動度 (%)": round(float(np.clip(expected_volatility, 0.1, 80)), 2),
            "來源分類": category_names.get(item.get("category", ""), "未分類"),
            "樣本來源": "基金篩選器網站同源資料",
        })

    return samples, ""


@st.cache_data(ttl=1800, show_spinner=False)
def load_real_bonds(selected_sources, show_all=False):
    """合併合庫與 MoneyDJ 公開債券資料，按 ISIN 去重且不補造 CDS。"""
    showall = "1" if show_all else "0"
    source_configs = {
        "合作金庫 MoneyDJ": {
            "url": (
                "https://tcbbankfund.moneydj.com/JsonData/bond/"
                f"bonddata.xdjjson?aspid=tcb&showall={showall}&x=search02"
            ),
            "ytm_fields": ("V52", "V13"),
            "price_fields": ("V51", "V28"),
        },
        "MoneyDJ 獨立資料": {
            "url": (
                "https://www.moneydj.com/JsonData/bond/"
                "bonddata.xdjjson?aspid=&showall=1&x=search02"
            ),
            "ytm_fields": ("V13", "V52"),
            "price_fields": ("V28", "V51"),
        },
    }
    rows = []
    refresh_times = []
    source_errors = []
    recognized_ratings = {
        'AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-',
        'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-',
        'B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-', 'CC', 'C', 'D',
    }

    def first_number(record, fields):
        for field in fields:
            value = pd.to_numeric(record.get(field), errors="coerce")
            if pd.notna(value):
                return value
        return np.nan

    for source_name in selected_sources:
        config = source_configs.get(source_name)
        if not config:
            continue
        try:
            response = subprocess.run(
                ["curl", "-L", "--fail", "--silent", "--show-error", "--max-time", "30", config["url"]],
                check=True,
                capture_output=True,
                text=True,
                timeout=35,
            )
            result_set = json.loads(response.stdout).get("ResultSet", {})
            records = result_set.get("Result", [])
            refresh_time = str(result_set.get("ExpireTime") or "").strip()
            if refresh_time:
                refresh_times.append(f"{source_name}：{refresh_time}")
            for record in records:
                price = first_number(record, config["price_fields"])
                ytm = first_number(record, config["ytm_fields"])
                coupon = pd.to_numeric(record.get("V20"), errors="coerce")
                years = pd.to_numeric(record.get("V18"), errors="coerce")
                rating = str(record.get("V34") or "").strip().upper()
                if rating not in recognized_ratings:
                    rating = "未評等"
                isin = str(record.get("V1") or "").strip()
                bond_code = str(record.get("V33") or isin).strip()
                bond_name = str(record.get("V2") or record.get("V3") or bond_code).strip()
                rows.append({
                    "債券名稱/發行機構 (ISIN)": f"{bond_code} {bond_name}（{isin}）",
                    "Ticker": bond_code,
                    "ISIN": isin,
                    "票面利率 (%)": coupon,
                    "參考淨價": price,
                    "到期殖利率 (YTM)": ytm,
                    "年化波動度 (%)": np.nan,
                    "CDS 利差 (bps)": np.nan,
                    "剩餘年限 (年)": years,
                    "信用評等": rating,
                    "幣別": str(record.get("V11") or record.get("V10") or "").strip(),
                    "產業": str(record.get("V7") or record.get("V9") or "未分類").strip(),
                    "發行人/國家": str(record.get("V5") or record.get("V6") or "").strip(),
                    "價格日期": str(record.get("V27") or "").strip(),
                    "是否可下單": "是" if record.get("V37") == "Y" else "否",
                    "投資人類型": str(record.get("V41") or "").strip(),
                    "資料來源": source_name,
                })
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
            source_errors.append(f"{source_name}：{exc}")

    if not rows:
        message = "；".join(source_errors) or "所選公開資料來源沒有回傳債券資料。"
        return pd.DataFrame(), "", message

    frame = pd.DataFrame(rows)
    frame["完整度"] = frame[["參考淨價", "到期殖利率 (YTM)", "剩餘年限 (年)", "票面利率 (%)"]].notna().sum(axis=1)
    frame = frame.sort_values(["ISIN", "價格日期", "完整度"], ascending=[True, False, False])
    source_map = frame.groupby("ISIN")["資料來源"].agg(lambda values: "、".join(dict.fromkeys(values)))
    frame = frame.drop_duplicates("ISIN", keep="first")
    frame["資料來源"] = frame["ISIN"].map(source_map)
    frame = frame.drop(columns="完整度")
    frame = frame.dropna(subset=["參考淨價", "到期殖利率 (YTM)", "剩餘年限 (年)"])
    warning = "；".join(source_errors)
    return frame, "｜".join(refresh_times), warning

# 基金池管理固定放在側邊欄，不受債券篩選結果影響。
with st.sidebar.expander("➕ 加入基金／ETF池", expanded=True):
    with st.form("sidebar_add_fund_form", clear_on_submit=True):
        sidebar_fund_name = st.text_input(
            "基金／ETF名稱",
            placeholder="例如：Vanguard Total World Stock ETF",
            key="sidebar_fund_name",
        )
        sidebar_fund_ticker = st.text_input(
            "交易代號／基金代碼",
            placeholder="例如：VT",
            key="sidebar_fund_ticker",
        ).upper().strip()
        sidebar_fund_return = st.number_input(
            "預期年化報酬率 (%)",
            min_value=-10.0,
            max_value=40.0,
            value=8.0,
            step=0.5,
            key="sidebar_fund_return",
        )
        sidebar_fund_vol = st.number_input(
            "預期年化波動度 (%)",
            min_value=0.1,
            max_value=80.0,
            value=12.0,
            step=0.5,
            key="sidebar_fund_vol",
        )
        sidebar_add_fund = st.form_submit_button(
            "確認加入基金池",
            width="stretch",
            type="primary",
        )

    if sidebar_add_fund:
        existing_tickers = {
            str(f.get("Ticker", "")).upper().strip()
            for f in st.session_state.custom_funds
        }
        if not sidebar_fund_ticker:
            st.error("請先輸入交易代號或基金代碼。")
        elif sidebar_fund_ticker in existing_tickers:
            st.warning(f"{sidebar_fund_ticker} 已在基金池內。")
        else:
            display_name = sidebar_fund_name.strip() or sidebar_fund_ticker
            st.session_state.custom_funds.append({
                "名稱/代號": f"{display_name} ({sidebar_fund_ticker})",
                "Ticker": sidebar_fund_ticker,
                "預期年化報酬率 (%)": float(sidebar_fund_return),
                "預期年化波動度 (%)": float(sidebar_fund_vol),
            })
            st.session_state["fund_add_message"] = (
                f"✅ 已將 {display_name}（{sidebar_fund_ticker}）加入基金池。"
            )
            st.rerun()

    st.caption(f"目前基金池共有 {len(st.session_state.custom_funds)} 檔")
    if st.session_state.get("fund_add_message"):
        st.success(st.session_state.pop("fund_add_message"))

    st.markdown("---")
    st.markdown("##### 🌐 匯入基金篩選器樣本")
    st.caption("來源為您既有基金篩選器網站的同源資料庫；歷史淨值足夠時會重新估算報酬與波動度。")
    if st.button("從基金篩選器載入樣本", width="stretch", key="import_fund_screener_samples"):
        imported_samples, import_error = load_fund_screener_samples()
        if import_error:
            st.error(import_error)
        else:
            existing_tickers = {
                str(f.get("Ticker", "")).upper().strip()
                for f in st.session_state.custom_funds
            }
            new_samples = [
                sample for sample in imported_samples
                if sample["Ticker"].upper().strip() not in existing_tickers
            ]
            st.session_state.custom_funds.extend(new_samples)
            st.session_state["fund_import_message"] = (
                f"✅ 已匯入 {len(new_samples)} 檔；略過 {len(imported_samples) - len(new_samples)} 檔重複樣本。"
            )
            st.rerun()

    if st.session_state.get("fund_import_message"):
        st.success(st.session_state.pop("fund_import_message"))
    st.link_button("開啟原基金篩選器網站", FUND_SAMPLE_SITE, width="stretch")

# =========================================================================
# 1. 側邊欄基礎條件篩選
# =========================================================================
st.sidebar.header("🔍 債券一級市場篩選")
yield_range = st.sidebar.slider('到期殖利率 YTM 區間 (%)', 0.0, 15.0, (0.0, 15.0), step=0.1)
maturity_range = st.sidebar.slider('剩餘到期年限 (年)', 1, 30, (1, 30), step=1)
price_type_filter = st.sidebar.multiselect(
    '價格型態（相對面額 100）',
    options=['溢價債券', '平價債券', '折價債券'],
    default=['溢價債券', '平價債券', '折價債券'],
)
rating_filter = st.sidebar.multiselect(
    '信用評等',
    options=['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-', 'CC', 'C', 'D', '未評等'],
    default=['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-', 'CC', 'C', 'D', '未評等'],
)
bond_category_filter = st.sidebar.multiselect(
    '債券類別（可複選）',
    options=['投資等級債', '非投資等級債', '新興市場債', '未評等債'],
    default=['投資等級債', '非投資等級債', '新興市場債', '未評等債'],
    help='新興市場債可同時帶有投資等級或非投資等級標籤。',
)
issuer_keyword = st.sidebar.text_input('發行人關鍵字', placeholder='例如：Apple、銀行、Toyota')
bond_data_sources = st.sidebar.multiselect(
    '真實債券資料來源',
    options=['合作金庫 MoneyDJ', 'MoneyDJ 獨立資料'],
    default=['合作金庫 MoneyDJ', 'MoneyDJ 獨立資料'],
    help='同時選取時會按 ISIN 合併去重，優先保留日期較新且欄位較完整的資料。',
)
show_all_real_bonds = st.sidebar.toggle(
    '合庫來源載入完整清單',
    value=False,
    help='僅影響合庫來源：關閉時載入主要清單，開啟時讀取完整公開紀錄；無價格或殖利率的紀錄不會納入模型。',
)

# 💡 全資產底層本地資料庫
bonds_list = [
    {"債券名稱/發行機構 (ISIN)": "蘋果公司 Apple 美元債（示意）", "Ticker": "AAPL", "票面利率 (%)": 4.85, "參考淨價": 102.40, "到期殖利率 (YTM)": 4.25, "年化波動度 (%)": 3.2, "CDS 利差 (bps)": 25, "剩餘年限 (年)": 3, "信用評等": "AA+", "幣別": "USD", "產業": "科技"},
    {"債券名稱/發行機構 (ISIN)": "微軟 Microsoft 美元債（示意）", "Ticker": "MSFT", "票面利率 (%)": 4.20, "參考淨價": 100.10, "到期殖利率 (YTM)": 4.18, "年化波動度 (%)": 2.5, "CDS 利差 (bps)": 15, "剩餘年限 (年)": 5, "信用評等": "AAA", "幣別": "USD", "產業": "科技"},
    {"債券名稱/發行機構 (ISIN)": "亞馬遜 Amazon 美元債（示意）", "Ticker": "AMZN", "票面利率 (%)": 4.55, "參考淨價": 98.60, "到期殖利率 (YTM)": 4.82, "年化波動度 (%)": 4.1, "CDS 利差 (bps)": 35, "剩餘年限 (年)": 7, "信用評等": "AA", "幣別": "USD", "產業": "科技"},
    {"債券名稱/發行機構 (ISIN)": "Alphabet 美元債（示意）", "Ticker": "GOOGL", "票面利率 (%)": 4.90, "參考淨價": 103.20, "到期殖利率 (YTM)": 4.35, "年化波動度 (%)": 3.4, "CDS 利差 (bps)": 22, "剩餘年限 (年)": 6, "信用評等": "AA+", "幣別": "USD", "產業": "科技"},
    {"債券名稱/發行機構 (ISIN)": "Meta Platforms 美元債（示意）", "Ticker": "META", "票面利率 (%)": 4.45, "參考淨價": 97.80, "到期殖利率 (YTM)": 5.05, "年化波動度 (%)": 5.5, "CDS 利差 (bps)": 65, "剩餘年限 (年)": 4, "信用評等": "A+", "幣別": "USD", "產業": "科技"},
    {"債券名稱/發行機構 (ISIN)": "英特爾 Intel 美元債（示意）", "Ticker": "INTC", "票面利率 (%)": 5.20, "參考淨價": 94.50, "到期殖利率 (YTM)": 6.35, "年化波動度 (%)": 11.2, "CDS 利差 (bps)": 210, "剩餘年限 (年)": 2, "信用評等": "BBB", "幣別": "USD", "產業": "科技"},
    {"債券名稱/發行機構 (ISIN)": "輝達 NVIDIA 美元債（示意）", "Ticker": "NVDA", "票面利率 (%)": 3.70, "參考淨價": 96.20, "到期殖利率 (YTM)": 4.60, "年化波動度 (%)": 4.8, "CDS 利差 (bps)": 30, "剩餘年限 (年)": 5, "信用評等": "A+", "幣別": "USD", "產業": "科技"},
    {"債券名稱/發行機構 (ISIN)": "摩根大通 JPMorgan 美元債（示意）", "Ticker": "JPM", "票面利率 (%)": 5.35, "參考淨價": 102.80, "到期殖利率 (YTM)": 4.88, "年化波動度 (%)": 4.0, "CDS 利差 (bps)": 48, "剩餘年限 (年)": 6, "信用評等": "A", "幣別": "USD", "產業": "金融"},
    {"債券名稱/發行機構 (ISIN)": "美國銀行 Bank of America 美元債（示意）", "Ticker": "BAC", "票面利率 (%)": 4.10, "參考淨價": 97.10, "到期殖利率 (YTM)": 4.75, "年化波動度 (%)": 4.4, "CDS 利差 (bps)": 58, "剩餘年限 (年)": 5, "信用評等": "A-", "幣別": "USD", "產業": "金融"},
    {"債券名稱/發行機構 (ISIN)": "高盛 Goldman Sachs 美元債（示意）", "Ticker": "GS", "票面利率 (%)": 5.95, "參考淨價": 104.20, "到期殖利率 (YTM)": 5.22, "年化波動度 (%)": 5.0, "CDS 利差 (bps)": 72, "剩餘年限 (年)": 8, "信用評等": "A", "幣別": "USD", "產業": "金融"},
    {"債券名稱/發行機構 (ISIN)": "可口可樂 Coca-Cola 美元債（示意）", "Ticker": "KO", "票面利率 (%)": 4.75, "參考淨價": 101.60, "到期殖利率 (YTM)": 4.52, "年化波動度 (%)": 2.9, "CDS 利差 (bps)": 32, "剩餘年限 (年)": 7, "信用評等": "A+", "幣別": "USD", "產業": "消費"},
    {"債券名稱/發行機構 (ISIN)": "沃爾瑪 Walmart 美元債（示意）", "Ticker": "WMT", "票面利率 (%)": 3.95, "參考淨價": 98.90, "到期殖利率 (YTM)": 4.15, "年化波動度 (%)": 2.6, "CDS 利差 (bps)": 28, "剩餘年限 (年)": 6, "信用評等": "AA", "幣別": "USD", "產業": "消費"},
    {"債券名稱/發行機構 (ISIN)": "埃克森美孚 Exxon Mobil 美元債（示意）", "Ticker": "XOM", "票面利率 (%)": 4.90, "參考淨價": 100.00, "到期殖利率 (YTM)": 4.90, "年化波動度 (%)": 4.2, "CDS 利差 (bps)": 40, "剩餘年限 (年)": 10, "信用評等": "AA-", "幣別": "USD", "產業": "能源"},
    {"債券名稱/發行機構 (ISIN)": "雪佛龍 Chevron 美元債（示意）", "Ticker": "CVX", "票面利率 (%)": 5.10, "參考淨價": 103.70, "到期殖利率 (YTM)": 4.62, "年化波動度 (%)": 4.1, "CDS 利差 (bps)": 38, "剩餘年限 (年)": 9, "信用評等": "AA-", "幣別": "USD", "產業": "能源"},
    {"債券名稱/發行機構 (ISIN)": "威訊 Verizon 美元債（示意）", "Ticker": "VZ", "票面利率 (%)": 4.35, "參考淨價": 92.80, "到期殖利率 (YTM)": 5.30, "年化波動度 (%)": 6.0, "CDS 利差 (bps)": 95, "剩餘年限 (年)": 11, "信用評等": "BBB+", "幣別": "USD", "產業": "電信"},
    {"債券名稱/發行機構 (ISIN)": "AT&T 美元債（示意）", "Ticker": "T", "票面利率 (%)": 5.40, "參考淨價": 99.20, "到期殖利率 (YTM)": 5.52, "年化波動度 (%)": 6.5, "CDS 利差 (bps)": 110, "剩餘年限 (年)": 12, "信用評等": "BBB", "幣別": "USD", "產業": "電信"},
    {"債券名稱/發行機構 (ISIN)": "豐田汽車 Toyota 美元債（示意）", "Ticker": "TM", "票面利率 (%)": 4.60, "參考淨價": 101.10, "到期殖利率 (YTM)": 4.38, "年化波動度 (%)": 3.3, "CDS 利差 (bps)": 36, "剩餘年限 (年)": 5, "信用評等": "A+", "幣別": "USD", "產業": "汽車"},
    {"債券名稱/發行機構 (ISIN)": "軟銀集團 SoftBank 美元債（示意）", "Ticker": "9984.T", "票面利率 (%)": 6.75, "參考淨價": 95.40, "到期殖利率 (YTM)": 7.65, "年化波動度 (%)": 9.8, "CDS 利差 (bps)": 285, "剩餘年限 (年)": 6, "信用評等": "BBB-", "幣別": "USD", "產業": "控股"},
    {"債券名稱/發行機構 (ISIN)": "台積電 TSMC 美元債（示意）", "Ticker": "TSM", "票面利率 (%)": 4.75, "參考淨價": 102.10, "到期殖利率 (YTM)": 4.31, "年化波動度 (%)": 3.1, "CDS 利差 (bps)": 24, "剩餘年限 (年)": 7, "信用評等": "AA-", "幣別": "USD", "產業": "半導體"},
    {"債券名稱/發行機構 (ISIN)": "三星電子 Samsung 美元債（示意）", "Ticker": "005930.KS", "票面利率 (%)": 4.25, "參考淨價": 99.80, "到期殖利率 (YTM)": 4.29, "年化波動度 (%)": 3.5, "CDS 利差 (bps)": 34, "剩餘年限 (年)": 4, "信用評等": "AA-", "幣別": "USD", "產業": "半導體"}
]

# 再建立 20 家發行人、每家 24 個不同到期系列，共 480 檔；連同上方 20 檔合計 500 檔。
# 所有數值均為介面與模型測試用的固定示意資料，不冒充券商即時報價。
generated_issuers = [
    ("思科 Cisco", "CSCO", "科技", "A+", 42, 4.55, 3.6),
    ("甲骨文 Oracle", "ORCL", "科技", "BBB", 78, 5.05, 4.8),
    ("IBM", "IBM", "科技", "A-", 64, 4.95, 4.4),
    ("博通 Broadcom", "AVGO", "科技", "BBB", 82, 5.15, 5.2),
    ("Visa", "V", "金融", "AA-", 31, 4.40, 3.0),
    ("Mastercard", "MA", "金融", "A+", 34, 4.45, 3.1),
    ("摩根士丹利 Morgan Stanley", "MS", "金融", "A-", 68, 5.10, 4.7),
    ("花旗 Citigroup", "C", "金融", "A-", 76, 5.25, 5.0),
    ("百事 PepsiCo", "PEP", "消費", "A+", 33, 4.50, 2.9),
    ("麥當勞 McDonald's", "MCD", "消費", "BBB+", 51, 4.80, 3.8),
    ("寶僑 P&G", "PG", "消費", "AA-", 29, 4.35, 2.7),
    ("迪士尼 Disney", "DIS", "媒體", "A-", 62, 5.00, 4.6),
    ("波音 Boeing", "BA", "工業", "BBB-", 245, 7.10, 9.5),
    ("開拓重工 Caterpillar", "CAT", "工業", "A", 45, 4.70, 3.7),
    ("福特 Ford", "F", "汽車", "BBB-", 195, 6.65, 8.1),
    ("通用汽車 GM", "GM", "汽車", "BBB", 145, 6.05, 7.0),
    ("殼牌 Shell", "SHEL", "能源", "A+", 41, 4.65, 3.5),
    ("道達爾能源 TotalEnergies", "TTE", "能源", "A+", 44, 4.72, 3.6),
    ("本田 Honda", "HMC", "汽車", "A", 49, 4.85, 3.9),
    ("索尼 Sony", "SONY", "科技", "A", 47, 4.78, 3.8),
]
# 2028–2051 共 24 個系列，剩餘年限 2–25 年；價格循環涵蓋溢價、平價與折價。
series_specs = []
price_cycle = [104.20, 102.60, 100.00, 98.40, 95.80, 92.50]
for series_index in range(24):
    years = series_index + 2
    series = str(2026 + years)
    clean_price = price_cycle[series_index % len(price_cycle)]
    term_spread = 0.12 + series_index * 0.065
    series_specs.append((series, years, clean_price, term_spread))

for issuer_index, (issuer, ticker, industry, rating, cds, base_yield, base_vol) in enumerate(generated_issuers):
    for series_index, (series, years, clean_price, term_spread) in enumerate(series_specs):
        ytm = base_yield + term_spread + (issuer_index % 3) * 0.04
        coupon_adjustment = [0.75, 0.35, 0.00, -0.25, -0.55, -0.90][series_index % 6]
        coupon = max(1.0, ytm + coupon_adjustment)
        bonds_list.append({
            "債券名稱/發行機構 (ISIN)": f"{issuer} {series} 美元債（示意）",
            "Ticker": ticker,
            "票面利率 (%)": round(coupon, 2),
            "參考淨價": round(clean_price - (issuer_index % 4) * 0.12, 2),
            "到期殖利率 (YTM)": round(ytm, 2),
            "年化波動度 (%)": round(base_vol + series_index * 0.55, 2),
            "CDS 利差 (bps)": cds + series_index * 4,
            "剩餘年限 (年)": years,
            "信用評等": rating,
            "幣別": "USD",
            "產業": industry,
        })

df_mock, real_bond_expire_time, real_bond_error = load_real_bonds(tuple(bond_data_sources), show_all_real_bonds)
if real_bond_error:
    if df_mock.empty:
        st.error(real_bond_error)
        st.info('為避免把示意值誤認為真實行情，目前不自動回退至示意債券。請稍後重新整理。')
        st.stop()
    st.warning(f'部分來源讀取失敗，但已保留其他來源的真實資料：{real_bond_error}')

investment_grade_ratings = {'AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-'}
non_investment_grade_ratings = {
    'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-', 'CC', 'C', 'D'
}
emerging_market_keywords = (
    '中國', '香港', '台灣', '印度', '印尼', '印度尼西亞', '泰國', '越南', '菲律賓',
    '馬來西亞', '巴西', '墨西哥', '智利', '哥倫比亞', '秘魯', '阿根廷',
    '南非', '土耳其', '沙烏地', '阿聯', '卡達', '阿曼', '巴林', '埃及',
    '波蘭', '匈牙利', '羅馬尼亞', '哈薩克', '巴基斯坦', '斯里蘭卡',
)


def classify_bond(row):
    tags = []
    rating = str(row.get('信用評等', '')).strip().upper()
    if rating in investment_grade_ratings:
        tags.append('投資等級債')
    elif rating in non_investment_grade_ratings:
        tags.append('非投資等級債')
    else:
        tags.append('未評等債')
    issuer_country = str(row.get('發行人/國家', ''))
    if any(keyword in issuer_country for keyword in emerging_market_keywords):
        tags.append('新興市場債')
    return '｜'.join(tags)


df_mock['債券分類'] = df_mock.apply(classify_bond, axis=1)

df_mock['價格型態'] = np.select(
    [df_mock['參考淨價'] > 100.25, df_mock['參考淨價'] < 99.75],
    ['溢價債券', '折價債券'],
    default='平價債券',
)
df_mock['資產類別'] = '海外債券'
df_mock['CDS風險警示'] = '⚪ 來源未提供'

# 依完整債券池動態建立產業選單；新增加的產業也會自動出現在選項中。
industry_options = sorted(df_mock['產業'].dropna().astype(str).unique().tolist())
industry_filter = st.sidebar.multiselect(
    '產業（可複選）',
    options=industry_options,
    default=industry_options,
    help='債券列表與後續投組標的會同步套用此產業條件。',
)

# 修正範圍滑桿解包
yield_min, yield_max = yield_range
maturity_min, maturity_max = maturity_range
filtered_df = df_mock[
    (df_mock['到期殖利率 (YTM)'] >= yield_min) & (df_mock['到期殖利率 (YTM)'] <= yield_max) &
    (df_mock['剩餘年限 (年)'] >= maturity_min) & (df_mock['剩餘年限 (年)'] <= maturity_max) &
    (df_mock['價格型態'].isin(price_type_filter)) &
    (df_mock['信用評等'].isin(rating_filter)) &
    (df_mock['產業'].isin(industry_filter)) &
    (df_mock['債券分類'].apply(lambda value: any(category in value for category in bond_category_filter)))
]
if issuer_keyword.strip():
    filtered_df = filtered_df[
        filtered_df['債券名稱/發行機構 (ISIN)'].str.contains(issuer_keyword.strip(), case=False, na=False)
    ]

st.subheader('📊 候選海外債券列表')
st.caption(
    f'真實債券池共 {len(df_mock):,} 檔，資料來源：{"、".join(bond_data_sources)}；'
    f'資料時間：{real_bond_expire_time or "來源未標示"}。價格為網站公開參考價，非保證成交價；'
    '溢折價以申購參考價相對面額100判定。'
)
if not filtered_df.empty:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('符合條件', f'{len(filtered_df)} 檔')
    c2.metric('溢價債', f"{filtered_df['價格型態'].eq('溢價債券').sum()} 檔")
    c3.metric('平價債', f"{filtered_df['價格型態'].eq('平價債券').sum()} 檔")
    c4.metric('折價債', f"{filtered_df['價格型態'].eq('折價債券').sum()} 檔")
    c5.metric('CDS 資料狀態', '來源未提供')
    grade1, grade2, grade3 = st.columns(3)
    grade1.metric('🏦 投資等級債', f"{filtered_df['債券分類'].apply(lambda value: '投資等級債' in value.split('｜')).sum():,} 檔")
    grade2.metric('⚠️ 非投資等級債', f"{filtered_df['債券分類'].apply(lambda value: '非投資等級債' in value.split('｜')).sum():,} 檔")
    grade3.metric('🌏 新興市場債', f"{filtered_df['債券分類'].apply(lambda value: '新興市場債' in value.split('｜')).sum():,} 檔")
    st.caption('此公開債券清單沒有 CDS 欄位，因此不以其他數值冒充 CDS。未來接上可靠 CDS 來源後，才會依 100／200 bps 門檻顯示橘色與紅色警示。')
    display_columns = ['債券名稱/發行機構 (ISIN)', '發行人/國家', '產業', '債券分類', '信用評等', '幣別', '票面利率 (%)', '參考淨價', '價格型態', '到期殖利率 (YTM)', '剩餘年限 (年)', '價格日期', '是否可下單', '投資人類型', '資料來源', 'CDS風險警示']

    def highlight_cds_warning(row):
        color = ''
        if row['CDS風險警示'] == '🔴 高風險':
            color = 'background-color: rgba(255, 75, 75, 0.28); color: #ffdddd; font-weight: 700;'
        elif row['CDS風險警示'] == '🟠 注意':
            color = 'background-color: rgba(255, 165, 0, 0.20); color: #ffe4b5;'
        return [color] * len(row)

    styled_bonds = filtered_df[display_columns].style.apply(highlight_cds_warning, axis=1)
    st.dataframe(styled_bonds, width='stretch', hide_index=True)
    
    # 資料流同步快取
    current_asset_details = {}
    all_asset_options = []
    for _, row in filtered_df.iterrows():
        name = f"【債券】{row['債券名稱/發行機構 (ISIN)']}"
        all_asset_options.append(name)
        duration_risk_proxy = float(np.clip(float(row['剩餘年限 (年)']) * 0.7, 2.0, 25.0))
        current_asset_details[name] = {
            "return": row['到期殖利率 (YTM)'],
            "vol": duration_risk_proxy,
            "cds": 0,
            "ticker": row['Ticker'],
            "type": "海外債券",
        }
    for f in st.session_state.custom_funds:
        name = f"【基金/ETF】{f['名稱/代號']}"
        all_asset_options.append(name)
        current_asset_details[name] = {"return": f['預期年化報酬率 (%)'], "vol": f['預期年化波動度 (%)'], "cds": 0, "ticker": f['Ticker'], "type": "基金/ETF"}
    st.session_state.cached_asset_details.update(current_asset_details)

    st.markdown("---")
    # =========================================================================
    # ⚙️ 2. 三合一中控分頁版型 (完美融合所有核心功能)
    # =========================================================================
    tab1, tab2, tab3 = st.tabs([
        "✍️ 第一步：選擇資產與設定目標", 
        "📊 第二步：智慧推薦與視覺化清單", 
        "🚨 第三步：台股估值與日圓平倉即時診斷"
    ])
    
    with tab1:
        col_panel1, col_panel2 = st.columns([1, 1.5])
        with col_panel1:
            st.markdown("##### ➕ 擴充共同基金/ETF池")
            with st.form("add_fund_form", clear_on_submit=True):
                fund_name = st.text_input("基金／ETF名稱", placeholder="例如：Vanguard Total World Stock ETF")
                fund_ticker = st.text_input("交易代號", placeholder="例如：VT").upper().strip()
                fund_return = st.number_input("預期年化報酬率 (%)", min_value=-10.0, max_value=40.0, value=8.0, step=0.5)
                fund_vol = st.number_input("預期年化波動度 (%)", min_value=0.1, max_value=80.0, value=12.0, step=0.5)
                add_fund = st.form_submit_button("確認加入基金池", width="stretch", type="primary")

            if add_fund:
                existing_tickers = {str(f.get('Ticker', '')).upper() for f in st.session_state.custom_funds}
                if not fund_ticker:
                    st.error("請先輸入基金或 ETF 的交易代號。")
                elif fund_ticker in existing_tickers:
                    st.warning(f"{fund_ticker} 已經在基金池內，不會重複新增。")
                else:
                    display_name = fund_name.strip() or f"{fund_ticker} 市場基金／ETF"
                    st.session_state.custom_funds.append({
                        "名稱/代號": f"{display_name} ({fund_ticker})",
                        "Ticker": fund_ticker,
                        "預期年化報酬率 (%)": float(fund_return),
                        "預期年化波動度 (%)": float(fund_vol),
                    })
                    st.session_state["fund_add_message"] = f"✅ 已將 {display_name}（{fund_ticker}）加入基金池。"
                    st.rerun()

            if st.session_state.get("fund_add_message"):
                st.success(st.session_state.pop("fund_add_message"))

            fund_pool_df = pd.DataFrame(st.session_state.custom_funds)
            st.caption(f"目前基金池：{len(fund_pool_df)} 檔")
            st.dataframe(fund_pool_df, width="stretch", hide_index=True, height=210)
        with col_panel2:
            st.markdown("##### 🎯 勾選本次要優化的組合標的")
            selected_assets = st.multiselect("可同時複選海外債券與基金：", options=all_asset_options, default=all_asset_options[:3] if len(all_asset_options)>=3 else all_asset_options)
            if selected_assets:
                rets = np.array([st.session_state.cached_asset_details[a]["return"] for a in selected_assets])
                vols = np.array([st.session_state.cached_asset_details[a]["vol"] for a in selected_assets])
                cdss = np.array([st.session_state.cached_asset_details[a]["cds"] for a in selected_assets])
                tickers = [st.session_state.cached_asset_details[a]["ticker"] for a in selected_assets]
                types = [st.session_state.cached_asset_details[a]["type"] for a in selected_assets]
                
                min_p, max_p = float(np.min(rets)), float(np.max(rets))
                st.caption(f"當前已勾選資產回報範圍：{min_p:.2f}% ~ {max_p:.2f}%")
                target_return = st.number_input("期望年化回報率 (%)", min_value=min_p, max_value=max_p, value=round((min_p+max_p)/2, 2), step=0.01)
                st.success("🎉 參數設定完畢！請點擊切換到「第二步」查看比例清單，或到「第三步」執行台股/日圓大診斷。")

    # 全域背景智慧分配權重計算
    if selected_assets and len(selected_assets) >= 1:
        distances = np.abs(rets - target_return)
        inv_dist = 1.0 / (distances + 1e-5)
        risk_penalty = 1.0 / (vols + 1e-5)
        cds_penalty = np.where(cdss > 0, 100.0 / (cdss + 1e-5), 1.0)
        combined_scores = inv_dist * risk_penalty * cds_penalty
        best_weights = combined_scores / np.sum(combined_scores)

    with tab2:
        if selected_assets:
            portfolio_return = np.dot(best_weights, rets)
            portfolio_volatility = np.dot(best_weights, vols)
            portfolio_cds = np.dot(best_weights, cdss)
            sharpe = (portfolio_return - 4.0) / portfolio_volatility if portfolio_volatility > 0 else 0

            st.markdown("### 🏆 系統自動調配之最佳黃金資產配置方案")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1: st.metric("🎯 投組預期年化報酬", f"{portfolio_return:.2f} %")
            with col_m2: st.metric("⚡ 投組年化波動度 (風險)", f"{portfolio_volatility:.2f} %")
            with col_m3: st.metric("🛡️ 投組平均 CDS 利差", f"{portfolio_cds:.2f} bps")
            with col_m4: st.metric("📊 綜合夏普值 (Sharpe)", f"{sharpe:.2f}")

            st.markdown("---")
            col_graph1, col_graph2 = st.columns(2)
            with col_graph1:
                st.write("📋 **建議分配黃金比例清單 (一體化初始化表格)**")
                result_df = pd.DataFrame({
                    "資產名稱": selected_assets,
                    "建議配置比例": [f"{w*100:.2f} %" for w in best_weights],
                    "資產大類": types,
                    "對沖風險 (CDS)": [f"{int(c)} bps" if c>0 else "無違約風險" for c in cdss],
                    "單體預期年化回報": [f"{r:.2f} %" for r in rets]
                })
                st.dataframe(result_df, width='stretch', height=280)
            with col_graph2:
                st.write("📊 **各資產投資權重配置直方圖**")
                chart_data = pd.DataFrame({"分配權重比例 (%)": np.round(best_weights * 100, 2)}, index=[a[:18]+"..." for a in selected_assets])
                st.bar_chart(chart_data, height=280)
        else:
            st.info("💡 請先到第一個分頁選取資產標的。")

    with tab3:
        st.markdown("### 🛡️ 聯網原生物理級風控：台股多空本質與日圓平倉壓力評估")
        st.write("點擊下方大按鈕，網頁將直接調用 Python 內建矩陣代數核心，即時下載台積電 ADR (`TSM`)、美股大盤 (`SPY`) 與美元/日圓匯率價格，為亞洲市場執行全自動診斷：")
        
        if st.button("🚀 啟動即時金融大診斷 (分析台股殺估值與日圓利差潮)", width='stretch', type="primary"):
            with st.spinner("⏳ 正在串接全球金融序列並進行 NumPy 矩陣 Least-Squares 迴歸運算..."):
                try:
                    # 聯網安全獲取數據
                    risk_tickers = ["TSM", "SPY", "JPY=X"]
                    raw_risk = yf.download(risk_tickers, period="1y", progress=False)
                    
                    # 相容新版 yfinance：auto_adjust=True 時通常只有 Close，
                    # 多代號下載則以 MultiIndex（價格欄位、Ticker）回傳。
                    if isinstance(raw_risk.columns, pd.MultiIndex):
                        price_fields = raw_risk.columns.get_level_values(0)
                        price_field = "Adj Close" if "Adj Close" in price_fields else "Close"
                        data_table = raw_risk.xs(price_field, axis=1, level=0)
                    else:
                        data_table = raw_risk["Adj Close"] if "Adj Close" in raw_risk.columns else raw_risk["Close"]

                    if isinstance(data_table, pd.Series):
                        data_table = data_table.to_frame()
                    
                    # 純粹提取對齊
                    risk_df = pd.DataFrame()
                    for t in risk_tickers:
                        if t in data_table.columns: risk_df[t] = data_table[t]
                    risk_df = risk_df.dropna()
                    missing_tickers = [t for t in risk_tickers if t not in risk_df.columns]
                    if missing_tickers:
                        raise RuntimeError(f"缺少行情欄位：{', '.join(missing_tickers)}")
                    if len(risk_df) < 30:
                        raise RuntimeError("可用歷史資料不足 30 筆")
                    risk_returns = risk_df.pct_change().dropna()
                    
                    # 1. 執行純矩陣 Least-Squares 線性回歸 (診斷殺估值)
                    spy_p = risk_df["SPY"].values
                    tsm_p = risk_df["TSM"].values
                    A_mat = np.vstack([np.ones(len(spy_p)), spy_p]).T
                    
                    # 完美修復：利用 *others 解包接收 NumPy 剩餘參數，徹底消滅 ValueError
                    (b0, b1), *others = np.linalg.lstsq(A_mat, tsm_p, rcond=None)
                    
                    current_spy = float(spy_p[-1])
                    current_tsm = float(tsm_p[-1])

                    fair_tsm = float(b0 + b1 * current_spy)
                    valuation_gap = (current_tsm / fair_tsm - 1.0) * 100 if fair_tsm else 0.0

                    # JPY=X 為美元兌日圓；下跌代表日圓升值與利差交易平倉壓力增加。
                    usd_jpy = risk_df["JPY=X"]
                    usd_jpy_20d = (float(usd_jpy.iloc[-1]) / float(usd_jpy.iloc[-21]) - 1.0) * 100 if len(usd_jpy) > 20 else 0.0
                    tsm_spy_corr = float(risk_returns[["TSM", "SPY"]].corr().iloc[0, 1])
                    tsm_jpy_corr = float(risk_returns[["TSM", "JPY=X"]].corr().iloc[0, 1])
                    tsm_vol = float(risk_returns["TSM"].std() * np.sqrt(252) * 100)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("TSM 最新價", f"{current_tsm:.2f} USD")
                    m2.metric("模型合理價", f"{fair_tsm:.2f} USD", f"偏離 {valuation_gap:+.2f}%")
                    m3.metric("美元兌日圓 20日", f"{usd_jpy.iloc[-1]:.2f}", f"{usd_jpy_20d:+.2f}%")
                    m4.metric("TSM 年化波動", f"{tsm_vol:.2f}%")

                    diagnostic = pd.DataFrame({
                        "診斷因子": ["TSM／SPY 日報酬相關", "TSM／USDJPY 日報酬相關", "TSM 相對模型估值偏離", "USDJPY 20日變動"],
                        "數值": [tsm_spy_corr, tsm_jpy_corr, valuation_gap / 100, usd_jpy_20d / 100],
                        "判讀": [
                            "越接近 1，TSM 越受美股系統性風險影響",
                            "正值表示美元兌日圓走高時 TSM 較有利",
                            "正值偏貴、負值偏低於模型合理價",
                            "負值代表日圓升值，需留意利差交易平倉",
                        ],
                    })
                    st.dataframe(
                        diagnostic,
                        width="stretch",
                        hide_index=True,
                        column_config={"數值": st.column_config.NumberColumn(format="%.3f")},
                    )

                    risk_points = int(valuation_gap > 10) + int(usd_jpy_20d < -3) + int(tsm_vol > 35)
                    if risk_points >= 2:
                        st.error("🔴 風險偏高：估值、日圓或波動因子中至少兩項亮紅燈，建議降低槓桿並分批布局。")
                    elif risk_points == 1:
                        st.warning("🟡 風險中性偏高：已有一項壓力訊號，建議保留現金並觀察匯率與美股方向。")
                    else:
                        st.success("🟢 風險相對溫和：目前未出現多因子同步惡化，但模型結果不代表保證報酬。")

                    chart_df = risk_df / risk_df.iloc[0] * 100
                    st.line_chart(chart_df, height=360)
                    st.caption("走勢均以分析期間起點標準化為 100；資料來源為 Yahoo Finance。")
                except Exception as exc:
                    st.error(f"即時資料暫時無法完成診斷：{type(exc).__name__}。請稍後再試，或確認網路連線與 Yahoo Finance 資料服務。")
else:
    st.warning("目前沒有符合殖利率與到期年限條件的債券，請放寬左側篩選區間。")
