"""
Data_Generation.py
KOSPI + KOSDAQ 시가총액 상위 1000개 종목의 일봉 데이터를 수집하고
LSTM 입력 특성으로 변환하여 저장

일별 모델: 59개 특성, 최근 30년

출력:
  daily_features.pkl  — {종목코드: DataFrame(날짜 × 71 features)}
"""

import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from pykrx import stock
from datetime import datetime, timedelta
import time as time_module
import os
import warnings
import requests
from io import StringIO
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── 설정 ────────────────────────────────────────────────
DAILY_YEARS = 30
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DAILY_OUTPUT = os.path.join(OUTPUT_DIR, "daily_features.pkl")
EPS = 1e-8


# ═══════════════════════════════════════════════════════════
#  유틸리티
# ═══════════════════════════════════════════════════════════

def safe_log_ratio(numerator, denominator):
    """ln(a/b) — 0이나 음수를 epsilon으로 방어"""
    a = np.maximum(np.asarray(numerator, dtype=np.float64), EPS)
    b = np.maximum(np.asarray(denominator, dtype=np.float64), EPS)
    return np.log(a / b)


def compute_rsi(series, period=14):
    """RSI 계산 (SMA 방식)"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + EPS)
    return 100 - (100 / (1 + rs))


def flatten_yf_columns(df):
    """yfinance MultiIndex 컬럼 -> 단일 레벨로 변환"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ═══════════════════════════════════════════════════════════
#  종목 목록
# ═══════════════════════════════════════════════════════════

TOP_N = 1000  # 시가총액 상위 N개 종목만 사용


def get_admin_tickers():
    """KRX KIND에서 현재 관리종목 코드 목록 반환"""
    try:
        url = "https://kind.krx.co.kr/investwarn/adminissue.do"
        params = {
            "method": "searchAdminIssueSub",
            "currentPageSize": 500,
            "pageIndex": 1,
            "orderMode": 0,
            "marketType": "",
        }
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://kind.krx.co.kr/",
        }
        r = requests.get(url, params=params, headers=headers, timeout=15)
        dfs = pd.read_html(StringIO(r.text))
        admin_names = dfs[0].iloc[:, 0].str.strip().tolist()

        # 종목명 -> 코드 매핑
        today = datetime.now()
        for i in range(10):
            date_str = (today - timedelta(days=i)).strftime("%Y%m%d")
            try:
                tickers = (stock.get_market_ticker_list(date_str, "KOSPI")
                           + stock.get_market_ticker_list(date_str, "KOSDAQ"))
                if len(tickers) > 0:
                    break
            except Exception:
                continue

        name_to_ticker = {}
        for t in tickers:
            try:
                name_to_ticker[stock.get_market_ticker_name(t)] = t
            except Exception:
                pass

        admin_codes = set()
        for name in admin_names:
            if name in name_to_ticker:
                admin_codes.add(name_to_ticker[name])
        return admin_codes
    except Exception as e:
        print(f"  [경고] 관리종목 조회 실패: {e} — 필터 없이 진행")
        return set()


def get_all_tickers():
    """KOSPI + KOSDAQ 시가총액 상위 1000개 종목 코드 반환 (관리종목 제외)"""
    print(f"KOSPI + KOSDAQ 시가총액 상위 {TOP_N}개 종목 조회 중...")

    today = datetime.now()
    date_str = None
    kospi_cap = None
    kosdaq_cap = None

    for i in range(10):
        date_str = (today - timedelta(days=i)).strftime("%Y%m%d")
        try:
            kospi_cap = stock.get_market_cap(date_str, market="KOSPI")
            kosdaq_cap = stock.get_market_cap(date_str, market="KOSDAQ")
            if (len(kospi_cap) > 0 and kospi_cap.iloc[:, 1].sum() > 0
                    and len(kosdaq_cap) > 0
                    and kosdaq_cap.iloc[:, 1].sum() > 0):
                break
            kospi_cap = None
            kosdaq_cap = None
        except Exception:
            kospi_cap = None
            kosdaq_cap = None
            continue

    if kospi_cap is None or kosdaq_cap is None:
        raise RuntimeError(
            "pykrx에서 시가총액 데이터를 가져올 수 없습니다.")

    # 관리종목 제외
    print("  관리종목 조회 중...")
    admin_tickers = get_admin_tickers()
    print(f"  관리종목: {len(admin_tickers)}개 제외")

    all_cap = pd.concat([kospi_cap, kosdaq_cap])
    all_cap = all_cap[~all_cap.index.isin(admin_tickers)]
    cap_col = all_cap.columns[0]  # '시가총액' 컬럼

    # 시가총액 기준 상위 TOP_N개 종목 선택
    all_cap = all_cap.sort_values(cap_col, ascending=False)
    top_tickers = all_cap.head(TOP_N).index.tolist()

    names = {}
    for t in top_tickers:
        try:
            names[t] = stock.get_market_ticker_name(t)
        except Exception:
            names[t] = t

    print(f"  기준일: {date_str}")
    print(f"  전체: KOSPI {len(kospi_cap)}개 + KOSDAQ {len(kosdaq_cap)}개"
          f" (관리종목 {len(admin_tickers)}개 제외)")
    print(f"  시가총액 상위 {TOP_N}개 선택")
    for i, t in enumerate(top_tickers[:5]):
        cap_val = all_cap.loc[t, cap_col]
        print(f"    {i+1:2d}. {t} ({names.get(t, t)}) "
              f"— 시총 {cap_val/1e12:.1f}조원")
    print(f"    ... 외 {len(top_tickers) - 5}개")

    return top_tickers, names


# ═══════════════════════════════════════════════════════════
#  일별 데이터 생성 (71 features)
# ═══════════════════════════════════════════════════════════

DAILY_FEATURE_COLS = [
    # OHLCV 로그 변화율 (5)
    "open_logret", "high_logret", "low_logret",
    "close_logret", "volume_logret",
    # 종가 대비 (3)
    "close_vs_open", "close_vs_high", "close_vs_low",
    # HL ratio (2)
    "hl_ratio", "hl_ratio_ma20",
    # 이동평균 로그변화율 (7): 5, 10, 20, 60, 120, 240, 480일
    "ma5_logret", "ma10_logret", "ma20_logret", "ma60_logret",
    "ma120_logret", "ma240_logret", "ma480_logret",
    # 종가 대비 이동평균 (7)
    "close_vs_ma5", "close_vs_ma10", "close_vs_ma20", "close_vs_ma60",
    "close_vs_ma120", "close_vs_ma240", "close_vs_ma480",
    # RSI (1)
    "rsi_norm",
    # 외부 지표 (4)
    "kospi_logret", "kospi_vs_avg", "usdkrw_logret", "usdkrw_vs_avg",
    # 거래대금 (1)
    "turnover_logret",
    # 볼린저밴드 (1)
    "bb_position",
    # 요일 인코딩 (2)
    "dow_sin", "dow_cos",
    # 주가 스케일 (1)
    "log_price",
    # 골든 크로스 (1): 5일선 & 20일선 기울기 양수 + 5일선 상향돌파
    "golden_cross",
    # 배열 수치 (1): 정배열(+1) ~ 역배열(-1)
    "alignment_score",
    # 외국인/기관 순매수 비율 (2)
    "foreign_net_ratio", "institution_net_ratio",
    # MACD (2)
    "macd_signal", "macd_histogram",
    # 스토캐스틱 오실레이터 (2)
    "stoch_k", "stoch_d",
    # 스토캐스틱 골든크로스 (1)
    "stoch_golden_cross",
    # 거래량 비율 (1)
    "volume_ratio",
    # ATR 정규화 (1)
    "atr_norm",
    # 가격 모멘텀 (2)
    "momentum_5d", "momentum_10d",
    # 거시 지표 — KOSDAQ (2)
    "kosdaq_logret", "kosdaq_vs_avg",
    # 거시 지표 — 미국 S&P500 (2)
    "sp500_logret", "sp500_vs_avg",
    # 거시 지표 — VIX 공포지수 (1)
    "vix_norm",
    # 거시 지표 — 미국 국채 10년물 (1)
    "tnx_logret",
    # 거시 지표 — 유가 WTI (2)
    "oil_logret", "oil_vs_avg",
    # 섹터 특성 (2)
    "sector_return", "sector_momentum_5d",
    # 시장 등락비율 (1)
    "advance_ratio",
]


def compute_daily_features(stock_df, kospi_df, usdkrw_df, inv_df=None,
                           kosdaq_df=None, sp500_df=None, vix_df=None,
                           tnx_df=None, oil_df=None,
                           sector_returns=None, advance_ratio_series=None):
    """일봉 -> 특성 DataFrame 반환 (현재 58개)

    inv_df: pykrx 투자자별 순매수 거래량 DataFrame (없으면 NaN → dropna()에서 제거)
    kosdaq_df, sp500_df, vix_df, tnx_df, oil_df: yfinance OHLCV DataFrame
    sector_returns: 해당 종목 섹터의 일별 평균 수익률 Series
    advance_ratio_series: 시장 전체 상승 종목 비율 Series
    """
    df = stock_df[["Open", "High", "Low", "Close", "Volume"]].copy()
    feat = pd.DataFrame(index=df.index)

    O, H, L, C, V = (
        df["Open"], df["High"], df["Low"], df["Close"], df["Volume"])

    # ── OHLCV 로그 변화율 (5) ──
    feat["open_logret"] = safe_log_ratio(O.values, O.shift(1).values)
    feat["high_logret"] = safe_log_ratio(H.values, H.shift(1).values)
    feat["low_logret"] = safe_log_ratio(L.values, L.shift(1).values)
    feat["close_logret"] = safe_log_ratio(C.values, C.shift(1).values)
    feat["volume_logret"] = safe_log_ratio(
        (V + EPS).values, (V.shift(1) + EPS).values
    )

    # ── 종가 대비 (3) ──
    feat["close_vs_open"] = safe_log_ratio(C.values, O.values)
    feat["close_vs_high"] = safe_log_ratio(C.values, H.values)
    feat["close_vs_low"] = safe_log_ratio(C.values, L.values)

    # ── HL ratio (2) ──
    feat["hl_ratio"] = ((H - L) / C).values
    feat["hl_ratio_ma20"] = feat["hl_ratio"].rolling(20).mean()

    # ── 이동평균 (14): 5, 10, 20, 60, 120, 240, 480일 ──
    ma_periods = [5, 10, 20, 60, 120, 240, 480]
    mas = {}
    for n in ma_periods:
        ma = C.rolling(n).mean()
        mas[n] = ma
        feat[f"ma{n}_logret"] = safe_log_ratio(
            ma.values, ma.shift(1).values)
        feat[f"close_vs_ma{n}"] = safe_log_ratio(C.values, ma.values)

    # ── RSI (1) ──
    feat["rsi_norm"] = (compute_rsi(C, 14) - 50).values / 100

    # ── 코스피 (2) ──
    kospi_c = kospi_df["Close"].reindex(df.index, method="ffill")
    kospi_ma40 = kospi_c.rolling(40).mean()
    feat["kospi_logret"] = safe_log_ratio(
        kospi_c.values, kospi_c.shift(1).values
    )
    feat["kospi_vs_avg"] = safe_log_ratio(
        kospi_c.values, kospi_ma40.values)

    # ── 환율 (2) ──
    fx_c = usdkrw_df["Close"].reindex(df.index, method="ffill")
    fx_ma40 = fx_c.rolling(40).mean()
    feat["usdkrw_logret"] = safe_log_ratio(
        fx_c.values, fx_c.shift(1).values)
    feat["usdkrw_vs_avg"] = safe_log_ratio(
        fx_c.values, fx_ma40.values)

    # ── 거래대금 (1) ──
    turnover = C * V
    feat["turnover_logret"] = safe_log_ratio(
        (turnover + EPS).values, (turnover.shift(1) + EPS).values
    )

    # ── 볼린저밴드 위치 (1) ──
    bb_std = C.rolling(20).std()
    bb_mid = C.rolling(20).mean()
    feat["bb_position"] = ((C - bb_mid) / (2 * bb_std + EPS)).values

    # ── 요일 인코딩 (2) ──
    dow = df.index.dayofweek  # 0=Monday, 4=Friday
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # ── 주가 스케일 (1): log(종가) / 10 ──
    feat["log_price"] = np.log(C.values.astype(np.float64) + EPS) / 10

    # ── 골든 크로스 (1) ──
    #   조건: 5일선 기울기 > 0, 20일선 기울기 > 0, 5일선이 20일선 상향돌파
    ma5, ma20 = mas[5], mas[20]
    ma5_rising = ma5 > ma5.shift(1)
    ma20_rising = ma20 > ma20.shift(1)
    prev_below = ma5.shift(1) <= ma20.shift(1)
    curr_above = ma5 > ma20
    feat["golden_cross"] = (
        ma5_rising & ma20_rising & prev_below & curr_above
    ).astype(np.float64)

    # ── 배열 수치 (1): 정배열(+1) ~ 역배열(-1) ──
    #   Close > MA5 > MA10 > MA20 > MA60 > MA120 > MA240 > MA480 → +1
    #   Close < MA5 < MA10 < MA20 < MA60 < MA120 < MA240 < MA480 → -1
    values_list = [C, mas[5], mas[10], mas[20], mas[60], mas[120],
                   mas[240], mas[480]]
    pair_count = len(values_list) - 1  # 7쌍
    score = pd.Series(0.0, index=df.index)
    for i in range(pair_count):
        score += (values_list[i] > values_list[i + 1]).astype(np.float64)
    feat["alignment_score"] = (score / pair_count * 2 - 1).values

    # ── 외국인/기관 순매수 비율 (2) ──
    #   데이터가 없는 날짜는 NaN → dropna()에서 자연 제거
    if inv_df is not None and not inv_df.empty:
        inv_idx = pd.DatetimeIndex(inv_df.index)
        inv_df = inv_df.set_index(inv_idx)

        foreign_col = None
        inst_col = None
        for col in inv_df.columns:
            if "외국인" in col:
                foreign_col = col
            if "기관" in col:
                inst_col = col

        if foreign_col:
            foreign_net = inv_df[foreign_col].reindex(df.index)
            feat["foreign_net_ratio"] = np.clip(
                foreign_net.values / (V.values + EPS), -5, 5)
        else:
            feat["foreign_net_ratio"] = np.nan

        if inst_col:
            inst_net = inv_df[inst_col].reindex(df.index)
            feat["institution_net_ratio"] = np.clip(
                inst_net.values / (V.values + EPS), -5, 5)
        else:
            feat["institution_net_ratio"] = np.nan
    else:
        feat["foreign_net_ratio"] = np.nan
        feat["institution_net_ratio"] = np.nan

    # ── MACD (2): (12, 26, 9) ──
    ema12 = C.ewm(span=12, adjust=False).mean()
    ema26 = C.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal_line = macd_line.ewm(span=9, adjust=False).mean()
    feat["macd_signal"] = (macd_signal_line / (C + EPS)).values
    feat["macd_histogram"] = ((macd_line - macd_signal_line) / (C + EPS)).values

    # ── 스토캐스틱 오실레이터 (2): 14일 ──
    low14 = L.rolling(14).min()
    high14 = H.rolling(14).max()
    stoch_k = (C - low14) / (high14 - low14 + EPS)
    stoch_d = stoch_k.rolling(3).mean()
    feat["stoch_k"] = (stoch_k - 0.5).values
    feat["stoch_d"] = (stoch_d - 0.5).values

    # ── 스토캐스틱 골든크로스 (1): %K가 %D를 상향돌파 ──
    prev_below = stoch_k.shift(1) <= stoch_d.shift(1)
    curr_above = stoch_k > stoch_d
    feat["stoch_golden_cross"] = (prev_below & curr_above).astype(np.float64)

    # ── 거래량 비율 (1): 거래량 / 20일 평균거래량 ──
    vol_ma20 = V.rolling(20).mean()
    feat["volume_ratio"] = safe_log_ratio((V + EPS).values, (vol_ma20 + EPS).values)

    # ── ATR 정규화 (1): ATR(14) / 종가 ──
    tr = pd.concat([
        H - L,
        (H - C.shift(1)).abs(),
        (L - C.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    feat["atr_norm"] = (atr14 / (C + EPS)).values

    # ── 가격 모멘텀 (2): 5일, 10일 누적 로그수익률 ──
    feat["momentum_5d"] = feat["close_logret"].rolling(5).sum()
    feat["momentum_10d"] = feat["close_logret"].rolling(10).sum()

    # ── 거시 지표 — KOSDAQ (2) ──
    if kosdaq_df is not None and not kosdaq_df.empty:
        kq_c = kosdaq_df["Close"].reindex(df.index, method="ffill")
        kq_ma40 = kq_c.rolling(40).mean()
        feat["kosdaq_logret"] = safe_log_ratio(
            kq_c.values, kq_c.shift(1).values)
        feat["kosdaq_vs_avg"] = safe_log_ratio(
            kq_c.values, kq_ma40.values)
    else:
        feat["kosdaq_logret"] = np.nan
        feat["kosdaq_vs_avg"] = np.nan

    # ── 거시 지표 — 미국 S&P500 (2) ──
    if sp500_df is not None and not sp500_df.empty:
        sp_c = sp500_df["Close"].reindex(df.index, method="ffill")
        sp_ma40 = sp_c.rolling(40).mean()
        feat["sp500_logret"] = safe_log_ratio(
            sp_c.values, sp_c.shift(1).values)
        feat["sp500_vs_avg"] = safe_log_ratio(
            sp_c.values, sp_ma40.values)
    else:
        feat["sp500_logret"] = np.nan
        feat["sp500_vs_avg"] = np.nan

    # ── 거시 지표 — VIX 공포지수 (1) ──
    if vix_df is not None and not vix_df.empty:
        vix_c = vix_df["Close"].reindex(df.index, method="ffill")
        # VIX를 정규화: (VIX - 20) / 20 (20이 평균적 수준)
        feat["vix_norm"] = ((vix_c - 20) / 20).values
    else:
        feat["vix_norm"] = np.nan

    # ── 거시 지표 — 미국 국채 10년물 (1) ──
    if tnx_df is not None and not tnx_df.empty:
        tnx_c = tnx_df["Close"].reindex(df.index, method="ffill")
        feat["tnx_logret"] = safe_log_ratio(
            tnx_c.values, tnx_c.shift(1).values)
    else:
        feat["tnx_logret"] = np.nan

    # ── 거시 지표 — 유가 WTI (2) ──
    if oil_df is not None and not oil_df.empty:
        oil_c = oil_df["Close"].reindex(df.index, method="ffill")
        oil_ma40 = oil_c.rolling(40).mean()
        feat["oil_logret"] = safe_log_ratio(
            oil_c.values, oil_c.shift(1).values)
        feat["oil_vs_avg"] = safe_log_ratio(
            oil_c.values, oil_ma40.values)
    else:
        feat["oil_logret"] = np.nan
        feat["oil_vs_avg"] = np.nan

    # ── 섹터 특성 (2) ──
    if sector_returns is not None and not sector_returns.empty:
        sr = sector_returns.reindex(df.index, method="ffill")
        feat["sector_return"] = sr.values
        feat["sector_momentum_5d"] = sr.rolling(5).sum().values
    else:
        feat["sector_return"] = np.nan
        feat["sector_momentum_5d"] = np.nan

    # ── 시장 등락비율 (1) ──
    if advance_ratio_series is not None and not advance_ratio_series.empty:
        ar = advance_ratio_series.reindex(df.index, method="ffill")
        # 0~1 범위를 -0.5~0.5로 중심화
        feat["advance_ratio"] = (ar - 0.5).values
    else:
        feat["advance_ratio"] = np.nan

    # 첫 행 NaN & warmup 제거
    feat.iloc[0] = np.nan
    feat = feat.dropna()

    return feat[DAILY_FEATURE_COLS]


def generate_daily_dataset(tickers):
    """전체 종목 일별 특성 데이터셋 생성 -> pickle 저장"""
    n_features = len(DAILY_FEATURE_COLS)
    print("\n" + "=" * 60)
    print(f" 일별 데이터 생성 ({n_features} features, 최대 30년)")
    print("=" * 60)

    end = datetime.now()
    start = end - timedelta(days=DAILY_YEARS * 365)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    pykrx_start = start.strftime("%Y%m%d")
    pykrx_end = end.strftime("%Y%m%d")

    # ── 외부 지표 다운로드 ──
    def download_yf(name, symbol):
        print(f"{name} 다운로드...")
        raw = yf.download(
            symbol, start=start_str, end=end_str,
            progress=False, auto_adjust=True)
        df = flatten_yf_columns(raw)
        print(f"  {name}: {len(df)}일")
        return df

    kospi_df = download_yf("코스피 지수", "^KS11")
    usdkrw_df = download_yf("USD/KRW 환율", "USDKRW=X")
    kosdaq_df = download_yf("코스닥 지수", "^KQ11")
    sp500_df = download_yf("S&P 500", "^GSPC")
    vix_df = download_yf("VIX 공포지수", "^VIX")
    tnx_df = download_yf("미국 국채 10Y", "^TNX")
    oil_df = download_yf("WTI 유가", "CL=F")

    # ── 섹터 분류 (pykrx) ──
    print("\n섹터 분류 조회 중...")
    ticker_to_sector = {}
    for market in ["KOSPI", "KOSDAQ"]:
        try:
            idx_list = stock.get_index_ticker_list(market=market)
            for idx_code in idx_list:
                idx_name = stock.get_index_ticker_name(idx_code)
                # 대형/중형/소형 등 규모 지수 제외, 업종 지수만 사용
                skip_keywords = ["대형", "중형", "소형", "100", "200", "50",
                                 "배당", "스타", "프리미어", "전체"]
                if any(kw in idx_name for kw in skip_keywords):
                    continue
                try:
                    members = stock.get_index_portfolio_deposit_file(idx_code)
                    for t in members:
                        if t not in ticker_to_sector:
                            ticker_to_sector[t] = idx_name
                except Exception:
                    pass
        except Exception as e:
            print(f"  [경고] {market} 섹터 조회 실패: {e}")
    print(f"  섹터 분류: {len(ticker_to_sector)}개 종목 매핑 완료")
    n_sectors = len(set(ticker_to_sector.values()))
    print(f"  섹터 수: {n_sectors}개")

    # ── 1차: 주가 데이터 다운로드 (섹터/등락비율 계산용) ──
    print("\n주가 데이터 다운로드 (1차: 수익률 계산용)...")
    stock_close_data = {}  # ticker -> Series(close)
    stock_raw_data = {}    # ticker -> DataFrame(OHLCV)
    stock_inv_data = {}    # ticker -> DataFrame(investor)
    failed = []

    pbar = tqdm(
        tickers, desc="일별 다운로드", unit="종목",
        bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}]"))
    for ticker in pbar:
        ticker_yf = f"{ticker}.KS"
        pbar.set_postfix_str(f"{ticker} 다운로드 중")

        try:
            raw = yf.download(
                ticker_yf, start=start_str, end=end_str,
                progress=False, auto_adjust=True,
            )
            raw = flatten_yf_columns(raw)

            if len(raw) < 500:
                tqdm.write(
                    f"  {ticker}: 데이터 부족 ({len(raw)}일) — skip")
                failed.append((ticker, f"데이터 부족 ({len(raw)}일)"))
                continue

            stock_raw_data[ticker] = raw
            stock_close_data[ticker] = raw["Close"]

            # pykrx 투자자별 순매수 거래량
            inv_df = None
            try:
                inv_df = stock.get_market_trading_volume_by_date(
                    pykrx_start, pykrx_end, ticker, on="순매수")
            except Exception:
                pass
            stock_inv_data[ticker] = inv_df

        except Exception as e:
            tqdm.write(f"  {ticker}: FAIL — {e}")
            failed.append((ticker, str(e)))

        time_module.sleep(0.05)
    pbar.close()
    print(f"  다운로드: 성공 {len(stock_raw_data)} | 실패 {len(failed)}")

    # ── 섹터별 일평균 수익률 계산 ──
    print("섹터 수익률 계산 중...")
    all_logrets = pd.DataFrame({
        t: np.log(s / s.shift(1))
        for t, s in stock_close_data.items()
    })
    sector_return_map = {}  # ticker -> Series(sector avg return)
    sectors = {}
    for t in stock_raw_data:
        sec = ticker_to_sector.get(t, "기타")
        sectors.setdefault(sec, []).append(t)
    for sec, members in sectors.items():
        if len(members) < 2:
            continue
        sector_avg = all_logrets[members].mean(axis=1)
        for t in members:
            # 자기 자신 제외한 섹터 평균
            others = [m for m in members if m != t]
            sector_return_map[t] = all_logrets[others].mean(axis=1)
    print(f"  {len(sectors)}개 섹터, {len(sector_return_map)}개 종목 매핑 완료")

    # ── 시장 등락비율 계산 ──
    print("시장 등락비율 계산 중...")
    advance_ratio_series = (all_logrets > 0).sum(axis=1) / all_logrets.count(axis=1)
    print(f"  등락비율: {len(advance_ratio_series)}일")

    # ── 2차: 특성 생성 ──
    print("\n특성 생성 중...")
    results = {}
    pbar2 = tqdm(
        stock_raw_data.items(), desc="특성 생성", unit="종목",
        total=len(stock_raw_data),
        bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}]"))
    for ticker, raw in pbar2:
        try:
            sec_ret = sector_return_map.get(ticker)
            feat = compute_daily_features(
                raw, kospi_df, usdkrw_df,
                inv_df=stock_inv_data.get(ticker),
                kosdaq_df=kosdaq_df, sp500_df=sp500_df,
                vix_df=vix_df, tnx_df=tnx_df, oil_df=oil_df,
                sector_returns=sec_ret,
                advance_ratio_series=advance_ratio_series,
            )
            results[ticker] = feat
            pbar2.set_postfix_str(f"{ticker} OK ({len(feat)}일)")
        except Exception as e:
            tqdm.write(f"  {ticker}: 특성 생성 FAIL — {e}")
            failed.append((ticker, str(e)))
    pbar2.close()

    with open(DAILY_OUTPUT, "wb") as f:
        pickle.dump(results, f)

    total_rows = sum(len(df) for df in results.values())
    print(f"\n일별 데이터 저장 완료: {DAILY_OUTPUT}")
    print(f"  성공: {len(results)}종목 | 실패: {len(failed)}종목")
    print(f"  총 데이터: {total_rows:,}행 x {n_features}열")
    if failed:
        print(f"  실패 상세:")
        for t, reason in failed[:10]:
            print(f"    - {t}: {reason}")
        if len(failed) > 10:
            print(f"    ... 외 {len(failed) - 10}개")
    return results


