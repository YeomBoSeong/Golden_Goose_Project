"""
Backtest_stored_model.py
저장된 일별 모델 백테스트

현실적인 수익률 계산:
  - 같은 날 시그널이 여러 개면 균등 분배 투자
  - 날짜별 포트폴리오 수익률로 계산
  - 초기 자본금 기반 실제 금액 표시

4가지 전략:
  A. 단순 1일 보유: 시그널 다음날 시가 매수 → 당일 종가 매도
  B. 익절/손절: 시가 매수 → +2% 익절 / -1% 손절 (최대 5일)
  C. 트레일링 스탑: 시가 매수 → 고점 대비 -1.5% 하락 시 매도 (최대 5일)
  D. 3일 보유: 시가 매수 → 3거래일 후 종가 매도

사용법:
  python Backtest_stored_model.py
  python Backtest_stored_model.py --threshold 0.6 0.65 0.7 0.8
  python Backtest_stored_model.py --threshold 0.7 --capital 200000
  python Backtest_stored_model.py --model daily_cls_model.pkl
"""

import argparse
import pickle
import os
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from pykrx import stock as pykrx_stock
from datetime import datetime, timedelta
from tqdm import tqdm
from train_utils import DEVICE, LSTMModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
DEFAULT_MODEL = os.path.join(BASE_DIR, "daily_cls_model.pkl")


def load_model(model_path):
    with open(model_path, "rb") as f:
        checkpoint = pickle.load(f)
    config = checkpoint["config"]
    model = LSTMModel(
        config["input_size"], config["hidden_size"],
        config["num_layers"], config["output_size"],
        config.get("dropout", 0.1),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def get_ohlcv_cached(ticker, start, end, cache):
    key = ticker
    if key not in cache:
        try:
            df = pykrx_stock.get_market_ohlcv_by_date(start, end, ticker)
            cache[key] = df if len(df) > 0 else None
        except Exception:
            cache[key] = None
    return cache[key]


# ─── 전략 함수들 ──────────────────────────────────────────

def simulate_strategy_a(ohlcv, signal_date):
    """전략 A: 단순 1일 보유"""
    dates = ohlcv.index.tolist()
    if signal_date not in dates:
        return None
    idx = dates.index(signal_date)
    if idx + 1 >= len(dates):
        return None
    next_day = dates[idx + 1]
    buy_price = ohlcv.loc[next_day, "시가"]
    sell_price = ohlcv.loc[next_day, "종가"]
    if buy_price <= 0:
        return None
    ret = (sell_price - buy_price) / buy_price
    return {
        "buy_date": next_day, "sell_date": next_day,
        "buy_price": buy_price, "sell_price": sell_price,
        "return": ret, "hold_days": 1, "exit": "종가매도",
    }


def simulate_strategy_b(ohlcv, signal_date, tp=0.02, sl=-0.01, max_hold=5):
    """전략 B: 익절/손절"""
    dates = ohlcv.index.tolist()
    if signal_date not in dates:
        return None
    idx = dates.index(signal_date)
    if idx + 1 >= len(dates):
        return None
    buy_date = dates[idx + 1]
    buy_price = ohlcv.loc[buy_date, "시가"]
    if buy_price <= 0:
        return None

    for d in range(max_hold):
        day_idx = idx + 1 + d
        if day_idx >= len(dates):
            break
        day = dates[day_idx]
        high = ohlcv.loc[day, "고가"]
        low = ohlcv.loc[day, "저가"]

        low_ret = (low - buy_price) / buy_price
        high_ret = (high - buy_price) / buy_price

        if low_ret <= sl:
            return {
                "buy_date": buy_date, "sell_date": day,
                "buy_price": buy_price,
                "sell_price": buy_price * (1 + sl),
                "return": sl, "hold_days": d + 1,
                "exit": f"손절({sl*100:.0f}%)",
            }
        if high_ret >= tp:
            return {
                "buy_date": buy_date, "sell_date": day,
                "buy_price": buy_price,
                "sell_price": buy_price * (1 + tp),
                "return": tp, "hold_days": d + 1,
                "exit": f"익절(+{tp*100:.0f}%)",
            }

    last_idx = min(idx + max_hold, len(dates) - 1)
    last_day = dates[last_idx]
    sell_price = ohlcv.loc[last_day, "종가"]
    ret = (sell_price - buy_price) / buy_price
    return {
        "buy_date": buy_date, "sell_date": last_day,
        "buy_price": buy_price, "sell_price": sell_price,
        "return": ret, "hold_days": last_idx - idx,
        "exit": f"보유만료({max_hold}일)",
    }


def simulate_strategy_c(ohlcv, signal_date, trail_pct=0.015, max_hold=5):
    """전략 C: 트레일링 스탑"""
    dates = ohlcv.index.tolist()
    if signal_date not in dates:
        return None
    idx = dates.index(signal_date)
    if idx + 1 >= len(dates):
        return None
    buy_date = dates[idx + 1]
    buy_price = ohlcv.loc[buy_date, "시가"]
    if buy_price <= 0:
        return None

    peak = buy_price
    for d in range(max_hold):
        day_idx = idx + 1 + d
        if day_idx >= len(dates):
            break
        day = dates[day_idx]
        high = ohlcv.loc[day, "고가"]
        low = ohlcv.loc[day, "저가"]

        if high > peak:
            peak = high
        trail_stop = peak * (1 - trail_pct)
        if low <= trail_stop:
            sell_price = trail_stop
            ret = (sell_price - buy_price) / buy_price
            return {
                "buy_date": buy_date, "sell_date": day,
                "buy_price": buy_price, "sell_price": sell_price,
                "return": ret, "hold_days": d + 1,
                "exit": f"트레일링({trail_pct*100:.1f}%)",
            }

    last_idx = min(idx + max_hold, len(dates) - 1)
    last_day = dates[last_idx]
    sell_price = ohlcv.loc[last_day, "종가"]
    ret = (sell_price - buy_price) / buy_price
    return {
        "buy_date": buy_date, "sell_date": last_day,
        "buy_price": buy_price, "sell_price": sell_price,
        "return": ret, "hold_days": last_idx - idx,
        "exit": f"보유만료({max_hold}일)",
    }


def simulate_strategy_d(ohlcv, signal_date, hold_days=3):
    """전략 D: 3일 보유"""
    dates = ohlcv.index.tolist()
    if signal_date not in dates:
        return None
    idx = dates.index(signal_date)
    if idx + 1 >= len(dates):
        return None
    buy_date = dates[idx + 1]
    buy_price = ohlcv.loc[buy_date, "시가"]
    if buy_price <= 0:
        return None

    sell_idx = min(idx + 1 + hold_days - 1, len(dates) - 1)
    sell_date = dates[sell_idx]
    sell_price = ohlcv.loc[sell_date, "종가"]
    actual_hold = sell_idx - idx
    ret = (sell_price - buy_price) / buy_price
    return {
        "buy_date": buy_date, "sell_date": sell_date,
        "buy_price": buy_price, "sell_price": sell_price,
        "return": ret, "hold_days": actual_hold, "exit": f"{hold_days}일보유",
    }


# ─── 현실적 포트폴리오 수익률 계산 ────────────────────────

def simulate_realistic_portfolio(trades, initial_capital=200000):
    """실제 주식 매매 시뮬레이션 (정수 주 단위, 잔고 차감)

    같은 날 시그널이 여러 개면 가용 현금을 균등 배분하여 각각 매수.
    주가보다 배분 금액이 작으면 해당 종목 매수를 건너뜀.

    Args:
        trades: 거래 리스트 (buy_date, sell_date, buy_price, sell_price 등)
        initial_capital: 초기 자본금 (원)

    Returns:
        dict with final_capital, cumulative_return, mdd, executed/skipped 등
    """
    if not trades:
        return {
            "cumulative_return": 0, "final_capital": initial_capital,
            "initial_capital": initial_capital, "mdd": 0,
            "trading_days": 0, "executed": 0, "skipped": 0,
            "executed_trades": [],
        }

    cash = initial_capital
    open_positions = []  # [{shares, buy_price, sell_price, sell_date, trade}]

    # 매수일 기준 그룹화
    by_buy_date = {}
    for t in trades:
        by_buy_date.setdefault(t["buy_date"], []).append(t)

    # 모든 이벤트 날짜 (매수일 + 매도일)
    all_dates = sorted(set(
        [t["buy_date"] for t in trades] + [t["sell_date"] for t in trades]
    ))

    equity_snapshots = [initial_capital]
    executed = 0
    skipped = 0
    executed_trades = []

    for date in all_dates:
        # 1) 매도일이 지난 포지션 청산 → 현금 회수
        remaining = []
        for pos in open_positions:
            if pos["sell_date"] <= date:
                proceeds = pos["shares"] * pos["sell_price"]
                cash += proceeds
            else:
                remaining.append(pos)
        open_positions = remaining

        # 2) 오늘 매수할 시그널 처리
        if date in by_buy_date:
            new_trades = by_buy_date[date]
            n = len(new_trades)
            alloc_per_trade = cash / n if n > 0 else 0

            for t in new_trades:
                shares = int(alloc_per_trade // t["buy_price"])
                if shares < 1:
                    skipped += 1
                    continue
                cost = shares * t["buy_price"]
                cash -= cost
                open_positions.append({
                    "shares": shares,
                    "buy_price": t["buy_price"],
                    "sell_price": t["sell_price"],
                    "sell_date": t["sell_date"],
                })
                actual_ret = (t["sell_price"] - t["buy_price"]) / t["buy_price"]
                executed_trades.append({
                    **t,
                    "shares": shares,
                    "invested": cost,
                    "proceeds": shares * t["sell_price"],
                    "actual_return": actual_ret,
                })
                executed += 1

        # 3) 자산 스냅샷 (현금 + 보유 포지션 원가 기준)
        pos_value = sum(p["shares"] * p["buy_price"] for p in open_positions)
        equity_snapshots.append(cash + pos_value)

    # 남은 포지션 모두 청산
    for pos in open_positions:
        cash += pos["shares"] * pos["sell_price"]
    open_positions = []

    final_equity = cash
    equity_snapshots.append(final_equity)

    # MDD 계산
    equity_arr = np.array(equity_snapshots)
    peak = np.maximum.accumulate(equity_arr)
    dd = (equity_arr - peak) / peak
    mdd = dd.min()

    return {
        "cumulative_return": (final_equity - initial_capital) / initial_capital,
        "final_capital": final_equity,
        "initial_capital": initial_capital,
        "mdd": mdd,
        "trading_days": len(by_buy_date),
        "executed": executed,
        "skipped": skipped,
        "executed_trades": executed_trades,
    }


def summarize_trades(trades, strategy_name, initial_capital=200000):
    """거래 결과 요약 (실제 주식 매매 시뮬레이션)"""
    if not trades:
        return {"strategy": strategy_name, "trades": 0}

    portfolio = simulate_realistic_portfolio(trades, initial_capital)
    exec_trades = portfolio["executed_trades"]

    if not exec_trades:
        return {
            "strategy": strategy_name, "trades": len(trades),
            "executed": 0, "skipped": portfolio["skipped"],
            "win_rate": 0, "avg_return": 0,
            "cumulative_return": 0,
            "final_capital": initial_capital,
            "initial_capital": initial_capital,
            "mdd": 0, "trading_days": 0,
        }

    returns = [t["actual_return"] for t in exec_trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    hold_days = [t["hold_days"] for t in exec_trades]

    # 월별 수익률 (실제 실행된 거래만)
    monthly = {}
    for t in exec_trades:
        m = t["buy_date"].strftime("%Y-%m")
        monthly.setdefault(m, []).append(t["actual_return"])

    # 매수일 기준 일별 수익률 (연속 손실 계산용)
    by_buy_date = {}
    for t in exec_trades:
        d = t["buy_date"]
        by_buy_date.setdefault(d, []).append(t["actual_return"])
    daily_avg_rets = {d: np.mean(r) for d, r in by_buy_date.items()}

    max_consec_loss = 0
    curr_loss = 0
    for date in sorted(daily_avg_rets.keys()):
        if daily_avg_rets[date] <= 0:
            curr_loss += 1
            max_consec_loss = max(max_consec_loss, curr_loss)
        else:
            curr_loss = 0

    exit_counts = {}
    for t in exec_trades:
        e = t["exit"]
        exit_counts[e] = exit_counts.get(e, 0) + 1

    return {
        "strategy": strategy_name,
        "trades": len(trades),
        "executed": portfolio["executed"],
        "skipped": portfolio["skipped"],
        "win_rate": len(wins) / len(exec_trades),
        "avg_return": np.mean(returns),
        "median_return": np.median(returns),
        "std_return": np.std(returns),
        "max_return": max(returns),
        "min_return": min(returns),
        "avg_hold_days": np.mean(hold_days),
        "mdd": portfolio["mdd"],
        "cumulative_return": portfolio["cumulative_return"],
        "final_capital": portfolio["final_capital"],
        "initial_capital": portfolio["initial_capital"],
        "trading_days": portfolio["trading_days"],
        "max_consec_loss": max_consec_loss,
        "exit_counts": exit_counts,
        "monthly": monthly,
        "profit_factor": (sum(wins) / abs(sum(losses))
                          if losses else float("inf")),
        "avg_signals_per_day": len(trades) / max(portfolio["trading_days"], 1),
    }


def print_summary(s, detail=False):
    if s["trades"] == 0:
        print(f"  {s['strategy']}: 거래 없음")
        return

    executed = s.get("executed", s["trades"])
    skipped = s.get("skipped", 0)

    if executed == 0:
        print(f"\n{'━' * 60}")
        print(f"  전략: {s['strategy']}")
        print(f"{'━' * 60}")
        print(f"  시그널 {s['trades']}건 중 매수 가능 0건 (자본 부족)")
        return

    print(f"\n{'━' * 60}")
    print(f"  전략: {s['strategy']}")
    print(f"{'━' * 60}")
    print(f"  시그널:        {s['trades']}건 → 실행 {executed}건 "
          f"/ 건너뜀 {skipped}건 (자본부족)")
    print(f"  거래일:        {s['trading_days']}일 "
          f"(일평균 {s['avg_signals_per_day']:.1f}건)")
    print(f"  승률:          {s['win_rate']:.1%}")
    print(f"  평균 수익률:   {s['avg_return']:+.2%} (건당)")
    print(f"  중간값 수익률: {s['median_return']:+.2%}")
    print(f"  표준편차:      {s['std_return']:.2%}")
    print(f"  최대 수익:     {s['max_return']:+.2%}")
    print(f"  최대 손실:     {s['min_return']:+.2%}")
    print(f"  초기 자본:     {s['initial_capital']:,.0f}원")
    print(f"  최종 자본:     {s['final_capital']:,.0f}원")
    print(f"  누적 수익률:   {s['cumulative_return']:+.1%} "
          f"({s['final_capital'] - s['initial_capital']:+,.0f}원)")
    print(f"  평균 보유:     {s['avg_hold_days']:.1f}일")
    print(f"  MDD:           {s['mdd']:+.1%}")
    print(f"  최대 연속손실: {s['max_consec_loss']}일")
    pf = s["profit_factor"]
    pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
    print(f"  손익비:        {pf_str}")
    print(f"  종료 유형:     ", end="")
    for exit_type, cnt in s["exit_counts"].items():
        print(f"{exit_type} {cnt}건  ", end="")
    print()

    if detail and s.get("monthly"):
        print(f"\n  월별 거래:")
        for month in sorted(s["monthly"].keys()):
            rets = s["monthly"][month]
            avg_r = np.mean(rets)
            wins = sum(1 for r in rets if r > 0)
            print(f"    {month}: {len(rets):>3}건 | "
                  f"승률 {wins/len(rets):.0%} | "
                  f"평균 {avg_r:+.2%}")


def save_results(all_summaries, threshold, output_path, model_name,
                 initial_capital):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f" 일별 모델 매매 전략 백테스트 결과\n")
        f.write(f" 모델: {model_name}\n")
        f.write(f" Threshold: {threshold}\n")
        f.write(f" 초기 자본: {initial_capital:,.0f}원\n")
        f.write(f" 수익률: 날짜별 균등분배 포트폴리오 기준\n")
        f.write(f" 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"{'전략':<22} {'시그널':>5} {'실행':>5} {'건너뜀':>5} "
                f"{'승률':>6} {'건당평균':>7} {'누적':>8} "
                f"{'최종자본':>12} {'MDD':>7} {'손익비':>6}\n")
        f.write("-" * 95 + "\n")
        for s in all_summaries:
            if s["trades"] == 0:
                continue
            executed = s.get("executed", s["trades"])
            skipped_cnt = s.get("skipped", 0)
            if executed == 0:
                f.write(f"{s['strategy']:<22} {s['trades']:>5} "
                        f"{executed:>5} {skipped_cnt:>5}  매수 불가\n")
                continue
            pf = s["profit_factor"]
            pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
            f.write(f"{s['strategy']:<22} {s['trades']:>5} "
                    f"{executed:>5} {skipped_cnt:>5} "
                    f"{s['win_rate']:>5.1%} "
                    f"{s['avg_return']:>+6.2%} "
                    f"{s['cumulative_return']:>+7.1%} "
                    f"{s['final_capital']:>11,.0f}원 "
                    f"{s['mdd']:>+6.1%} "
                    f"{pf_str:>6}\n")

        for s in all_summaries:
            if s["trades"] == 0:
                continue
            executed = s.get("executed", s["trades"])
            skipped_cnt = s.get("skipped", 0)
            f.write(f"\n{'─' * 60}\n")
            f.write(f" {s['strategy']}\n")
            f.write(f"{'─' * 60}\n")
            f.write(f"  시그널:        {s['trades']}건 → 실행 {executed}건"
                    f" / 건너뜀 {skipped_cnt}건 (자본부족)\n")
            if executed == 0:
                f.write(f"  매수 가능 종목 없음 (자본 부족)\n")
                continue
            f.write(f"  거래일:        {s['trading_days']}일 "
                    f"(일평균 {s['avg_signals_per_day']:.1f}건)\n")
            f.write(f"  승률:          {s['win_rate']:.1%}\n")
            f.write(f"  평균 수익률:   {s['avg_return']:+.2%} (건당)\n")
            f.write(f"  중간값 수익률: {s['median_return']:+.2%}\n")
            f.write(f"  초기 자본:     {s['initial_capital']:,.0f}원\n")
            f.write(f"  최종 자본:     {s['final_capital']:,.0f}원\n")
            f.write(f"  누적 수익률:   {s['cumulative_return']:+.1%} "
                    f"({s['final_capital'] - s['initial_capital']:+,.0f}원)\n")
            f.write(f"  MDD:           {s['mdd']:+.1%}\n")
            pf = s["profit_factor"]
            pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
            f.write(f"  손익비:        {pf_str}\n")

            if s.get("monthly"):
                f.write(f"\n  월별 거래:\n")
                for month in sorted(s["monthly"].keys()):
                    rets = s["monthly"][month]
                    avg_r = np.mean(rets)
                    wins = sum(1 for r in rets if r > 0)
                    f.write(f"    {month}: {len(rets):>3}건 | "
                            f"승률 {wins/len(rets):.0%} | "
                            f"평균 {avg_r:+.2%}\n")

    print(f"\n  결과 저장: {output_path}")


def collect_signals(model, features_dict, seq_len, input_size,
                    threshold, start_dt, end_dt):
    """모든 종목에서 시그널 수집"""
    signals = []
    for ticker, df in tqdm(features_dict.items(), desc="시그널 스캔",
                           unit="종목"):
        data = df.values[:, :input_size]
        dates = df.index

        for i in range(seq_len, len(data)):
            pred_date = dates[i]
            if pred_date < start_dt or pred_date > end_dt:
                continue

            x = data[i - seq_len:i]
            x_tensor = torch.tensor(
                x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(x_tensor).cpu().numpy()
            probs = softmax(logits, axis=1)
            prob_up = probs[0, 1]

            if prob_up >= threshold:
                signals.append((pred_date, ticker, float(prob_up)))

    signals.sort(key=lambda x: x[0])
    return signals


def run_backtest_for_threshold(signals, threshold, ohlcv_cache,
                               ohlcv_start, ohlcv_end,
                               initial_capital, model_name):
    """특정 threshold에 대한 백테스트 실행"""
    # 시그널 필터
    filtered = [(d, t, p) for d, t, p in signals if p >= threshold]
    print(f"\n{'=' * 70}")
    print(f" Threshold = {threshold} | 시그널 {len(filtered)}건")
    print(f"{'=' * 70}")

    if not filtered:
        print("  시그널이 없습니다.")
        return

    # 시그널 날짜 분포
    signal_dates = {}
    for s in filtered:
        d = s[0].strftime("%Y-%m-%d")
        signal_dates[d] = signal_dates.get(d, 0) + 1
    print(f"  시그널 발생일: {len(signal_dates)}일")
    top_dates = sorted(signal_dates.items(), key=lambda x: -x[1])[:5]
    for d, cnt in top_dates:
        print(f"    {d}: {cnt}건")

    # 매매 시뮬레이션
    trades_a, trades_b, trades_c, trades_d = [], [], [], []

    for signal_date, ticker, prob in tqdm(filtered, desc="매매 시뮬레이션"):
        ohlcv = get_ohlcv_cached(ticker, ohlcv_start, ohlcv_end, ohlcv_cache)
        if ohlcv is None:
            continue

        for fn, tlist in [
            (simulate_strategy_a, trades_a),
            (simulate_strategy_b, trades_b),
            (simulate_strategy_c, trades_c),
            (simulate_strategy_d, trades_d),
        ]:
            result = fn(ohlcv, signal_date)
            if result:
                result["ticker"] = ticker
                result["prob"] = prob
                result["signal_date"] = signal_date
                tlist.append(result)

    summaries = [
        summarize_trades(trades_a, "A. 단순 1일 보유", initial_capital),
        summarize_trades(trades_b, "B. 익절/손절(+2%/-1%)", initial_capital),
        summarize_trades(trades_c, "C. 트레일링(-1.5%)", initial_capital),
        summarize_trades(trades_d, "D. 3일 보유", initial_capital),
    ]

    # 비교표
    print(f"\n{'전략':<22} {'시그널':>5} {'실행':>5} {'건너뜀':>5} "
          f"{'승률':>6} {'건당평균':>7} {'누적':>8} "
          f"{'최종자본':>12} {'MDD':>7} {'손익비':>6}")
    print("-" * 95)
    for s in summaries:
        if s["trades"] == 0:
            continue
        executed = s.get("executed", s["trades"])
        skipped_cnt = s.get("skipped", 0)
        if executed == 0:
            print(f"{s['strategy']:<22} {s['trades']:>5} "
                  f"{executed:>5} {skipped_cnt:>5}  매수 불가")
            continue
        pf = s["profit_factor"]
        pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
        print(f"{s['strategy']:<22} {s['trades']:>5} "
              f"{executed:>5} {skipped_cnt:>5} "
              f"{s['win_rate']:>5.1%} "
              f"{s['avg_return']:>+6.2%} "
              f"{s['cumulative_return']:>+7.1%} "
              f"{s['final_capital']:>11,.0f}원 "
              f"{s['mdd']:>+6.1%} "
              f"{pf_str:>6}")

    for s in summaries:
        print_summary(s, detail=True)

    output_path = os.path.join(BASE_DIR,
                               f"backtest_exp12_th{threshold}.txt")
    save_results(summaries, threshold, output_path, model_name,
                 initial_capital)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="일별 모델 백테스트")
    parser.add_argument("--threshold", type=float, nargs="+",
                        default=[0.6, 0.65, 0.7, 0.8],
                        help="threshold 값 (여러 개 가능, 예: 0.6 0.65 0.7 0.8)")
    parser.add_argument("--start", type=str, default="20190101")
    parser.add_argument("--end", type=str, default="20221231")
    parser.add_argument("--capital", type=int, default=200000,
                        help="초기 자본금 (원, 기본값: 200000)")
    parser.add_argument("--model", type=str, default=None,
                        help="모델 파일명 (LSTM 폴더 기준, 예: daily_cls_model.pkl)")
    args = parser.parse_args()

    # 모델 경로 결정
    if args.model:
        model_path = os.path.join(BASE_DIR, args.model)
        model_name = args.model
    else:
        model_path = DEFAULT_MODEL
        model_name = "daily_cls_model.pkl"

    min_threshold = min(args.threshold)

    print("=" * 70)
    print(f" 일별 모델 백테스트")
    print(f" 모델: {model_name}")
    print(f" Threshold: {args.threshold}")
    print(f" 초기 자본: {args.capital:,}원")
    print(f" 기간: {args.start} ~ {args.end}")
    print(f" 수익률: 날짜별 균등분배 포트폴리오")
    print("=" * 70)

    model, config = load_model(model_path)
    seq_len = config["seq_len"]
    input_size = config["input_size"]

    pkl_path = os.path.join(DATA_DIR, "daily_features.pkl")
    with open(pkl_path, "rb") as f:
        features_dict = pickle.load(f)
    print(f"  {len(features_dict)}개 종목 로드")

    # 1단계: 최소 threshold로 시그널 수집 (한 번만 스캔)
    print(f"\n[1/2] 시그널 수집 중 (threshold >= {min_threshold})...")
    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end)

    all_signals = collect_signals(
        model, features_dict, seq_len, input_size,
        min_threshold, start_dt, end_dt)
    print(f"  총 {len(all_signals)}건 시그널 (threshold >= {min_threshold})")

    if not all_signals:
        print("  시그널이 없습니다.")
        exit()

    # 2단계: OHLCV 캐시 준비
    ohlcv_cache = {}
    ohlcv_start = (start_dt - timedelta(days=30)).strftime("%Y%m%d")
    ohlcv_end = (end_dt + timedelta(days=30)).strftime("%Y%m%d")

    # 3단계: 각 threshold에 대해 백테스트
    print(f"\n[2/2] 백테스트 실행...")
    for th in sorted(args.threshold):
        run_backtest_for_threshold(
            all_signals, th, ohlcv_cache, ohlcv_start, ohlcv_end,
            args.capital, model_name)

    print(f"\n{'=' * 70}")
    print(f" 백테스트 완료!")
    print(f" 결과 파일:")
    for th in sorted(args.threshold):
        print(f"   backtest_exp12_th{th}.txt")
    print(f"{'=' * 70}")
