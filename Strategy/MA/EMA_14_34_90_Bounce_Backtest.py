from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_NAMES = {
    "1m": "NIFTY_1m_ANALYSIS.csv",
    "5m": "NIFTY_5m_ANALYSIS.csv",
    "10m": "NIFTY_10m_ANALYSIS.csv",
    "15m": "NIFTY_15m_ANALYSIS.csv",
    "1h": "NIFTY_1h_ANALYSIS.csv",
}

COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"
USE_COLOR = True


def resolve_data_dir(data_dir: str | None = None) -> Path:
    if data_dir:
        return Path(data_dir).expanduser().resolve()

    candidates = [
        Path.cwd() / "Nifty50_FullData",
        Path.cwd() / "Nifty50_FullData" / "Nifty50_FullData",
        PROJECT_ROOT / "Nifty50_FullData",
        PROJECT_ROOT / "Nifty50_FullData" / "Nifty50_FullData",
    ]
    for candidate in candidates:
        if all((candidate / name).exists() for name in CSV_NAMES.values()):
            return candidate.resolve()
    return candidates[0].resolve()


def data_files(data_dir: Path) -> dict[str, Path]:
    return {timeframe: data_dir / name for timeframe, name in CSV_NAMES.items()}


def bool_arg(value: str) -> bool:
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false")


def color_text(value: str, color: str) -> str:
    return f"{color}{value}{COLOR_RESET}" if USE_COLOR else value


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["datetime", "open", "high", "low", "close"]).reset_index(drop=True)


def add_indicators(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = out["close"].ewm(span=args.fast_ema, adjust=False).mean()
    out["ema_mid"] = out["close"].ewm(span=args.mid_ema, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=args.slow_ema, adjust=False).mean()
    out["ema_channel_high"] = out["high"].ewm(span=args.mid_ema, adjust=False).mean()
    out["ema_channel_low"] = out["low"].ewm(span=args.mid_ema, adjust=False).mean()
    out["ema_mid_slope"] = (out["ema_mid"] - out["ema_mid"].shift(args.slope_lookback)) / args.slope_lookback
    out["ema_fast_slope"] = (out["ema_fast"] - out["ema_fast"].shift(args.slope_lookback)) / args.slope_lookback
    out["ribbon_width"] = (out["ema_fast"] - out["ema_slow"]).abs()
    out["ribbon_width_avg"] = out["ribbon_width"].rolling(args.ribbon_lookback).mean()
    if "volume" in out.columns:
        out["volume_avg"] = out["volume"].rolling(args.volume_lookback).mean()
    else:
        out["volume_avg"] = np.nan
    prev_close = out["close"].shift(1)
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = true_range.rolling(args.atr_stop_len).mean()
    out["body"] = (out["close"] - out["open"]).abs()
    out["range"] = out["high"] - out["low"]
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    prev_open = out["open"].shift(1)
    prev_close = out["close"].shift(1)
    out["bullish_engulfing"] = (out["close"] > out["open"]) & (prev_close < prev_open) & (out["open"] <= prev_close) & (out["close"] >= prev_open)
    out["bearish_engulfing"] = (out["close"] < out["open"]) & (prev_close > prev_open) & (out["open"] >= prev_close) & (out["close"] <= prev_open)
    out["distance_from_fast"] = (out["close"] - out["ema_fast"]).abs()
    out["amd_range_high"] = out["high"].shift(1).rolling(args.amd_lookback).max()
    out["amd_range_low"] = out["low"].shift(1).rolling(args.amd_lookback).min()
    out["amd_range_width"] = out["amd_range_high"] - out["amd_range_low"]
    out["sweep_down"] = (out["low"] < out["amd_range_low"] - args.smc_sweep_buffer) & (out["close"] > out["amd_range_low"])
    out["sweep_up"] = (out["high"] > out["amd_range_high"] + args.smc_sweep_buffer) & (out["close"] < out["amd_range_high"])
    out["recent_sweep_down"] = out["sweep_down"].shift(1).rolling(args.smc_sweep_lookback).max()
    out["recent_sweep_up"] = out["sweep_up"].shift(1).rolling(args.smc_sweep_lookback).max()
    out["choch_high"] = out["high"].shift(1).rolling(args.choch_lookback).max()
    out["choch_low"] = out["low"].shift(1).rolling(args.choch_lookback).min()
    out["bullish_choch"] = out["close"] > out["choch_high"] + args.choch_buffer
    out["bearish_choch"] = out["close"] < out["choch_low"] - args.choch_buffer
    out["bullish_fvg"] = out["low"] > out["high"].shift(2)
    out["bearish_fvg"] = out["high"] < out["low"].shift(2)
    out["recent_bullish_fvg"] = out["bullish_fvg"].shift(1).rolling(args.fvg_lookback).max()
    out["recent_bearish_fvg"] = out["bearish_fvg"].shift(1).rolling(args.fvg_lookback).max()
    last_bear_high = out["high"].where(out["close"] < out["open"]).ffill().shift(1)
    last_bear_low = out["low"].where(out["close"] < out["open"]).ffill().shift(1)
    last_bull_high = out["high"].where(out["close"] > out["open"]).ffill().shift(1)
    last_bull_low = out["low"].where(out["close"] > out["open"]).ffill().shift(1)
    out["bullish_ob_retest"] = (out["low"] <= last_bear_high + args.ob_buffer) & (out["close"] >= last_bear_low)
    out["bearish_ob_retest"] = (out["high"] >= last_bull_low - args.ob_buffer) & (out["close"] <= last_bull_high)
    return out


def parse_clock(value: str) -> tuple[int, int]:
    hour, minute = value.split(":", 1)
    return int(hour), int(minute)


def in_session(ts: pd.Timestamp, start_text: str, end_text: str) -> bool:
    start_h, start_m = parse_clock(start_text)
    end_h, end_m = parse_clock(end_text)
    after_start = ts.hour > start_h or (ts.hour == start_h and ts.minute >= start_m)
    before_end = ts.hour < end_h or (ts.hour == end_h and ts.minute <= end_m)
    return after_start and before_end


def force_exit_bar(ts: pd.Timestamp, exit_text: str) -> bool:
    hour, minute = parse_clock(exit_text)
    return ts.hour > hour or (ts.hour == hour and ts.minute >= minute)


def selected_stop(side: str, entry: float, row: pd.Series, prev: pd.Series, swing_window: pd.DataFrame, args: argparse.Namespace) -> float:
    if side == "BUY":
        signal_sl = float(prev.low) - args.sl_buffer
        fixed_sl = entry - args.fixed_stop_points
        swing_sl = float(swing_window["low"].min()) - args.sl_buffer
        atr_sl = entry - float(row.atr) * args.atr_stop_mult
        ema34_sl = float(row.ema_mid) - args.sl_buffer
        if args.stop_loss_mode == "fixed":
            return fixed_sl
        if args.stop_loss_mode == "swing":
            return swing_sl
        if args.stop_loss_mode == "atr":
            return atr_sl
        if args.stop_loss_mode == "ema34":
            return ema34_sl
        return signal_sl

    signal_sl = float(prev.high) + args.sl_buffer
    fixed_sl = entry + args.fixed_stop_points
    swing_sl = float(swing_window["high"].max()) + args.sl_buffer
    atr_sl = entry + float(row.atr) * args.atr_stop_mult
    ema34_sl = float(row.ema_mid) + args.sl_buffer
    if args.stop_loss_mode == "fixed":
        return fixed_sl
    if args.stop_loss_mode == "swing":
        return swing_sl
    if args.stop_loss_mode == "atr":
        return atr_sl
    if args.stop_loss_mode == "ema34":
        return ema34_sl
    return signal_sl


def rejection_ok(row: pd.Series, args: argparse.Namespace, side: str) -> bool:
    if args.rejection_mode == "none":
        return True

    candle_range = float(row.range)
    body = max(float(row.body), 0.01)
    if candle_range <= 0:
        return False

    if side == "BUY":
        wick_ok = (
            row.lower_wick >= body * args.min_wick_body_ratio
            and row.lower_wick >= candle_range * args.min_wick_range_pct
        )
        engulfing_ok = bool(row.bullish_engulfing)
    else:
        wick_ok = (
            row.upper_wick >= body * args.min_wick_body_ratio
            and row.upper_wick >= candle_range * args.min_wick_range_pct
        )
        engulfing_ok = bool(row.bearish_engulfing)

    if args.rejection_mode == "wick":
        return bool(wick_ok)
    if args.rejection_mode == "engulfing":
        return bool(engulfing_ok)
    if args.rejection_mode == "both":
        return bool(wick_ok and engulfing_ok)
    return bool(wick_ok or engulfing_ok)


def low_risk_retest_ok(row: pd.Series, args: argparse.Namespace, side: str) -> bool:
    if not args.use_low_risk_retest:
        return False
    if side == "BUY":
        signal_risk = row.close - row.low
    else:
        signal_risk = row.high - row.close
    return bool(0 < signal_risk <= args.max_signal_risk_points)


def trend_quality_ok(row: pd.Series, args: argparse.Namespace, side: str) -> bool:
    if args.use_slope_filter:
        if side == "BUY":
            slope_ok = row.ema_mid_slope >= args.min_slope_points and row.ema_fast_slope > 0
        else:
            slope_ok = row.ema_mid_slope <= -args.min_slope_points and row.ema_fast_slope < 0
        if not bool(slope_ok):
            return False

    if args.use_ribbon_filter:
        if pd.isna(row.ribbon_width_avg) or row.ribbon_width_avg <= 0:
            return False
        if row.ribbon_width < row.ribbon_width_avg * args.min_ribbon_width_mult:
            return False

    if args.use_volume_filter and "volume" in row.index and not pd.isna(row.volume_avg):
        if row.volume < row.volume_avg * args.min_volume_mult:
            return False

    return True


def signal_on_bar(row: pd.Series, args: argparse.Namespace, side: str) -> bool:
    if pd.isna(row.ema_slow) or pd.isna(row.atr) or pd.isna(row.ema_mid_slope):
        return False

    distance_ok = row.distance_from_fast <= args.max_distance_from_fast
    if not distance_ok:
        return False

    if args.use_chop_filter and not bool(row.chop_ok):
        return False

    if not trend_quality_ok(row, args, side):
        return False

    if side == "BUY":
        stack_ok = row.ema_fast > row.ema_mid > row.ema_slow
        support_level = row.ema_channel_low if args.use_ma_channel else row.ema_mid
        support_retest_ok = row.low <= support_level + args.bounce_buffer and row.close >= support_level
        bounce_ok = support_retest_ok and row.close > row.ema_fast
        candle_ok = not args.require_candle_direction or row.close > row.open
        break_ok = not args.entry_on_rejection_break or row.close > row.high - row.range * args.close_break_range_pct
        price_action_ok = rejection_ok(row, args, side) or low_risk_retest_ok(row, args, side)
        return bool(stack_ok and bounce_ok and candle_ok and break_ok and price_action_ok)

    stack_ok = row.ema_fast < row.ema_mid < row.ema_slow
    resistance_level = row.ema_channel_high if args.use_ma_channel else row.ema_mid
    resistance_retest_ok = row.high >= resistance_level - args.bounce_buffer and row.close <= resistance_level
    bounce_ok = resistance_retest_ok and row.close < row.ema_fast
    candle_ok = not args.require_candle_direction or row.close < row.open
    break_ok = not args.entry_on_rejection_break or row.close < row.low + row.range * args.close_break_range_pct
    price_action_ok = rejection_ok(row, args, side) or low_risk_retest_ok(row, args, side)
    return bool(stack_ok and bounce_ok and candle_ok and break_ok and price_action_ok)


def smc_signal_on_bar(row: pd.Series, args: argparse.Namespace, side: str) -> bool:
    if not args.use_smc_setup:
        return False
    if pd.isna(row.amd_range_width) or pd.isna(row.choch_high) or pd.isna(row.choch_low):
        return False
    if args.use_chop_filter and not bool(row.chop_ok):
        return False
    if not trend_quality_ok(row, args, side):
        return False

    accumulation_ok = row.amd_range_width <= args.amd_max_range_points
    if not bool(accumulation_ok):
        return False

    if side == "BUY":
        manipulation_ok = bool(row.recent_sweep_down)
        choch_ok = bool(row.bullish_choch)
        fvg_ob_ok = bool(row.recent_bullish_fvg) or bool(row.bullish_ob_retest)
        candle_ok = not args.require_candle_direction or row.close > row.open
        return bool(manipulation_ok and choch_ok and candle_ok and (fvg_ob_ok or not args.require_fvg_or_ob))

    manipulation_ok = bool(row.recent_sweep_up)
    choch_ok = bool(row.bearish_choch)
    fvg_ob_ok = bool(row.recent_bearish_fvg) or bool(row.bearish_ob_retest)
    candle_ok = not args.require_candle_direction or row.close < row.open
    return bool(manipulation_ok and choch_ok and candle_ok and (fvg_ob_ok or not args.require_fvg_or_ob))


def simulate(timeframe: str, path: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    df = add_indicators(load_data(path), args)
    df["ema_overlap_count"] = (
        ((df["low"] <= df["ema_fast"]) & (df["high"] >= df["ema_fast"])).astype(int)
        + ((df["low"] <= df["ema_mid"]) & (df["high"] >= df["ema_mid"])).astype(int)
        + ((df["low"] <= df["ema_slow"]) & (df["high"] >= df["ema_slow"])).astype(int)
    )
    df["choppy_overlap_bar"] = df["ema_overlap_count"] >= args.chop_min_ema_overlaps
    df["choppy_overlap_count"] = df["choppy_overlap_bar"].shift(1).rolling(args.chop_lookback).sum()
    df["chop_ok"] = df["choppy_overlap_count"] <= args.max_chop_overlap_bars
    trades: list[dict[str, object]] = []
    counts = {
        "buy_signals": 0,
        "sell_signals": 0,
        "smc_buy_signals": 0,
        "smc_sell_signals": 0,
        "buy_entries": 0,
        "sell_entries": 0,
        "ignored": 0,
    }

    in_pos = False
    side = ""
    entry = np.nan
    sl = np.nan
    tp = np.nan
    risk = np.nan
    entry_bar: int | None = None
    best_move = 0.0
    open_trade: dict[str, object] | None = None

    for i, row in df.iterrows():
        ts = row.datetime

        if in_pos and open_trade is not None and entry_bar is not None and i > entry_bar:
            if side == "BUY":
                best_move = max(best_move, row.high - entry)
                hit_sl = row.low <= sl
                hit_tp = row.high >= tp
                close_ema34_exit = row.close <= row.ema_mid
                day_exit = force_exit_bar(ts, args.force_exit)
                profit_lock_active = args.use_trailing_stop and best_move >= args.trail_activate_points
                exit_reason = None
                exit_price = np.nan
                if hit_tp:
                    exit_reason, exit_price = "TARGET", tp
                elif args.conservative_intrabar and hit_sl:
                    exit_reason, exit_price = "SL", sl
                elif hit_sl:
                    exit_reason, exit_price = "SL", sl
                elif close_ema34_exit:
                    exit_reason, exit_price = ("PROFIT LOCK", entry + args.trail_lock_points) if profit_lock_active else ("CLOSE BELOW EMA34", row.close)
                elif day_exit:
                    exit_reason, exit_price = ("PROFIT LOCK", entry + args.trail_lock_points) if profit_lock_active else ("DAY", row.close)
                if exit_reason:
                    pnl = exit_price - entry
                    open_trade.update({"exit_time": ts, "exit": exit_price, "pnl": pnl, "profit_risk_multiple": pnl / risk if risk else np.nan, "best_move": best_move, "final_sl": sl, "bars": i - entry_bar, "exit_type": exit_reason})
                    trades.append(open_trade)
                    in_pos = False
                    open_trade = None
            else:
                best_move = max(best_move, entry - row.low)
                hit_sl = row.high >= sl
                hit_tp = row.low <= tp
                close_ema34_exit = row.close >= row.ema_mid
                day_exit = force_exit_bar(ts, args.force_exit)
                profit_lock_active = args.use_trailing_stop and best_move >= args.trail_activate_points
                exit_reason = None
                exit_price = np.nan
                if hit_tp:
                    exit_reason, exit_price = "TARGET", tp
                elif args.conservative_intrabar and hit_sl:
                    exit_reason, exit_price = "SL", sl
                elif hit_sl:
                    exit_reason, exit_price = "SL", sl
                elif close_ema34_exit:
                    exit_reason, exit_price = ("PROFIT LOCK", entry - args.trail_lock_points) if profit_lock_active else ("CLOSE ABOVE EMA34", row.close)
                elif day_exit:
                    exit_reason, exit_price = ("PROFIT LOCK", entry - args.trail_lock_points) if profit_lock_active else ("DAY", row.close)
                if exit_reason:
                    pnl = entry - exit_price
                    open_trade.update({"exit_time": ts, "exit": exit_price, "pnl": pnl, "profit_risk_multiple": pnl / risk if risk else np.nan, "best_move": best_move, "final_sl": sl, "bars": i - entry_bar, "exit_type": exit_reason})
                    trades.append(open_trade)
                    in_pos = False
                    open_trade = None

        if i == 0 or in_pos:
            continue

        prev = df.iloc[i - 1]
        buy_ema_setup = signal_on_bar(prev, args, "BUY")
        sell_ema_setup = signal_on_bar(prev, args, "SELL")
        buy_smc_setup = smc_signal_on_bar(prev, args, "BUY")
        sell_smc_setup = smc_signal_on_bar(prev, args, "SELL")
        buy_setup = buy_ema_setup or (args.trade_smc_setup and buy_smc_setup)
        sell_setup = sell_ema_setup or (args.trade_smc_setup and sell_smc_setup)

        if buy_setup:
            counts["buy_signals"] += 1
        if sell_setup:
            counts["sell_signals"] += 1
        if buy_smc_setup:
            counts["smc_buy_signals"] += 1
        if sell_smc_setup:
            counts["smc_sell_signals"] += 1

        if not in_session(ts, args.entry_start, args.entry_end):
            if buy_setup or sell_setup:
                counts["ignored"] += 1
            continue

        swing_start = max(0, i - args.swing_lookback)
        swing_window = df.iloc[swing_start:i]
        if swing_window.empty:
            continue

        if buy_setup and args.side in {"both", "buy"}:
            setup_name = "EMA" if buy_ema_setup else "SMC"
            entry = row.open if args.entry_price == "open" else row.close
            sl = selected_stop("BUY", entry, row, prev, swing_window, args)
            risk = entry - sl
            if risk <= 0:
                counts["ignored"] += 1
                continue
            tp = entry + args.target_points
            side = "BUY"
            in_pos = True
            entry_bar = i
            best_move = 0.0
            counts["buy_entries"] += 1
            reason = "EMA34 dynamic support retest with bullish rejection" if setup_name == "EMA" else "AMD sweep + bullish CHOCH with FVG/OB confirmation"
            open_trade = {"timeframe": timeframe, "setup": setup_name, "side": side, "entry_time": ts, "entry": entry, "initial_sl": sl, "target": tp, "risk": risk, "ema_fast": row.ema_fast, "ema_mid": row.ema_mid, "ema_slow": row.ema_slow, "reason": reason}
        elif sell_setup and args.side in {"both", "sell"}:
            setup_name = "EMA" if sell_ema_setup else "SMC"
            entry = row.open if args.entry_price == "open" else row.close
            sl = selected_stop("SELL", entry, row, prev, swing_window, args)
            risk = sl - entry
            if risk <= 0:
                counts["ignored"] += 1
                continue
            tp = entry - args.target_points
            side = "SELL"
            in_pos = True
            entry_bar = i
            best_move = 0.0
            counts["sell_entries"] += 1
            reason = "EMA34 dynamic resistance retest with bearish rejection" if setup_name == "EMA" else "AMD sweep + bearish CHOCH with FVG/OB confirmation"
            open_trade = {"timeframe": timeframe, "setup": setup_name, "side": side, "entry_time": ts, "entry": entry, "initial_sl": sl, "target": tp, "risk": risk, "ema_fast": row.ema_fast, "ema_mid": row.ema_mid, "ema_slow": row.ema_slow, "reason": reason}

    return df, pd.DataFrame(trades), counts


def profit_factor(trades: pd.DataFrame) -> float:
    losses = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
    wins = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    return wins if losses == 0 else wins / losses


def max_drawdown(points: pd.Series) -> float:
    equity = pd.concat([pd.Series([0.0]), points.reset_index(drop=True)]).cumsum()
    drawdown = equity - equity.cummax()
    return float(drawdown.min()) if len(drawdown) else 0.0


def fmt(value: object, header: str) -> str:
    if isinstance(value, float):
        text = f"{value:.2f}"
        if header in {"NetPts", "AvgPts", "PnL", "ProfitRiskMultiple"}:
            if value > 0:
                return color_text(text, COLOR_GREEN)
            if value < 0:
                return color_text(text, COLOR_RED)
        if header == "MaxDrawDown" and value < 0:
            return color_text(text, COLOR_RED)
        if header == "Win%":
            if value >= 50:
                return color_text(text, COLOR_GREEN)
            if value < 30:
                return color_text(text, COLOR_RED)
        return text
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M")
    if header == "Side":
        return color_text(str(value), COLOR_GREEN if value == "BUY" else COLOR_RED)
    return str(value)


def print_table(title: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        print(f"\n{title}\n(no rows)")
        return
    headers = list(rows[0].keys())
    str_rows = [[fmt(row.get(header, ""), header) for header in headers] for row in rows]
    clean = lambda text: text.replace(COLOR_GREEN, "").replace(COLOR_RED, "").replace(COLOR_RESET, "")
    widths = [max(len(header), *(len(clean(cell)) for cell in col)) for header, col in zip(headers, zip(*str_rows))]
    print(f"\n{title}")
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in str_rows:
        print(" | ".join(row[i].rjust(widths[i]) for i in range(len(headers))))


def summary_row(timeframe: str, df: pd.DataFrame, trades: pd.DataFrame, counts: dict[str, int]) -> dict[str, object]:
    if trades.empty:
        return {"TF": timeframe, "Bars": len(df), "Buy": counts["buy_entries"], "Sell": counts["sell_entries"], "Ignored": counts["ignored"], "Closed": 0, "WinTrades": 0, "LossTrades": 0, "Win%": 0.0, "NetPts": 0.0, "AvgPts": 0.0, "AvgRiskPts": 0.0, "PF": 0.0, "MaxDrawDown": 0.0}
    wins = int((trades["pnl"] > 0).sum())
    losses = int((trades["pnl"] < 0).sum())
    return {"TF": timeframe, "Bars": len(df), "Buy": counts["buy_entries"], "Sell": counts["sell_entries"], "Ignored": counts["ignored"], "Closed": len(trades), "WinTrades": wins, "LossTrades": losses, "Win%": wins / len(trades) * 100, "NetPts": trades["pnl"].sum(), "AvgPts": trades["pnl"].mean(), "AvgRiskPts": trades["risk"].mean(), "PF": profit_factor(trades), "MaxDrawDown": max_drawdown(trades.sort_values("exit_time")["pnl"])}


def group_rows(trades: pd.DataFrame, group_col: str, label: str) -> list[dict[str, object]]:
    rows = []
    for key, group in trades.groupby(group_col):
        wins = int((group["pnl"] > 0).sum())
        rows.append({label: key, "Trades": len(group), "Win%": wins / len(group) * 100, "NetPts": group["pnl"].sum(), "AvgPts": group["pnl"].mean(), "AvgRiskPts": group["risk"].mean()})
    return rows


def trade_rows(trades: pd.DataFrame, recent: int, best: bool = False) -> list[dict[str, object]]:
    rows = []
    data = trades.sort_values(["pnl", "profit_risk_multiple"], ascending=[False, False]).head(recent) if best else trades.sort_values("exit_time").tail(recent)
    for row in data.itertuples():
        rows.append({"Setup": row.setup, "Side": row.side, "EntryTime": row.entry_time, "Entry": row.entry, "RiskPts": row.risk, "SL": row.initial_sl, "Target": row.target, "BestMovePts": row.best_move, "ExitTime": row.exit_time, "Exit": row.exit, "PnL": row.pnl, "ProfitRiskMultiple": row.profit_risk_multiple, "Bars": row.bars, "ExitType": row.exit_type})
    return rows


def config_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    return [
        {"Setting": "Fast EMA", "Value": args.fast_ema},
        {"Setting": "Mid/Bounce EMA", "Value": args.mid_ema},
        {"Setting": "Slow EMA", "Value": args.slow_ema},
        {"Setting": "Target Points", "Value": args.target_points},
        {"Setting": "Max Distance From EMA14", "Value": args.max_distance_from_fast},
        {"Setting": "Bounce Buffer", "Value": args.bounce_buffer},
        {"Setting": "Require Candle Direction", "Value": args.require_candle_direction},
        {"Setting": "Rejection Mode", "Value": args.rejection_mode},
        {"Setting": "Min Wick/Body Ratio", "Value": args.min_wick_body_ratio},
        {"Setting": "Min Wick/Range %", "Value": args.min_wick_range_pct},
        {"Setting": "Low Risk Retest", "Value": args.use_low_risk_retest},
        {"Setting": "Max Signal Risk Points", "Value": args.max_signal_risk_points},
        {"Setting": "MA High/Low Channel", "Value": args.use_ma_channel},
        {"Setting": "Slope Filter", "Value": args.use_slope_filter},
        {"Setting": "Slope Lookback", "Value": args.slope_lookback},
        {"Setting": "Min Slope Points", "Value": args.min_slope_points},
        {"Setting": "Ribbon Filter", "Value": args.use_ribbon_filter},
        {"Setting": "Ribbon Lookback", "Value": args.ribbon_lookback},
        {"Setting": "Min Ribbon Width Mult", "Value": args.min_ribbon_width_mult},
        {"Setting": "Volume Filter", "Value": args.use_volume_filter},
        {"Setting": "Min Volume Mult", "Value": args.min_volume_mult},
        {"Setting": "Close Break Filter", "Value": args.entry_on_rejection_break},
        {"Setting": "SMC Setup", "Value": args.use_smc_setup},
        {"Setting": "Trade SMC Setup", "Value": args.trade_smc_setup},
        {"Setting": "AMD Lookback", "Value": args.amd_lookback},
        {"Setting": "AMD Max Range Points", "Value": args.amd_max_range_points},
        {"Setting": "SMC Sweep Lookback", "Value": args.smc_sweep_lookback},
        {"Setting": "SMC Sweep Buffer", "Value": args.smc_sweep_buffer},
        {"Setting": "CHOCH Lookback", "Value": args.choch_lookback},
        {"Setting": "CHOCH Buffer", "Value": args.choch_buffer},
        {"Setting": "FVG Lookback", "Value": args.fvg_lookback},
        {"Setting": "Require FVG/OB", "Value": args.require_fvg_or_ob},
        {"Setting": "Chop Filter", "Value": args.use_chop_filter},
        {"Setting": "Chop Lookback", "Value": args.chop_lookback},
        {"Setting": "Chop Min EMA Overlaps", "Value": args.chop_min_ema_overlaps},
        {"Setting": "Max Choppy Overlap Bars", "Value": args.max_chop_overlap_bars},
        {"Setting": "Stop Loss Mode", "Value": args.stop_loss_mode},
        {"Setting": "Fixed Stop Points", "Value": args.fixed_stop_points},
        {"Setting": "Trailing Stop", "Value": args.use_trailing_stop},
        {"Setting": "Trail Activate Points", "Value": args.trail_activate_points},
        {"Setting": "Trail Lock Points", "Value": args.trail_lock_points},
        {"Setting": "Entry Price", "Value": args.entry_price},
        {"Setting": "Entry Window", "Value": f"{args.entry_start}-{args.entry_end}"},
        {"Setting": "Force Exit", "Value": args.force_exit},
    ]


def run(args: argparse.Namespace) -> None:
    global USE_COLOR
    USE_COLOR = not args.no_color
    files = data_files(resolve_data_dir(args.data_dir))
    print_table("Active Config", config_rows(args))

    all_trades: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for timeframe in args.timeframes:
        df, trades, counts = simulate(timeframe, files[timeframe], args)
        all_trades.append(trades)
        summaries.append(summary_row(timeframe, df, trades, counts))
        print(f"\n=== {timeframe} | EMA {args.fast_ema}/{args.mid_ema}/{args.slow_ema} Bounce ===")
        print(f"Data: {df.datetime.min()} -> {df.datetime.max()} | bars={len(df)}")
        print(f"Signals: buy={counts['buy_signals']} sell={counts['sell_signals']} | SMC candidates: buy={counts['smc_buy_signals']} sell={counts['smc_sell_signals']} | Entries: buy={counts['buy_entries']} sell={counts['sell_entries']} | Ignored={counts['ignored']}")
        if not trades.empty:
            print_table(f"{timeframe} By Setup", group_rows(trades, "setup", "Setup"))
            print_table(f"{timeframe} By Side", group_rows(trades, "side", "Side"))
            print_table(f"{timeframe} By Entry Hour", group_rows(trades.assign(hour=trades["entry_time"].dt.hour), "hour", "Hour"))
            print_table(f"{timeframe} Best Trades", trade_rows(trades, args.best_trades, best=True))
            print_table(f"{timeframe} Recent Trades", trade_rows(trades, args.recent))
            if args.export:
                output = Path(f"ema_14_34_90_trades_{timeframe}.csv")
                trades.to_csv(output, index=False)
                print(f"Exported: {output}")
        else:
            print_table(f"{timeframe} Recent Trades", [])

    print_table("Timeframe Comparison", summaries)
    combined = pd.concat([t for t in all_trades if not t.empty], ignore_index=True) if any(not t.empty for t in all_trades) else pd.DataFrame()
    if not combined.empty:
        overall = summary_row("ALL", pd.DataFrame(index=range(sum(len(load_data(files[tf])) for tf in args.timeframes))), combined, {"buy_entries": 0, "sell_entries": 0, "ignored": 0})
        print_table("Overall P/L", [overall])
        if args.export:
            combined.to_csv("ema_14_34_90_trades_all.csv", index=False)
            print("Exported: ema_14_34_90_trades_all.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest EMA 14/34/90 bounce strategy on NIFTY CSVs.")
    parser.add_argument("--data-dir", help="Folder containing NIFTY_*_ANALYSIS.csv files. Auto-detected when omitted.")
    parser.add_argument("--timeframes", nargs="+", choices=sorted(CSV_NAMES), default=["1m", "5m"])
    parser.add_argument("--fast-ema", type=int, default=14)
    parser.add_argument("--mid-ema", type=int, default=34)
    parser.add_argument("--slow-ema", type=int, default=90)
    parser.add_argument("--target-points", type=float, default=50.0)
    parser.add_argument("--max-distance-from-fast", type=float, default=15.0)
    parser.add_argument("--bounce-buffer", type=float, default=2.0)
    parser.add_argument("--require-candle-direction", type=bool_arg, default=True)
    parser.add_argument("--rejection-mode", choices=["none", "wick", "engulfing", "either", "both"], default="wick")
    parser.add_argument("--min-wick-body-ratio", type=float, default=0.8)
    parser.add_argument("--min-wick-range-pct", type=float, default=0.25)
    parser.add_argument("--use-low-risk-retest", type=bool_arg, default=True)
    parser.add_argument("--max-signal-risk-points", type=float, default=7.0)
    parser.add_argument("--use-ma-channel", type=bool_arg, default=False)
    parser.add_argument("--use-slope-filter", type=bool_arg, default=True)
    parser.add_argument("--slope-lookback", type=int, default=5)
    parser.add_argument("--min-slope-points", type=float, default=0.5)
    parser.add_argument("--use-ribbon-filter", type=bool_arg, default=True)
    parser.add_argument("--ribbon-lookback", type=int, default=20)
    parser.add_argument("--min-ribbon-width-mult", type=float, default=0.9)
    parser.add_argument("--use-volume-filter", type=bool_arg, default=False)
    parser.add_argument("--volume-lookback", type=int, default=20)
    parser.add_argument("--min-volume-mult", type=float, default=0.9)
    parser.add_argument("--entry-on-rejection-break", type=bool_arg, default=False)
    parser.add_argument("--close-break-range-pct", type=float, default=0.25)
    parser.add_argument("--use-smc-setup", type=bool_arg, default=False)
    parser.add_argument("--trade-smc-setup", type=bool_arg, default=False)
    parser.add_argument("--amd-lookback", type=int, default=20)
    parser.add_argument("--amd-max-range-points", type=float, default=120.0)
    parser.add_argument("--smc-sweep-lookback", type=int, default=8)
    parser.add_argument("--smc-sweep-buffer", type=float, default=0.0)
    parser.add_argument("--choch-lookback", type=int, default=5)
    parser.add_argument("--choch-buffer", type=float, default=0.0)
    parser.add_argument("--fvg-lookback", type=int, default=5)
    parser.add_argument("--ob-buffer", type=float, default=2.0)
    parser.add_argument("--require-fvg-or-ob", type=bool_arg, default=True)
    parser.add_argument("--use-bos-filter", type=bool_arg, default=False, help=argparse.SUPPRESS)
    parser.add_argument("--bos-lookback", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--use-chop-filter", type=bool_arg, default=False)
    parser.add_argument("--chop-lookback", type=int, default=5)
    parser.add_argument("--chop-min-ema-overlaps", type=int, choices=[2, 3], default=3)
    parser.add_argument("--max-chop-overlap-bars", type=int, default=2)
    parser.add_argument("--swing-lookback", type=int, default=5)
    parser.add_argument("--stop-loss-mode", choices=["signal", "fixed", "swing", "atr", "ema34"], default="signal")
    parser.add_argument("--fixed-stop-points", type=float, default=10.0)
    parser.add_argument("--use-trailing-stop", type=bool_arg, default=True)
    parser.add_argument("--trail-activate-points", type=float, default=30.0)
    parser.add_argument("--trail-lock-points", type=float, default=30.0)
    parser.add_argument("--atr-stop-len", type=int, default=14)
    parser.add_argument("--atr-stop-mult", type=float, default=1.0)
    parser.add_argument("--sl-buffer", type=float, default=0.0)
    parser.add_argument("--entry-price", choices=["open", "close"], default="open")
    parser.add_argument("--entry-start", default="12:00")
    parser.add_argument("--entry-end", default="13:59")
    parser.add_argument("--force-exit", default="15:15")
    parser.add_argument("--side", choices=["both", "buy", "sell"], default="both")
    parser.add_argument("--conservative-intrabar", type=bool_arg, default=True)
    parser.add_argument("--recent", type=int, default=20)
    parser.add_argument("--best-trades", type=int, default=10)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
