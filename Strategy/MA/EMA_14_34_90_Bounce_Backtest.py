from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_NAMES = {
    "1m": "nifty_1min_clean.csv",
    "5m": "nifty_5min_clean.csv",
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
    out["rsi_fast"] = rsi(out["close"], args.rsi_fast_len)
    out["rsi_trend"] = rsi(out["close"], args.rsi_trend_len)
    out["rsi_fast_prev"] = out["rsi_fast"].shift(1)
    out["rsi_prev_low"] = out["rsi_fast"].shift(1).rolling(args.rsi_retest_bars).min()
    out["rsi_prev_high"] = out["rsi_fast"].shift(1).rolling(args.rsi_retest_bars).max()
    out["rsi_break_high"] = out["high"].shift(1).rolling(args.rsi_breakout_lookback).max()
    out["rsi_break_low"] = out["low"].shift(1).rolling(args.rsi_breakout_lookback).min()
    return add_smc_indicators(out, args)


def rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    value = value.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    value = value.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    value = value.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return value


def add_smc_indicators(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()

    day_key = out["datetime"].dt.normalize()
    out["day_high"] = out.groupby(day_key)["high"].cummax()
    out["day_low"] = out.groupby(day_key)["low"].cummin()
    daily = out.groupby(day_key).agg(day_full_high=("high", "max"), day_full_low=("low", "min"))
    out["prev_day_high"] = day_key.map(daily["day_full_high"].shift(1))
    out["prev_day_low"] = day_key.map(daily["day_full_low"].shift(1))

    htf_key = out["datetime"].dt.floor(f"{args.smc_htf_minutes}min")
    htf = out.groupby(htf_key).agg(htf_full_high=("high", "max"), htf_full_low=("low", "min"))
    out["prev_htf_high"] = htf_key.map(htf["htf_full_high"].shift(1))
    out["prev_htf_low"] = htf_key.map(htf["htf_full_low"].shift(1))

    swing_lb = args.smc_swing_lookback
    out["swing_high_level"] = out["high"].shift(1).rolling(swing_lb).max()
    out["swing_low_level"] = out["low"].shift(1).rolling(swing_lb).min()
    out["support_level"] = out["swing_low_level"]
    out["resistance_level"] = out["swing_high_level"]

    sweep_buffer = args.smc_sweep_buffer
    out["buy_liquidity_sweep"] = (
        ((out["low"] < out["swing_low_level"] - sweep_buffer) | (out["low"] < out["prev_day_low"] - sweep_buffer) | (out["low"] < out["prev_htf_low"] - sweep_buffer))
        & (out["close"] > out[["swing_low_level", "prev_day_low", "prev_htf_low"]].min(axis=1))
    )
    out["sell_liquidity_sweep"] = (
        ((out["high"] > out["swing_high_level"] + sweep_buffer) | (out["high"] > out["prev_day_high"] + sweep_buffer) | (out["high"] > out["prev_htf_high"] + sweep_buffer))
        & (out["close"] < out[["swing_high_level", "prev_day_high", "prev_htf_high"]].max(axis=1))
    )
    out["turtle_soup_buy"] = (out["low"] < out["prev_day_low"] - sweep_buffer) & (out["close"] > out["prev_day_low"])
    out["turtle_soup_sell"] = (out["high"] > out["prev_day_high"] + sweep_buffer) & (out["close"] < out["prev_day_high"])
    out["crt_buy"] = (out["low"] < out["low"].shift(1) - sweep_buffer) & (out["close"] > out["high"].shift(1))
    out["crt_sell"] = (out["high"] > out["high"].shift(1) + sweep_buffer) & (out["close"] < out["low"].shift(1))

    out["bos_buy"] = out["close"] > out["swing_high_level"] + args.smc_break_buffer
    out["bos_sell"] = out["close"] < out["swing_low_level"] - args.smc_break_buffer
    recent_buy_sweep = out["buy_liquidity_sweep"].shift(1).rolling(args.smc_choch_lookback).max().fillna(False).astype(bool)
    recent_sell_sweep = out["sell_liquidity_sweep"].shift(1).rolling(args.smc_choch_lookback).max().fillna(False).astype(bool)
    out["choch_buy"] = recent_buy_sweep & out["bos_buy"]
    out["choch_sell"] = recent_sell_sweep & out["bos_sell"]
    out["cisd_buy"] = recent_buy_sweep & (out["close"] > out[["open", "close"]].max(axis=1).shift(1).rolling(args.smc_cisd_lookback).max())
    out["cisd_sell"] = recent_sell_sweep & (out["close"] < out[["open", "close"]].min(axis=1).shift(1).rolling(args.smc_cisd_lookback).min())

    out["bull_fvg"] = (out["low"] > out["high"].shift(2) + args.smc_zone_buffer) & (out["close"] > out["open"])
    out["bear_fvg"] = (out["high"] < out["low"].shift(2) - args.smc_zone_buffer) & (out["close"] < out["open"])
    out["bull_fvg_low"] = out["high"].shift(2).where(out["bull_fvg"]).ffill()
    out["bull_fvg_high"] = out["low"].where(out["bull_fvg"]).ffill()
    out["bear_fvg_low"] = out["high"].where(out["bear_fvg"]).ffill()
    out["bear_fvg_high"] = out["low"].shift(2).where(out["bear_fvg"]).ffill()
    out["in_bull_fvg"] = (out["low"] <= out["bull_fvg_high"] + args.smc_zone_buffer) & (out["high"] >= out["bull_fvg_low"] - args.smc_zone_buffer)
    out["in_bear_fvg"] = (out["high"] >= out["bear_fvg_low"] - args.smc_zone_buffer) & (out["low"] <= out["bear_fvg_high"] + args.smc_zone_buffer)
    out["bull_ifvg"] = out["bear_fvg"].shift(1).rolling(args.smc_zone_lookback).max().fillna(False).astype(bool) & (out["close"] > out["bear_fvg_high"])
    out["bear_ifvg"] = out["bull_fvg"].shift(1).rolling(args.smc_zone_lookback).max().fillna(False).astype(bool) & (out["close"] < out["bull_fvg_low"])

    out["bull_ob"] = (out["close"].shift(1) < out["open"].shift(1)) & out["bos_buy"]
    out["bear_ob"] = (out["close"].shift(1) > out["open"].shift(1)) & out["bos_sell"]
    out["bull_ob_low"] = out["low"].shift(1).where(out["bull_ob"]).ffill()
    out["bull_ob_high"] = out["open"].shift(1).where(out["bull_ob"]).ffill()
    out["bear_ob_low"] = out["open"].shift(1).where(out["bear_ob"]).ffill()
    out["bear_ob_high"] = out["high"].shift(1).where(out["bear_ob"]).ffill()
    out["in_bull_ob"] = (out["low"] <= out["bull_ob_high"] + args.smc_zone_buffer) & (out["high"] >= out["bull_ob_low"] - args.smc_zone_buffer)
    out["in_bear_ob"] = (out["high"] >= out["bear_ob_low"] - args.smc_zone_buffer) & (out["low"] <= out["bear_ob_high"] + args.smc_zone_buffer)
    out["bull_breaker"] = out["bear_ob"].shift(1).rolling(args.smc_zone_lookback).max().fillna(False).astype(bool) & (out["close"] > out["bear_ob_high"])
    out["bear_breaker"] = out["bull_ob"].shift(1).rolling(args.smc_zone_lookback).max().fillna(False).astype(bool) & (out["close"] < out["bull_ob_low"])

    range_avg = out["range"].rolling(args.smc_amd_lookback).mean()
    range_high = out["high"].shift(1).rolling(args.smc_amd_lookback).max()
    range_low = out["low"].shift(1).rolling(args.smc_amd_lookback).min()
    out["accumulation_zone"] = (range_high - range_low) <= args.smc_amd_max_range_points
    accumulation_prev = pd.Series(np.r_[False, out["accumulation_zone"].to_numpy(dtype=bool)[:-1]], index=out.index)
    out["amd_buy"] = accumulation_prev & out["buy_liquidity_sweep"] & (out["close"] > range_high)
    out["amd_sell"] = accumulation_prev & out["sell_liquidity_sweep"] & (out["close"] < range_low)
    out["displacement_buy"] = out["close"] > out["open"] + range_avg * args.smc_displacement_mult
    out["displacement_sell"] = out["close"] < out["open"] - range_avg * args.smc_displacement_mult

    sweep_count = (out["buy_liquidity_sweep"] | out["sell_liquidity_sweep"]).shift(1).rolling(args.smc_trap_lookback).sum()
    displacement_count = (out["displacement_buy"] | out["displacement_sell"]).shift(1).rolling(args.smc_trap_lookback).sum()
    out["trap_zone"] = (sweep_count >= args.smc_trap_min_sweeps) & (displacement_count == 0)
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


def setup_entry_session_ok(ts: pd.Timestamp, setup: str, side: str, args: argparse.Namespace) -> bool:
    if setup == "EMA" and side == "BUY":
        return in_session(ts, args.ema_buy_entry_start, args.ema_buy_entry_end)
    if setup == "EMA" and side == "SELL":
        return in_session(ts, args.ema_sell_entry_start, args.ema_sell_entry_end)
    if setup == "RSI":
        return in_session(ts, args.rsi_entry_start, args.rsi_entry_end)
    if setup == "SMC":
        return in_session(ts, args.smc_entry_start, args.smc_entry_end)
    return True


def force_exit_bar(ts: pd.Timestamp, exit_text: str) -> bool:
    hour, minute = parse_clock(exit_text)
    return ts.hour > hour or (ts.hour == hour and ts.minute >= minute)


def timeframe_minutes(timeframe: str) -> int:
    if timeframe.endswith("m"):
        return int(timeframe[:-1])
    if timeframe.endswith("h"):
        return int(timeframe[:-1]) * 60
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def rsi_trade_allowed(timeframe: str, side: str, args: argparse.Namespace) -> bool:
    if timeframe_minutes(timeframe) < args.rsi_min_trade_timeframe_minutes:
        return False
    if args.rsi_trade_side == "buy" and side != "BUY":
        return False
    if args.rsi_trade_side == "sell" and side != "SELL":
        return False
    return True


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


def rsi_price_action_ok(row: pd.Series, args: argparse.Namespace, side: str) -> bool:
    if args.rsi_price_action_confirm == "none":
        return True

    if side == "BUY":
        breakout_ok = not pd.isna(row.rsi_break_high) and row.close > row.rsi_break_high + args.rsi_breakout_buffer
        rejection_confirm = row.close > row.open and rejection_ok(row, args, side)
    else:
        breakout_ok = not pd.isna(row.rsi_break_low) and row.close < row.rsi_break_low - args.rsi_breakout_buffer
        rejection_confirm = row.close < row.open and rejection_ok(row, args, side)

    if args.rsi_price_action_confirm == "breakout":
        return bool(breakout_ok)
    if args.rsi_price_action_confirm == "rejection":
        return bool(rejection_confirm)
    return bool(breakout_ok or rejection_confirm)


def rsi_signal_on_bar(row: pd.Series, args: argparse.Namespace, side: str) -> bool:
    if not args.use_rsi_setup:
        return False
    needed = [row.rsi_fast, row.rsi_fast_prev, row.rsi_trend, row.rsi_prev_low, row.rsi_prev_high]
    if any(pd.isna(value) for value in needed):
        return False
    if args.use_chop_filter and not bool(row.chop_ok):
        return False
    if args.rsi_use_ema_trend_filter:
        if side == "BUY":
            if not bool(row.ema_fast > row.ema_mid > row.ema_slow):
                return False
        else:
            if not bool(row.ema_fast < row.ema_mid < row.ema_slow):
                return False
        if not trend_quality_ok(row, args, side):
            return False
    if not rsi_price_action_ok(row, args, side):
        return False
    if args.rsi_avoid_displacement_candle:
        if side == "BUY" and bool(row.displacement_buy):
            return False
        if side == "SELL" and bool(row.displacement_sell):
            return False

    if side == "BUY":
        trend_ok = row.rsi_trend >= 50 + args.rsi_trend_buffer
        momentum_cross = row.rsi_fast_prev <= args.rsi_buy_momentum_level and row.rsi_fast > args.rsi_buy_momentum_level
        pullback_bounce = row.rsi_prev_low <= args.rsi_buy_pullback_level and row.rsi_fast > row.rsi_fast_prev and row.rsi_fast >= args.rsi_buy_pullback_level
        return bool(trend_ok and (momentum_cross or pullback_bounce))

    trend_ok = row.rsi_trend <= 50 - args.rsi_trend_buffer
    momentum_cross = row.rsi_fast_prev >= args.rsi_sell_momentum_level and row.rsi_fast < args.rsi_sell_momentum_level
    pullback_reject = row.rsi_prev_high >= args.rsi_sell_pullback_level and row.rsi_fast < row.rsi_fast_prev and row.rsi_fast <= args.rsi_sell_pullback_level
    return bool(trend_ok and (momentum_cross or pullback_reject))


def smc_price_near_level(row: pd.Series, args: argparse.Namespace, side: str) -> bool:
    levels = ["support_level", "prev_day_low", "prev_htf_low"] if side == "BUY" else ["resistance_level", "prev_day_high", "prev_htf_high"]
    price = row.low if side == "BUY" else row.high
    for name in levels:
        level = row.get(name, np.nan)
        if not pd.isna(level) and abs(price - level) <= args.smc_level_buffer:
            return True
    return False


def smc_signal_on_bar(row: pd.Series, args: argparse.Namespace, side: str) -> bool:
    if not args.use_smc_setup:
        return False
    if args.use_chop_filter and not bool(row.chop_ok):
        return False
    if args.smc_avoid_trap_zones and bool(row.trap_zone):
        return False

    if side == "BUY":
        sweep_ok = bool(row.buy_liquidity_sweep or row.turtle_soup_buy or row.crt_buy)
        structure_ok = bool(row.choch_buy or row.cisd_buy or row.bos_buy)
        zone_ok = bool(row.in_bull_fvg or row.bull_ifvg or row.in_bull_ob or row.bull_breaker or row.amd_buy or smc_price_near_level(row, args, side))
        displacement_ok = not args.smc_require_displacement or bool(row.displacement_buy)
        candle_ok = row.close > row.open
        return bool(sweep_ok and structure_ok and zone_ok and displacement_ok and candle_ok)

    sweep_ok = bool(row.sell_liquidity_sweep or row.turtle_soup_sell or row.crt_sell)
    structure_ok = bool(row.choch_sell or row.cisd_sell or row.bos_sell)
    zone_ok = bool(row.in_bear_fvg or row.bear_ifvg or row.in_bear_ob or row.bear_breaker or row.amd_sell or smc_price_near_level(row, args, side))
    displacement_ok = not args.smc_require_displacement or bool(row.displacement_sell)
    candle_ok = row.close < row.open
    return bool(sweep_ok and structure_ok and zone_ok and displacement_ok and candle_ok)


def smc_trade_allowed(timeframe: str, side: str, args: argparse.Namespace) -> bool:
    if timeframe_minutes(timeframe) < args.smc_min_trade_timeframe_minutes:
        return False
    if args.smc_trade_side == "buy" and side != "BUY":
        return False
    if args.smc_trade_side == "sell" and side != "SELL":
        return False
    return True


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
        "buy_entries": 0,
        "sell_entries": 0,
        "ignored": 0,
        "rsi_buy_candidates": 0,
        "rsi_sell_candidates": 0,
        "rsi_buy_entries": 0,
        "rsi_sell_entries": 0,
        "smc_buy_candidates": 0,
        "smc_sell_candidates": 0,
        "smc_buy_entries": 0,
        "smc_sell_entries": 0,
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
        ema_buy_setup = signal_on_bar(prev, args, "BUY")
        ema_sell_setup = signal_on_bar(prev, args, "SELL")
        rsi_buy_setup = rsi_signal_on_bar(prev, args, "BUY")
        rsi_sell_setup = rsi_signal_on_bar(prev, args, "SELL")
        smc_buy_setup = smc_signal_on_bar(prev, args, "BUY")
        smc_sell_setup = smc_signal_on_bar(prev, args, "SELL")
        trade_rsi_buy = args.trade_rsi_setup and rsi_buy_setup and rsi_trade_allowed(timeframe, "BUY", args)
        trade_rsi_sell = args.trade_rsi_setup and rsi_sell_setup and rsi_trade_allowed(timeframe, "SELL", args)
        trade_smc_buy = args.trade_smc_setup and smc_buy_setup and smc_trade_allowed(timeframe, "BUY", args)
        trade_smc_sell = args.trade_smc_setup and smc_sell_setup and smc_trade_allowed(timeframe, "SELL", args)
        buy_setup = ema_buy_setup or trade_rsi_buy or trade_smc_buy
        sell_setup = ema_sell_setup or trade_rsi_sell or trade_smc_sell
        buy_setup_name = "EMA" if ema_buy_setup else "RSI" if trade_rsi_buy else "SMC"
        sell_setup_name = "EMA" if ema_sell_setup else "RSI" if trade_rsi_sell else "SMC"

        if ema_buy_setup:
            counts["buy_signals"] += 1
        if ema_sell_setup:
            counts["sell_signals"] += 1
        if rsi_buy_setup:
            counts["rsi_buy_candidates"] += 1
        if rsi_sell_setup:
            counts["rsi_sell_candidates"] += 1
        if smc_buy_setup:
            counts["smc_buy_candidates"] += 1
        if smc_sell_setup:
            counts["smc_sell_candidates"] += 1

        if not in_session(ts, args.entry_start, args.entry_end):
            if buy_setup or sell_setup:
                counts["ignored"] += 1
            continue

        if buy_setup and not setup_entry_session_ok(ts, buy_setup_name, "BUY", args):
            counts["ignored"] += 1
            buy_setup = False
        if sell_setup and not setup_entry_session_ok(ts, sell_setup_name, "SELL", args):
            counts["ignored"] += 1
            sell_setup = False
        if not buy_setup and not sell_setup:
            continue

        swing_start = max(0, i - args.swing_lookback)
        swing_window = df.iloc[swing_start:i]
        if swing_window.empty:
            continue

        if buy_setup and args.side in {"both", "buy"}:
            entry = row.open if args.entry_price == "open" else row.close
            sl = selected_stop("BUY", entry, row, prev, swing_window, args)
            risk = entry - sl
            if risk <= 0:
                counts["ignored"] += 1
                continue
            if buy_setup_name == "EMA" and risk < args.min_ema_buy_risk_points:
                counts["ignored"] += 1
                continue
            if buy_setup_name == "RSI" and risk > args.max_rsi_risk_points:
                counts["ignored"] += 1
                continue
            if buy_setup_name == "SMC" and risk > args.max_smc_risk_points:
                counts["ignored"] += 1
                continue
            if buy_setup_name == "SMC" and risk < args.min_smc_risk_points:
                counts["ignored"] += 1
                continue
            trade_target_points = args.rsi_target_points if buy_setup_name == "RSI" else args.smc_target_points if buy_setup_name == "SMC" else args.target_points
            tp = entry + trade_target_points
            side = "BUY"
            in_pos = True
            entry_bar = i
            best_move = 0.0
            counts["buy_entries"] += 1
            if buy_setup_name == "RSI":
                counts["rsi_buy_entries"] += 1
            if buy_setup_name == "SMC":
                counts["smc_buy_entries"] += 1
            reason = (
                "EMA34 dynamic support retest with bullish rejection"
                if buy_setup_name == "EMA"
                else "RSI50 bullish bias with RSI10 momentum/pullback confirmation"
                if buy_setup_name == "RSI"
                else "SMC liquidity sweep with CHOCH/CISD and zone confirmation"
            )
            open_trade = {"timeframe": timeframe, "setup": buy_setup_name, "side": side, "entry_time": ts, "entry": entry, "initial_sl": sl, "target": tp, "risk": risk, "ema_fast": row.ema_fast, "ema_mid": row.ema_mid, "ema_slow": row.ema_slow, "rsi_fast": row.rsi_fast, "rsi_trend": row.rsi_trend, "reason": reason}
        elif sell_setup and args.side in {"both", "sell"}:
            entry = row.open if args.entry_price == "open" else row.close
            sl = selected_stop("SELL", entry, row, prev, swing_window, args)
            risk = sl - entry
            if risk <= 0:
                counts["ignored"] += 1
                continue
            if sell_setup_name == "EMA" and risk < args.min_ema_sell_risk_points:
                counts["ignored"] += 1
                continue
            if sell_setup_name == "RSI" and risk > args.max_rsi_risk_points:
                counts["ignored"] += 1
                continue
            if sell_setup_name == "SMC" and risk > args.max_smc_risk_points:
                counts["ignored"] += 1
                continue
            if sell_setup_name == "SMC" and risk < args.min_smc_risk_points:
                counts["ignored"] += 1
                continue
            trade_target_points = args.rsi_target_points if sell_setup_name == "RSI" else args.smc_target_points if sell_setup_name == "SMC" else args.target_points
            tp = entry - trade_target_points
            side = "SELL"
            in_pos = True
            entry_bar = i
            best_move = 0.0
            counts["sell_entries"] += 1
            if sell_setup_name == "RSI":
                counts["rsi_sell_entries"] += 1
            if sell_setup_name == "SMC":
                counts["smc_sell_entries"] += 1
            reason = (
                "EMA34 dynamic resistance retest with bearish rejection"
                if sell_setup_name == "EMA"
                else "RSI50 bearish bias with RSI10 momentum/pullback confirmation"
                if sell_setup_name == "RSI"
                else "SMC liquidity sweep with CHOCH/CISD and zone confirmation"
            )
            open_trade = {"timeframe": timeframe, "setup": sell_setup_name, "side": side, "entry_time": ts, "entry": entry, "initial_sl": sl, "target": tp, "risk": risk, "ema_fast": row.ema_fast, "ema_mid": row.ema_mid, "ema_slow": row.ema_slow, "rsi_fast": row.rsi_fast, "rsi_trend": row.rsi_trend, "reason": reason}

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
        {"Setting": "Min EMA Buy Risk Points", "Value": args.min_ema_buy_risk_points},
        {"Setting": "Min EMA Sell Risk Points", "Value": args.min_ema_sell_risk_points},
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
        {"Setting": "RSI Setup Enabled", "Value": args.use_rsi_setup},
        {"Setting": "Trade RSI Setup", "Value": args.trade_rsi_setup},
        {"Setting": "RSI Fast/Trend", "Value": f"{args.rsi_fast_len}/{args.rsi_trend_len}"},
        {"Setting": "RSI Price Action", "Value": args.rsi_price_action_confirm},
        {"Setting": "RSI EMA Trend Filter", "Value": args.rsi_use_ema_trend_filter},
        {"Setting": "RSI Target Points", "Value": args.rsi_target_points},
        {"Setting": "Max RSI Risk Points", "Value": args.max_rsi_risk_points},
        {"Setting": "RSI Avoid Displacement Candle", "Value": args.rsi_avoid_displacement_candle},
        {"Setting": "RSI Trade Side", "Value": args.rsi_trade_side},
        {"Setting": "RSI Min Trade TF Minutes", "Value": args.rsi_min_trade_timeframe_minutes},
        {"Setting": "RSI Entry Window", "Value": f"{args.rsi_entry_start}-{args.rsi_entry_end}"},
        {"Setting": "SMC Setup Enabled", "Value": args.use_smc_setup},
        {"Setting": "Trade SMC Setup", "Value": args.trade_smc_setup},
        {"Setting": "SMC Components", "Value": "AMD/CISD/CRT/TS/OB/BB/FVG/IFVG/CHOCH/levels"},
        {"Setting": "SMC Target Points", "Value": args.smc_target_points},
        {"Setting": "Max SMC Risk Points", "Value": args.max_smc_risk_points},
        {"Setting": "Min SMC Risk Points", "Value": args.min_smc_risk_points},
        {"Setting": "SMC Trade Side", "Value": args.smc_trade_side},
        {"Setting": "SMC Min Trade TF Minutes", "Value": args.smc_min_trade_timeframe_minutes},
        {"Setting": "SMC Avoid Trap Zones", "Value": args.smc_avoid_trap_zones},
        {"Setting": "SMC Require Displacement", "Value": args.smc_require_displacement},
        {"Setting": "SMC Entry Window", "Value": f"{args.smc_entry_start}-{args.smc_entry_end}"},
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
        {"Setting": "EMA Buy Entry Window", "Value": f"{args.ema_buy_entry_start}-{args.ema_buy_entry_end}"},
        {"Setting": "EMA Sell Entry Window", "Value": f"{args.ema_sell_entry_start}-{args.ema_sell_entry_end}"},
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
        print(f"Signals: buy={counts['buy_signals']} sell={counts['sell_signals']} | Entries: buy={counts['buy_entries']} sell={counts['sell_entries']} | Ignored={counts['ignored']}")
        if args.use_rsi_setup:
            print(f"RSI Candidates: buy={counts['rsi_buy_candidates']} sell={counts['rsi_sell_candidates']} | RSI Entries: buy={counts['rsi_buy_entries']} sell={counts['rsi_sell_entries']} | Trading={'on' if args.trade_rsi_setup else 'off'}")
        if args.use_smc_setup:
            print(f"SMC Candidates: buy={counts['smc_buy_candidates']} sell={counts['smc_sell_candidates']} | SMC Entries: buy={counts['smc_buy_entries']} sell={counts['smc_sell_entries']} | Trading={'on' if args.trade_smc_setup else 'off'}")
        if not trades.empty:
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
    parser.add_argument("--min-ema-buy-risk-points", type=float, default=10.0)
    parser.add_argument("--min-ema-sell-risk-points", type=float, default=5.0)
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
    parser.add_argument("--use-rsi-setup", type=bool_arg, default=False)
    parser.add_argument("--trade-rsi-setup", type=bool_arg, default=False)
    parser.add_argument("--rsi-fast-len", type=int, default=10)
    parser.add_argument("--rsi-trend-len", type=int, default=50)
    parser.add_argument("--rsi-trend-buffer", type=float, default=5.0)
    parser.add_argument("--rsi-buy-momentum-level", type=float, default=60.0)
    parser.add_argument("--rsi-buy-pullback-level", type=float, default=40.0)
    parser.add_argument("--rsi-sell-momentum-level", type=float, default=40.0)
    parser.add_argument("--rsi-sell-pullback-level", type=float, default=60.0)
    parser.add_argument("--rsi-retest-bars", type=int, default=3)
    parser.add_argument("--rsi-breakout-lookback", type=int, default=3)
    parser.add_argument("--rsi-breakout-buffer", type=float, default=0.0)
    parser.add_argument("--rsi-price-action-confirm", choices=["none", "breakout", "rejection", "either"], default="breakout")
    parser.add_argument("--rsi-use-ema-trend-filter", type=bool_arg, default=True)
    parser.add_argument("--rsi-target-points", type=float, default=30.0)
    parser.add_argument("--max-rsi-risk-points", type=float, default=20.0)
    parser.add_argument("--rsi-avoid-displacement-candle", type=bool_arg, default=True)
    parser.add_argument("--rsi-trade-side", choices=["both", "buy", "sell"], default="sell")
    parser.add_argument("--rsi-min-trade-timeframe-minutes", type=int, default=5)
    parser.add_argument("--rsi-entry-start", default="9:15")
    parser.add_argument("--rsi-entry-end", default="14:59")
    parser.add_argument("--use-smc-setup", type=bool_arg, default=False)
    parser.add_argument("--trade-smc-setup", type=bool_arg, default=False)
    parser.add_argument("--smc-swing-lookback", type=int, default=10)
    parser.add_argument("--smc-sweep-buffer", type=float, default=1.0)
    parser.add_argument("--smc-break-buffer", type=float, default=0.5)
    parser.add_argument("--smc-level-buffer", type=float, default=8.0)
    parser.add_argument("--smc-zone-buffer", type=float, default=1.0)
    parser.add_argument("--smc-zone-lookback", type=int, default=20)
    parser.add_argument("--smc-choch-lookback", type=int, default=10)
    parser.add_argument("--smc-cisd-lookback", type=int, default=3)
    parser.add_argument("--smc-htf-minutes", type=int, default=15)
    parser.add_argument("--smc-amd-lookback", type=int, default=12)
    parser.add_argument("--smc-amd-max-range-points", type=float, default=45.0)
    parser.add_argument("--smc-displacement-mult", type=float, default=0.8)
    parser.add_argument("--smc-trap-lookback", type=int, default=8)
    parser.add_argument("--smc-trap-min-sweeps", type=int, default=3)
    parser.add_argument("--smc-avoid-trap-zones", type=bool_arg, default=True)
    parser.add_argument("--smc-require-displacement", type=bool_arg, default=True)
    parser.add_argument("--smc-target-points", type=float, default=30.0)
    parser.add_argument("--max-smc-risk-points", type=float, default=20.0)
    parser.add_argument("--min-smc-risk-points", type=float, default=16.0)
    parser.add_argument("--smc-trade-side", choices=["both", "buy", "sell"], default="sell")
    parser.add_argument("--smc-min-trade-timeframe-minutes", type=int, default=5)
    parser.add_argument("--smc-entry-start", default="9:15")
    parser.add_argument("--smc-entry-end", default="14:59")
    parser.add_argument("--use-bos-filter", type=bool_arg, default=False, help=argparse.SUPPRESS)
    parser.add_argument("--bos-lookback", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--use-chop-filter", type=bool_arg, default=True)
    parser.add_argument("--chop-lookback", type=int, default=10)
    parser.add_argument("--chop-min-ema-overlaps", type=int, choices=[2, 3], default=2)
    parser.add_argument("--max-chop-overlap-bars", type=int, default=3)
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
    parser.add_argument("--entry-start", default="9:15")
    parser.add_argument("--entry-end", default="15:30")
    parser.add_argument("--ema-buy-entry-start", default="9:15")
    parser.add_argument("--ema-buy-entry-end", default="15:30")
    parser.add_argument("--ema-sell-entry-start", default="9:15")
    parser.add_argument("--ema-sell-entry-end", default="15:30")
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
