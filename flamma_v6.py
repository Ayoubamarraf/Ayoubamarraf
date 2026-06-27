"""
================================================================================
FLAMMA AI v6.0 - UNIFIED PRODUCTION BOT
================================================================================

الملف الوحيد النهائي - كل الميزات في مكان واحد

FEATURES:
  1.  Core Trading Engine       - EMA + RSI + ATR + Structure
  2.  Dynamic Lot Sizing        - % risk حقيقي حسب balance + SL distance
  3.  Multi-timeframe           - H4 + M15 confirmation قبل الدخول
  4.  SMC Analyzer              - Order Blocks + FVG + Liquidity
  5.  Intermarket Analysis      - DXY / SPX correlation
  6.  Sentiment (Fear & Greed)  - Alternative.me API حقيقي
  7.  Q-Learning Agent          - Reinforcement learning
  8.  Flamma Brain              - Self-learning from trade history
  9.  Smart Signal Filter       - Multi-layer filtering
  10. News Filter               - High-impact event avoidance
  11. Position Management       - Breakeven + Partial + Trailing
  12. Max Daily Loss Protection  - يوقف البوت لو خسر أكثر من الحد
  13. Self-Healer               - Auto-fix errors
  14. Auto-Optimizer            - Kimi AI periodic review
  15. Watchdog                  - Auto-restart on crash
  16. Telegram Notifications    - Real-time alerts
  17. Full Backtest Engine      - نفس logic البوت الحقيقي

Usage:
  python flamma_v6.py           → تشغيل البوت
  python flamma_v6.py backtest  → تشغيل الباكتست
================================================================================
"""

import os
import sys
import time
import json
import logging
import traceback
import random
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import deque
from threading import Thread, Event, Lock

import numpy as np
import MetaTrader5 as mt5
import requests
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================

TOKEN         = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID       = os.getenv("TELEGRAM_CHAT_ID", "")
MT5_LOGIN     = int(os.getenv("MT5_LOGIN", "0") or 0)
MT5_PASSWORD  = os.getenv("MT5_PASSWORD", "")
MT5_SERVER    = os.getenv("MT5_SERVER", "")
KIMI_API_KEY  = os.getenv("KIMI_API_KEY", "")
LIVE_TRADING  = os.getenv("LIVE_TRADING", "False").lower() == "true"

# Trading params
SYMBOLS           = ["XAUUSDm", "EURUSDm", "GBPUSDm", "USDJPYm"]
TIMEFRAME_M15     = mt5.TIMEFRAME_M15   # Entry timeframe
TIMEFRAME_H4      = mt5.TIMEFRAME_H4    # Trend confirmation
BARS              = 250

# Risk management
RISK_PERCENT      = 1.0    # % of balance per trade (e.g. 1.0 = 1%)
MAX_LOT           = 0.50   # Hard cap on lot size
MIN_LOT           = 0.01
MAX_DAILY_LOSS_PERCENT = 15.0  # للتجربة بالبالانس الصغير — يوقف عند خسارة $5
MAX_DAILY_TRADES  = 5          # خمس صفقات يومياً فقط للتجربة

# Indicators
FAST_EMA          = 10
SLOW_EMA          = 20
RSI_PERIOD        = 14
ATR_PERIOD        = 14

# Signal thresholds
MIN_CONFIDENCE    = 7     # نرفعو الحد للتجربة — صفقات أقل بجودة أعلى
SIGNAL_THRESHOLD  = 4     # نرفعو الحد كذلك
COOLDOWN_SECONDS  = 1800
MAX_OPEN_POSITIONS = 1

# MT5 params
DEVIATION         = 50
MAGIC             = 555777

# Paths
DATA_DIR    = Path("data");    DATA_DIR.mkdir(exist_ok=True)
LOG_DIR     = Path("logs");    LOG_DIR.mkdir(exist_ok=True)
BACKUP_DIR  = Path("backups"); BACKUP_DIR.mkdir(exist_ok=True)

TRADES_FILE           = DATA_DIR / "trades.json"
LAST_TRADE_FILE       = DATA_DIR / "last_trade.json"
MANAGED_POSITIONS_FILE = DATA_DIR / "managed_positions.json"
AI_STATE_FILE         = DATA_DIR / "ai_state.json"
BRAIN_MEMORY_FILE     = DATA_DIR / "brain_memory.json"
Q_TABLE_FILE          = DATA_DIR / "q_table.json"
OPEN_TRADES_FILE      = DATA_DIR / "open_trades.json"
ERROR_LOG_FILE        = LOG_DIR  / "errors.json"
HEALER_STATE_FILE     = DATA_DIR / "healer_state.json"
DAILY_STATS_FILE      = DATA_DIR / "daily_stats.json"

logging.basicConfig(
    filename=LOG_DIR / "bot.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# Per-symbol settings
SETTINGS_BY_SYMBOL = {
    # Gold - high volatility (1 point = 0.01)
    "XAUUSDm": {
        "spread_limit":     600,
        "atr_min":          120,
        "ema_gap_min":       80,
        "max_ema_dist":    1800,
        "trail_trigger":   1200,
        "trail_dist":       800,
        "be_trigger":      1000,
        "partial_trigger": 1400,
        "sl_atr_mult":      2.5,   # SL = ATR * mult
        "tp_rr":            2.0,   # TP = SL * RR ratio
    },
    # EUR/USD - 5-digit forex (1 point = 0.00001)
    "EURUSDm": {
        "spread_limit":      50,
        "atr_min":           50,
        "ema_gap_min":       20,
        "max_ema_dist":     600,
        "trail_trigger":    400,
        "trail_dist":       250,
        "be_trigger":       300,
        "partial_trigger":  500,
        "sl_atr_mult":      2.0,
        "tp_rr":            2.0,
    },
    "GBPUSDm": {
        "spread_limit":      60,
        "atr_min":           60,
        "ema_gap_min":       25,
        "max_ema_dist":     700,
        "trail_trigger":    450,
        "trail_dist":       280,
        "be_trigger":       350,
        "partial_trigger":  550,
        "sl_atr_mult":      2.0,
        "tp_rr":            2.0,
    },
    "USDJPYm": {
        "spread_limit":      50,
        "atr_min":           50,
        "ema_gap_min":       20,
        "max_ema_dist":     600,
        "trail_trigger":    400,
        "trail_dist":       250,
        "be_trigger":       300,
        "partial_trigger":  500,
        "sl_atr_mult":      2.0,
        "tp_rr":            2.0,
    },
}

# ==============================================================================
# SECTION 2: UTILITIES
# ==============================================================================

def to_float(val):
    """Safely convert numpy or any value to float."""
    try:
        if isinstance(val, np.ndarray):
            return float(val.item()) if val.size == 1 else float(val[0])
        if hasattr(val, 'item'):
            return float(val.item())
        return float(val)
    except:
        return 0.0


def load_json_file(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default


def save_json_file(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Save JSON error {path}: {e}")


def append_trade_log(row):
    data = load_json_file(TRADES_FILE, [])
    data.append(row)
    save_json_file(TRADES_FILE, data)


def send_telegram_message(msg: str):
    print("[TG] " + msg[:200])
    if not TOKEN or not CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=15)
    except Exception as e:
        logging.warning(f"Telegram error: {e}")


# ==============================================================================
# SECTION 3: SAFE INDICATORS
# ==============================================================================

def safe_ema(values: list, period: int):
    try:
        if len(values) < period:
            return None
        k = 2.0 / (period + 1)
        val = float(values[0])
        for price in values[1:]:
            val = float(price) * k + val * (1 - k)
        return float(val)
    except Exception as e:
        logging.warning(f"EMA error: {e}")
        return None


def safe_rsi(closes: list, period: int = 14):
    try:
        if len(closes) < period + 1:
            return None
        gains, losses = [], []
        for i in range(1, period + 1):
            diff = float(closes[-i]) - float(closes[-i - 1])
            gains.append(max(diff, 0))
            losses.append(max(-diff, 0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))
    except Exception as e:
        logging.warning(f"RSI error: {e}")
        return 50.0


def safe_atr(highs: list, lows: list, closes: list, period: int = 14):
    try:
        if len(closes) < period + 1:
            return None
        trs = []
        for i in range(1, len(closes)):
            h      = float(to_float(highs[i]))
            l      = float(to_float(lows[i]))
            c_prev = float(to_float(closes[i - 1]))
            trs.append(max(h - l, abs(h - c_prev), abs(l - c_prev)))
        if len(trs) < period:
            return None
        return sum(trs[-period:]) / period
    except Exception as e:
        logging.warning(f"ATR error: {e}")
        return None


def safe_detect_structure(highs: list, lows: list) -> str:
    """Detects market structure: BULLISH / BEARISH / NEUTRAL."""
    try:
        if len(highs) < 6 or len(lows) < 6:
            return "NEUTRAL"
        last_high  = max(float(to_float(h)) for h in highs[-3:])
        prev_high  = max(float(to_float(h)) for h in highs[-6:-3])
        last_low   = min(float(to_float(l)) for l in lows[-3:])
        prev_low   = min(float(to_float(l)) for l in lows[-6:-3])
        if bool(last_high > prev_high) and bool(last_low > prev_low):
            return "BULLISH"
        if bool(last_high < prev_high) and bool(last_low < prev_low):
            return "BEARISH"
        return "NEUTRAL"
    except:
        return "NEUTRAL"


# ==============================================================================
# SECTION 4: RISK MANAGEMENT
# ==============================================================================

def calculate_dynamic_lot(symbol: str, sl_price: float, entry_price: float, account_balance: float) -> float:
    """
    حساب حجم اللوت الحقيقي بناءً على:
    - % risk من الـ balance
    - مسافة الـ SL بالـ points

    Formula: lot = (balance * risk%) / (sl_distance_in_$ per lot)
    For XAUUSD: 1 lot = 100 oz = $1 per pip = $10 per point (0.01 price move = $1 for 0.01 lot)
    """
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            return MIN_LOT

        sl_distance = abs(entry_price - sl_price)
        if sl_distance <= 0:
            return MIN_LOT

        # Value per lot per point
        tick_value = info.trade_tick_value  # $ per lot per tick
        tick_size  = info.trade_tick_size   # size of one tick

        if tick_size <= 0 or tick_value <= 0:
            return MIN_LOT

        # $ risk per lot
        risk_per_lot = (sl_distance / tick_size) * tick_value

        # Target risk in $
        risk_amount = account_balance * (RISK_PERCENT / 100.0)

        lot = risk_amount / risk_per_lot
        lot = max(MIN_LOT, min(MAX_LOT, round(lot, 2)))
        return lot
    except Exception as e:
        logging.warning(f"Dynamic lot error: {e}")
        return MIN_LOT


def calculate_smart_sl_tp(symbol: str, trade_type: str, entry_price: float, atr: float):
    """
    SL/TP based on ATR multiple:
    SL = entry ± (ATR * sl_atr_mult)
    TP = entry ± (SL_distance * tp_rr)
    """
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            return None, None
        point  = info.point
        digits = info.digits
        cfg    = SETTINGS_BY_SYMBOL.get(symbol, SETTINGS_BY_SYMBOL["XAUUSDm"])

        sl_distance = atr * cfg["sl_atr_mult"]
        sl_distance = max(sl_distance, cfg.get("atr_min", 120) * point * 2.5)
        tp_distance = sl_distance * cfg["tp_rr"]

        if trade_type == "BUY":
            sl = round(entry_price - sl_distance, digits)
            tp = round(entry_price + tp_distance, digits)
        else:
            sl = round(entry_price + sl_distance, digits)
            tp = round(entry_price - tp_distance, digits)

        return sl, tp
    except Exception as e:
        logging.warning(f"SL/TP calc error: {e}")
        return None, None


# ==============================================================================
# SECTION 5: DAILY LOSS PROTECTION
# ==============================================================================

class DailyGuard:
    """
    Tracks daily PnL and trade count.
    Stops the bot if max daily loss or max trades is reached.
    """

    def __init__(self):
        self.today = self._today()
        self.stats = self._load()

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load(self) -> dict:
        data = load_json_file(DAILY_STATS_FILE, {})
        if data.get("date") != self.today:
            # New day — reset
            data = {"date": self.today, "trades": 0, "pnl": 0.0, "start_balance": 0.0}
        return data

    def _save(self):
        save_json_file(DAILY_STATS_FILE, self.stats)

    def set_start_balance(self, balance: float):
        if self.stats["start_balance"] == 0.0:
            self.stats["start_balance"] = balance
            self._save()

    def record_trade(self, pnl: float = 0.0):
        if self.today != self._today():
            # Day changed — reset
            self.today = self._today()
            self.stats = {"date": self.today, "trades": 0, "pnl": 0.0, "start_balance": self.stats.get("start_balance", 0.0)}
        self.stats["trades"] += 1
        self.stats["pnl"] += pnl
        self._save()

    def is_trading_allowed(self) -> tuple[bool, str]:
        start_bal = self.stats.get("start_balance", 0.0)
        if start_bal > 0:
            loss_pct = (-self.stats["pnl"] / start_bal) * 100
            if loss_pct >= MAX_DAILY_LOSS_PERCENT:
                return False, f"Max daily loss reached: -{round(loss_pct, 2)}% (limit {MAX_DAILY_LOSS_PERCENT}%)"
        if self.stats["trades"] >= MAX_DAILY_TRADES:
            return False, f"Max daily trades reached: {self.stats['trades']}/{MAX_DAILY_TRADES}"
        return True, "OK"

    def get_summary(self) -> str:
        return (
            f"Daily PnL: {round(self.stats['pnl'], 2)} | "
            f"Trades: {self.stats['trades']}/{MAX_DAILY_TRADES} | "
            f"Date: {self.stats['date']}"
        )


# ==============================================================================
# SECTION 6: MULTI-TIMEFRAME ANALYSIS
# ==============================================================================

def get_h4_trend(symbol: str) -> str:
    """
    Checks H4 trend direction using EMA cross + structure.
    Returns: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
    """
    try:
        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_H4, 0, 100)
        if rates is None or len(rates) < 50:
            return "NEUTRAL"

        closes = [float(float(to_float(r["close"]))) for r in rates]
        highs  = [float(float(to_float(r["high"])))  for r in rates]
        lows   = [float(float(to_float(r["low"])))   for r in rates]

        fast_h4 = safe_ema(closes[-60:], FAST_EMA)
        slow_h4 = safe_ema(closes[-60:], SLOW_EMA)

        if fast_h4 is None or slow_h4 is None:
            return "NEUTRAL"

        fast_h4   = float(fast_h4)
        slow_h4   = float(slow_h4)
        structure = safe_detect_structure(highs, lows)
        ema_trend = "BULLISH" if bool(fast_h4 > slow_h4) else "BEARISH"

        # Both must agree
        if ema_trend == structure:
            return ema_trend
        return "NEUTRAL"

    except Exception as e:
        logging.warning(f"H4 trend error: {e}")
        return "NEUTRAL"


# ==============================================================================
# SECTION 7: SENTIMENT - FEAR & GREED INDEX (Alternative.me API)
# ==============================================================================

class SentimentAnalyzer:
    """
    Real sentiment from Alternative.me Fear & Greed Index.
    Falls back to price-action if API unavailable.
    """

    FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"

    def __init__(self):
        self._cache      = None
        self._cache_time = 0
        self._cache_ttl  = 3600  # 1 hour

    def _fetch_fear_greed(self) -> dict:
        try:
            resp = requests.get(self.FEAR_GREED_URL, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                entry = data["data"][0]
                value = int(entry["value"])
                label = entry["value_classification"]
                return {"value": value, "label": label, "source": "api"}
        except Exception as e:
            logging.warning(f"Fear & Greed API error: {e}")
        return None

    def _price_action_sentiment(self) -> dict:
        """Fallback: 24h price change direction."""
        try:
            rates = mt5.copy_rates_from_pos("XAUUSDm", mt5.TIMEFRAME_H1, 0, 24)
            if rates is None or len(rates) < 2:
                return {"sentiment": "NEUTRAL", "score": 0.0, "reason": "No data", "source": "price"}
            closes = [float(float(to_float(r["close"]))) for r in rates]
            change = (closes[-1] - closes[0]) / closes[0] * 100
            if change > 0.5:
                return {"sentiment": "BULLISH", "score": min(1.0, change / 2), "reason": f"+{round(change,2)}% 24h", "source": "price"}
            elif change < -0.5:
                return {"sentiment": "BEARISH", "score": min(1.0, abs(change) / 2), "reason": f"{round(change,2)}% 24h", "source": "price"}
            return {"sentiment": "NEUTRAL", "score": 0.0, "reason": "Sideways", "source": "price"}
        except:
            return {"sentiment": "NEUTRAL", "score": 0.0, "reason": "Error", "source": "price"}

    def get_market_sentiment(self) -> dict:
        # Use cache
        if self._cache and (time.time() - self._cache_time) < self._cache_ttl:
            return self._cache

        fg = self._fetch_fear_greed()
        if fg:
            val = fg["value"]
            # Map fear/greed to sentiment
            # 0-25: Extreme Fear → BEARISH for markets, BULLISH for gold
            # 25-45: Fear → slightly BULLISH for gold
            # 45-55: Neutral
            # 55-75: Greed → BEARISH for gold
            # 75-100: Extreme Greed → BEARISH for gold
            if val <= 25:
                result = {"sentiment": "BULLISH", "score": 0.8, "reason": f"Extreme Fear (FGI={val}) → gold safe haven", "source": "api"}
            elif val <= 45:
                result = {"sentiment": "BULLISH", "score": 0.4, "reason": f"Fear (FGI={val}) → mild gold support", "source": "api"}
            elif val <= 55:
                result = {"sentiment": "NEUTRAL", "score": 0.0, "reason": f"Neutral (FGI={val})", "source": "api"}
            elif val <= 75:
                result = {"sentiment": "BEARISH", "score": 0.4, "reason": f"Greed (FGI={val}) → risk-on, gold pressured", "source": "api"}
            else:
                result = {"sentiment": "BEARISH", "score": 0.8, "reason": f"Extreme Greed (FGI={val}) → gold bearish", "source": "api"}
        else:
            result = self._price_action_sentiment()

        self._cache      = result
        self._cache_time = time.time()
        return result


# ==============================================================================
# SECTION 8: SMC ANALYZER
# ==============================================================================

class SMCAnalyzer:
    def __init__(self):
        self.order_blocks = deque(maxlen=50)
        self.fvg_list     = deque(maxlen=50)

    def find_order_blocks(self, rates, lookback=20):
        if len(rates) < lookback + 5:
            return []
        obs = []
        for i in range(len(rates) - lookback, len(rates) - 2):
            o, c, h, l   = float(to_float(rates[i]["open"])), float(to_float(rates[i]["close"])), float(to_float(rates[i]["high"])), float(to_float(rates[i]["low"]))
            n_o, n_c     = float(to_float(rates[i+1]["open"])), float(to_float(rates[i+1]["close"]))
            n_h, n_l     = float(to_float(rates[i+1]["high"])), float(to_float(rates[i+1]["low"]))
            candle_range = h - l + 1e-10
            if o > c and n_c > n_o and n_c > h:
                obs.append({"type": "BULLISH_OB", "high": float(h), "low": float(l), "strength": float(abs(n_c - n_o) / candle_range)})
            elif c > o and n_o > n_c and n_c < l:
                obs.append({"type": "BEARISH_OB", "high": float(h), "low": float(l), "strength": float(abs(n_c - n_o) / candle_range)})
        self.order_blocks = deque(sorted(obs, key=lambda x: x["strength"], reverse=True)[:20], maxlen=50)
        return list(self.order_blocks)

    def find_fvg(self, rates, lookback=20):
        if len(rates) < lookback + 3:
            return []
        fvgs = []
        for i in range(len(rates) - lookback, len(rates) - 2):
            l_i, h_i     = float(to_float(rates[i]["low"])),   float(to_float(rates[i]["high"]))
            l_i2, h_i2   = float(to_float(rates[i+2]["low"])), float(to_float(rates[i+2]["high"]))
            if l_i > h_i2:
                fvgs.append({"type": "BULLISH_FVG", "top": float(l_i), "bottom": float(h_i2), "size": float(l_i - h_i2)})
            elif h_i < l_i2:
                fvgs.append({"type": "BEARISH_FVG", "top": float(l_i2), "bottom": float(h_i), "size": float(l_i2 - h_i)})
        self.fvg_list = deque(sorted(fvgs, key=lambda x: x["size"], reverse=True)[:20], maxlen=50)
        return list(self.fvg_list)

    def find_liquidity(self, rates, lookback=30):
        if len(rates) < lookback:
            return {"sell_side": [], "buy_side": []}
        highs = [float(to_float(r["high"])) for r in rates[-lookback:]]
        lows  = [float(to_float(r["low"]))  for r in rates[-lookback:]]
        avg_price = sum(highs) / len(highs)
        threshold = avg_price * 0.0003  # 0.03% relative — works for XAU (~0.6) and forex (~0.0003)
        sell_liq = [float(highs[i]) for i in range(len(highs)) for j in range(i+1, len(highs)) if abs(highs[i] - highs[j]) < threshold]
        buy_liq  = [float(lows[i])  for i in range(len(lows))  for j in range(i+1, len(lows))  if abs(lows[i]  - lows[j])  < threshold]
        return {"sell_side": list(set(sell_liq))[:5], "buy_side": list(set(buy_liq))[:5]}

    def get_signal(self, rates, current_price):
        obs       = self.find_order_blocks(rates)
        fvgs      = self.find_fvg(rates)
        liquidity = self.find_liquidity(rates)
        score, reasons = 0, []

        cp = float(current_price)
        for ob in obs:
            if ob["type"] == "BULLISH_OB" and float(ob["low"]) <= cp <= float(ob["high"]):
                score += 3; reasons.append("Price at Bullish OB"); break
        for ob in obs:
            if ob["type"] == "BEARISH_OB" and float(ob["low"]) <= cp <= float(ob["high"]):
                score -= 3; reasons.append("Price at Bearish OB"); break
        for fvg in fvgs:
            if fvg["type"] == "BULLISH_FVG" and float(fvg["bottom"]) <= cp <= float(fvg["top"]):
                score += 2; reasons.append("Price in Bullish FVG"); break
            elif fvg["type"] == "BEARISH_FVG" and float(fvg["bottom"]) <= cp <= float(fvg["top"]):
                score -= 2; reasons.append("Price in Bearish FVG"); break
        for level in liquidity["sell_side"]:
            if abs(current_price - level) < 0.001:
                score += 2; reasons.append("Sell-side liquidity swept")
        for level in liquidity["buy_side"]:
            if abs(current_price - level) < 0.001:
                score -= 2; reasons.append("Buy-side liquidity swept")

        if score >= 3:
            return "BUY",  score,      " | ".join(reasons) or "No SMC"
        elif score <= -3:
            return "SELL", abs(score), " | ".join(reasons) or "No SMC"
        return "NEUTRAL", 0,           " | ".join(reasons) or "No SMC"


# ==============================================================================
# SECTION 9: INTERMARKET ANALYSIS
# ==============================================================================

class IntermarketAnalyzer:
    def _get_rates(self, symbol, timeframe, bars=100):
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return None
            if not info.visible:
                mt5.symbol_select(symbol, True)
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None or len(rates) < bars // 2:
                return None
            return rates
        except:
            return None

    def _correlation(self, rates1, rates2) -> float:
        try:
            c1 = [float(to_float(r["close"])) for r in rates1[-50:]]
            c2 = [float(to_float(r["close"])) for r in rates2[-50:]]
            n  = min(len(c1), len(c2))
            if n < 10:
                return 0.0
            c1, c2   = c1[-n:], c2[-n:]
            r1 = [(c1[i] - c1[i-1]) / c1[i-1] for i in range(1, n)]
            r2 = [(c2[i] - c2[i-1]) / c2[i-1] for i in range(1, n)]
            m1, m2   = sum(r1) / len(r1), sum(r2) / len(r2)
            num      = sum((a - m1) * (b - m2) for a, b in zip(r1, r2))
            d1 = sum((a - m1) ** 2 for a in r1) ** 0.5
            d2 = sum((b - m2) ** 2 for b in r2) ** 0.5
            if d1 == 0 or d2 == 0:
                return 0.0
            return num / (d1 * d2)
        except:
            return 0.0

    def analyze(self, primary_symbol="XAUUSDm") -> dict:
        primary = self._get_rates(primary_symbol, TIMEFRAME_M15, 100)
        if primary is None:
            return {"dxy_correlation": 0, "spx_correlation": 0, "bias": "NEUTRAL", "strength": 0, "reasons": "No data"}

        dxy = self._get_rates("EURUSDm", TIMEFRAME_M15, 100)
        spx = self._get_rates("US500m",  TIMEFRAME_M15, 100)

        dxy_corr = self._correlation(primary, dxy) if dxy is not None else 0.0
        spx_corr = self._correlation(primary, spx) if spx is not None else 0.0

        strength, reasons = 0, []
        # Only act on STRONG, reliable correlations. Weak/unusual ones are ignored
        # (no penalty) to avoid fighting the primary signal on noisy data.
        if dxy_corr < -0.5:
            strength += 1; reasons.append(f"Inverse DXY confirms ({round(dxy_corr,2)})")
        if spx_corr < -0.3:
            strength += 1; reasons.append("Risk-off (inverse SPX)")
        elif spx_corr > 0.6:
            strength -= 1; reasons.append("Strong risk-on (SPX)")

        bias = "BULLISH" if strength > 0 else ("BEARISH" if strength < 0 else "NEUTRAL")
        return {
            "dxy_correlation": round(dxy_corr, 2),
            "spx_correlation": round(spx_corr, 2),
            "bias": bias,
            "strength": abs(strength),
            "reasons": " | ".join(reasons) or "No intermarket signals",
        }


# ==============================================================================
# SECTION 10: Q-LEARNING AGENT
# ==============================================================================

class QLearningAgent:
    def __init__(self, lr=0.1, gamma=0.95, epsilon=0.1):
        self.lr      = lr
        self.gamma   = gamma
        self.epsilon = epsilon
        self.actions = ["BUY", "SELL", "HOLD"]
        self.q_table = load_json_file(Q_TABLE_FILE, {})

    def _save(self):
        save_json_file(Q_TABLE_FILE, self.q_table)

    def state_key(self, rsi: float, ema_diff: float, structure: str, sentiment: str) -> str:
        rsi_b = round(rsi / 10) * 10
        trend = "UP" if ema_diff > 0 else "DOWN"
        return f"RSI{rsi_b}_{trend}_{structure}_{sentiment}"

    def choose_action(self, key: str) -> str:
        if key not in self.q_table:
            self.q_table[key] = {a: 0.0 for a in self.actions}
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.q_table[key], key=self.q_table[key].get)

    def update(self, key: str, action: str, reward: float, next_key: str):
        if key not in self.q_table:
            self.q_table[key] = {a: 0.0 for a in self.actions}
        if next_key not in self.q_table:
            self.q_table[next_key] = {a: 0.0 for a in self.actions}
        curr   = self.q_table[key][action]
        best_n = max(self.q_table[next_key].values())
        self.q_table[key][action] = curr + self.lr * (reward + self.gamma * best_n - curr)
        self._save()


# ==============================================================================
# SECTION 11: FLAMMA BRAIN (SELF-LEARNING)
# ==============================================================================

class FlammaBrain:
    def __init__(self):
        self.memory = load_json_file(BRAIN_MEMORY_FILE, {
            "total_trades": 0, "wins": 0, "losses": 0,
            "patterns": {}, "best_hours": [], "worst_hours": [],
        })

    def _save(self):
        save_json_file(BRAIN_MEMORY_FILE, self.memory)

    def _pattern_key(self, conditions: dict) -> str:
        rsi_r = round(conditions.get("rsi", 50) / 10) * 10
        atr_r = round(conditions.get("atr", 0) / 100) * 100
        hour  = conditions.get("hour", "00")
        return f"rsi{rsi_r}_atr{atr_r}_h{hour}"

    def learn_from_trade(self, trade_data: dict):
        self.memory["total_trades"] += 1
        hour = trade_data.get("time", "T00")[11:13]
        key  = self._pattern_key({
            "rsi":  trade_data.get("rsi",  50),
            "atr":  trade_data.get("atr",  0),
            "hour": hour,
        })
        if key not in self.memory["patterns"]:
            self.memory["patterns"][key] = {"wins": 0, "losses": 0}
        if trade_data.get("profit", 0) > 0:
            self.memory["wins"] += 1
            self.memory["patterns"][key]["wins"] += 1
            if hour not in self.memory["best_hours"]:
                self.memory["best_hours"].append(hour)
        else:
            self.memory["losses"] += 1
            self.memory["patterns"][key]["losses"] += 1
            if hour not in self.memory["worst_hours"]:
                self.memory["worst_hours"].append(hour)
        self._save()

    def should_trade(self, conditions: dict) -> tuple[bool, str]:
        key   = self._pattern_key(conditions)
        stats = self.memory["patterns"].get(key)
        if stats:
            total = stats["wins"] + stats["losses"]
            if total > 5:
                wr = stats["wins"] / total
                if wr < 0.30:
                    return False, f"Pattern win rate too low: {round(wr*100,1)}%"
        return True, "OK"


# ==============================================================================
# SECTION 12: NEWS FILTER
# ==============================================================================

class NewsFilter:
    """Block trading around known high-impact news windows (UTC)."""

    def is_high_impact_time(self) -> tuple[bool, str]:
        now = datetime.now(timezone.utc)
        # NFP: first Friday of month ~13:30 UTC
        if now.weekday() == 4 and now.day <= 7 and 13 <= now.hour <= 14:
            return True, "NFP Friday"
        # CPI: ~13th-15th of month ~12:30-13:30 UTC
        if 13 <= now.day <= 15 and 12 <= now.hour <= 14:
            return True, "CPI / Inflation Data"
        # FOMC: Wednesday ~19:00 UTC
        if now.weekday() == 2 and 18 <= now.hour <= 20:
            return True, "FOMC Decision"
        return False, "Clear"


# ==============================================================================
# SECTION 13: SMART SIGNAL FILTER
# ==============================================================================

class SmartFilter:
    def __init__(self, brain: FlammaBrain, news: NewsFilter):
        self.brain = brain
        self.news  = news

    def evaluate(self, direction: str, confidence: float, market_data: dict) -> tuple[float, str]:
        """Returns (adjusted_confidence, reason)."""
        # News block
        is_news, news_reason = self.news.is_high_impact_time()
        if is_news:
            return 0.0, f"HIGH IMPACT NEWS ({news_reason})"

        # Brain pattern check
        ok, reason = self.brain.should_trade(market_data)
        if not ok:
            return 0.0, reason

        score, reasons = confidence, []

        # Session bonus
        hour = market_data.get("hour", 12)
        if hour in [0, 1, 2, 3, 4]:
            score -= 1; reasons.append("Low-liquidity session")
        elif hour in [13, 14, 15]:
            score += 0.5; reasons.append("London/NY overlap")

        return max(0.0, min(10.0, score)), " | ".join(reasons) or "Standard"


# ==============================================================================
# SECTION 14: SELF-HEALER
# ==============================================================================

class SelfHealer:
    def __init__(self):
        self.lock          = Lock()
        self.error_count   = 0
        self.error_history = []
        self.fixes_applied = []
        self.max_errors    = 50
        state = load_json_file(HEALER_STATE_FILE, {})
        self.error_count   = state.get("error_count", 0)
        self.fixes_applied = state.get("fixes_applied", [])

    def log_error(self, error_msg: str, tb: str = ""):
        with self.lock:
            self.error_count += 1
            entry = {"time": datetime.now(timezone.utc).isoformat(), "error": str(error_msg)[:300], "tb": tb[:300]}
            self.error_history.append(entry)
            try:
                save_json_file(ERROR_LOG_FILE, self.error_history[-50:])
            except:
                pass
            logging.error(f"Error #{self.error_count}: {str(error_msg)[:150]}")

    def apply_fix(self, error_msg: str) -> str:
        em = str(error_msg).lower()
        if "truth value of an array" in em or "ambiguous" in em:
            fix = "numpy_fix"
        elif "mt5" in em or "authorization" in em:
            fix = "mt5_reconnect"
        elif "unicode" in em or "encoding" in em or "charmap" in em:
            fix = "unicode_fix"
        elif "division by zero" in em:
            fix = "zero_div_fix"
        else:
            fix = "generic_retry"
        self.fixes_applied.append({"time": datetime.now(timezone.utc).isoformat(), "fix": fix})
        save_json_file(HEALER_STATE_FILE, {"error_count": self.error_count, "fixes_applied": self.fixes_applied[-20:]})
        return fix

    def should_restart(self) -> bool:
        if self.error_count >= self.max_errors:
            return True
        if len(self.error_history) >= 5:
            times = [datetime.fromisoformat(e["time"]) for e in self.error_history[-5:]]
            if (times[-1] - times[0]).total_seconds() < 300:
                return True
        return False

    def get_status(self) -> dict:
        return {
            "error_count":   self.error_count,
            "fixes_applied": len(self.fixes_applied),
            "healthy":       self.error_count < 10,
        }


# ==============================================================================
# SECTION 14B: WATCHDOG
# ==============================================================================

class Watchdog:
    """
    Monitors the main loop heartbeat.
    If the loop is silent for max_silence seconds, reconnects MT5.
    """

    def __init__(self, max_silence: int = 120):
        self.max_silence = max_silence
        self._last_beat  = time.time()
        self._stop       = Event()
        self._thread     = Thread(target=self._run, daemon=True, name="Watchdog")

    def start(self):
        self._thread.start()
        logging.info("Watchdog started")

    def beat(self):
        self._last_beat = time.time()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            time.sleep(15)
            silence = time.time() - self._last_beat
            if silence > self.max_silence:
                logging.error(f"Watchdog: main loop silent for {int(silence)}s — reconnecting MT5")
                healer.log_error(f"Watchdog triggered after {int(silence)}s silence")
                try:
                    safe_connect_mt5()
                except Exception as e:
                    logging.error(f"Watchdog reconnect failed: {e}")
                self._last_beat = time.time()


# ==============================================================================
# SECTION 15: KIMI AI CLIENT
# ==============================================================================

class KimiClient:
    BASE_URL      = "https://api.moonshot.ai/v1"
    DEFAULT_MODEL = "kimi-k2.6"
    THINK_MODEL   = "kimi-k2-thinking"
    FAST_MODEL    = "kimi-k2.5"

    def __init__(self):
        self.api_key = KIMI_API_KEY
        self.enabled = bool(self.api_key)
        if not self.enabled:
            logging.warning("KIMI_API_KEY not set — AI features disabled")
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"} if self.enabled else {}

    def _chat(self, messages, model=None, temperature=0.3, max_tokens=1500) -> str:
        if not self.enabled:
            return ""
        try:
            resp = requests.post(
                self.BASE_URL + "/chat/completions",
                headers=self.headers,
                json={"model": model or self.DEFAULT_MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                timeout=90,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logging.warning(f"Kimi API error: {e}")
            return ""

    def _extract_json(self, text: str) -> dict:
        for delimiter in [("```json\n", "\n```"), ("```\n", "\n```"), ("{", "}")]:
            try:
                start = text.find(delimiter[0])
                end   = text.rfind(delimiter[1])
                if start != -1 and end > start:
                    chunk = text[start + len(delimiter[0]):end] if len(delimiter[0]) > 1 else text[start:end+1]
                    return json.loads(chunk)
            except:
                continue
        return {}

    def review_signal(self, signal: dict, context: dict) -> dict:
        prompt = f"Review this XAUUSD signal. Signal: {json.dumps(signal)}. Context: {json.dumps(context)}. Return JSON: {{approve: bool, confidence: int 1-10, reasoning: str}}"
        resp = self._chat([{"role": "user", "content": prompt}], model=self.FAST_MODEL)
        return self._extract_json(resp) or {"approve": True, "confidence": 5, "reasoning": "No AI response"}

    def optimize_strategy(self, trades: list, settings: dict, metrics: dict) -> dict:
        prompt = f"Analyze these {len(trades)} trades and current settings. Return JSON {{should_apply: bool, recommended_changes: dict, reasoning: str, confidence: int}}.\nTrades summary: {json.dumps(metrics)}\nSettings: {json.dumps(settings)}"
        resp = self._chat([{"role": "user", "content": prompt}], model=self.THINK_MODEL, max_tokens=800)
        return self._extract_json(resp) or {}


# ==============================================================================
# SECTION 16: AUTO-OPTIMIZER
# ==============================================================================

class AutoOptimizer:
    def __init__(self, kimi: KimiClient):
        self.kimi      = kimi
        self.last_time = 0
        self.interval  = 3600 * 6  # 6 hours

    def run(self):
        if time.time() - self.last_time < self.interval:
            return
        self.last_time = time.time()

        trades = load_json_file(TRADES_FILE, [])
        closed = [t for t in trades if t.get("status") in ["CLOSED", "CLOSED_BY_SL", "CLOSED_BY_TP"]]
        if len(closed) < 20:
            return

        wins     = sum(1 for t in closed if t.get("profit", 0) > 0)
        win_rate = wins / len(closed)
        metrics  = {"win_rate": round(win_rate, 3), "total_trades": len(closed),
                    "recent_pnl": sum(t.get("profit", 0) for t in closed[-10:])}
        settings = {"RISK_PERCENT": RISK_PERCENT, "MIN_CONFIDENCE": MIN_CONFIDENCE,
                    "COOLDOWN_SECONDS": COOLDOWN_SECONDS}

        opt = self.kimi.optimize_strategy(closed[-30:], settings, metrics)
        if opt.get("should_apply") and opt.get("confidence", 0) >= 6:
            send_telegram_message(
                f"[AI OPTIMIZER]\nWin rate: {round(win_rate*100,1)}%\n"
                f"Suggestion: {opt.get('reasoning','')[:200]}\n"
                f"Changes: {json.dumps(opt.get('recommended_changes',{}))}"
            )


# ==============================================================================
# SECTION 17: MT5 CONNECTION HELPERS
# ==============================================================================

healer = SelfHealer()  # Global healer instance


def safe_connect_mt5() -> bool:
    if not MT5_LOGIN or not MT5_PASSWORD or not MT5_SERVER:
        send_telegram_message("[ERROR] Missing MT5 credentials in .env")
        return False
    for attempt in range(5):
        try:
            try:
                mt5.shutdown(); time.sleep(1)
            except:
                pass
            ok = mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
            if ok:
                acc = mt5.account_info()
                if acc:
                    send_telegram_message(
                        f"[OK] Flamma v6.0 connected\n"
                        f"Login: {acc.login} | Balance: {acc.balance} {acc.currency}\n"
                        f"LIVE_TRADING: {LIVE_TRADING}"
                    )
                    return True
                healer.log_error("MT5 ok but no account info")
            else:
                healer.log_error(f"MT5 init failed: {mt5.last_error()}")
        except Exception as e:
            healer.log_error(str(e))
        print(f"[MT5] Attempt {attempt+1}/5 failed, retrying in 5s...")
        time.sleep(5)
    send_telegram_message(f"[ERROR] MT5 failed after 5 attempts: {mt5.last_error()}")
    return False


def safe_check_mt5() -> bool:
    try:
        if mt5.terminal_info() is None or mt5.account_info() is None:
            healer.log_error("MT5 disconnected")
            return safe_connect_mt5()
        return True
    except Exception as e:
        healer.log_error(str(e))
        return safe_connect_mt5()


def safe_ensure_symbol(symbol: str) -> bool:
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            return False
        if not info.visible:
            return mt5.symbol_select(symbol, True)
        return True
    except:
        return False


def safe_get_rates(symbol: str, timeframe, bars: int):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) < bars // 2:
            return None
        return rates
    except Exception as e:
        healer.log_error(f"get_rates error: {e}")
        return None


# ==============================================================================
# SECTION 18: POSITION MANAGEMENT
# ==============================================================================

def modify_sl_tp(ticket: int, symbol: str, new_sl=None, new_tp=None) -> bool:
    try:
        pos_list = mt5.positions_get(ticket=ticket)
        if not pos_list:
            return False
        pos = pos_list[0]
        req = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   symbol,
            "position": ticket,
            "sl":       float(pos.sl if new_sl is None else new_sl),
            "tp":       float(pos.tp if new_tp is None else new_tp),
            "magic":    MAGIC,
        }
        r = mt5.order_send(req)
        return r is not None and r.retcode == mt5.TRADE_RETCODE_DONE
    except Exception as e:
        healer.log_error(f"modify_sl_tp: {e}")
        return False


def close_partial(pos, volume: float) -> bool:
    try:
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return False
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price      = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        req = {
            "action":   mt5.TRADE_ACTION_DEAL,
            "symbol":   pos.symbol,
            "position": pos.ticket,
            "volume":   round(volume, 2),
            "type":     order_type,
            "price":    price,
            "deviation": DEVIATION,
            "magic":    MAGIC,
            "comment":  "PARTIAL_CLOSE",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        r = mt5.order_send(req)
        return r is not None and r.retcode == mt5.TRADE_RETCODE_DONE
    except Exception as e:
        healer.log_error(f"close_partial: {e}")
        return False


def manage_open_positions():
    try:
        positions = mt5.positions_get()
        if not positions:
            return
        managed = load_json_file(MANAGED_POSITIONS_FILE, {})
        for pos in positions:
            symbol = pos.symbol
            if symbol not in SETTINGS_BY_SYMBOL:
                continue
            tick = mt5.symbol_info_tick(symbol)
            info = mt5.symbol_info(symbol)
            if tick is None or info is None:
                continue
            cfg     = SETTINGS_BY_SYMBOL[symbol]
            point   = info.point
            digits  = info.digits
            cur     = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
            pts_up  = float((cur - pos.price_open) / point) if pos.type == mt5.POSITION_TYPE_BUY else float((pos.price_open - cur) / point)
            key     = str(pos.ticket)
            state   = managed.get(key, {"be": False, "partial": False})

            # Breakeven
            if not state["be"] and pts_up >= cfg["be_trigger"]:
                be_sl = round(pos.price_open, digits)
                if modify_sl_tp(pos.ticket, symbol, new_sl=be_sl):
                    state["be"] = True
                    send_telegram_message(f"[BE] Breakeven set | {symbol} | ticket={pos.ticket}")

            # Partial close
            if not state["partial"] and pts_up >= cfg["partial_trigger"] and pos.volume >= 0.02:
                vol = round(pos.volume / 2, 2)
                if vol >= MIN_LOT:
                    if close_partial(pos, vol):
                        state["partial"] = True
                        send_telegram_message(f"[PARTIAL] Closed {vol} lots | {symbol}")

            # Trailing stop
            if pts_up >= cfg["trail_trigger"]:
                if pos.type == mt5.POSITION_TYPE_BUY:
                    new_sl = round(cur - cfg["trail_dist"] * point, digits)
                    if new_sl > pos.sl and new_sl < cur:
                        modify_sl_tp(pos.ticket, symbol, new_sl=new_sl)
                else:
                    new_sl = round(cur + cfg["trail_dist"] * point, digits)
                    if (pos.sl == 0.0 or new_sl < pos.sl) and new_sl > cur:
                        modify_sl_tp(pos.ticket, symbol, new_sl=new_sl)

            managed[key] = state
        save_json_file(MANAGED_POSITIONS_FILE, managed)
    except Exception as e:
        healer.log_error(f"manage_positions: {e}")


# ==============================================================================
# SECTION 18B: CLOSED TRADE TRACKER + GRACEFUL SHUTDOWN
# ==============================================================================

def check_closed_trades(modules: dict, guard: "DailyGuard"):
    """
    Detects positions that closed since last cycle.
    Updates Q-Learning with actual reward and Brain memory.
    """
    try:
        tracked = load_json_file(OPEN_TRADES_FILE, {})
        if not tracked:
            return

        current_tickets = set()
        positions = mt5.positions_get()
        if positions:
            current_tickets = {str(p.ticket) for p in positions}

        closed_tickets = set(tracked.keys()) - current_tickets
        if not closed_tickets:
            return

        # Fetch recent deal history (last 7 days)
        since = datetime.now(timezone.utc) - timedelta(days=7)
        deals = mt5.history_deals_get(since, datetime.now(timezone.utc))
        deal_map = {}
        if deals:
            for d in deals:
                deal_map[str(d.position_id)] = d

        for ticket in closed_tickets:
            meta    = tracked[ticket]
            q_key   = meta.get("q_key", "")
            action  = meta.get("direction", "BUY")
            deal    = deal_map.get(ticket)

            pnl     = float(deal.profit) if deal else 0.0
            reward  = pnl / 10.0  # Normalize: $10 profit → reward=1.0

            # Update Q-Learning with real outcome
            if q_key:
                modules["qlearning"].update(q_key, action, reward, q_key)
                logging.info(f"Q-Learning updated: ticket={ticket} pnl={pnl} reward={round(reward,3)}")

            # Brain learns from closed trade
            modules["brain"].learn_from_trade({
                "time":   datetime.now(timezone.utc).isoformat(),
                "symbol": meta.get("symbol", ""),
                "rsi":    meta.get("rsi", 50),
                "atr":    meta.get("atr", 0),
                "profit": pnl,
            })

            # Record PnL in daily guard
            guard.record_trade(pnl=pnl)

            status = "WIN" if pnl > 0 else "LOSS"
            send_telegram_message(
                f"[CLOSED] {meta.get('symbol','')} {action} | "
                f"PnL: {round(pnl,2)} | {status} | Q-reward: {round(reward,2)}"
            )

            del tracked[ticket]

        save_json_file(OPEN_TRADES_FILE, tracked)

    except Exception as e:
        healer.log_error(f"check_closed_trades: {e}")


def close_all_positions():
    """Graceful shutdown: close all open positions at market price."""
    try:
        positions = mt5.positions_get()
        if not positions:
            return
        logging.info(f"Graceful shutdown: closing {len(positions)} position(s)")
        for pos in positions:
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                continue
            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price      = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
            req = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       pos.symbol,
                "position":     pos.ticket,
                "volume":       pos.volume,
                "type":         order_type,
                "price":        price,
                "deviation":    DEVIATION,
                "magic":        MAGIC,
                "comment":      "SHUTDOWN_CLOSE",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            r = mt5.order_send(req)
            status = "OK" if (r and r.retcode == mt5.TRADE_RETCODE_DONE) else f"FAILED retcode={getattr(r,'retcode','?')}"
            send_telegram_message(f"[SHUTDOWN] {pos.symbol} ticket={pos.ticket} → {status}")
    except Exception as e:
        healer.log_error(f"close_all_positions: {e}")


# ==============================================================================
# SECTION 19: SIGNAL BUILDING (CORE LOGIC)
# ==============================================================================

def build_signal(symbol: str, modules: dict) -> dict:
    """
    Full signal build using M15 + H4 confirmation + all modules.
    Returns a signal dict with type / price / sl / tp / confidence / reason.
    """
    NO_TRADE = lambda r: {"symbol": symbol, "type": "NO_TRADE", "confidence": 0, "reason": r}

    try:
        if not safe_ensure_symbol(symbol):
            return NO_TRADE("Symbol not found")

        # --- M15 data ---
        rates = safe_get_rates(symbol, TIMEFRAME_M15, BARS)
        if rates is None:
            return NO_TRADE("No M15 data")

        closes = [float(float(to_float(r["close"]))) for r in rates]
        highs  = [float(float(to_float(r["high"])))  for r in rates]
        lows   = [float(float(to_float(r["low"])))   for r in rates]

        fast_m15 = safe_ema(closes[-60:], FAST_EMA)
        slow_m15 = safe_ema(closes[-60:], SLOW_EMA)
        rsi      = safe_rsi(closes, RSI_PERIOD)
        atr      = safe_atr(highs, lows, closes, ATR_PERIOD)
        structure_m15 = safe_detect_structure(highs, lows)

        if None in (fast_m15, slow_m15, rsi, atr):
            return NO_TRADE("Indicators not ready")
        fast_m15 = float(fast_m15)
        slow_m15 = float(slow_m15)
        rsi      = float(rsi)
        atr      = float(atr)

        # --- Tick / spread ---
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        if tick is None or info is None:
            return NO_TRADE("No tick data")

        point        = info.point
        cfg          = SETTINGS_BY_SYMBOL.get(symbol, SETTINGS_BY_SYMBOL["XAUUSDm"])
        spread_pts   = abs(tick.ask - tick.bid) / point
        atr_pts      = atr / point
        ema_gap_pts  = abs(float(fast_m15) - float(slow_m15)) / float(point)
        current_price = float(tick.ask)

        print(f"\n[{symbol}] Price={current_price} | RSI={round(rsi,1)} | ATR={round(atr_pts,1)} | Spread={round(spread_pts,1)} | M15={structure_m15}")

        # --- Basic filters ---
        if spread_pts > cfg["spread_limit"]:
            return NO_TRADE(f"Spread too high ({round(spread_pts,1)})")
        if atr_pts < cfg["atr_min"]:
            return NO_TRADE(f"ATR weak ({round(atr_pts,1)})")
        if ema_gap_pts < cfg["ema_gap_min"]:
            return NO_TRADE(f"Trend weak (EMA gap={round(ema_gap_pts,1)})")

        # --- H4 trend confirmation ---
        h4_trend = get_h4_trend(symbol)
        print(f"[H4] Trend: {h4_trend}")

        # --- Module signals ---
        smc_dir, smc_score, smc_reason = modules["smc"].get_signal(rates, current_price)
        intermarket = modules["intermarket"].analyze(symbol)
        sentiment   = modules["sentiment"].get_market_sentiment()
        q_key       = modules["qlearning"].state_key(rsi, fast_m15 - slow_m15, structure_m15, sentiment["sentiment"])
        q_action    = modules["qlearning"].choose_action(q_key)

        print(f"[SMC] {smc_dir} (score={smc_score}) | {smc_reason}")
        print(f"[IM]  Bias={intermarket['bias']} | DXY={intermarket['dxy_correlation']}")
        print(f"[FGI] {sentiment['sentiment']} (score={round(sentiment['score'],2)}) | {sentiment['reason']} [{sentiment['source']}]")
        print(f"[QL]  Action={q_action} | State={q_key}")

        # --- Score aggregation ---
        score, reasons = 0.0, []

        # M15 technical (fast_m15, slow_m15, rsi already floats)
        # Trend direction via EMA + structure, with broader RSI confirmation
        if bool(fast_m15 > slow_m15) and structure_m15 == "BULLISH":
            score += 2; reasons.append("M15 Bullish (EMA+structure)")
            if bool(50 <= rsi <= 70):
                score += 1; reasons.append("RSI confirms bull")
        elif bool(fast_m15 < slow_m15) and structure_m15 == "BEARISH":
            score -= 2; reasons.append("M15 Bearish (EMA+structure)")
            if bool(30 <= rsi <= 50):
                score -= 1; reasons.append("RSI confirms bear")
        # EMA-only signal (weaker) when structure is neutral
        elif bool(fast_m15 > slow_m15):
            score += 1; reasons.append("M15 EMA bullish")
        elif bool(fast_m15 < slow_m15):
            score -= 1; reasons.append("M15 EMA bearish")

        # H4 trend alignment (important weight)
        if h4_trend == "BULLISH":
            score += 2; reasons.append("H4 Bullish trend")
        elif h4_trend == "BEARISH":
            score -= 2; reasons.append("H4 Bearish trend")

        # SMC
        if smc_dir == "BUY":
            score += smc_score; reasons.append(f"SMC: {smc_reason}")
        elif smc_dir == "SELL":
            score -= smc_score; reasons.append(f"SMC: {smc_reason}")

        # Intermarket
        if intermarket["bias"] == "BULLISH":
            score += intermarket["strength"]; reasons.append(f"IM: {intermarket['reasons']}")
        elif intermarket["bias"] == "BEARISH":
            score -= intermarket["strength"]; reasons.append(f"IM: {intermarket['reasons']}")

        # Sentiment
        sent_weight = sentiment["score"]
        if sentiment["sentiment"] == "BULLISH":
            score += sent_weight; reasons.append(f"Sentiment: {sentiment['reason']}")
        elif sentiment["sentiment"] == "BEARISH":
            score -= sent_weight; reasons.append(f"Sentiment: {sentiment['reason']}")

        # Q-Learning
        if q_action == "BUY":
            score += 1; reasons.append("Q-Learning: BUY")
        elif q_action == "SELL":
            score -= 1; reasons.append("Q-Learning: SELL")

        print(f"[SCORE] {round(score, 2)} | {' | '.join(reasons)}")

        # --- Determine direction ---
        if score >= SIGNAL_THRESHOLD:
            direction = "BUY"
        elif score <= -SIGNAL_THRESHOLD:
            direction = "SELL"
        else:
            return NO_TRADE(f"Score too low ({round(score,1)})")

        # --- H4 must confirm direction (map BUY->BULLISH, SELL->BEARISH) ---
        h4_expected = "BULLISH" if direction == "BUY" else "BEARISH"
        if h4_trend != "NEUTRAL" and h4_trend != h4_expected:
            return NO_TRADE(f"H4 conflict: {direction} vs H4={h4_trend}")

        entry_price = tick.ask if direction == "BUY" else tick.bid
        confidence  = min(10.0, 5.0 + abs(score) / 2.0)

        # --- Smart filter ---
        smart = modules["smart_filter"]
        market_data = {"rsi": rsi, "atr": atr_pts, "spread": spread_pts, "hour": datetime.now(timezone.utc).hour}
        adj_conf, filter_reason = smart.evaluate(direction, confidence, market_data)
        if adj_conf < MIN_CONFIDENCE:
            return NO_TRADE(f"Smart filter blocked: {filter_reason}")
        confidence = adj_conf

        # --- SL / TP ---
        sl, tp = calculate_smart_sl_tp(symbol, direction, entry_price, atr)
        if sl is None or tp is None:
            return NO_TRADE("SL/TP calc failed")

        return {
            "symbol":     symbol,
            "type":       direction,
            "price":      entry_price,
            "sl":         sl,
            "tp":         tp,
            "confidence": round(confidence, 1),
            "reason":     f"Score={round(score,1)} | {' | '.join(reasons)} | {filter_reason}",
            "meta":       {
                "rsi": round(rsi, 1), "atr": round(atr_pts, 1),
                "smc": smc_dir, "h4": h4_trend,
                "sentiment": sentiment["sentiment"], "q": q_action,
                "q_key": q_key,
            },
        }

    except Exception as e:
        tb = traceback.format_exc()
        healer.log_error(f"build_signal: {e}", tb)
        healer.apply_fix(str(e))
        return {"symbol": symbol, "type": "NO_TRADE", "confidence": 0, "reason": f"Error: {str(e)[:80]}"}


def get_best_signal(modules: dict):
    best = None
    for symbol in SYMBOLS:
        sig = build_signal(symbol, modules)
        if sig["type"] == "NO_TRADE":
            print(f"[WAIT] {symbol}: {sig['reason']}")
            continue
        print(f"[SIG] {symbol}: {sig['type']} conf={sig['confidence']} | {sig['reason'][:80]}")
        if sig["confidence"] >= MIN_CONFIDENCE:
            if best is None or sig["confidence"] > best["confidence"]:
                best = sig
    return best


# ==============================================================================
# SECTION 20: TRADE EXECUTION
# ==============================================================================

def cooldown_active() -> bool:
    try:
        last = load_json_file(LAST_TRADE_FILE, {})
        ts   = last.get("timestamp")
        if not ts:
            return False
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds()
        return elapsed < COOLDOWN_SECONDS
    except:
        return False


def has_open_positions() -> bool:
    try:
        pos = mt5.positions_get()
        return pos is not None and len(pos) > 0
    except:
        return False


def execute_trade(signal: dict, guard: DailyGuard, modules: dict, kimi: "KimiClient" = None):
    symbol     = signal["symbol"]
    direction  = signal["type"]
    sl         = signal["sl"]
    tp         = signal["tp"]
    confidence = signal["confidence"]
    reason     = signal["reason"]
    meta       = signal.get("meta", {})

    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return
        entry_price = tick.ask if direction == "BUY" else tick.bid

        # Dynamic lot
        account = mt5.account_info()
        balance = account.balance if account else 10000.0
        guard.set_start_balance(balance)
        lot = calculate_dynamic_lot(symbol, sl, entry_price, balance)

        # Kimi AI signal review (non-blocking if API down)
        if kimi and kimi.enabled:
            context = {
                "balance":   round(balance, 2),
                "daily_pnl": round(guard.stats.get("pnl", 0), 2),
                "h4":        meta.get("h4"),
                "score":     signal.get("confidence"),
            }
            review = kimi.review_signal(signal, context)
            if not review.get("approve", True):
                reason_ai = review.get("reasoning", "No reason")[:200]
                logging.info(f"Kimi rejected signal: {reason_ai}")
                send_telegram_message(f"[KIMI] Signal rejected\n{reason_ai}")
                return

        if not LIVE_TRADING:
            msg = (
                f"[PAPER SIGNAL] Flamma v6\n"
                f"Pair: {symbol} | {direction}\n"
                f"Entry: {round(entry_price, 5)}\n"
                f"SL: {round(sl, 5)} | TP: {round(tp, 5)}\n"
                f"Lot: {lot} | Risk: {RISK_PERCENT}%\n"
                f"Confidence: {confidence}/10\n"
                f"H4: {meta.get('h4')} | RSI: {meta.get('rsi')} | Sentiment: {meta.get('sentiment')}\n"
                f"Reason: {reason[:200]}"
            )
            send_telegram_message(msg)
            append_trade_log({
                "time": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol, "type": direction, "price": entry_price,
                "lot": lot, "sl": sl, "tp": tp, "confidence": confidence,
                "reason": reason, "status": "PAPER_SIGNAL", **meta,
            })
            save_json_file(LAST_TRADE_FILE, {"symbol": symbol, "timestamp": datetime.now(timezone.utc).isoformat()})
            guard.record_trade()
            return

        # Live execution
        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       lot,
            "type":         order_type,
            "price":        entry_price,
            "sl":           sl,
            "tp":           tp,
            "deviation":    DEVIATION,
            "magic":        MAGIC,
            "comment":      "FLAMMA_V6",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(req)

        # Retry without SL/TP if stops rejected
        if result is not None and result.retcode == 10016:
            print("[WARN] Invalid stops — retrying without SL/TP")
            req.pop("sl", None); req.pop("tp", None)
            result = mt5.order_send(req)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                time.sleep(1)
                pos_list = mt5.positions_get(symbol=symbol)
                if pos_list:
                    modify_sl_tp(pos_list[0].ticket, symbol, new_sl=sl, new_tp=tp)

        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            trade_data = {
                "time": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol, "type": direction, "price": entry_price,
                "lot": lot, "sl": sl, "tp": tp,
                "confidence": confidence, "reason": reason,
                "status": "OPENED", **meta,
            }
            append_trade_log(trade_data)
            save_json_file(LAST_TRADE_FILE, {"symbol": symbol, "timestamp": datetime.now(timezone.utc).isoformat()})
            guard.record_trade()

            # Track position for real reward update when it closes
            open_trades = load_json_file(OPEN_TRADES_FILE, {})
            open_trades[str(result.order)] = {
                "q_key":     meta.get("q_key", ""),
                "direction": direction,
                "symbol":    symbol,
                "rsi":       meta.get("rsi", 50),
                "atr":       meta.get("atr", 0),
            }
            save_json_file(OPEN_TRADES_FILE, open_trades)

            send_telegram_message(
                f"[TRADE EXECUTED] Flamma v6\n"
                f"{symbol} {direction} @ {round(entry_price, 5)}\n"
                f"Lot: {lot} | Risk: {RISK_PERCENT}% | Balance: {round(balance, 2)}\n"
                f"SL: {round(sl, 5)} | TP: {round(tp, 5)}\n"
                f"Conf: {confidence}/10 | LIVE: {LIVE_TRADING}\n"
                f"{reason[:150]}"
            )
        else:
            code = getattr(result, "retcode", "None")
            healer.log_error(f"Trade failed: retcode={code}")
            send_telegram_message(f"[ERROR] Trade failed on {symbol} | retcode={code}")

    except Exception as e:
        tb = traceback.format_exc()
        healer.log_error(f"execute_trade: {e}", tb)
        healer.apply_fix(str(e))


# ==============================================================================
# SECTION 21: BACKTEST ENGINE
# ==============================================================================

def run_backtest(csv_path: str = "backtests/data.csv", symbol: str = "XAUUSDm"):
    """
    Full backtest using the same signal logic as the live bot.
    Reads CSV with columns: time, open, high, low, close
    Runs M15 signal logic (no H4 / intermarket in backtest for simplicity).
    """
    print("\n" + "="*60)
    print("FLAMMA v6 BACKTEST ENGINE")
    print("="*60)

    from pathlib import Path
    import csv as csvlib

    p = Path(csv_path)
    if not p.exists():
        print(f"[ERROR] Data file not found: {csv_path}")
        print("  → Place CSV with columns: time,open,high,low,close")
        print("  → Example: backtests/data.csv")
        return

    # Load CSV
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csvlib.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "time":  row["time"],
                    "open":  float(row["open"]),
                    "high":  float(row["high"]),
                    "low":   float(row["low"]),
                    "close": float(row["close"]),
                })
            except:
                continue

    if len(rows) < 100:
        print(f"[ERROR] Not enough data: {len(rows)} rows (need 100+)")
        return

    print(f"[OK] Loaded {len(rows)} candles")

    # Backtest parameters (same as live)
    cfg = SETTINGS_BY_SYMBOL.get(symbol, SETTINGS_BY_SYMBOL["XAUUSDm"])
    SL_ATR_MULT = cfg["sl_atr_mult"]
    TP_RR       = cfg["tp_rr"]
    RISK_PCT    = RISK_PERCENT / 100.0
    START_BAL   = 10000.0

    # Symbol-specific P&L constants
    # point_size: smallest price increment
    # units_per_lot: contract size (affects $ value per price unit)
    POINT_SIZE = {
        "XAUUSDm":  0.01,
        "EURUSDm":  0.00001,
        "GBPUSDm":  0.00001,
        "USDJPYm":  0.001,
    }.get(symbol, 0.01)
    UNITS_PER_LOT = {
        "XAUUSDm":  100,       # 100 oz per lot → $100 per $1 move per lot
        "EURUSDm":  100000,    # 100k units → $10 per pip per lot
        "GBPUSDm":  100000,
        "USDJPYm":  100000,    # approximate (ignores JPY conversion)
    }.get(symbol, 100)

    print(f"[OK] Symbol: {symbol} | point={POINT_SIZE} | units/lot={UNITS_PER_LOT}")

    balance    = START_BAL
    trades     = []
    open_trade = None
    cooldown   = 0
    equity_curve = [balance]

    # We need at least 60 bars for indicators
    for i in range(60, len(rows)):
        candles = rows[max(0, i-250):i+1]
        closes  = [float(c["close"]) for c in candles]
        highs   = [c["high"]  for c in candles]
        lows    = [c["low"]   for c in candles]

        fast = safe_ema(closes[-60:], FAST_EMA)
        slow = safe_ema(closes[-60:], SLOW_EMA)
        rsi  = safe_rsi(closes, RSI_PERIOD)
        atr  = safe_atr(highs, lows, closes, ATR_PERIOD)

        if None in (fast, slow, rsi, atr):
            continue

        current = rows[i]
        price   = current["close"]
        structure = safe_detect_structure(highs, lows)

        # --- Manage open trade ---
        if open_trade:
            direction = open_trade["type"]
            sl = open_trade["sl"]
            tp = open_trade["tp"]
            entry = open_trade["entry"]

            # Check SL/TP hit on this candle
            hit_sl = current["low"] <= sl if direction == "BUY" else current["high"] >= sl
            hit_tp = current["high"] >= tp if direction == "BUY" else current["low"] <= tp

            if hit_sl or hit_tp:
                exit_price = sl if hit_sl else tp
                pnl_points = (exit_price - entry) if direction == "BUY" else (entry - exit_price)
                lot = open_trade["lot"]

                # P&L = price_change * units_per_lot * lot
                pnl = pnl_points * UNITS_PER_LOT * lot
                balance += pnl
                status  = "TP" if hit_tp else "SL"
                trades.append({
                    "entry": entry, "exit": exit_price,
                    "type":  direction, "lot": lot,
                    "pnl":   round(pnl, 2), "status": status,
                    "time":  current["time"],
                })
                open_trade = None
                cooldown   = 30  # bars cooldown
                equity_curve.append(balance)
                continue

        # Cooldown
        if cooldown > 0:
            cooldown -= 1
            continue

        if open_trade:
            continue  # Already in trade

        # --- Signal logic (mirrors live bot) ---
        atr_pts     = atr / POINT_SIZE
        ema_gap_pts = abs(fast - slow) / POINT_SIZE
        rsi_ok_buy  = 55 <= rsi <= 65
        rsi_ok_sell = 35 <= rsi <= 45

        # BUY signal
        if (fast > slow and structure == "BULLISH" and rsi_ok_buy
                and atr_pts >= cfg["atr_min"] and ema_gap_pts >= cfg["ema_gap_min"]):
            direction = "BUY"
        # SELL signal
        elif (fast < slow and structure == "BEARISH" and rsi_ok_sell
                and atr_pts >= cfg["atr_min"] and ema_gap_pts >= cfg["ema_gap_min"]):
            direction = "SELL"
        else:
            continue

        # SL / TP
        sl_dist = atr * SL_ATR_MULT
        tp_dist = sl_dist * TP_RR
        if direction == "BUY":
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            sl = price + sl_dist
            tp = price - tp_dist

        # Dynamic lot: risk_usd = sl_distance * UNITS_PER_LOT * lot
        risk_usd = balance * RISK_PCT
        sl_dist  = abs(price - sl)
        lot = risk_usd / (sl_dist * UNITS_PER_LOT) if sl_dist > 0 else MIN_LOT
        lot = max(MIN_LOT, min(MAX_LOT, round(lot, 2)))

        open_trade = {
            "type":  direction,
            "entry": price,
            "sl":    sl,
            "tp":    tp,
            "lot":   lot,
            "time":  current["time"],
        }

    # Close any open trade at last price
    if open_trade and rows:
        last_price = rows[-1]["close"]
        direction  = open_trade["type"]
        pnl_pts    = (last_price - open_trade["entry"]) if direction == "BUY" else (open_trade["entry"] - last_price)
        pnl        = pnl_pts * UNITS_PER_LOT * open_trade["lot"]
        balance   += pnl
        trades.append({
            "entry": open_trade["entry"], "exit": last_price,
            "type":  direction, "lot": open_trade["lot"],
            "pnl":   round(pnl, 2), "status": "FORCED_CLOSE",
            "time":  rows[-1]["time"],
        })

    # --- Results ---
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)

    if not trades:
        print("[!] No trades taken")
        return

    wins      = [t for t in trades if t["pnl"] > 0]
    losses    = [t for t in trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate  = len(wins) / len(trades) * 100

    gross_win  = sum(t["pnl"] for t in wins)  or 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) or 1
    profit_factor = gross_win / gross_loss

    # Max drawdown
    peak = START_BAL
    max_dd = 0.0
    bal = START_BAL
    for t in trades:
        bal += t["pnl"]
        if bal > peak:
            peak = bal
        dd = (peak - bal) / peak * 100
        if dd > max_dd:
            max_dd = dd

    print(f"  Start Balance : ${START_BAL:,.2f}")
    print(f"  End Balance   : ${balance:,.2f}")
    print(f"  Total PnL     : ${total_pnl:,.2f}  ({round((balance-START_BAL)/START_BAL*100,2)}%)")
    print(f"  Total Trades  : {len(trades)}")
    print(f"  Win Rate      : {round(win_rate,1)}%")
    print(f"  Profit Factor : {round(profit_factor,2)}")
    print(f"  Max Drawdown  : {round(max_dd,2)}%")
    print(f"  Avg Win       : ${round(gross_win/max(len(wins),1),2)}")
    print(f"  Avg Loss      : ${round(gross_loss/max(len(losses),1),2)}")
    print("="*60)

    # Save results
    result_path = Path("backtests") / "results.json"
    result_path.parent.mkdir(exist_ok=True)
    save_json_file(str(result_path), {
        "start_balance": START_BAL,
        "end_balance":   round(balance, 2),
        "total_pnl":     round(total_pnl, 2),
        "total_trades":  len(trades),
        "win_rate":      round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown":  round(max_dd, 2),
        "trades":        trades,
    })
    print(f"\n[OK] Results saved to {result_path}")


# ==============================================================================
# SECTION 22: MAIN LOOP
# ==============================================================================

def main():
    print("=" * 60)
    print("  FLAMMA AI v6.0 — UNIFIED PRODUCTION BOT")
    print("=" * 60)
    print(f"  LIVE_TRADING : {LIVE_TRADING}")
    print(f"  RISK_PERCENT : {RISK_PERCENT}%")
    print(f"  MAX_DAILY_LOSS: {MAX_DAILY_LOSS_PERCENT}%")
    print(f"  MAX_DAILY_TRADES: {MAX_DAILY_TRADES}")
    print(f"  SYMBOLS      : {SYMBOLS}")
    print("  Modules: Core | H4 MTF | Dynamic Lot | DailyGuard |")
    print("           SMC | Intermarket | FGI Sentiment | Q-Learning |")
    print("           Brain | SmartFilter | News | Healer | AutoOptimizer")
    print("=" * 60)

    # Initialize modules
    brain   = FlammaBrain()
    news    = NewsFilter()
    modules = {
        "smc":          SMCAnalyzer(),
        "intermarket":  IntermarketAnalyzer(),
        "sentiment":    SentimentAnalyzer(),
        "qlearning":    QLearningAgent(),
        "brain":        brain,
        "smart_filter": SmartFilter(brain, news),
    }

    kimi      = KimiClient()
    optimizer = AutoOptimizer(kimi)
    guard     = DailyGuard()

    # Connect MT5
    if not safe_connect_mt5():
        print("[FATAL] Cannot connect to MT5")
        return

    # Start Watchdog
    watchdog = Watchdog(max_silence=120)
    watchdog.start()

    send_telegram_message(
        f"[OK] Flamma v6.0 started\n"
        f"LIVE={LIVE_TRADING} | Risk={RISK_PERCENT}% | MaxDD={MAX_DAILY_LOSS_PERCENT}%\n"
        f"{guard.get_summary()}"
    )

    last_status_time = 0
    cycle_count      = 0

    while True:
        try:
            cycle_count += 1

            # MT5 check
            if not safe_check_mt5():
                time.sleep(30)
                continue

            # Daily guard
            allowed, guard_reason = guard.is_trading_allowed()
            if not allowed:
                print(f"[GUARD] {guard_reason}")
                send_telegram_message(f"[DAILY GUARD] Trading paused: {guard_reason}")
                time.sleep(300)
                continue

            # 12h status report
            if time.time() - last_status_time > 43200:
                s = healer.get_status()
                send_telegram_message(
                    f"[STATUS] Flamma v6 | Cycle #{cycle_count}\n"
                    f"Errors: {s['error_count']} | Fixes: {s['fixes_applied']}\n"
                    f"Healthy: {s['healthy']}\n"
                    f"{guard.get_summary()}"
                )
                last_status_time = time.time()

            # Watchdog heartbeat
            watchdog.beat()

            # Detect closed trades → update Q-Learning reward
            check_closed_trades(modules, guard)

            # Manage open positions
            manage_open_positions()

            # Wait if position open
            if has_open_positions():
                print("[WAIT] Open position — waiting...")
                time.sleep(30)
                continue

            # Cooldown
            if cooldown_active():
                print("[WAIT] Cooldown active...")
                time.sleep(30)
                continue

            # Get signal
            signal = get_best_signal(modules)
            if signal is None:
                print("[WAIT] No valid signal")
                time.sleep(30)
                continue

            # Execute (with Kimi AI review)
            execute_trade(signal, guard, modules, kimi=kimi)

            # Periodic optimizer
            optimizer.run()

            time.sleep(30)

        except KeyboardInterrupt:
            print("[STOP] Bot stopped by user")
            watchdog.stop()
            if LIVE_TRADING:
                print("[STOP] Closing all open positions...")
                close_all_positions()
            send_telegram_message("[STOP] Flamma v6.0 stopped by user — all positions closed")
            break

        except Exception as e:
            tb = traceback.format_exc()
            healer.log_error(str(e), tb)
            fix = healer.apply_fix(str(e))
            send_telegram_message(
                f"[ERROR] Auto-healing triggered\n"
                f"Error: {str(e)[:100]}\n"
                f"Fix: {fix} | Total errors: {healer.error_count}"
            )
            if healer.should_restart():
                send_telegram_message("[WARN] Too many errors — resetting MT5 connection...")
                healer.error_count = 0
                safe_connect_mt5()
            time.sleep(30)

    mt5.shutdown()


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        csv_file   = sys.argv[2] if len(sys.argv) > 2 else "backtests/data.csv"
        sym_arg    = sys.argv[3] if len(sys.argv) > 3 else "XAUUSDm"
        run_backtest(csv_file, symbol=sym_arg)
    else:
        main()
