
import json

def build_compact_memo(signal_data: dict, trade_data: dict) -> str:
    """
    Builds a compact JSON memo for Injective transaction.
    Max 256 characters for Cosmos SDK memo field.
    """
    memo = {
        "v": 1,                                    # version
        "m": trade_data.get("market", "INJ/USDT"),
        "d": trade_data.get("direction", "long"),
        "e": round(trade_data.get("entry_price", 0), 4),
        "f": round(signal_data.get("funding_rate", 0), 6),
        "r": round(signal_data.get("rsi", 50), 2),
        "H": round(signal_data.get("hurst_H", 0.5), 3),
        "p": signal_data.get("corrected_period", 14),
        "sl": trade_data.get("stop_loss_pct", 1.5),
        "sz": trade_data.get("size", 0.5),
        "c": signal_data.get("confidence", "medium")
    }
    # Use separators to remove whitespace and save characters
    return json.dumps(memo, separators=(',', ':'))
