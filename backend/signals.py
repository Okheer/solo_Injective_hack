import numpy as np
import aiohttp
import os
import logging
import time
from dataclasses import dataclass
from pyinjective.indexer_client import IndexerClient
from pyinjective.core.network import Network
from pyinjective.client.model.pagination import PaginationOption
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("signals")

# ── Price staleness tracking 
_last_new_data_time = 0.0
_stale_count = 0



@dataclass
class SignalResult:
    action: str           
    funding_rate: float
    carry_signal: int     
    rsi: float
    hurst_H: float
    corrected_period: int
    standard_period: int = 14
    confidence: str = "medium"
    reasoning_inputs: dict = None


def estimate_hurst(prices: list[float], max_lag: int = 20) -> float:
    """
    Generalized Hurst Exponent via GHE method.
    Di Matteo, Aste, Dacorogna (2005).
    Returns H ∈ [0,1]. Crypto typical: H ≈ 0.08–0.15.
    """
    ts = np.array(prices)
    if len(ts) < max_lag * 2:
        logger.info(f"   [Hurst] Insufficient data ({len(ts)} pts < {max_lag*2}), defaulting H=0.50")
        return 0.5  # default to Brownian if insufficient data
    lags = range(2, min(max_lag, len(ts) // 2))
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    if not tau or min(tau) <= 0:
        logger.info(f"   [Hurst] Zero variance detected, defaulting H=0.50")
        return 0.5
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    H = float(np.clip(poly[0], 0.01, 0.99))
    logger.info(f"   [Hurst] GHE regression: slope={poly[0]:.4f} → H={H:.4f} | lags=[2..{max(lags)}] | n={len(prices)} pts")
    return H


def compute_rsi(prices: list[float], period: int) -> float:
    """Standard RSI computation for given period."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean() if gains.mean() > 0 else 1e-9
    avg_loss = losses.mean() if losses.mean() > 0 else 1e-9
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def hurst_corrected_rsi(prices: list[float], standard_period: int = 14) -> tuple[float, float, int]:
    """
    RSI with lookback corrected for measured Hurst exponent.
    
    Rationale:
      Standard RSI assumes H = 0.5 (Brownian) → 14-period is calibrated for this.
      For H < 0.5 (anti-persistent crypto markets), mean reversion is faster.
      Correct period ≈ standard_period × (H / 0.5).
      
      H=0.10 → period = 14 × 0.2 = ~3  (very short, captures fast snap-backs)
      H=0.25 → period = 14 × 0.5 = ~7
      H=0.50 → period = 14             (standard RSI, no correction)
    
    Returns: (rsi_value, hurst_H, corrected_period)
    """
    H = estimate_hurst(prices[-60:] if len(prices) >= 60 else prices)
    corrected_period = max(2, int(standard_period * (H / 0.5)))
    rsi = compute_rsi(prices, corrected_period)
    return rsi, H, corrected_period


def funding_carry_signal(funding_rate: float, threshold: float = 0.001) -> int:
    """
    Carry signal from perpetual funding rate.
    
    When F < -threshold: shorts paying longs → long has positive carry → BUY
    When F >  threshold: longs paying shorts → short has positive carry → SELL
    
    Economic basis: Fama (1984) forward premium / UIP deviation.
    Applied to crypto perps: Frazzini & Pedersen (2014) carry factor.
    """
    if funding_rate < -threshold:
        return 1
    elif funding_rate > threshold:
        return -1
    return 0


def generate_signal(
    funding_rate: float,
    prices: list[float],
    rsi_oversold: float = 35.0,
    rsi_overbought: float = 65.0,
    funding_threshold: float = 0.001,
) -> SignalResult:
    """
    Combined signal: funding carry AND H-corrected RSI must agree.
    Both signals required to reduce false positives.
    """
    carry = funding_carry_signal(funding_rate, funding_threshold)
    rsi, H, period = hurst_corrected_rsi(prices)

    long_signal  = (carry == 1)  and (rsi < rsi_oversold)
    short_signal = (carry == -1) and (rsi > rsi_overbought)

    if long_signal:
        action = "long"
    elif short_signal:
        action = "short"
    else:
        action = "hold"

    confidence = (
        "high"   if abs(funding_rate) > 0.002 and (rsi < 28 or rsi > 72)
        else "medium" if action != "hold"
        else "low"
    )

    return SignalResult(
        action=action,
        funding_rate=funding_rate,
        carry_signal=carry,
        rsi=round(rsi, 2),
        hurst_H=round(H, 3),
        corrected_period=period,
        confidence=confidence,
        reasoning_inputs={
            "funding_rate": funding_rate,
            "funding_threshold": funding_threshold,
            "rsi": round(rsi, 2),
            "rsi_oversold_threshold": rsi_oversold,
            "rsi_overbought_threshold": rsi_overbought,
            "hurst_H": round(H, 3),
            "standard_rsi_period": 14,
            "corrected_rsi_period": period,
        }
    )


async def get_indexer_client(network_name: str = None) -> IndexerClient:
    """Helper to instantiate IndexerClient based on configured network."""
    if network_name is None:
        network_name = os.getenv("INJECTIVE_NETWORK", "testnet")
    network = Network.testnet() if network_name == "testnet" else Network.mainnet()
    return IndexerClient(network)


async def fetch_funding_rate(market_id: str, client: IndexerClient = None) -> float:
    """
    Fetch current funding rate from Injective Indexer gRPC.
    Returns funding rate as decimal (e.g. -0.00012 = -0.012%)
    """
    try:
        if client is None:
            client = await get_indexer_client()
        res = await client.fetch_funding_rates(market_id=market_id)
        rates = res.get("fundingRates", [])
        if rates:
            rate = float(rates[0].get("rate", "0"))
            logger.info(f"   [gRPC] Funding rate fetched: {rate:.10f} ({rate*100:.6f}%)")
            return rate
        logger.info(f"   [gRPC] No funding rates returned for market")
        return 0.0
    except Exception as e:
        logger.warning(f"   [gRPC] Funding rate fetch error: {e}")
        return 0.0 


async def fetch_recent_prices(market_id: str, n: int = 100, client: IndexerClient = None) -> list[float]:
    """
    Fetch recent trade prices from Injective Indexer gRPC for Hurst estimation.
    Returns list of floats ordered oldest→newest.
    
    Uses pagination to fetch 500 trades and filters out liquidations and consecutive duplicates
    to yield organic price movements. Tracks data staleness.
    """
    global _last_price_hash, _last_new_data_time, _stale_count

    try:
        if client is None:
            client = await get_indexer_client()
        
        pagination = PaginationOption(limit=500)
        res = await client.fetch_derivative_trades(market_ids=[market_id], pagination=pagination)
        trades = res.get("trades", [])
        
        # The gRPC Trades are returned newest first.
        # We reverse them to get oldest first.
        prices = []
        for t in reversed(trades):
            # Skip liquidations to capture organic market behavior
            if t.get("tradeExecutionType") == "marketLiquidation":
                continue
            if "positionDelta" in t and "executionPrice" in t["positionDelta"]:
                p = float(t["positionDelta"]["executionPrice"]) / 1e6
                # Filter out consecutive duplicates to capture clean tick updates
                if not prices or prices[-1] != p:
                    prices.append(p)

        if prices:
            price_hash = hash(tuple(prices[-20:]))  # hash of last 20 prices
            if price_hash != _last_price_hash:
                _last_price_hash = price_hash
                _last_new_data_time = time.time()
                _stale_count = 0
                logger.info(f"   [gRPC] ✅ Fresh price data: {len(prices)} unique ticks | Latest: ${prices[-1]:.4f}")
            else:
                _stale_count += 1
                stale_secs = time.time() - _last_new_data_time if _last_new_data_time else 0
                if _stale_count % 5 == 1:  # log every 5th stale fetch to avoid spam
                    logger.info(f"   [gRPC] ♻️ Same trade data (stale {stale_secs:.0f}s, {_stale_count} checks) — no new trades on-chain")
                
        if len(prices) >= n:
            return prices[-n:]
        elif len(prices) > 0:
            return [prices[0]] * (n - len(prices)) + prices
        
        # Fallback to general price if no organic trades found
        fallback_prices = []
        for t in reversed(trades):
            if "positionDelta" in t and "executionPrice" in t["positionDelta"]:
                fallback_prices.append(float(t["positionDelta"]["executionPrice"]) / 1e6)
        if len(fallback_prices) >= n:
            return fallback_prices[-n:]
        elif len(fallback_prices) > 0:
            return [fallback_prices[0]] * (n - len(fallback_prices)) + fallback_prices
            
        return [5.1] * n 
    except Exception as e:
        logger.warning(f"   [gRPC] Price fetch error: {e}")
        return [5.1] * n

