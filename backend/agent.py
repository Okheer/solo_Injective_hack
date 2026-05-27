"""
agent.py — Main trading agent loop with rich terminal logging
"""

import asyncio
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

from signals import generate_signal, fetch_funding_rate, fetch_recent_prices
from injective_client import InjectiveClient
from memo_store import build_compact_memo

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

class TradingAgent:
    def __init__(self, config: dict):
        self.market_id = config.get("market_id", os.getenv("INJ_MARKET_ID"))
        self.market_name = config.get("market", "INJ/USDT PERP")
        self.strategy_desc = config.get("strategy", "Funding rate carry + Hurst-corrected RSI")
        self.size_inj = config.get("size_inj", 0.5)
        self.leverage = config.get("leverage", 2)
        self.stop_loss_pct = config.get("stop_loss_pct", 1.5)
        self.dry_run = config.get("dry_run", True)
        
        self.running = False
        self.client = InjectiveClient()
        self.last_signal = None
        self.trades = []
        self.cycle_count = 0
        
        mode = "[DRY RUN]" if self.dry_run else "[LIVE EXECUTION]"
        logger.info(f"🚀 TradingAgent initialized | Mode: {mode}")
        logger.info(f"   Market:     {self.market_name}")
        logger.info(f"   Strategy:   {self.strategy_desc}")
        logger.info(f"   Size:       {self.size_inj} INJ | Leverage: {self.leverage}x | Stop: {self.stop_loss_pct}%")

        # Report which AI providers are available
        gemini_key = os.getenv("GEMINI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if gemini_key:
            from gemini_client import GEMINI_MODEL
            logger.info(f"   AI Provider: ✅ Gemini (primary) — model={GEMINI_MODEL}, key loaded ({gemini_key[:8]}...)")
        if anthropic_key:
            logger.info(f"   AI Provider: ✅ Anthropic Claude (fallback) — key loaded ({anthropic_key[:8]}...)")
        if not gemini_key and not anthropic_key:
            logger.warning("   AI Provider: ⚠️ No API keys found — using local quantitative fallback")

    async def run(self):
        self.running = True
        logger.info(f"───────────── Agent Loop Started ─────────────")
        
        poll_interval = int(os.getenv("POLL_INTERVAL_SECONDS", 30))
        logger.info(f"   Poll interval: {poll_interval}s")
        
        last_hurst = None
        
        while self.running:
            try:
                self.cycle_count += 1
                
                # 1. Fetch Market Data
                logger.info(f"── Cycle #{self.cycle_count} ──────────────────────")
                logger.info(f"📡 Fetching live market data from Injective gRPC...")
                
                funding_rate = await fetch_funding_rate(self.market_id)
                logger.info(f"   Funding Rate (raw):   {funding_rate:.8f}")
                
                prices = await fetch_recent_prices(self.market_id)
                if prices:
                    logger.info(f"   Prices fetched:       {len(prices)} candles | Latest: ${prices[0]:.4f}")
                else:
                    logger.warning(f"   ⚠️ No price data returned")
                
                # 2. Generate Signal — this is where the math happens
                logger.info(f"🧮 Running signal engine...")
                signal = generate_signal(funding_rate, prices)
                self.last_signal = signal
                
                # Detect if Hurst has changed
                hurst_changed = last_hurst is None or abs(signal.hurst_H - last_hurst) > 0.001
                last_hurst = signal.hurst_H
                
                # Log detailed math breakdown
                logger.info(f"   ┌─ SIGNAL COMPUTATION RESULTS ─────────────")
                logger.info(f"   │ Funding Rate:     {signal.funding_rate:.8f} ({signal.funding_rate*100:.5f}%)")
                carry_dir = "LONG (shorts→longs)" if signal.carry_signal == 1 else ("SHORT (longs→shorts)" if signal.carry_signal == -1 else "NEUTRAL")
                logger.info(f"   │ Carry Signal:     {carry_dir}")
                logger.info(f"   │ Hurst Exponent:   H = {signal.hurst_H:.4f}  {'(anti-persistent ✓)' if signal.hurst_H < 0.4 else '(trending/random)'} {'[UPDATED]' if hurst_changed else '[unchanged]'}")
                logger.info(f"   │ RSI Period Adj:   14 → {signal.corrected_period} (scaled by H={signal.hurst_H:.3f})")
                logger.info(f"   │ H-Corrected RSI:  {signal.rsi:.2f}  {'[OVERSOLD]' if signal.rsi < 35 else ('[OVERBOUGHT]' if signal.rsi > 65 else '[NEUTRAL]')}")
                logger.info(f"   │ Confidence:       {signal.confidence.upper()}")
                logger.info(f"   │ ══════════════════════════════════")
                logger.info(f"   └─ ACTION:          {signal.action.upper()}")

                # 3. Check for Execution
                if signal.action != "hold":
                    await self.execute_trade(signal)
                else:
                    logger.info(f"   ⏸ HOLD — Conditions not met. Next check in {poll_interval}s...")
                
                await asyncio.sleep(poll_interval)
            except Exception as e:
                logger.error(f"❌ Error in agent loop: {e}")
                await asyncio.sleep(5)

    async def execute_trade(self, signal):
        logger.info(f"🔥 TRIGGERED: {signal.action.upper()} on {self.market_name}")
        
        # Build trade data for memo
        recent_prices = await fetch_recent_prices(self.market_id, 1)
        current_price = recent_prices[0] if recent_prices else 5.1
        
        tx_hash = "pending"
        
        trade_data = {
            "market": self.market_name,
            "direction": signal.action,
            "entry_price": current_price,
            "size": self.size_inj,
            "leverage": self.leverage,
            "stop_loss_pct": self.stop_loss_pct,
            "tx_hash": tx_hash
        }
        
        # Build compact memo for on-chain storage
        memo = build_compact_memo(signal.__dict__, trade_data)
        logger.info(f"📝 On-chain memo built ({len(memo)}/256 bytes): {memo}")

        # Generate full reasoning log via AI
        logger.info(f"🤖 Generating AI reasoning journal...")
        from reasoning import write_reasoning
        full_reasoning_log = write_reasoning(signal, trade_data)
        
        # Check which AI was used
        try:
            import json
            log_parsed = json.loads(full_reasoning_log)
            provider = log_parsed.get("ai_provider", "unknown")
            logger.info(f"   AI Provider used: {provider.upper()}")
            reasoning_text = log_parsed.get("reasoning", "")
            if reasoning_text:
                # Show first 200 chars of the reasoning
                preview = reasoning_text[:200].replace('\n', ' ')
                logger.info(f"   Reasoning preview: {preview}...")
        except:
            pass

        if self.dry_run:
            logger.info(f"[DRY RUN] Simulating {signal.action.upper()} execution on-chain...")
            import uuid
            tx_hash = f"0xsimulated_{uuid.uuid4().hex}"
            trade_data["tx_hash"] = tx_hash
            logger.info(f"   Simulated tx hash: {tx_hash}")
            # Re-generate reasoning log with simulated hash
            full_reasoning_log = write_reasoning(signal, trade_data)
        else:
            logger.info(f"[LIVE] Executing {signal.action.upper()} on-chain...")
            try:
                tx_result = await self.client.place_derivative_order(
                    market_id=self.market_id,
                    price=current_price,
                    quantity=self.size_inj,
                    leverage=self.leverage,
                    is_buy=(signal.action == "long")
                )
                if tx_result and "txHash" in tx_result:
                    tx_hash = tx_result["txHash"]
                    trade_data["tx_hash"] = tx_hash
                    logger.info(f"   ✅ On-chain tx: {tx_hash}")
                    full_reasoning_log = write_reasoning(signal, trade_data)
            except Exception as ex:
                logger.error(f"   ❌ On-chain order execution failed: {ex}")
        
        # Log trade locally
        self.trades.append({
            "timestamp": datetime.now().isoformat(),
            "trade_data": trade_data,
            "signal": signal.__dict__,
            "memo": memo,
            "reasoning_log": full_reasoning_log
        })
        logger.info(f"✅ Trade #{len(self.trades)} logged locally. Total trades: {len(self.trades)}")

    def get_state(self):
        """Returns current agent state for API/WS"""
        return {
            "running": self.running,
            "market": self.market_name,
            "last_signal": self.last_signal.__dict__ if self.last_signal else None,
            "trades_count": len(self.trades),
            "trades": self.trades,
            "cycle_count": self.cycle_count,
        }
