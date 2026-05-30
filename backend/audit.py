import os
import aiohttp
import json
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("audit")


@dataclass
class TradeLog:
    tx_hash: str
    timestamp: str
    market: str
    direction: str
    entry_price: float
    reasoning: str
    signal_data: dict
    exit_price: float | None = None
    pnl_pct: float | None = None


async def fetch_trade_logs(wallet_address: str, limit: int = 20) -> list[TradeLog]:
    """
    Fetches all tx memos written by this wallet from Injective LCD.
    Parses those matching the TRADE_LOG_v1 format.
    """
    url = (
        f"https://lcd.injective.network/cosmos/tx/v1beta1/txs"
        f"?events=message.sender='{wallet_address}'&limit={limit}&order_by=ORDER_BY_DESC"
    )
    logs = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as r:
                if r.status != 200:
                    return []
                data = await r.json()
    except Exception:
        return []
    
    for tx in data.get("tx_responses", []):
        memo = tx.get("tx", {}).get("body", {}).get("memo", "")
        if not memo:
            continue
        try:
            log_data = json.loads(memo)
            # Check for version 1 format (compact or full)
            if log_data.get("v") == 1 or log_data.get("version") == 1:
                logs.append(TradeLog(
                    tx_hash=tx["txhash"],
                    timestamp=tx.get("timestamp", ""),
                    market=log_data.get("m") or log_data.get("market"),
                    direction=log_data.get("d") or log_data.get("direction"),
                    entry_price=log_data.get("e") or log_data.get("entry_price"),
                    reasoning=log_data.get("reasoning", ""),
                    signal_data={
                        "funding_rate": log_data.get("f") or log_data.get("signal", {}).get("funding_rate"),
                        "rsi": log_data.get("r") or log_data.get("signal", {}).get("rsi"),
                        "hurst_H": log_data.get("H") or log_data.get("signal", {}).get("hurst_H"),
                        "confidence": log_data.get("c") or log_data.get("signal", {}).get("confidence"),
                    }
                ))
        except (json.JSONDecodeError, KeyError):
            continue
    return logs


# ── AI Provider Calls ────────────────────────────────────────────────────

def _call_gemini_audit(prompt: str) -> str:
    """Call Google Gemini for audit via REST."""
    from gemini_client import call_gemini
    return call_gemini(prompt, max_tokens=600)


def _call_anthropic_audit(prompt: str) -> str:
    """Call Anthropic Claude for audit."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def _call_ai(prompt: str) -> str | None:
    """Try all AI providers in order. Returns None if all fail."""
    # Try Gemini
    if os.getenv("GEMINI_API_KEY"):
        try:
            result = _call_gemini_audit(prompt)
            logger.info("AI audit response generated via Gemini")
            return result
        except Exception as e:
            logger.warning(f"Gemini audit failed: {e}")

    # Try Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            result = _call_anthropic_audit(prompt)
            logger.info("AI audit response generated via Anthropic")
            return result
        except Exception as e:
            logger.warning(f"Anthropic audit failed: {e}")

    return None


# ── Prompt Builders ──────────────────────────────────────────────────────

def _build_audit_prompt_from_logs(logs_text: str, count: int, question: str) -> str:
    return f"""You are auditing an AI trading agent's performance on Injective blockchain.
You have access to the agent's trade history with exact signal values and reasoning.

Here are the agent's {count} most recent trades with their reasoning:

{logs_text}

The user asks: "{question}"

Answer directly and quantitatively. Reference specific trades by number.
If asked about win rate or P&L, note that exit data may not be available and be transparent about limitations.
Be honest about any patterns — good or bad. You are conducting a scientific review, not selling performance."""


def _build_general_chat_prompt(question: str, agent_state: dict = None) -> str:
    """Build a prompt for general questions about the agent, even when no trades exist."""
    state_context = ""
    if agent_state:
        sig = agent_state.get("last_signal")
        if sig:
            state_context = f"""
CURRENT LIVE AGENT STATE:
  Agent Status:     {"ACTIVE (running)" if agent_state.get("running") else "IDLE (stopped)"}
  Market:           {agent_state.get("market", "INJ/USDT PERP")}
  Cycles Completed: {agent_state.get("cycle_count", 0)}
  Trades Executed:  {agent_state.get("trades_count", 0)}
  
  LATEST SIGNAL READINGS:
    Funding Rate:     {sig.get("funding_rate", "N/A")} ({float(sig.get("funding_rate", 0))*100:.5f}%)
    Carry Signal:     {"LONG (shorts→longs)" if sig.get("carry_signal") == 1 else ("SHORT (longs→shorts)" if sig.get("carry_signal") == -1 else "NEUTRAL")}
    Hurst Exponent H: {sig.get("hurst_H", "N/A")} {"(anti-persistent ✓)" if sig.get("hurst_H", 0.5) < 0.4 else "(random walk)"}
    H-Corrected RSI:  {sig.get("rsi", "N/A")} (period adjusted from 14 → {sig.get("corrected_period", "N/A")})
    Confidence:       {sig.get("confidence", "N/A").upper()}
    Current Action:   {sig.get("action", "N/A").upper()}
"""
        else:
            state_context = f"""
CURRENT AGENT STATE:
  Agent Status:     {"ACTIVE (running)" if agent_state.get("running") else "IDLE (stopped)"}
  Market:           {agent_state.get("market", "INJ/USDT PERP")}
  Cycles Completed: {agent_state.get("cycle_count", 0)}
  Trades Executed:  {agent_state.get("trades_count", 0)}
  Last Signal:      No signal data yet
"""

    return f"""You are the AI auditor for an autonomous trading agent on Injective blockchain.
The agent trades INJ/USDT perpetual futures using a combination of:
1. Funding Rate Carry (based on Fama/Frazzini-Pedersen UIP deviation)
2. Hurst-Corrected RSI (GHE method adjusts RSI lookback for anti-persistent markets)

The agent only executes trades when BOTH carry signal AND RSI agree (to reduce false positives).
The current market is INJ/USDT PERP on Injective mainnet.
The agent is running in DRY-RUN mode (simulation, no real on-chain trades).
{state_context}

STRATEGY DETAILS:
- When Hurst H < 0.5 (anti-persistent), markets mean-revert faster → RSI period shortened
- H ≈ 0.21 means strong anti-persistence, RSI period scaled from 14 to ~5
- Funding rate threshold is 0.001 (0.1%) — below that, carry signal is NEUTRAL
- Both carry AND RSI must confirm for trade execution
- Current market: funding rate is very small (~0.0004%), so carry = NEUTRAL → agent HOLDs

The user asks: "{question}"

Answer helpfully, accurately, and concisely. Explain the math/strategy if asked.
If asked about trades when none exist, explain WHY the agent is holding (conditions not met).
Reference actual signal values from the live state above when relevant.
Format your response with markdown for readability."""


# ── Local Fallback ───────────────────────────────────────────────────────

def _local_audit_fallback(logs_data: list[dict], question: str, agent_state: dict = None) -> str:
    """Quantitative fallback when no AI API key is available."""
    if not logs_data:
        # Even without AI, provide a useful response based on agent state
        if agent_state and agent_state.get("last_signal"):
            sig = agent_state["last_signal"]
            return (
                f"📊 **Live Agent Status** (Local Analysis — no AI API key active)\n\n"
                f"The agent is currently **{sig.get('action', 'hold').upper()}ING** on {agent_state.get('market', 'INJ/USDT PERP')}.\n\n"
                f"**Latest Signal Readings:**\n"
                f"- Funding Rate: `{float(sig.get('funding_rate', 0)):.8f}` ({float(sig.get('funding_rate', 0))*100:.5f}%)\n"
                f"- Carry Signal: `{'NEUTRAL' if sig.get('carry_signal') == 0 else ('LONG' if sig.get('carry_signal') == 1 else 'SHORT')}`\n"
                f"- Hurst H: `{sig.get('hurst_H', 'N/A')}` {'(anti-persistent ✓)' if sig.get('hurst_H', 0.5) < 0.4 else ''}\n"
                f"- RSI: `{sig.get('rsi', 'N/A')}` (period: {sig.get('corrected_period', 14)})\n"
                f"- Confidence: `{sig.get('confidence', 'N/A').upper()}`\n\n"
                f"**Why HOLD?** The funding rate ({float(sig.get('funding_rate', 0))*100:.5f}%) is below the 0.1% threshold, "
                f"so carry signal is NEUTRAL. Both carry AND RSI must agree for trade execution.\n\n"
                f"You asked: *\"{question}\"*\n\n"
                f"💡 For AI-powered natural language analysis, ensure `GEMINI_API_KEY` is set and working."
            )
        return (
            "🔍 **Agent Not Started**\n\n"
            "Click **Start Agent** to begin. The agent will fetch live market data from "
            "Injective gRPC, compute Hurst exponents and RSI signals, and execute trades "
            "when carry + momentum signals align.\n\n"
            f"You asked: *\"{question}\"*"
        )

    long_count = sum(1 for l in logs_data if l.get("direction", "").lower() == "long")
    short_count = sum(1 for l in logs_data if l.get("direction", "").lower() == "short")

    hurst_vals = [l["signal"].get("hurst_H") for l in logs_data if l.get("signal", {}).get("hurst_H") is not None]
    avg_hurst = sum(hurst_vals) / len(hurst_vals) if hurst_vals else 0.5

    rsi_vals = [l["signal"].get("rsi") for l in logs_data if l.get("signal", {}).get("rsi") is not None]
    avg_rsi = sum(rsi_vals) / len(rsi_vals) if rsi_vals else 50.0

    report = (
        f"📊 **LOCAL QUANTITATIVE AUDIT** (No AI API key active)\n\n"
        f"Scanned **{len(logs_data)}** trades:\n"
        f"- **Direction:** {long_count} LONG | {short_count} SHORT\n"
        f"- **Mean Hurst H:** `{avg_hurst:.4f}` ({'Anti-persistent ✅' if avg_hurst < 0.4 else 'Random walk ⚠️'})\n"
        f"- **Mean RSI:** `{avg_rsi:.2f}`\n\n"
        f"You asked: *\"{question}\"*\n\n"
        f"💡 For AI-powered natural language analysis, ensure `GEMINI_API_KEY` is set and working."
    )
    return report


# ── Main Audit Functions ─────────────────────────────────────────────────

def run_general_chat(question: str, agent_state: dict = None) -> str:
    """
    General purpose AI chat — works even when no trades exist.
    Uses live agent state to answer questions about current market conditions.
    """
    prompt = _build_general_chat_prompt(question, agent_state)

    # Try AI providers
    ai_response = _call_ai(prompt)
    if ai_response:
        return ai_response

    # Local fallback
    return _local_audit_fallback([], question, agent_state)


def run_self_audit_from_memory(agent_trades: list[dict], question: str, agent_state: dict = None) -> str:
    """
    Run self-audit using the agent's in-memory trade list.
    This works even in dry-run mode with no on-chain data.
    """
    if not agent_trades:
        # No trades yet — use general chat instead
        return run_general_chat(question, agent_state)

    # Build structured trade summaries for the LLM
    parsed = []
    logs_text_parts = []  # ← FIXED: was incorrectly inside the for loop before

    for i, trade in enumerate(agent_trades):
        td = trade.get("trade_data", {})
        sig = trade.get("signal", {})
        reasoning_raw = trade.get("reasoning_log", "")

        # Parse reasoning if it's JSON
        reasoning_text = ""
        try:
            r = json.loads(reasoning_raw)
            reasoning_text = r.get("reasoning", reasoning_raw)
        except (json.JSONDecodeError, TypeError):
            reasoning_text = str(reasoning_raw)

        parsed.append({
            "direction": td.get("direction", "unknown"),
            "signal": sig,
        })

        logs_text_parts.append(
            f"Trade {i+1}: {td.get('direction', '?').upper()} {td.get('market', '?')} @ ${td.get('entry_price', 0):.4f}\n"
            f"  Timestamp:     {trade.get('timestamp', 'N/A')}\n"
            f"  Leverage:      {td.get('leverage', '?')}x | Size: {td.get('size', '?')} INJ\n"
            f"  Funding Rate:  {sig.get('funding_rate', 'N/A')}\n"
            f"  Hurst H:       {sig.get('hurst_H', 'N/A')}\n"
            f"  RSI:           {sig.get('rsi', 'N/A')} (period={sig.get('corrected_period', 'N/A')})\n"
            f"  Confidence:    {sig.get('confidence', 'N/A')}\n"
            f"  Stop Loss:     {td.get('stop_loss_pct', 'N/A')}%\n"
            f"  Reasoning:     {reasoning_text[:400]}"
        )

    logs_text = "\n\n".join(logs_text_parts)
    prompt = _build_audit_prompt_from_logs(logs_text, len(agent_trades), question)

    # Try AI providers
    ai_response = _call_ai(prompt)
    if ai_response:
        return ai_response

    # Local fallback
    return _local_audit_fallback(parsed, question, agent_state)


def run_self_audit(logs: list[TradeLog], question: str) -> str:
    """
    Run self-audit from on-chain TradeLog objects.
    """
    if not logs:
        return _local_audit_fallback([], question)

    logs_text = "\n\n".join([
        f"Trade {i+1}: {log.direction.upper()} {log.market} @ ${log.entry_price:.4f}\n"
        f"  Funding rate: {log.signal_data.get('funding_rate', 'N/A')}\n"
        f"  Hurst H: {log.signal_data.get('hurst_H', 'N/A')}\n"
        f"  RSI (H-corrected): {log.signal_data.get('rsi', 'N/A')}\n"
        f"  Confidence: {log.signal_data.get('confidence', 'N/A')}\n"
        f"  Reasoning: {log.reasoning[:300]}..."
        for i, log in enumerate(logs)
    ])

    prompt = _build_audit_prompt_from_logs(logs_text, len(logs), question)

    # Try AI providers
    ai_response = _call_ai(prompt)
    if ai_response:
        return ai_response

    # Local fallback for on-chain logs
    parsed = [{"direction": l.direction, "signal": l.signal_data} for l in logs]
    return _local_audit_fallback(parsed, question)
