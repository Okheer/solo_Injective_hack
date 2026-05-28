"""
reasoning.py — AI-powered on-chain reasoning log writer
Supports: Google Gemini (primary) → Anthropic Claude (fallback) → Local quantitative (last resort)
"""

import os
import json
from dotenv import load_dotenv
from signals import SignalResult

load_dotenv()


def build_reasoning_prompt(
    signal: SignalResult,
    trade: dict,
    market_context: dict,
) -> str:
    return f"""You are the reasoning module of an autonomous trading agent on Injective blockchain.
A trade was just executed. Write a concise, precise reasoning log (max 200 words) that will be stored permanently on-chain.

TRADE EXECUTED:
  Market:     {trade['market']}
  Direction:  {trade['direction'].upper()}
  Entry:      ${trade['entry_price']:.4f}
  Size:       {trade['size']} INJ
  Leverage:   {trade['leverage']}x
  Stop Loss:  {trade['stop_loss_pct']}%

SIGNALS THAT TRIGGERED THIS TRADE:
  Funding Rate:        {signal.funding_rate:.5f} ({signal.funding_rate*100:.4f}%)
  Carry Signal:        {"LONG (shorts paying longs)" if signal.carry_signal == 1 else "SHORT (longs paying shorts)"}
  H-Corrected RSI:     {signal.rsi:.1f} (period={signal.corrected_period}, vs standard 14)
  Hurst Exponent H:    {signal.hurst_H:.3f} (anti-persistent, fast mean-reverting)
  Confidence:          {signal.confidence.upper()}

MARKET CONTEXT:
  Mark Price: ${market_context.get('mark_price', trade['entry_price']):.4f}

Write the log in this exact structure:
SIGNAL: [one sentence on what triggered the trade]
RATIONALE: [why this signal is statistically meaningful — reference the math]
EXPECTED: [specific expected outcome with timeframe]
RISK: [stop loss and why it was set at this level]

Be precise, not promotional. This is a scientific record."""


def _call_gemini(prompt: str) -> str:
    """Call Google Gemini API via REST (avoids protobuf conflicts)."""
    from gemini_client import call_gemini
    return call_gemini(prompt, max_tokens=300)


def _call_anthropic(prompt: str) -> str:
    """Call Anthropic Claude API."""
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def _local_fallback(signal: SignalResult, trade: dict) -> str:
    """Quantitative fallback when no API key is available."""
    rsi_status = "oversold" if signal.rsi < 35 else ("overbought" if signal.rsi > 65 else "neutral")
    return (
        f"SIGNAL: Combined momentum {rsi_status} and carry triggered {trade['direction'].upper()} {trade['market']}.\n"
        f"RATIONALE: Hurst exponent of {signal.hurst_H:.3f} indicates anti-persistent mean-reversion. "
        f"Standard 14-period RSI calibrated down to {signal.corrected_period} due to fractal roughness, confirming {rsi_status} ({signal.rsi:.2f}).\n"
        f"EXPECTED: Mean reversion snapback to spot price within next 4 hours.\n"
        f"RISK: Stop loss placed at {trade['stop_loss_pct']}% to mitigate tail risk."
    )


def write_reasoning(
    signal: SignalResult,
    trade: dict,
    market_context: dict = None
) -> str:
    """
    Generate on-chain reasoning log using best available AI provider.
    Priority: Gemini → Claude → Local fallback
    """
    if market_context is None:
        market_context = {}

    prompt = build_reasoning_prompt(signal, trade, market_context)
    reasoning_text = None
    provider_used = "local_fallback"

    # Try Gemini first
    if os.getenv("GEMINI_API_KEY"):
        try:
            reasoning_text = _call_gemini(prompt)
            provider_used = "gemini"
        except Exception as e:
            print(f"[reasoning] Gemini call failed: {e}")

    # Try Anthropic second
    if reasoning_text is None and os.getenv("ANTHROPIC_API_KEY"):
        try:
            reasoning_text = _call_anthropic(prompt)
            provider_used = "anthropic"
        except Exception as e:
            print(f"[reasoning] Anthropic call failed: {e}")

    # Local fallback
    if reasoning_text is None:
        reasoning_text = _local_fallback(signal, trade)
        provider_used = "local_fallback"

    log = {
        "version": 1,
        "trade_tx": trade.get("tx_hash", "pending"),
        "market": trade["market"],
        "direction": trade["direction"],
        "entry_price": trade["entry_price"],
        "signal": {
            "funding_rate": signal.funding_rate,
            "hurst_H": signal.hurst_H,
            "rsi": signal.rsi,
            "rsi_period_used": signal.corrected_period,
            "confidence": signal.confidence,
        },
        "reasoning": reasoning_text,
        "ai_provider": provider_used,
    }
    return json.dumps(log)
