from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
import logging
from datetime import datetime
from collections import deque

from agent import TradingAgent
from audit import fetch_trade_logs, run_self_audit, run_self_audit_from_memory, run_general_chat

# ---------------------------------------------------------------------------
# Custom log handler that captures logs into a ring buffer for the frontend
# ---------------------------------------------------------------------------
class TerminalLogBuffer(logging.Handler):
    """Captures log records into a bounded deque for frontend streaming."""
    def __init__(self, maxlen: int = 500):
        super().__init__()
        self.buffer = deque(maxlen=maxlen)

    def emit(self, record):
        entry = {
            "ts": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "level": record.levelname,
            "name": record.name,
            "msg": self.format(record),
        }
        self.buffer.append(entry)

    def get_recent(self, n: int = 200) -> list[dict]:
        return list(self.buffer)[-n:]

    def get_new_since(self, count: int) -> list[dict]:
        """Return entries added after `count` total entries were seen."""
        all_entries = list(self.buffer)
        if count >= len(all_entries):
            return []
        return all_entries[count:]


# Setup logging
log_buffer = TerminalLogBuffer(maxlen=500)
log_buffer.setFormatter(logging.Formatter("%(name)s | %(message)s"))

# Attach buffer handler to root logger so it captures everything
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(log_buffer)

# Also add a stream handler for actual terminal output
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
root_logger.addHandler(stream_handler)

logger = logging.getLogger("main")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_instance: TradingAgent | None = None
agent_task: asyncio.Task | None = None


class StartRequest(BaseModel):
    market: str = "INJ/USDT PERP"
    market_id: str = "0x9b9980167ecc3645ff1a5517886652d94a0825e54a77d2057cbbe3ebee015963"
    strategy: str = "Funding rate carry + Hurst-corrected RSI"
    size_inj: float = 0.5
    leverage: int = 2
    stop_loss_pct: float = 1.5
    dry_run: bool = True


@app.post("/start")
async def start_agent(req: StartRequest):
    global agent_instance, agent_task
    
    if agent_instance and agent_instance.running:
        return {"status": "error", "message": "Agent already running"}
    
    agent_instance = TradingAgent(req.model_dump())
    agent_task = asyncio.create_task(agent_instance.run())
    logger.info(f"🚀 Agent started: {req.strategy} on {req.market} (dry_run={req.dry_run})")
    return {"status": "agent started", "strategy": req.strategy}


@app.post("/stop")
async def stop_agent():
    global agent_instance, agent_task
    if agent_instance:
        agent_instance.running = False
        if agent_task:
            agent_task.cancel()
            agent_task = None
        logger.info("🛑 Agent stopped by user")
        return {"status": "stopped"}
    return {"status": "error", "message": "No agent running"}


@app.get("/audit")
async def audit(wallet: str = "", question: str = "How did my last trades perform?"):
    """
    Self-audit / general chat endpoint.
    - If agent has trades → audit them with AI
    - If no trades but agent running → general AI chat about current state
    - Falls back to on-chain logs if wallet provided
    """
    # Get current agent state for context
    agent_state = agent_instance.get_state() if agent_instance else {"running": False}

    # Prefer in-memory trades (always available during dry-run)
    if agent_instance and agent_instance.trades:
        logger.info(f"🔍 Auditing {len(agent_instance.trades)} in-memory trades: '{question}'")
        answer = run_self_audit_from_memory(agent_instance.trades, question, agent_state)
        return {"logs_count": len(agent_instance.trades), "source": "in_memory", "answer": answer}
    
    # Fall back to on-chain logs if wallet provided
    if wallet:
        logger.info(f"🔍 Fetching on-chain logs for wallet: {wallet}")
        logs = await fetch_trade_logs(wallet)
        if logs:
            answer = run_self_audit(logs, question)
            return {"logs_count": len(logs), "source": "on_chain", "answer": answer}

    # General AI chat — even with no trades, AI can answer about current state/strategy
    logger.info(f"💬 General AI chat (no trades yet): '{question}'")
    answer = run_general_chat(question, agent_state)
    return {
        "logs_count": 0,
        "source": "ai_chat",
        "answer": answer
    }


@app.get("/status")
async def get_status():
    if agent_instance:
        return agent_instance.get_state()
    return {"running": False}


# Cache test-ai results to avoid burning free-tier quota (20 RPD)
_test_ai_cache = {"result": None, "timestamp": 0}
_TEST_AI_CACHE_TTL = 300  # 5 minutes


@app.get("/test-ai")
async def test_ai(force: bool = False):
    """Quick test to verify AI API key works. Cached for 5 minutes to preserve free-tier quota."""
    import os
    import time as _time

    # Return cached result if fresh
    if not force and _test_ai_cache["result"] and (_time.time() - _test_ai_cache["timestamp"]) < _TEST_AI_CACHE_TTL:
        logger.info("🤖 Returning cached test-ai result (use /test-ai?force=true to re-test)")
        return _test_ai_cache["result"]

    gemini_key = os.getenv("GEMINI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    result = {"gemini_key_set": bool(gemini_key), "anthropic_key_set": bool(anthropic_key)}

    if gemini_key:
        try:
            from gemini_client import call_gemini, GEMINI_MODEL
            resp_text = call_gemini("Reply with exactly: GEMINI_OK", max_tokens=50, retries=1)
            result["gemini_status"] = "✅ working"
            result["gemini_model"] = GEMINI_MODEL
            result["gemini_response"] = resp_text.strip()[:100]
            logger.info(f"🤖 Gemini API test: ✅ SUCCESS (model={GEMINI_MODEL})")
        except Exception as e:
            result["gemini_status"] = f"❌ error: {str(e)[:200]}"
            logger.error(f"🤖 Gemini API test: ❌ {e}")

    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            msg = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=20, messages=[{"role":"user","content":"Reply with exactly: CLAUDE_OK"}])
            result["anthropic_status"] = "✅ working"
            result["anthropic_response"] = msg.content[0].text.strip()[:100]
            logger.info(f"🤖 Anthropic API test: ✅ SUCCESS")
        except Exception as e:
            result["anthropic_status"] = f"❌ error: {str(e)[:200]}"
            logger.error(f"🤖 Anthropic API test: ❌ {e}")

    # Cache the result
    _test_ai_cache["result"] = result
    _test_ai_cache["timestamp"] = _time.time()

    return result


@app.get("/logs")
async def get_logs(since: int = 0):
    """REST endpoint to fetch recent terminal logs."""
    entries = log_buffer.get_new_since(since)
    return {"total": len(list(log_buffer.buffer)), "entries": entries}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Streams live agent state + terminal logs to frontend.
    """
    await websocket.accept()
    logger.info("WebSocket client connected")
    log_cursor = len(list(log_buffer.buffer))  # start from current position
    try:
        while True:
            state = {}
            if agent_instance:
                state = agent_instance.get_state()
            else:
                state = {"running": False}

            # Gather new log entries since last push
            new_logs = log_buffer.get_new_since(log_cursor)
            log_cursor += len(new_logs)

            payload = {
                **state,
                "terminal_logs": new_logs,
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass
