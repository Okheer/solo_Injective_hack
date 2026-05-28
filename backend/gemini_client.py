
import os
import json
import logging
import time
import threading
import urllib.request
import urllib.error

logger = logging.getLogger("gemini")

# ── Rate Limiter (5 RPM on free tier → at most 1 call per 12s) ──────────
_rate_lock = threading.Lock()
_last_call_time = 0.0
_MIN_INTERVAL = 13.0  # seconds between calls (conservative for 5 RPM)

# ── Model config ─────────────────────────────────────────────────────────
# Free tier model — matches what's shown in Google AI Studio
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def _wait_for_rate_limit():
    """Block until the rate limit window allows a new request."""
    global _last_call_time
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_call_time
        if elapsed < _MIN_INTERVAL:
            wait = _MIN_INTERVAL - elapsed
            logger.info(f"⏳ Rate limit: waiting {wait:.1f}s before next Gemini call...")
            time.sleep(wait)
        _last_call_time = time.time()


def call_gemini(prompt: str, max_tokens: int = 400, retries: int = 2) -> str:
    """
    Call Google Gemini API via direct REST, bypassing the SDK
    to avoid protobuf descriptor conflicts with pyinjective.

    Features:
    - Uses gemini-2.5-flash (free tier compatible)
    - Built-in rate limiting (5 RPM)
    - Exponential backoff on 429 errors
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    url = f"{GEMINI_API_BASE}/{GEMINI_MODEL}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.7,
        }
    }

    data = json.dumps(payload).encode("utf-8")

    last_error = None
    for attempt in range(retries + 1):
        _wait_for_rate_limit()

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                candidates = body.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    finish_reason = candidates[0].get("finishReason", "")
                    if parts and parts[0].get("text"):
                        text = parts[0]["text"]
                        logger.info(f"✅ Gemini REST call succeeded ({len(text)} chars) [model={GEMINI_MODEL}]")
                        return text
                    # Model returned but with no text (e.g. MAX_TOKENS hit before generating)
                    if finish_reason == "MAX_TOKENS":
                        logger.warning(f"⚠️ Gemini returned MAX_TOKENS with no text — retrying with more tokens")
                        # Retry with doubled token budget
                        payload["generationConfig"]["maxOutputTokens"] = max_tokens * 3
                        data = json.dumps(payload).encode("utf-8")
                        continue
                raise ValueError(f"Empty Gemini response (finishReason={candidates[0].get('finishReason', '?') if candidates else 'no_candidates'})")

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            last_error = f"Gemini API error {e.code}: {error_body[:200]}"

            if e.code == 429 and attempt < retries:
                # Exponential backoff: 30s, 60s
                backoff = 30 * (2 ** attempt)
                logger.warning(f"⚠️ Gemini 429 rate limit hit (attempt {attempt+1}/{retries+1}). "
                               f"Backing off {backoff}s...")
                time.sleep(backoff)
                continue
            else:
                logger.error(f"Gemini HTTP error {e.code}: {error_body[:300]}")
                raise ValueError(last_error)

        except Exception as e:
            last_error = str(e)
            if attempt < retries:
                backoff = 15 * (2 ** attempt)
                logger.warning(f"⚠️ Gemini call failed (attempt {attempt+1}): {e}. Retrying in {backoff}s...")
                time.sleep(backoff)
                continue
            raise

    raise ValueError(last_error or "Gemini call failed after all retries")
