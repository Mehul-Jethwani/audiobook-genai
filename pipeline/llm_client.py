"""
llm_client.py (v2)

Provides unified LLM invocation with automatic fallback across provider keys:
- Groq (openai/gpt-oss-120b / qwen/qwen3.8-27b)
- Gemini (gemini-3.6-flash / gemini-3.1-flash-lite-preview)
- Cerebras (gpt-oss-120b)

Features:
- Distinguishes temporary rate limits (429) with cooldown from permanent key errors.
- Automatic fallback across keys and providers.
- JSON response stripping and extraction.
"""

import os
import json
import time
from dotenv import load_dotenv
from google import genai
from groq import Groq
from cerebras.cloud.sdk import Cerebras

from config import RATE_LIMIT_COOLDOWN_SECONDS, MAX_WAIT_ROUNDS

# Load .env from finish dir and parent dir
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

GEMINI_MODEL = "gemini-3.6-flash"
GROQ_MODEL = "openai/gpt-oss-120b"
CEREBRAS_MODEL = "gpt-oss-120b"


def _load_keys(env_var_multi, env_var_single):
    multi = os.getenv(env_var_multi)
    if multi:
        return [k.strip() for k in multi.split(",") if k.strip()]
    single = os.getenv(env_var_single)
    return [single.strip()] if single and single.strip() else []


gemini_keys = _load_keys("GEMINI_API_KEYS", "GEMINI_API_KEY")
groq_keys = _load_keys("GROQ_API_KEYS", "GROQ_API_KEY")
cerebras_keys = _load_keys("CEREBRAS_API_KEYS", "CEREBRAS_API_KEY")

PROVIDER_CHAIN = []

for idx, key in enumerate(groq_keys, start=1):
    client = Groq(api_key=key)

    def make_groq_call(c=client):
        def call(prompt):
            response = c.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        return call

    PROVIDER_CHAIN.append((f"Groq-{idx}", make_groq_call()))

for idx, key in enumerate(cerebras_keys, start=1):
    client = Cerebras(api_key=key)

    def make_cerebras_call(c=client):
        def call(prompt):
            response = c.chat.completions.create(
                model=CEREBRAS_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        return call

    PROVIDER_CHAIN.append((f"Cerebras-{idx}", make_cerebras_call()))

for idx, key in enumerate(gemini_keys, start=1):
    client = genai.Client(api_key=key)

    def make_gemini_call(c=client):
        def call(prompt):
            response = c.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return response.text
        return call

    PROVIDER_CHAIN.append((f"Gemini-{idx}", make_gemini_call()))


# name -> {"dead": bool, "cooldown_until": float (epoch seconds)}
_provider_state = {}


def _get_state(name):
    return _provider_state.setdefault(name, {"dead": False, "cooldown_until": 0.0})


def _is_available(name):
    state = _get_state(name)
    return (not state["dead"]) and state["cooldown_until"] <= time.time()


def _classify_error(error_text):
    text = error_text.lower()
    if "429" in text or "rate_limit" in text or "rate limit" in text or "too many requests" in text:
        return "rate_limit"
    if "quota" in text or "resource_exhausted" in text or "insufficient_quota" in text or "payment_required" in text:
        return "quota"
    return "other"


def _shorten(error_text):
    quota_id_start = error_text.find("quotaId")
    if quota_id_start != -1:
        return error_text[quota_id_start:quota_id_start + 80]
    return error_text[:150]


def call_llm_with_fallback(prompt):
    """
    Tries each available provider/key in PROVIDER_CHAIN order. Rate-limited
    providers cool down and come back automatically; only genuinely dead
    providers (bad key, daily quota gone) are skipped for the rest of the
    run. If EVERY provider is cooling (none dead), waits for the soonest
    one to clear and retries, up to MAX_WAIT_ROUNDS times.
    """
    last_error = None

    for round_num in range(MAX_WAIT_ROUNDS + 1):
        tried_any = False

        for name, call_fn in PROVIDER_CHAIN:
            if not _is_available(name):
                continue
            tried_any = True

            try:
                return call_fn(prompt)
            except Exception as e:
                error_text = str(e)
                kind = _classify_error(error_text)
                detail = _shorten(error_text)
                state = _get_state(name)

                if kind == "rate_limit":
                    state["cooldown_until"] = time.time() + RATE_LIMIT_COOLDOWN_SECONDS
                    print(f"  ({name} rate-limited - {detail} - cooling down "
                          f"{RATE_LIMIT_COOLDOWN_SECONDS}s, will retry, NOT marked dead)")
                elif kind == "quota":
                    state["dead"] = True
                    print(f"  ({name} quota/billing exhausted - {detail} - dead for rest of run)")
                else:
                    state["dead"] = True
                    print(f"  ({name} failed - {detail} - dead for rest of run)")

                last_error = e
                continue

        # If any are ONLY cooling (not permanently dead), wait for soonest
        cooling = [
            s["cooldown_until"] for s in _provider_state.values()
            if not s["dead"] and s["cooldown_until"] > time.time()
        ]
        if not cooling or round_num >= MAX_WAIT_ROUNDS:
            break

        wait_time = max(1.0, min(cooling) - time.time())
        print(f"  All providers busy/cooling - waiting {wait_time:.0f}s "
              f"(retry round {round_num + 1}/{MAX_WAIT_ROUNDS})...")
        time.sleep(wait_time)

    raise RuntimeError(f"All providers/keys exhausted. Last error: {last_error}")


def provider_status():
    """Debug helper: see which providers are dead/cooling/available right now."""
    now = time.time()
    status = {}
    for name, _ in PROVIDER_CHAIN:
        state = _get_state(name)
        if state["dead"]:
            status[name] = "dead"
        elif state["cooldown_until"] > now:
            status[name] = f"cooling ({state['cooldown_until'] - now:.0f}s left)"
        else:
            status[name] = "available"
    return status


def strip_json_fence(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return raw


def call_llm_json(prompt, fallback_value):
    raw = call_llm_with_fallback(prompt)
    raw = strip_json_fence(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback_value


if __name__ == "__main__":
    print("Configured providers:", [name for name, _ in PROVIDER_CHAIN])
    print("Sending one test prompt...")
    result = call_llm_with_fallback("Reply with exactly the word: pong")
    print("Result:", result)
    print("Provider status:", provider_status())