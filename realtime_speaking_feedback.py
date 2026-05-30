"""
Per-turn speaking evaluation + emoji rewards published to clients via LiveKit data channel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Optional

from livekit import rtc
from openai import AsyncOpenAI

logger = logging.getLogger("zabano.agent.feedback")

EVENT_SPEAKING_TURN_EVAL = "speaking_turn_eval"
EVENT_EMOJI_REWARD = "emoji_reward"

REALTIME_FEEDBACK_MODEL = os.getenv("REALTIME_FEEDBACK_MODEL", "gpt-4o-mini")
REALTIME_FEEDBACK_TIMEOUT_S = float(os.getenv("REALTIME_FEEDBACK_TIMEOUT_S", "8"))
MIN_WORDS_FOR_EVAL = int(os.getenv("REALTIME_FEEDBACK_MIN_WORDS", "3"))

REALTIME_TURN_EVAL_SYSTEM = """You evaluate ONE learner utterance in a live speaking session.
Output ONE JSON object only (no markdown).

Fields (all required unless noted):
- score: integer 0-100 for this utterance only
- pass: boolean — true if understandable and reasonably correct for level
- feedback: ONE short encouraging sentence in the learner's native language
- grammar_note: optional ONE short tip in native language, or empty string
- corrected_text: when pass is false OR there is a fixable mistake, the full corrected sentence in the speaking/target language; otherwise empty string
- reason: short snake_case tag e.g. good_pronunciation, good_fluency, needs_clarity, grammar_fix
- emojis: REQUIRED array of exactly 3-4 emoji characters that PRECISELY match what the learner talked about. Rules:
  * Country, city, or nationality → use that place's flag plus one related icon (Turkey/Türkiye/Turkish → 🇹🇷; France/French/Paris → 🇫🇷; Japan/Japanese/Tokyo → 🇯🇵; USA/America/American → 🇺🇸; Germany/German/Berlin → 🇩🇪; Italy/Italian/Rome → 🇮🇹; Spain/Spanish/Madrid → 🇪🇸; UK/Britain/England/London → 🇬🇧)
  * Video games / gaming / consoles → 🎮 🕹️ 👾 (not generic praise)
  * A specific food, sport, job, hobby, or object → emojis for THAT thing only
  * Do NOT mix unrelated topics. Do NOT use only 👍 👏 ✨ unless the sentence has no specific topic at all.
- emoji: same as emojis[0] (primary emoji — must match the main topic)
- intensity: integer 1 (subtle, 3 emojis), 2 (good), or 3 (exceptional answer — still max 4 topical emojis)

Be warm and brief. Do not repeat the full user sentence in feedback."""

_llm_semaphore: Optional[asyncio.Semaphore] = None
_openai_client: Optional[AsyncOpenAI] = None


def _get_semaphore(max_concurrent: int) -> asyncio.Semaphore:
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(max(1, max_concurrent))
    return _llm_semaphore


def _get_openai() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def should_evaluate_utterance(user_text: str) -> bool:
    t = (user_text or "").strip()
    if not t:
        return False
    return _word_count(t) >= MIN_WORDS_FOR_EVAL


def _clean_json_response(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object")
    return parsed


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _normalize_emojis(parsed: dict[str, Any]) -> list[str]:
    out: list[str] = []
    emojis_raw = parsed.get("emojis")
    if isinstance(emojis_raw, list):
        for e in emojis_raw:
            if isinstance(e, str) and e.strip():
                out.append(e.strip())
    single = parsed.get("emoji")
    if isinstance(single, str) and single.strip() and single.strip() not in out:
        out.insert(0, single.strip())
    if not out:
        out = ["👍"]
    return out[:5]


async def publish_client_event(room: rtc.Room, payload: dict[str, Any]) -> None:
    lp = getattr(room, "local_participant", None)
    if lp is None:
        logger.warning("publish_client_event_no_local_participant")
        return
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    publish = getattr(lp, "publish_data", None)
    if not callable(publish):
        logger.warning("publish_client_event_no_publish_data")
        return
    await publish(data, reliable=True)


async def _call_eval_llm(
    *,
    user_text: str,
    config: dict[str, Any],
    max_concurrent: int,
) -> dict[str, Any]:
    native = str(config.get("native_language") or "English")[:64]
    speaking = str(
        config.get("speaking_language")
        or config.get("learning_language")
        or config.get("target_language")
        or "English"
    )[:64]
    level = str(config.get("learner_level") or config.get("user_level") or "B1")[:12]

    user_prompt = (
        f"Native language for feedback: {native}\n"
        f"Speaking / target language: {speaking}\n"
        f"Learner level: {level}\n\n"
        f"Learner utterance to evaluate:\n{user_text[:2000]}"
    )

    async with _get_semaphore(max_concurrent):
        client = _get_openai()
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=REALTIME_FEEDBACK_MODEL,
                messages=[
                    {"role": "system", "content": REALTIME_TURN_EVAL_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=400,
            ),
            timeout=REALTIME_FEEDBACK_TIMEOUT_S,
        )
    content = (resp.choices[0].message.content or "").strip()
    return _clean_json_response(content)


async def evaluate_and_publish(
    room: rtc.Room,
    *,
    user_text: str,
    config: dict[str, Any],
    turn_id: str,
    max_concurrent: int = 1,
) -> None:
    if not should_evaluate_utterance(user_text):
        logger.info("realtime_feedback_skip_short turn_id=%s", turn_id)
        return

    try:
        parsed = await _call_eval_llm(
            user_text=user_text,
            config=config,
            max_concurrent=max_concurrent,
        )
    except Exception:
        logger.exception("realtime_feedback_llm_failed turn_id=%s", turn_id)
        return

    score = _clamp_int(parsed.get("score"), 0, 100, 50)
    passed = bool(parsed.get("pass"))
    feedback = str(parsed.get("feedback") or "").strip()[:500]
    grammar_note = str(parsed.get("grammar_note") or "").strip()[:300]
    corrected_text = str(parsed.get("corrected_text") or "").strip()[:2000]
    reason = str(parsed.get("reason") or "turn_eval").strip()[:64]
    emojis = _normalize_emojis(parsed)
    intensity = _clamp_int(parsed.get("intensity"), 1, 3, 2 if passed else 1)

    eval_payload = {
        "type": EVENT_SPEAKING_TURN_EVAL,
        "turn_id": turn_id,
        "user_text": user_text[:2000],
        "score": score,
        "pass": passed,
        "feedback": feedback,
        "grammar_note": grammar_note,
        "corrected_text": corrected_text,
        "reason": reason,
    }

    emoji_payload: dict[str, Any] = {
        "type": EVENT_EMOJI_REWARD,
        "turn_id": turn_id,
        "user_text": user_text[:500],
        "intensity": intensity if passed else max(1, min(intensity, 2)),
        "reason": reason,
        "emojis": emojis,
        "emoji": emojis[0] if emojis else "👍",
    }
    try:
        await publish_client_event(room, eval_payload)
        if passed:
            await publish_client_event(room, emoji_payload)
        logger.info(
            "realtime_feedback_published turn_id=%s score=%s pass=%s emojis=%s intensity=%s",
            turn_id,
            score,
            passed,
            emojis,
            intensity if passed else 0,
        )
    except Exception:
        logger.exception("realtime_feedback_publish_failed turn_id=%s", turn_id)
