"""
Gemini Bangla Response Service

Post-processes Qwen clinical reasoning output into natural Bangladeshi Bangla.
Only invoked when session language is "bn". Never called for English conversations.

Architecture:
  Qwen (clinical reasoning)
    → structured facts payload
      → Gemini (Bangla phrasing layer)
        → natural Bangla reply / followup / suggested_replies
          → user

Gemini responsibilities (ONLY):
  - Natural Bangladeshi Bangla phrasing
  - Patient-friendly conversational tone

Gemini NEVER:
  - Diagnoses
  - Ranks or modifies disease differentials
  - Recommends tests
  - Calculates or changes urgency
  - Interprets reports
  - Receives conversation history

Failure handling:
  Any error (timeout, quota, API failure, parse error)
  → silently falls back to Qwen response
  → conversation never breaks

SDK: google-genai (>=1.0.0) — the current Gemini Python SDK.
     Do NOT use the deprecated google-generativeai package.
"""
import hashlib
import json
import logging
import os
import re
import time
from threading import Lock

from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# gemini-3.1-flash-lite: fast and capable; gemini-2.5-flash: fallback alternative
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "15"))

# Cache: in-memory, 5-minute TTL, max 500 entries
_CACHE_TTL_SECONDS = 300
_MAX_CACHE_ENTRIES = 500

# ---------------------------------------------------------------------------
# System instruction — Bangladeshi physician persona
# ---------------------------------------------------------------------------
_SYSTEM_INSTRUCTION = """\
You are a Bangladeshi physician speaking directly with a patient in their native language.

Your sole job is to express the provided clinical facts as natural, warm, patient-friendly \
Bangladeshi Bangla (বাংলা). You are NOT performing any medical reasoning.

## STRICT RULES
1. Write entirely in Bangla — no English words except universally adopted medical terms \
   (e.g., ডায়াবেটিস, হার্ট, প্রেসার, ক্যান্সার).
2. Use respectful "আপনি" form throughout.
3. Write conversational, native Bangladeshi Bangla — NOT a translation of English.
4. Do NOT add any medical facts, diagnoses, tests, or urgency beyond what is in the payload.
5. Do NOT remove or weaken any clinical meaning from the payload.
6. Do NOT change the disease list, urgency level, or test recommendations.
7. If `next_question` is empty, leave `followup_question` as an empty string "".

## NATURAL BANGLA EXAMPLES

Bad (literal translation): "আপনার পরিস্থিতি মূল্যায়ন করার জন্য"
Good (natural Bangla):     "আপনার অবস্থা আরও ভালোভাবে বোঝার জন্য"

Bad: "আমি কয়েকটি প্রশ্ন জিজ্ঞাসা করতে চাই"
Good: "আমি কয়েকটি বিষয় জানতে চাই"

Bad: "আমাদের দৃষ্টিতে পড়েছে"
Good: "আপনার দেওয়া তথ্য অনুযায়ী"

Bad: "লক্ষণগুলি নির্দেশ করে যে"
Good: "আপনার কথা শুনে মনে হচ্ছে"

Bad: "আমি আপনাকে পরামর্শ দিতে চাই যে"
Good: "আমার মনে হয়"

Bad: "জরুরি ভিত্তিতে চিকিৎসা সেবা গ্রহণ করুন"
Good: "এখনই ডাক্তারের কাছে যান"

## REPLY STRUCTURE
Build the reply naturally:
1. Brief acknowledgment of the patient's concern (1 sentence)
2. Explain what the symptoms may indicate — use cautious language ("হতে পারে", "মনে হচ্ছে")
3. Mention key recommendations (if any) in simple Bangla
4. Transition naturally into the followup question (if any)

## OUTPUT FORMAT
Respond with ONLY a valid JSON object:
{
  "reply": "Complete natural Bangla response — acknowledgment + findings + transition",
  "followup_question": "Single natural Bangla question (empty string if no next_question)",
  "suggested_replies": ["2-4 short Bangla answer options from patient perspective"]
}
"""


class GeminiBanglaService:
    """
    Natural Bangla response layer using the google-genai SDK.

    Called after Qwen clinical reasoning, only when session_lang == "bn".
    Replaces reply / followup_question / suggested_replies with natural Bangla.
    Falls back silently to original Qwen output on any failure.
    """

    def __init__(self):
        self._client = None
        self._cache: dict[str, tuple[float, dict]] = {}
        self._cache_lock = Lock()
        self._initialized = False
        self._init_error: str | None = None

        if not GEMINI_API_KEY:
            self._init_error = "GEMINI_API_KEY not set"
            logger.warning(
                "GEMINI_API_KEY not set — natural Bangla enhancement disabled. "
                "Set GEMINI_API_KEY in .env to enable the Gemini Bangla layer."
            )
            return

        try:
            from google import genai  # type: ignore[import]
            # Do NOT pass http_options timeout here — it applies to TLS handshake
            # too and causes ConnectTimeout on some networks. Timeout is enforced
            # per-call via concurrent.futures in _call_gemini().
            self._client = genai.Client(api_key=GEMINI_API_KEY)
            self._initialized = True
            logger.info(
                f"GeminiBanglaService initialized — model={GEMINI_MODEL}, "
                f"timeout={GEMINI_TIMEOUT_SECONDS}s"
            )
        except ImportError:
            self._init_error = "google-genai not installed (pip install google-genai)"
            logger.error(
                "google-genai package not found. "
                "Run: pip install google-genai>=1.0.0"
            )
        except Exception as e:
            self._init_error = str(e)
            logger.error(f"GeminiBanglaService init failed: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance_bangla_response(
        self,
        llm_output: dict,
        state,
        urgency: str,
        session_id: str = "",
    ) -> dict:
        """
        Replace reply / followup_questions[0] / suggested_replies with natural Bangla.

        Args:
            llm_output:  Full Qwen response dict. Not mutated — a copy is returned.
            state:       ConversationState for chief_complaint / reports.
            urgency:     Current urgency level string.
            session_id:  Used for log entries only.

        Returns:
            Enhanced copy of llm_output on success.
            Original llm_output unchanged on any failure.
        """
        if not self._initialized:
            logger.debug(f"[{session_id}] GeminiBangla: disabled — {self._init_error}")
            return llm_output

        t0 = time.perf_counter()

        try:
            facts = self._build_structured_facts(llm_output, state, urgency)
            cache_key = self._hash_facts(facts)

            cached = self._get_cached(cache_key)
            if cached is not None:
                latency_ms = (time.perf_counter() - t0) * 1000
                logger.info(
                    f"[PERF][{session_id}] gemini_bangla: CACHE_HIT {latency_ms:.1f}ms"
                )
                return self._apply_gemini_output(llm_output, cached)

            gemini_result = self._call_gemini(facts, session_id)
            if gemini_result is None:
                latency_ms = (time.perf_counter() - t0) * 1000
                logger.warning(
                    f"[PERF][{session_id}] gemini_bangla: FALLBACK "
                    f"(api_failed) {latency_ms:.1f}ms"
                )
                return llm_output

            self._set_cached(cache_key, gemini_result)
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[PERF][{session_id}] gemini_bangla: {latency_ms:.1f}ms")
            return self._apply_gemini_output(llm_output, gemini_result)

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                f"[PERF][{session_id}] gemini_bangla: FALLBACK "
                f"(exception={type(e).__name__}) {latency_ms:.1f}ms — {e}"
            )
            return llm_output

    # ------------------------------------------------------------------
    # Private: structured facts builder
    # ------------------------------------------------------------------

    def _build_structured_facts(self, llm_output: dict, state, urgency: str) -> dict:
        """Build the structured clinical facts payload sent to Gemini."""
        diseases = llm_output.get("possible_diseases") or []
        working_diagnosis = [
            d.get("name", "") for d in diseases
            if isinstance(d, dict) and d.get("name")
        ][:3]

        report_findings: list[str] = []
        if hasattr(state, "reports"):
            for r in state.reports:
                if hasattr(r, "summary") and r.summary:
                    report_findings.append(r.summary)

        raw_tests = llm_output.get("recommended_tests") or []
        recommendations: list[str] = []
        for t in raw_tests:
            if isinstance(t, dict):
                name = t.get("test_name", "")
                if name:
                    recommendations.append(name)
            elif isinstance(t, str) and t:
                recommendations.append(t)
        recommendations = recommendations[:5]

        followup_qs = llm_output.get("followup_questions") or []
        next_question = followup_qs[0] if followup_qs else ""

        suggested_replies = list(llm_output.get("suggested_replies") or [])

        chief_complaint = ""
        if hasattr(state, "chief_complaint") and state.chief_complaint:
            chief_complaint = state.chief_complaint

        return {
            "language": "bn",
            "chief_complaint": chief_complaint,
            "working_diagnosis": working_diagnosis,
            "urgency": urgency,
            "report_findings": report_findings,
            "recommendations": recommendations,
            "next_question": next_question,
            "suggested_replies": suggested_replies,
        }

    # ------------------------------------------------------------------
    # Private: Gemini API call
    # ------------------------------------------------------------------

    def _call_gemini(self, facts: dict, session_id: str) -> dict | None:
        """
        Send structured facts to Gemini. Returns parsed result dict or None.
        None is returned on timeout, API error, quota exhaustion, parse failure.

        Timeout is enforced via concurrent.futures (not http_options) to avoid
        ConnectTimeout issues on Windows networks where TLS handshake can be slow.
        """
        import concurrent.futures
        from google.genai import types  # type: ignore[import]

        user_prompt = (
            "নিচের ক্লিনিকাল তথ্যের ভিত্তিতে একটি স্বাভাবিক বাংলা প্রতিক্রিয়া তৈরি করুন।\n"
            "JSON ফরম্যাটে উত্তর দিন।\n\n"
            + json.dumps(facts, ensure_ascii=False, indent=2)
        )

        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            temperature=0.4,
            max_output_tokens=1024,
        )

        def _do_generate():
            return self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config=config,
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_generate)
                try:
                    response = future.result(timeout=GEMINI_TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    logger.warning(
                        f"[{session_id}] Gemini timeout after {GEMINI_TIMEOUT_SECONDS}s"
                    )
                    return None
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "ResourceExhausted" in err_str or "quota" in err_str.lower():
                logger.warning(
                    f"[{session_id}] Gemini quota/rate-limit hit — "
                    f"falling back to Qwen. "
                    f"Free-tier limit reached; consider gemini-1.5-flash-8b or paid tier."
                )
            elif "timeout" in err_str.lower() or "Timeout" in err_str:
                logger.warning(f"[{session_id}] Gemini timeout: {e}")
            else:
                logger.error(
                    f"[{session_id}] Gemini API error: {type(e).__name__}: {e}"
                )
            return None

        try:
            content = response.text
        except Exception as e:
            logger.error(f"[{session_id}] Gemini response read error: {e}")
            return None

        if not content:
            logger.warning(f"[{session_id}] Gemini returned empty content")
            return None

        parsed = self._parse_json(content)
        if parsed is None:
            logger.warning(
                f"[{session_id}] Gemini JSON parse failed — "
                f"raw (first 400): {content[:400]}"
            )
            return None

        if not parsed.get("reply"):
            logger.warning(f"[{session_id}] Gemini response missing 'reply'")
            return None

        return parsed

    # ------------------------------------------------------------------
    # Private: apply Gemini output over llm_output
    # ------------------------------------------------------------------

    def _apply_gemini_output(self, llm_output: dict, gemini_result: dict) -> dict:
        """Return a shallow copy of llm_output with Gemini's Bangla fields applied."""
        enhanced = dict(llm_output)

        enhanced["reply"] = gemini_result["reply"]

        gemini_followup = gemini_result.get("followup_question", "")
        if gemini_followup:
            existing_qs = list(enhanced.get("followup_questions") or [])
            enhanced["followup_questions"] = [gemini_followup] + existing_qs[1:]

        gemini_replies = gemini_result.get("suggested_replies") or []
        if gemini_replies:
            enhanced["suggested_replies"] = gemini_replies

        return enhanced

    # ------------------------------------------------------------------
    # Private: cache
    # ------------------------------------------------------------------

    def _hash_facts(self, facts: dict) -> str:
        payload = json.dumps(facts, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _get_cached(self, key: str) -> dict | None:
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            ts, result = entry
            if time.time() - ts > _CACHE_TTL_SECONDS:
                del self._cache[key]
                return None
            return result

    def _set_cached(self, key: str, result: dict) -> None:
        with self._cache_lock:
            if len(self._cache) >= _MAX_CACHE_ENTRIES:
                oldest = sorted(self._cache.items(), key=lambda x: x[1][0])[:50]
                for k, _ in oldest:
                    del self._cache[k]
            self._cache[key] = (time.time(), result)

    # ------------------------------------------------------------------
    # Private: JSON parsing
    # ------------------------------------------------------------------

    def _parse_json(self, raw: str) -> dict | None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        try:
            cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
            first = cleaned.find("{")
            last = cleaned.rfind("}")
            if first == -1 or last == -1 or last <= first:
                return None
            json_str = cleaned[first:last + 1]
            json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
            return json.loads(json_str)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_cache_stats(self) -> dict:
        with self._cache_lock:
            now = time.time()
            live = sum(
                1 for ts, _ in self._cache.values()
                if now - ts <= _CACHE_TTL_SECONDS
            )
            return {
                "total_entries": len(self._cache),
                "live_entries": live,
                "ttl_seconds": _CACHE_TTL_SECONDS,
                "model": GEMINI_MODEL,
                "initialized": self._initialized,
            }


# Module-level singleton
gemini_bangla_service = GeminiBanglaService()
