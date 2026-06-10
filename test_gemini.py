"""
Quick test for GeminiBanglaService.
Run: venv/Scripts/python test_gemini.py
"""
import sys
import os

# Force UTF-8 output on Windows so Bangla text prints correctly
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(override=True)

key = os.getenv("GEMINI_API_KEY", "")
model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
print(f"GEMINI_API_KEY: {key[:20]}...")
print(f"GEMINI_MODEL:   {model}")

# ── 1. Direct SDK connectivity test ──────────────────────────────────────────
print("\n--- Direct google.genai SDK test ---")
try:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=model,
        contents='Reply with valid JSON: {"status": "ok", "message": "hello"}',
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=64,
        ),
    )
    print("SUCCESS:", response.text)
except Exception as e:
    print(f"ERROR ({type(e).__name__}): {e}")

# ── 2. GeminiBanglaService integration test ───────────────────────────────────
print("\n--- GeminiBanglaService test ---")
from app.services.gemini_bangla_service import gemini_bangla_service
print(f"initialized: {gemini_bangla_service._initialized}")
print(f"init_error:  {gemini_bangla_service._init_error}")

if not gemini_bangla_service._initialized:
    print("Service not initialized — check API key and package installation")
    sys.exit(1)

dummy_output = {
    "reply": "আপনার জ্বর এবং মাথাব্যথার কারণ টাইফয়েড হতে পারে। দয়া করে পরীক্ষা করুন।",
    "followup_questions": ["কতদিন ধরে জ্বর হচ্ছে আপনার?"],
    "suggested_replies": ["৩ দিন ধরে", "এক সপ্তাহ ধরে", "দুই সপ্তাহেরও বেশি"],
    "possible_diseases": [
        {"name": "Typhoid", "concern_level": "উচ্চ ঝুঁকি"},
        {"name": "Malaria", "concern_level": "মাঝারি ঝুঁকি"},
    ],
    "urgency": "MEDIUM",
    "recommended_tests": [
        {"test_name": "Complete Blood Count"},
        {"test_name": "Malaria RDT"},
    ],
}

class FakeState:
    chief_complaint = "জ্বর এবং মাথাব্যথা"
    reports = []

result = gemini_bangla_service.enhance_bangla_response(
    llm_output=dummy_output,
    state=FakeState(),
    urgency="MEDIUM",
    session_id="test-001",
)

print("\nreply:")
print(result.get("reply"))
print("\nfollowup_questions:")
print(result.get("followup_questions"))
print("\nsuggested_replies:")
print(result.get("suggested_replies"))

was_enhanced = result.get("reply") != dummy_output["reply"]
print(f"\nEnhanced by Gemini: {was_enhanced}")
print(f"Cache stats: {gemini_bangla_service.get_cache_stats()}")
