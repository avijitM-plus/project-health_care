"""
V2 Conversational Triage - Integration Test Suite
Tests: symptom accumulation, qualifier capture, urgency escalation,
       session management, predictor fallback flag.
"""
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import requests
import json
import time
import sys

BASE = "http://127.0.0.1:8000"
SESSION_ID = f"v2-test-{int(time.time())}"
PASS = 0
FAIL = 0


def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")


def chat(msg, sid=None):
    r = requests.post(f"{BASE}/chat", json={
        "message": msg,
        "conversation_id": sid or SESSION_ID,
    })
    return r.json()


print("=" * 60)
print("IASIS AI v2.0 -- Integration Test Suite")
print("=" * 60)

# ---- Test 1: Health + Root ----
print("\n[1] Health & Root")
r = requests.get(f"{BASE}/health")
test("GET /health -> 200", r.status_code == 200)
test("status == ok", r.json().get("status") == "ok")

r = requests.get(f"{BASE}/")
test("GET / -> v2.0 message", "v2.0" in r.json().get("message", ""))

# ---- Test 2: Session stats (empty) ----
print("\n[2] Session stats (before any chat)")
r = requests.get(f"{BASE}/session/stats")
data = r.json()
test("GET /session/stats -> 200", r.status_code == 200)
test("max_sessions = 500", data.get("max_sessions") == 500)
test("ttl_seconds = 1800", data.get("ttl_seconds") == 1800)

# ---- Test 3: Turn 1 -- "I have fever and cough" ----
print("\n[3] Turn 1: 'I have fever and cough'")
resp1 = chat("I have fever and cough")
test("turn_number == 1", resp1.get("turn_number") == 1)
test("predictor_available == true", resp1.get("predictor_available") is True)
acc1 = resp1.get("accumulated_symptoms", [])
test("accumulated has 'fever'", any("fever" in s for s in acc1), f"got {acc1}")
test("accumulated has 'cough'", any("cough" in s for s in acc1), f"got {acc1}")
test("urgency is MEDIUM", resp1.get("urgency") == "MEDIUM", f"got {resp1.get('urgency')}")
test("reply is non-empty", len(resp1.get("reply", "")) > 20)
test("disclaimer present", "medical diagnosis" in resp1.get("disclaimer", "").lower())
print(f"  -> accumulated_symptoms: {acc1}")

# ---- Test 4: Turn 2 -- "The cough is dry" (qualifier capture) ----
print("\n[4] Turn 2: 'The cough is dry' (qualifier merge)")
resp2 = chat("The cough is dry")
test("turn_number == 2", resp2.get("turn_number") == 2)
acc2 = resp2.get("accumulated_symptoms", [])
test("accumulated has 'fever'", any("fever" in s for s in acc2), f"got {acc2}")
has_dry_cough = any("dry" in s and "cough" in s for s in acc2)
test("'dry cough' merged into accumulated", has_dry_cough, f"got {acc2}")
test("still 2 symptoms (no dup)", len(acc2) == 2, f"got {len(acc2)}: {acc2}")
print(f"  -> accumulated_symptoms: {acc2}")

# ---- Test 5: Turn 3 -- "It's been 3 days and I have headache" (accumulation) ----
print("\n[5] Turn 3: 'It has been 3 days and I have headache'")
resp3 = chat("It has been 3 days and I have headache")
test("turn_number == 3", resp3.get("turn_number") == 3)
acc3 = resp3.get("accumulated_symptoms", [])
test("accumulated has 'headache'", any("headache" in s for s in acc3), f"got {acc3}")
test("still has 'fever'", any("fever" in s for s in acc3), f"got {acc3}")
test("3 total symptoms", len(acc3) == 3, f"got {len(acc3)}: {acc3}")
print(f"  -> accumulated_symptoms: {acc3}")

# ---- Test 6: Urgency escalation (never downgrades) ----
print("\n[6] Urgency escalation")
test("Turn 1 urgency was MEDIUM", resp1.get("urgency") == "MEDIUM")
test("Turn 2 urgency stayed MEDIUM (not downgraded)", resp2.get("urgency") == "MEDIUM")
test("Turn 3 urgency >= MEDIUM", resp3.get("urgency") in ("MEDIUM", "HIGH", "EMERGENCY"))

# ---- Test 7: Session stats (active) ----
print("\n[7] Session stats (after chat)")
r = requests.get(f"{BASE}/session/stats")
data = r.json()
test("active_sessions >= 1", data.get("active_sessions", 0) >= 1)
print(f"  -> active_sessions: {data.get('active_sessions')}")

# ---- Test 8: Delete session ----
print("\n[8] Session deletion")
r = requests.delete(f"{BASE}/session/{SESSION_ID}")
test("DELETE /session -> 200", r.status_code == 200)
test("response has 'deleted'", "deleted" in r.json().get("detail", "").lower())

r = requests.delete(f"{BASE}/session/nonexistent-session-xyz")
test("DELETE unknown -> 404", r.status_code == 404)

# ---- Test 9: Session stats (after delete) ----
print("\n[9] Session stats (after delete)")
r = requests.get(f"{BASE}/session/stats")
data = r.json()
print(f"  -> active_sessions: {data.get('active_sessions')}")

# ---- Summary ----
print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print("=" * 60)

sys.exit(0 if FAIL == 0 else 1)
