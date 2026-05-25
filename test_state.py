import requests
import json
import time

URL = "http://127.0.0.1:8000/chat"

def test():
    print("--- TURN 1 ---")
    payload1 = {
        "message": "I feel fever",
        "conversation_id": "test_session_slot_v1"
    }
    r1 = requests.post(URL, json=payload1)
    if r1.status_code != 200:
        print("Error:", r1.text)
        return
    print(json.dumps(r1.json(), indent=2))

    time.sleep(2)

    print("\n--- TURN 2 ---")
    payload2 = {
        "message": "I am a male and I have 100F fever",
        "conversation_id": "test_session_slot_v1",
        "gender": "male"
    }
    r2 = requests.post(URL, json=payload2)
    print(json.dumps(r2.json(), indent=2))

if __name__ == "__main__":
    test()
