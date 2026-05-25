import asyncio
import httpx
import uuid

async def test_v3_state_machine():
    session_id = str(uuid.uuid4())
    url = "http://127.0.0.1:8000/chat"
    
    print(f"--- Starting V3 State Machine Test ---")
    print(f"Session ID: {session_id}")
    
    # Turn 1
    req1 = {
        "message": "I have a cough and fever",
        "conversation_id": session_id
    }
    
    async with httpx.AsyncClient() as client:
        print("\n--- TURN 1 ---")
        print(f"User: {req1['message']}")
        resp1 = await client.post(url, json=req1, timeout=60.0)
        data1 = resp1.json()
        print(f"AI: {data1.get('reply', '')}")
        print(f"Follow-ups Generated: {data1.get('followup_questions', [])}")
        print(f"Resolved Questions: {data1.get('resolved_questions', [])}")
        
        # Turn 2 - User answers one of the follow-ups (e.g. about dry cough)
        req2 = {
            "message": "The cough is dry and I've had the fever for 3 days.",
            "conversation_id": session_id
        }
        
        print("\n--- TURN 2 ---")
        print(f"User: {req2['message']}")
        resp2 = await client.post(url, json=req2, timeout=60.0)
        data2 = resp2.json()
        print(f"AI: {data2.get('reply', '')}")
        print(f"Follow-ups Generated: {data2.get('followup_questions', [])}")
        print(f"Resolved Questions: {data2.get('resolved_questions', [])}")
        
        # Verify deduplication
        f_1 = set(data1.get('followup_questions', []))
        f_2 = set(data2.get('followup_questions', []))
        overlap = f_1.intersection(f_2)
        print(f"Overlap in follow-ups: {overlap}")
        
        if not overlap:
            print("SUCCESS: V3 State machine is deduplicating follow-ups!")
        else:
            print("WARNING: Follow-ups repeated.")

if __name__ == "__main__":
    asyncio.run(test_v3_state_machine())
