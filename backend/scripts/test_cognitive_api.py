
import json
import uuid
import urllib.request
import urllib.error
from datetime import datetime, timedelta

def test_cognitive_api():
    url = "http://localhost:8000/api/cognitive/analyze"
    
    # Construct a valid payload
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    start_time = datetime.utcnow()
    end_time = start_time + timedelta(seconds=60)
    
    payload = {
        "session_id": session_id,
        "patient_id": None,  # Will trigger fallback user logic
        "user_metadata": {
            "userAgent": "TestScript/1.0"
        },
        "tasks": [
            {
                "task_id": "reaction_time_v1",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "events": [
                    {
                        "timestamp": 1000,
                        "event_type": "test_start",
                        "payload": {}
                    },
                    {
                        "timestamp": 2000,
                        "event_type": "stimulus_shown",
                        "payload": {"trial": 1}
                    },
                    {
                        "timestamp": 2250,
                        "event_type": "response_received",
                        "payload": {"trial": 1, "rt": 250}
                    },
                    {
                         "timestamp": 3000,
                         "event_type": "test_end",
                         "payload": {}
                    }
                ],
                "metadata": {}
            }
        ]
    }
    
    print(f"Sending request to {url}...")
    
    req = urllib.request.Request(
        url, 
        data=json.dumps(payload).encode('utf-8'), 
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            status = response.status
            print(f"Status: {status}")
            
            data = json.loads(response.read().decode('utf-8'))
            print("Success!")
            print(f"Risk Assessment: {data.get('risk_assessment')}")
            print(f"Status: {data.get('status')}")
            
            if data.get('status') == 'success':
                print("✅ End-to-end API test passed")
            else:
                 print("⚠️ API returned success but payload status was not 'success'")

    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")
    except urllib.error.URLError as e:
        print(f"Connection failed: {e.reason}")
        print("Ensure the backend server is running on localhost:8000")

if __name__ == "__main__":
    test_cognitive_api()
