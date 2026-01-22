import asyncio
import aiohttp
import json
import uuid

API_URL = "http://localhost:8000/api/patients/"

async def verify_patient_api():
    print(f"🚀 Starting Patient API Verification...")
    
    # Randomize to ensure unique phone
    unique_suffix = uuid.uuid4().hex[:6]
    payload = {
        "full_name": f"API Test Patient {unique_suffix}",
        "phone_number": f"555-{unique_suffix}",
        "date_of_birth": "1990-01-01",
        "gender": "Female",
        "medical_notes": "Created via API verification script"
    }
    
    print(f"📤 Sending payload: {json.dumps(payload, indent=2)}")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(API_URL, json=payload) as resp:
                print(f"📥 Status Code: {resp.status}")
                if resp.status in [200, 201]:
                    data = await resp.json()
                    print(f"✅ Patient Created Successfully!")
                    print(f"   ID: {data['id']}")
                    print(f"   Name: {data['full_name']}")
                    print(f"   DOB: {data.get('date_of_birth')}")
                else:
                    text = await resp.text()
                    print(f"❌ Failed to create patient.")
                    print(f"   Response: {text}")
        except Exception as e:
            print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    asyncio.run(verify_patient_api())
