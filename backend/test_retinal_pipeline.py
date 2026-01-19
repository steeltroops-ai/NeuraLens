"""Test the retinal pipeline endpoint."""
import requests
import numpy as np
from PIL import Image
import io

# Create a simple test image (512x512 fundus-like)
img = np.zeros((512, 512, 3), dtype=np.uint8)
# Add reddish fundus color
img[:,:,0] = 180  # Red
img[:,:,1] = 80   # Green
img[:,:,2] = 60   # Blue
# Add bright disc area
center_y, center_x = 256, 380
for y in range(512):
    for x in range(512):
        dist = ((x-center_x)**2 + (y-center_y)**2)**0.5
        if dist < 50:
            img[y,x] = [255, 220, 180]

# Save to bytes
pil_img = Image.fromarray(img)
buffer = io.BytesIO()
pil_img.save(buffer, format='JPEG')
image_bytes = buffer.getvalue()

print(f"Created test image: {len(image_bytes)} bytes")

# Send request
files = {'image': ('test_fundus.jpg', image_bytes, 'image/jpeg')}
data = {'session_id': 'test-123', 'patient_id': 'ANON'}


try:
    resp = requests.post('http://localhost:8001/api/retinal/analyze', files=files, data=data, timeout=120)

    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        result = resp.json()
        print(f"Success: {result.get('success', False)}")
        if result.get('diabetic_retinopathy'):
            dr = result['diabetic_retinopathy']
            print(f"DR Grade: {dr.get('grade', 'N/A')}")
        if result.get('risk_assessment'):
            risk = result['risk_assessment']
            print(f"Risk Score: {risk.get('score', 'N/A')}")
            print(f"Risk Level: {risk.get('category', 'N/A')}")
        print("PIPELINE TEST PASSED!")
    else:
        print(f"Error: {resp.text[:500]}")
except Exception as e:
    print(f"Request failed: {e}")
