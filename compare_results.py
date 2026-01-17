import json

r1 = json.load(open('result1.json'))
r2 = json.load(open('result2.json'))

print('=== ORIGINAL (440Hz, 5s) ===')
print(f"Risk: {r1['risk_score']:.4f}")
print(f"Jitter: {r1['biomarkers']['jitter']['value']:.6f}")
print(f"HNR: {r1['biomarkers']['hnr']['value']:.4f}")
print(f"Duration: {r1['file_info']['duration']}s")

print()
print('=== DIFFERENT (880Hz, 3s) ===')
print(f"Risk: {r2['risk_score']:.4f}")
print(f"Jitter: {r2['biomarkers']['jitter']['value']:.6f}")
print(f"HNR: {r2['biomarkers']['hnr']['value']:.4f}")
print(f"Duration: {r2['file_info']['duration']}s")

print()
if r1['risk_score'] != r2['risk_score']:
    print('RESULTS ARE DIFFERENT - Real analysis confirmed!')
else:
    print('WARNING: Same results - might be cached!')
