#!/usr/bin/env python3
"""
Simplified cross-file test: upload 2 files, ask same question on both.
"""

import requests, time, json
from pathlib import Path

BASE_URL = "http://127.0.0.1:5000"
DATA_DIR = Path(r"c:\Users\wania\OneDrive\Documents\Semester-8 (FAST)\AIPD\Talk2Decision\backend\context-extraction\data")

def wait(file_id):
    for _ in range(60):
        r = requests.get(f"{BASE_URL}/status/{file_id}")
        if r.status_code == 200 and r.json()['status'] == 'completed':
            return True
        time.sleep(0.5)
    return False

files = list(DATA_DIR.glob("*.json"))[:2]
print(f"Using files: {[f.name for f in files]}\n")

# Upload & query first file
print(f"=== FILE 1: {files[0].name} ===")
with open(files[0], 'rb') as f:
    r = requests.post(f"{BASE_URL}/upload", files={'file': (files[0].name, f)})
fid1 = r.json()['file_id']
print(f"Uploaded, file_id={fid1}")
wait(fid1)

query = "What is the decision?"
print(f"Query: {query}")
r = requests.post(f"{BASE_URL}/query",
    headers={'Content-Type': 'application/json'},
    json={'file_id': fid1, 'query': query, 'top_k': 8}
)
d1 = r.json()
print(f"Result: Decision #{d1['decision_id']}, Action: {d1['action']}\n")

time.sleep(1)

# Upload & query second file with SAME query
print(f"=== FILE 2: {files[1].name} ===")
with open(files[1], 'rb') as f:
    r = requests.post(f"{BASE_URL}/upload", files={'file': (files[1].name, f)})
fid2 = r.json()['file_id']
print(f"Uploaded, file_id={fid2}")
wait(fid2)

print(f"Query: {query}")
r = requests.post(f"{BASE_URL}/query",
    headers={'Content-Type': 'application/json'},
    json={'file_id': fid2, 'query': query, 'top_k': 8}
)
d2 = r.json()
print(f"Result: Decision #{d2['decision_id']}, Action: {d2['action']}\n")

# Check result
if d1['decision_id'] == d2['decision_id']:
    print(f"SUCCESS! Both files updated Decision #{d1['decision_id']}")
else:
    print(f"FAILED! File 1 created Decision #{d1['decision_id']}, File 2 created Decision #{d2['decision_id']}")

# List all decisions
r = requests.get(f"{BASE_URL}/decisions")
print(f"\nTotal decisions in DB: {r.json()['count']}")
