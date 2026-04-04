#!/usr/bin/env python3
"""
Simple test: Just query and check response status.
"""

import requests
import time
from pathlib import Path

BASE_URL = "http://127.0.0.1:5000"
DATA_DIR = Path(r"c:\Users\wania\OneDrive\Documents\Semester-8 (FAST)\AIPD\Talk2Decision\backend\context-extraction\data")

files = list(DATA_DIR.glob("*.json"))[:1]

def wait(file_id):
    for i in range(60):
        r = requests.get(f"{BASE_URL}/status/{file_id}")
        if r.status_code == 200:
            data = r.json()
            if data['status'] == 'completed':
                return True
            elif data['status'] == 'failed':
                print(f"Processing failed: {data.get('error')}")
                return False
        time.sleep(0.5)
    return False

print(f"File: {files[0].name}")

with open(files[0], 'rb') as f:
    r = requests.post(f"{BASE_URL}/upload", files={'file': (files[0].name, f)})

print(f"Upload status: {r.status_code}")
data = r.json()
file_id = data['file_id']
print(f"File ID: {file_id}")

print("Waiting for processing...")
if not wait(file_id):
    print("Processing failed or timed out")
else:
    print("Processing complete")
    
    query = "What is the decision?"
    r = requests.post(f"{BASE_URL}/query", headers={'Content-Type': 'application/json'},
        json={'file_id': file_id, 'query': query, 'top_k': 8})
    
    print(f"Query status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Decision ID: {data['decision_id']}")
        print(f"Action: {data['action']}")
        print(f"Message: {data['message']}")
    else:
        print(f"Error: {r.text}")
