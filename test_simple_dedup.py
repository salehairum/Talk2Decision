#!/usr/bin/env python3
"""
Simple test to debug the deduplication logic.
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://127.0.0.1:5000"
DATA_DIR = Path(r"c:\Users\wania\OneDrive\Documents\Semester-8 (FAST)\AIPD\Talk2Decision\backend\context-extraction\data")

def wait_for_processing(file_id):
    for i in range(60):
        response = requests.get(f"{BASE_URL}/status/{file_id}")
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'completed':
                return True
            elif data['status'] == 'failed':
                return False
        time.sleep(0.5)
    return False

# Get first test file
test_file = list(DATA_DIR.glob("*.json"))[0]
print(f"Test file: {test_file.name}\n")

# Check initial state
response = requests.get(f"{BASE_URL}/decisions")
print(f"Initial decisions: {response.json()['count']}\n")

# Upload file
print(f"Uploading {test_file.name}...")
with open(test_file, 'rb') as f:
    response = requests.post(f"{BASE_URL}/upload", files={'file': (test_file.name, f)})
file_id = response.json()['file_id']
print(f"File uploaded as: {file_id}")

# Wait for processing
print("Waiting for processing...")
wait_for_processing(file_id)
print("Processing complete\n")

# Query 1
query = "What is the decision?"
print(f"QUERY 1: {query}")
response = requests.post(f"{BASE_URL}/query",
    headers={'Content-Type': 'application/json'},
    json={'file_id': file_id, 'query': query, 'top_k': 8}
)
if response.status_code == 200:
    r1 = response.json()
    print(f"  Decision ID: {r1['decision_id']}")
    print(f"  Action: {r1['action']}\n")
else:
    print(f"  ERROR: {response.status_code}\n")

time.sleep(1)

# Query 2 - Same query
print(f"QUERY 2 (same as Query 1): {query}")
response = requests.post(f"{BASE_URL}/query",
    headers={'Content-Type': 'application/json'},
    json={'file_id': file_id, 'query': query, 'top_k': 8}
)
if response.status_code == 200:
    r2 = response.json()
    print(f"  Decision ID: {r2['decision_id']}")
    print(f"  Action: {r2['action']}\n")
else:
    print(f"  ERROR: {response.status_code}\n")

# Check if same ID
if r1['decision_id'] == r2['decision_id']:
    print("✅ SUCCESS: Same query used same decision!")
else:
    print("❌ FAILED: Same query created different decisions!")

# Check final count
response = requests.get(f"{BASE_URL}/decisions")
final_count = response.json()['count']
print(f"Final decisions: {final_count}")

# Show decision details
response = requests.get(f"{BASE_URL}/decisions/{r1['decision_id']}")
if response.status_code == 200:
    d = response.json()
    print(f"\nDecision #{r1['decision_id']} details:")
    print(f"  Query: {d['query']}")
    print(f"  Source files: {d.get('source_files', 'N/A')}")
