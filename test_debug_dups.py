#!/usr/bin/env python3
"""
Debug test: Run two queries and check database state after each.
"""

import requests
import time
from pathlib import Path

BASE_URL = "http://127.0.0.1:5000"
DATA_DIR = Path(r"c:\Users\wania\OneDrive\Documents\Semester-8 (FAST)\AIPD\Talk2Decision\backend\context-extraction\data")

def check_db():
    """Check what's in the database."""
    r = requests.get(f"{BASE_URL}/decisions")
    if r.status_code == 200:
        decisions = r.json()['decisions']
        print(f"DB has {len(decisions)} decisions:")
        for d in decisions:
            print(f"  - ID {d['id']}: query={d['query']!r}")
        return decisions
    return []

def wait(file_id):
    for _ in range(60):
        r = requests.get(f"{BASE_URL}/status/{file_id}")
        if r.status_code == 200 and r.json()['status'] == 'completed':
            return True
        time.sleep(0.5)
    return False

files = list(DATA_DIR.glob("*.json"))[:2]
query = "What is the decision?"

print("=== INITIAL STATE ===")
check_db()

print("\n=== UPLOAD & QUERY FILE 1 ===")
with open(files[0], 'rb') as f:
    r = requests.post(f"{BASE_URL}/upload", files={'file': (files[0].name, f)})
file_id_1 = r.json()['file_id']
print(f"File uploaded: {file_id_1}")
wait(file_id_1)

r = requests.post(f"{BASE_URL}/query", headers={'Content-Type': 'application/json'},
    json={'file_id': file_id_1, 'query': query, 'top_k': 8})
if r.status_code == 200:
    print(f"Query result: Decision #{r.json()['decision_id']}, Action: {r.json()['action']}")

print("\nDatabase state after File 1:")
check_db()

print("\n=== UPLOAD & QUERY FILE 2 ===")
with open(files[1], 'rb') as f:
    r = requests.post(f"{BASE_URL}/upload", files={'file': (files[1].name, f)})
file_id_2 = r.json()['file_id']
print(f"File uploaded: {file_id_2}")
wait(file_id_2)

r = requests.post(f"{BASE_URL}/query", headers={'Content-Type': 'application/json'},
    json={'file_id': file_id_2, 'query': query, 'top_k': 8})
if r.status_code == 200:
    print(f"Query result: Decision #{r.json()['decision_id']}, Action: {r.json()['action']}")

print("\nDatabase state after File 2:")
check_db()
