#!/usr/bin/env python3
"""
Test the full decision export flow: upload file, run query, run same query again.
This simulates what the user sees when exporting the same decision twice.
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://127.0.0.1:5000"
DATA_DIR = Path(r"c:\Users\wania\OneDrive\Documents\Semester-8 (FAST)\AIPD\Talk2Decision\backend\context-extraction\data")

def wait_for_file_processing(file_id, max_attempts=60):
    """Poll until file is processed."""
    for i in range(max_attempts):
        response = requests.get(f"{BASE_URL}/status/{file_id}")
        if response.status_code == 200:
            data = response.json()
            print(f"[{i+1}] Status: {data['status']} - {data['message']}")
            if data['status'] == 'completed':
                return True
            elif data['status'] == 'failed':
                print(f"Processing failed: {data.get('error')}")
                return False
        time.sleep(0.5)
    return False

def test_export_flow():
    print("[TEST] Starting decision export deduplication test...")
    print("[TEST] This simulates: upload file → ask question → ask same question again\n")
    
    # Find an existing test file
    test_files = list(DATA_DIR.glob("*.json"))
    if not test_files:
        print("[ERROR] No test files found in data folder")
        return
    
    # Use first test file
    test_file = test_files[0]
    print(f"[TEST] Using test file: {test_file.name}")
    
    # Upload file
    print("[TEST] Uploading file...")
    with open(test_file, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/upload",
            files={'file': (test_file.name, f, 'application/json')}
        )
    
    if response.status_code != 202:
        print(f"[ERROR] Upload failed: {response.status_code}")
        return
    
    upload_data = response.json()
    file_id = upload_data['file_id']
    print(f"[TEST] File uploaded. file_id='{file_id}'")
    
    # Wait for processing
    print("[TEST] Waiting for file processing...")
    if not wait_for_file_processing(file_id):
        print("[ERROR] File processing failed or timed out")
        return
    
    print("[TEST] File is ready\n")
    
    # Get initial decisions count
    response = requests.get(f"{BASE_URL}/decisions")
    initial_count = response.json()['count'] if response.status_code == 200 else 0
    print(f"[TEST] Initial decision count: {initial_count}\n")
    
    # Run query first time
    test_query = "What are the key decisions?"
    print(f'[TEST] FIRST EXPORT: Running query: "{test_query}"')
    response = requests.post(
        f"{BASE_URL}/query",
        headers={'Content-Type': 'application/json'},
        json={
            'file_id': file_id,
            'query': test_query,
            'top_k': 8
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        decision_id_1 = data['decision_id']
        action_1 = data['action']
        print(f"  → Decision ID: {decision_id_1}, Action: {action_1}\n")
    else:
        print(f"[ERROR] Query failed: {response.status_code} - {response.text}")
        return
    
    # Wait a moment
    time.sleep(1)
    
    # Run SAME query second time
    print(f'[TEST] SECOND EXPORT: Running SAME query: "{test_query}"')
    response = requests.post(
        f"{BASE_URL}/query",
        headers={'Content-Type': 'application/json'},
        json={
            'file_id': file_id,
            'query': test_query,
            'top_k': 8
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        decision_id_2 = data['decision_id']
        action_2 = data['action']
        print(f"  → Decision ID: {decision_id_2}, Action: {action_2}\n")
    else:
        print(f"[ERROR] Query failed: {response.status_code} - {response.text}")
        return
    
    # Check results
    print("[TEST] Results:")
    print(f"  First export decision ID:  {decision_id_1}")
    print(f"  Second export decision ID: {decision_id_2}")
    
    if decision_id_1 == decision_id_2:
        print(f"\n✓ SUCCESS! Same decision was UPDATED (not duplicated)")
        print(f"  Action 1: {action_1}")
        print(f"  Action 2: {action_2}")
    else:
        print(f"\n✗ FAILED! Created two different decisions instead of updating")
        print(f"  This means duplicates are being created!")
    
    # Check final decision count
    response = requests.get(f"{BASE_URL}/decisions")
    final_count = response.json()['count'] if response.status_code == 200 else 0
    print(f"\n[TEST] Final decision count: {final_count}")
    if final_count == initial_count + 2:
        print(f"  → Created 2 new decisions (BAD - should be 1)")
    elif final_count == initial_count + 1:
        print(f"  → Created 1 new decision (GOOD - was updated on second export)")
    else:
        print(f"  → Something unexpected happened")

if __name__ == "__main__":
    test_export_flow()
