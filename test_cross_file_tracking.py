#!/usr/bin/env python3
"""
Test cross-file decision deduplication: same topic from different days 
should update ONE decision, not create multiple.
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
            if data['status'] == 'completed':
                return True
            elif data['status'] == 'failed':
                return False
        time.sleep(0.5)
    return False

def upload_and_query(file_path, query_text):
    """Upload a file and run a query on it."""
    filename = Path(file_path).name
    print(f"\n  📤 Uploading: {filename}")
    
    with open(file_path, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/upload",
            files={'file': (filename, f, 'application/json')}
        )
    
    if response.status_code != 202:
        print(f"  ❌ Upload failed: {response.status_code}")
        return None
    
    upload_data = response.json()
    file_id = upload_data['file_id']
    print(f"  ✓ File uploaded (file_id='{file_id}')")
    
    # Wait for processing
    print(f"  ⏳ Processing...")
    if not wait_for_file_processing(file_id):
        print(f"  ❌ Processing failed")
        return None
    print(f"  ✓ Processing complete")
    
    # Run query
    print(f"  🔍 Query: \"{query_text}\"")
    response = requests.post(
        f"{BASE_URL}/query",
        headers={'Content-Type': 'application/json'},
        json={
            'file_id': file_id,
            'query': query_text,
            'top_k': 8
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        return {
            'file_id': file_id,
            'decision_id': data['decision_id'],
            'action': data['action'],
            'decision_text': data['decision']['decision'] if 'decision' in data else 'N/A'
        }
    else:
        print(f"  ❌ Query failed: {response.status_code}")
        return None

def test_cross_file_tracking():
    print("[TEST] Cross-File Decision Tracking Test")
    print("=" * 60)
    print("Scenario: Same topic asked across different days/files")
    print("Expected: ONE decision updated, not multiple created\n")
    
    # Find test files
    test_files = sorted(list(DATA_DIR.glob("*.json")))[:3]  # Use first 3 files as different days
    
    if len(test_files) < 2:
        print("❌ Need at least 2 test files to run this test")
        return
    
    print(f"Found {len(test_files)} test files:")
    for f in test_files:
        print(f"  - {f.name}")
    
    # Get initial decision count
    response = requests.get(f"{BASE_URL}/decisions")
    initial_count = response.json()['count'] if response.status_code == 200 else 0
    print(f"\nInitial decisions: {initial_count}\n")
    
    # Simulate exporting same topic from different days
    test_query = "What are the key decisions?"
    results = []
    
    for i, test_file in enumerate(test_files, 1):
        print(f"DAY {i}: {test_file.name}")
        result = upload_and_query(test_file, test_query)
        if result:
            results.append(result)
            print(f"  Result: Decision #{result['decision_id']}, Action: {result['action']}")
            time.sleep(1)  # Simulate time between exports
        else:
            print(f"  ❌ Failed to process day {i}")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    if len(results) < len(test_files):
        print(f"❌ Only processed {len(results)}/{len(test_files)} files")
        return
    
    decision_ids = [r['decision_id'] for r in results]
    unique_decision_ids = set(decision_ids)
    
    print(f"\nDecision IDs across days:")
    for i, result in enumerate(results, 1):
        print(f"  Day {i} ({result['file_id']}): Decision #{result['decision_id']}")
    
    if len(unique_decision_ids) == 1:
        decision_id = list(unique_decision_ids)[0]
        print(f"\n✅ SUCCESS! All days updated the SAME decision #{decision_id}")
        print(f"   Actions: {' → '.join([r['action'] for r in results])}")
        
        # Get the decision details
        response = requests.get(f"{BASE_URL}/decisions/{decision_id}")
        if response.status_code == 200:
            decision = response.json()
            print(f"\n   Decision Details:")
            print(f"   - Query: \"{decision['query']}\"")
            print(f"   - Status: {decision['status']}")
            print(f"   - Created: {decision['created_at']}")
            print(f"   - Updated: {decision['updated_at']}")
            print(f"   - Source Files: {decision.get('source_files', 'N/A')}")
    else:
        print(f"\n❌ FAILED! Created {len(unique_decision_ids)} different decisions:")
        for i, decision_id in enumerate(unique_decision_ids, 1):
            print(f"   - Decision #{decision_id}")
        print(f"\n   This means duplicate decisions are still being created!")
    
    # Check final count
    response = requests.get(f"{BASE_URL}/decisions")
    final_count = response.json()['count'] if response.status_code == 200 else 0
    
    new_decisions = final_count - initial_count
    print(f"\nFinal decision count: {final_count}")
    print(f"New decisions created: {new_decisions}")
    if new_decisions == 1:
        print("✅ Only 1 new decision (GOOD - cross-file deduplication works!)")
    elif new_decisions == len(test_files):
        print("❌ Created one per file (BAD - no cross-file deduplication)")
    else:
        print(f"⚠️  Unexpected: {new_decisions} new decisions")

if __name__ == "__main__":
    test_cross_file_tracking()
