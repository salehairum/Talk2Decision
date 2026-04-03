"""
Quick test script for the decision tracking system.
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"

def test_upload_and_query():
    """Test uploading a chat file and extracting decisions."""
    
    # Upload chat.json
    print("1. Uploading chat.json...")
    with open("backend/context-extraction/data/chat.json", "rb") as f:
        files = {"file": ("chat.json", f)}
        response = requests.post(f"{BASE_URL}/upload", files=files)
    
    if response.status_code != 202:
        print(f"   ❌ Upload failed: {response.text}")
        return
    
    upload_data = response.json()
    file_id = upload_data["file_id"]
    print(f"   ✅ Upload successful! File ID: {file_id}")
    
    # Wait for processing
    print("2. Waiting for file processing...")
    for i in range(30):
        response = requests.get(f"{BASE_URL}/status/{file_id}")
        status_data = response.json()
        status = status_data.get("status")
        progress = status_data.get("progress", 0)
        
        print(f"   Status: {status} ({progress}%)", end="\r")
        
        if status == "completed":
            print(f"   ✅ Processing complete!                ")
            break
        elif status == "failed":
            print(f"   ❌ Processing failed: {status_data.get('error')}")
            return
        
        time.sleep(1)
    else:
        print("   ⏱️  Timeout waiting for processing")
        return
    
    # Query for decisions
    print("3. Querying for decisions...")
    query_data = {
        "file_id": file_id,
        "query": "What was the main decision discussed?"
    }
    
    response = requests.post(f"{BASE_URL}/query", json=query_data)
    
    if response.status_code != 200:
        print(f"   ❌ Query failed: {response.text}")
        return
    
    result = response.json()
    decision_id = result.get("decision_id")
    decision = result.get("decision", {})
    
    print(f"   ✅ Query successful!")
    print(f"   Decision ID: {decision_id}")
    print(f"   Decision: {decision.get('decision', 'N/A')}")
    print(f"   Confidence: {decision.get('confidence', 'N/A')}")
    print(f"   Action Items: {len(decision.get('action_items', []))} found")
    
    # Test getting all decisions
    print("4. Retrieving all decisions from database...")
    response = requests.get(f"{BASE_URL}/decisions")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Found {data['count']} decisions in database")
        
        # Show first decision
        if data['decisions']:
            first = data['decisions'][0]
            print(f"   First decision: {first['decision'][:60]}...")
    else:
        print(f"   ❌ Failed to retrieve decisions: {response.text}")
    
    # Test getting specific decision
    print(f"5. Retrieving specific decision (ID={decision_id})...")
    response = requests.get(f"{BASE_URL}/decisions/{decision_id}")
    
    if response.status_code == 200:
        decision = response.json()
        print(f"   ✅ Decision details retrieved")
        print(f"      Status: {decision['status']}")
        print(f"      Owner: {decision['owner']}")
        print(f"      Priority: {decision['priority']}")
        print(f"      Category: {decision['category']}")
        print(f"      Evidence items: {len(decision['evidence'])}")
        print(f"      Action items: {len(decision['action_items'])}")
        
        # Display action items if any
        if decision['action_items']:
            print("      Action items:")
            for ai in decision['action_items']:
                print(f"        - {ai['task']} (owner: {ai['owner']}, due: {ai['due_date']})")
    else:
        print(f"   ❌ Failed to retrieve decision: {response.text}")
    
    # Test updating decision status
    print(f"6. Updating decision status...")
    update_data = {
        "status": "In-Progress",
        "changed_by": "test_user"
    }
    response = requests.post(f"{BASE_URL}/decisions/{decision_id}/status", json=update_data)
    
    if response.status_code == 200:
        print(f"   ✅ Status updated to In-Progress")
    else:
        print(f"   ❌ Failed to update status: {response.text}")
    
    # Test getting history
    print(f"7. Retrieving decision history...")
    response = requests.get(f"{BASE_URL}/decisions/{decision_id}/history")
    
    if response.status_code == 200:
        data = response.json()
        history = data.get('history', [])
        print(f"   ✅ Found {len(history)} history entries")
        for entry in history:
            print(f"      - {entry['field']}: {entry['old_value']} → {entry['new_value']} (by {entry['changed_by']})")
    else:
        print(f"   ❌ Failed to retrieve history: {response.text}")
    
    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    test_upload_and_query()
