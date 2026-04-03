"""
Quick test for the decision tracking system - query existing processed file.
"""

import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def test_query_and_track():
    """Test querying a processed file and tracking the decision."""
    
    file_id = "chat"  # The file we already processed
    
    # Query for decisions
    print("1. Querying for decisions...")
    query_data = {
        "file_id": file_id,
        "query": "What are the main decisions?"
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
    print(f"   Decision: {decision.get('decision', 'N/A')[:80]}...")
    print(f"   Confidence: {decision.get('confidence', 'N/A')}")
    print(f"   Action Items: {len(decision.get('action_items', []))} found")
    
    # List all decisions from database
    print("\n2. Retrieving all decisions from database...")
    response = requests.get(f"{BASE_URL}/decisions")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Found {data['count']} decisions in database")
        for i, d in enumerate(data['decisions'][:3], 1):
            print(f"      {i}. ID={d['id']}, Status={d['status']}, Priority={d['priority']}")
    else:
        print(f"   ❌ Failed: {response.text}")
    
    if not decision_id:
        print("\n❌ No decision ID to proceed with further tests")
        return
    
    # Get specific decision
    print(f"\n3. Retrieving decision ID={decision_id}...")
    response = requests.get(f"{BASE_URL}/decisions/{decision_id}")
    
    if response.status_code == 200:
        decision = response.json()
        print(f"   ✅ Decision details retrieved")
        print(f"      Status: {decision['status']}")
        print(f"      Owner: {decision['owner']}")
        print(f"      Priority: {decision['priority']}")
        print(f"      Evidence items: {len(decision['evidence'])}")
        print(f"      Action items: {len(decision['action_items'])}")
    else:
        print(f"   ❌ Failed: {response.text}")
        return
    
    # Update decision status
    print(f"\n4. Updating decision status to 'In-Progress'...")
    update_data = {
        "status": "In-Progress",
        "changed_by": "test_user"
    }
    response = requests.post(f"{BASE_URL}/decisions/{decision_id}/status", json=update_data)
    
    if response.status_code == 200:
        print(f"   ✅ Status updated")
    else:
        print(f"   ❌ Failed: {response.text}")
        return
    
    # Get history
    print(f"\n5. Retrieving decision history...")
    response = requests.get(f"{BASE_URL}/decisions/{decision_id}/history")
    
    if response.status_code == 200:
        data = response.json()
        history = data.get('history', [])
        print(f"   ✅ Found {len(history)} history entries")
        for entry in history:
            print(f"      - {entry['field']}: {entry['old_value']} → {entry['new_value']}")
    else:
        print(f"   ❌ Failed: {response.text}")
        return
    
    # Update owner
    print(f"\n6. Assigning decision owner...")
    update_data = {
        "owner": "Alice Wong",
        "changed_by": "admin"
    }
    response = requests.post(f"{BASE_URL}/decisions/{decision_id}/owner", json=update_data)
    
    if response.status_code == 200:
        print(f"   ✅ Owner assigned to Alice Wong")
    else:
        print(f"   ❌ Failed: {response.text}")
        return
    
    # Update metadata
    print(f"\n7. Updating metadata (priority and category)...")
    update_data = {
        "priority": "High",
        "category": "Technical Decision",
        "changed_by": "admin"
    }
    response = requests.post(f"{BASE_URL}/decisions/{decision_id}/metadata", json=update_data)
    
    if response.status_code == 200:
        decision = response.json()
        print(f"   ✅ Metadata updated")
        print(f"      Priority: {decision['priority']}")
        print(f"      Category: {decision['category']}")
    else:
        print(f"   ❌ Failed: {response.text}")
        return
    
    # Add action item
    print(f"\n8. Adding action item...")
    action_data = {
        "task": "Implement the decision",
        "owner": "Alice Wong",
        "due_date": "2026-04-10"
    }
    response = requests.post(f"{BASE_URL}/decisions/{decision_id}/actions", json=action_data)
    
    if response.status_code == 201:
        action = response.json()
        print(f"   ✅ Action item created")
        print(f"      ID: {action['id']}")
        print(f"      Task: {action['task']}")
        print(f"      Owner: {action['owner']}")
        print(f"      Due: {action['due_date']}")
    else:
        print(f"   ❌ Failed: {response.text}")
        return
    
    # Get final state
    print(f"\n9. Final decision state:")
    response = requests.get(f"{BASE_URL}/decisions/{decision_id}")
    
    if response.status_code == 200:
        decision = response.json()
        print(f"   ✅ Decision fully tracked")
        print(f"      Status: {decision['status']}")
        print(f"      Owner: {decision['owner']}")
        print(f"      Priority: {decision['priority']}")
        print(f"      Category: {decision['category']}")
        print(f"      Action items: {len(decision['action_items'])}")
        print(f"      History entries: {len(decision.get('history', []))}") 
    else:
        print(f"   ❌ Failed: {response.text}")
        return
    
    print("\n✅ ALL TESTS PASSED!")

if __name__ == "__main__":
    test_query_and_track()
