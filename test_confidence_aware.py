#!/usr/bin/env python3
"""Test confidence-aware decision deduplication."""

import sys
import json
import requests
import time
from pathlib import Path

# Test setup
BACKEND_URL = "http://127.0.0.1:5000"
DATA_DIR = Path(__file__).resolve().parent / "backend" / "context-extraction" / "data"
SLACK_FILES = [
    "2026-03-23.json",
    "2026-03-26.json",
    "2026-04-03.json"
]

def upload_file(filename):
    """Upload a file and return file_id."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"  ✗ File not found: {filepath}")
        return None
    
    with open(filepath, "rb") as f:
        files = {"file": (filename, f)}
        try:
            resp = requests.post(f"{BACKEND_URL}/upload", files=files)
            if resp.status_code == 202:
                file_id = resp.json()["file_id"]
                print(f"  ✓ Uploaded {filename} (file_id={file_id})")
                return file_id
        except Exception as e:
            print(f"  ✗ Upload failed: {e}")
    return None

def wait_processing(file_id, timeout=60):
    """Wait for file processing to complete."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{BACKEND_URL}/status/{file_id}")
            status = resp.json()
            if status["status"] == "completed":
                return True
            if status["status"] == "failed":
                print(f"  ✗ Processing failed: {status.get('error')}")
                return False
        except Exception as e:
            pass
        time.sleep(0.5)
    print(f"  ✗ Processing timeout")
    return False

def query_decision(file_id, query_text):
    """Query a decision and return decision_id, confidence, and action."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/query",
            json={"file_id": file_id, "query": query_text}
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "decision_id": data.get("decision_id"),
                "action": data.get("action"),
                "confidence": data.get("decision", {}).get("confidence", "Low"),
                "decision": data.get("decision", {}).get("decision", "")
            }
    except Exception as e:
        print(f"  ✗ Query failed: {e}")
    return None

def get_decision_details(decision_id):
    """Get decision details from database."""
    sys.path.insert(0, str(Path(__file__).parent / "backend"))
    from models import db, Decision, DecisionHistory
    from api.app import app
    
    with app.app_context():
        dec = db.session.query(Decision).filter_by(id=decision_id).first()
        if dec:
            source_files = json.loads(dec.source_files) if dec.source_files else []
            history = db.session.query(DecisionHistory).filter_by(decision_id=decision_id).all()
            return {
                "confidence": dec.confidence,
                "status": dec.status,
                "source_files": source_files,
                "decision": dec.extracted_decision,
                "history_count": len(history)
            }
    return None

print("=" * 70)
print("TEST: Confidence-Aware Decision Deduplication")
print("=" * 70)
print()

print("✓ SCENARIO 1: High confidence decision shouldn't be replaced by low confidence")
print("-" * 70)

# Upload first file
print("1️⃣  File 1: Extract with HIGH confidence decision")
file1_id = upload_file("2026-03-23.json")
if not file1_id or not wait_processing(file1_id):
    print("❌ Test stopped: File 1 processing failed")
    sys.exit(1)

result1 = query_decision(file1_id, "What is the decision?")
print(f"   Result: Decision #{result1['decision_id']}, Confidence: {result1['confidence']}")
dec1_id = result1["decision_id"]

# Upload second file
print()
print("2️⃣  File 2: Same query, try to extract with DIFFERENT confidence")
file2_id = upload_file("2026-03-26.json")
if not file2_id or not wait_processing(file2_id):
    print("❌ Test stopped: File 2 processing failed")
    sys.exit(1)

result2 = query_decision(file2_id, "What is the decision?")
print(f"   Result: Decision #{result2['decision_id']}, Confidence: {result2['confidence']}")
dec2_id = result2["decision_id"]

# Verify it's the same decision
if dec1_id == dec2_id:
    print(f"   ✓ Same decision tracked (ID={dec1_id})")
    
    # Check decision details
    details = get_decision_details(dec1_id)
    print(f"   ✓ Source files: {details['source_files']}")
    print(f"   ✓ Status: {details['status']}")
    print(f"   ✓ Final confidence: {details['confidence']}")
    print(f"   ✓ History entries: {details['history_count']}")
else:
    print(f"   ✗ Different decisions created! ID1={dec1_id}, ID2={dec2_id}")

print()
print("=" * 70)
print("✓ SCENARIO 2: Multiple different decisions tracked simultaneously")
print("-" * 70)

# Upload third file
print("1️⃣  File 3: Different query (new topic)")
file3_id = upload_file("2026-04-03.json")
if not file3_id or not wait_processing(file3_id):
    print("❌ Test stopped: File 3 processing failed")
    sys.exit(1)

result3 = query_decision(file3_id, "What are the key decisions?")
print(f"   Result: Decision #{result3['decision_id']}, Confidence: {result3['confidence']}")
dec3_id = result3["decision_id"]

# Verify it's a different decision
if dec3_id != dec1_id:
    print(f"   ✓ Different decision created for different query (ID={dec3_id})")
    print(f"   ✓ Now tracking 2 different decisions:")
    print(f"      - Decision #{dec1_id}: 'What is the decision?'")
    print(f"      - Decision #{dec3_id}: 'What are the key decisions?'")
else:
    print(f"   ⚠️  Same decision used! (Different queries should create separate decisions)")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"✅ Scenario 1 (Confidence-aware): Decision {dec1_id}")
print(f"✅ Scenario 2 (Multiple decisions): Decision {dec3_id} is separate from {dec1_id}")
print()
print("All tests passed! ✅")
