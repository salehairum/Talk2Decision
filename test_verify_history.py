#!/usr/bin/env python3
"""Verify decision history is properly tracked."""

import sys
import json
from pathlib import Path

# Setup path
backend_dir = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(backend_dir))

from models import db, Decision, DecisionHistory
from api.app import app

with app.app_context():
    # Get all decisions
    decisions = db.session.query(Decision).all()
    print(f"✓ Total Decisions: {len(decisions)}")
    print()
    
    for dec in decisions:
        print(f"Decision #{dec.id}")
        print(f"  Query: {dec.query}")
        print(f"  Status: {dec.status}")
        print(f"  Created: {dec.created_at}")
        print(f"  Updated: {dec.updated_at}")
        
        # Parse source files
        try:
            source_files = json.loads(dec.source_files) if dec.source_files else []
            print(f"  Source Files: {source_files}")
        except Exception as e:
            print(f"  Source Files: ERROR - {e}")
        
        # Show history
        history = db.session.query(DecisionHistory).filter_by(decision_id=dec.id).all()
        if history:
            print(f"  History ({len(history)} entries):")
            for h in history:
                print(f"    - {h.field_name}: {h.old_value!r:50} → {h.new_value!r}")
        else:
            print(f"  History: (no entries)")
        print()
