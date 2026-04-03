#!/usr/bin/env python3
"""
Test the SQL matching logic directly.
"""

import sys
sys.path.insert(0, 'backend')

from backend.api.app import app
from models import db, Decision
from sqlalchemy import select, func

with app.app_context():
    # Create a test decision
    d1 = Decision(
        query="What is the decision?",
        extracted_decision="First answer",
        confidence="High",
        file_id="test-file-1",
        status="Open"
    )
    db.session.add(d1)
    db.session.commit()
    print(f"Created Decision #{d1.id}: {d1.query!r}")
    
    # Try to find it using normalized matching
    test_queries = [
        "What is the decision?",
        "what is the decision?",
        "What is the decision? ",
        " What is the decision?",
        "WHAT IS THE DECISION?"
    ]
    
    for q in test_queries:
        normalized = q.strip().lower()
        print(f"\nTesting: {q!r} (normalized: {normalized!r})")
        
        result = db.session.execute(
            select(Decision).where(
                func.lower(func.trim(Decision.query)) == normalized
            )
        ).scalars().first()
        
        if result:
            print(f"  ✓ FOUND Decision #{result.id}")
        else:
            print(f"  ✗ NOT FOUND")
            
            # Debug: Check what's in the database
            all_decs = db.session.execute(select(Decision)).scalars().all()
            for d in all_decs:
                db_normalized = d.query.strip().lower()
                db_func = func.lower(func.trim(d.query))
                print(f"    DB has: {d.query!r} (normalized: {db_normalized!r})")
                print(f"    Comparing: {normalized!r} == ?")
