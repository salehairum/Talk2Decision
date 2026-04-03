#!/usr/bin/env python3
"""
Reset the database by dropping all tables and recreating them.
"""

import sys
sys.path.insert(0, 'backend')

from backend.api.app import app
from models import db

with app.app_context():
    # Drop all tables
    print("Dropping all tables...")
    db.drop_all()
    
    # Recreate all tables
    print("Creating all tables...")
    db.create_all()
    
    print("Database reset complete!")
