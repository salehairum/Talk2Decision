#!/usr/bin/env python3
"""
Debug script to list all decisions in the database.
"""

import requests

BASE_URL = "http://127.0.0.1:5000"

response = requests.get(f"{BASE_URL}/decisions")
if response.status_code == 200:
    data = response.json()
    print(f"Total decisions: {data['count']}\n")
    for d in data['decisions']:
        print(f"ID: {d['id']}")
        print(f"  Query: {d['query']}")
        print(f"  File: {d['file_id']}")
        print(f"  Status: {d['status']}")
        print(f"  Decision: {d['decision'][:60]}...")
        print()
else:
    print(f"Error: {response.status_code}")
