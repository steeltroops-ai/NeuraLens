#!/usr/bin/env python
"""Test script to check if the app can be imported"""

import sys
import traceback

try:
    print("Testing imports...")
    print("1. Importing FastAPI...")
    from fastapi import FastAPI
    print("   ✅ FastAPI imported")
    
    print("2. Importing app.main...")
    from app.main import app
    print("   ✅ app.main imported")
    
    print("3. Importing speech analyzer...")
    from app.pipelines.speech.analyzer import RealtimeSpeechAnalyzer
    print("   ✅ Speech analyzer imported")
    
    print("\n✅ All imports successful!")
    print(f"App type: {type(app)}")
    
except Exception as e:
    print(f"\n❌ Import failed!")
    print(f"Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
