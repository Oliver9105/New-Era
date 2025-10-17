#!/usr/bin/env python3
"""
Simple wrapper to run the aviator app with proper environment
"""
import subprocess
import sys
import os

# Change to the correct directory
os.chdir('/workspace/aviator_predictor')

# Run the app
try:
    subprocess.run([sys.executable, 'app.py'], check=True)
except KeyboardInterrupt:
    print("\nShutting down...")
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)