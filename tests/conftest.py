# tests/conftest.py
# Pytest path setup for DP-ADAGE test suite
import os
import sys
from pathlib import Path

# Add project root directory to sys.path so src is directly importable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
