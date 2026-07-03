import sys
from pathlib import Path

# Add src to sys.path so tests can import from fraud_detection
src_path = str(Path(__file__).resolve().parents[1] / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
