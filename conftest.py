"""Pytest config: make `documind` importable when running tests from the project root.

The project root *itself* is the `documind` package (it has __init__.py), so we
add its parent directory to sys.path. That way `import documind.x.y` resolves
regardless of where pytest is invoked from.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
