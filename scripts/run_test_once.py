"""Final one-shot run on the TEST split with the same frozen protocol (requires --final).

    python scripts/run_test_once.py reports/protocol-freeze/<id>/freeze.json --kind natural --final --authorized
Refuses unless both validation kinds already ran for this freeze.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location('run_validation_once', Path(__file__).with_name('run_validation_once.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

if __name__ == '__main__':
    raise SystemExit(_mod.main(sys.argv[1:], split='test'))
