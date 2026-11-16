"""Freeze the v3 ordinal evaluation system before any validation / test run (PROTOCOL-V3-ORDINAL section 6).

    python scripts/freeze_v3.py --judge deepseek-pro --call-mode single --prompt-version v3-dimension-2

Writes reports/protocol-freeze-v3/<freeze_id>/freeze.json (never overwritten). run_ordinal_v3.py refuses a
validation/test run unless every field below equals the current system byte for byte.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from medical_jury.evaluation import ordinal_run as R  # noqa: E402
from medical_jury.evaluation.freeze import git_commit, seal  # noqa: E402
from medical_jury.v3.evaluator import EVALUATOR_PROMPT_VERSION, PROMPT_VERSIONS  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--judge', required=True)
    p.add_argument('--call-mode', choices=R.CALL_MODES, required=True)
    p.add_argument('--prompt-version', choices=list(PROMPT_VERSIONS), default=EVALUATOR_PROMPT_VERSION)
    p.add_argument('--note', default='')
    args = p.parse_args(argv)
    from run_ordinal_v3 import judge_spec
    fields = R.system_fields(args.root, judge_spec(args.root, args.judge), args.call_mode, args.prompt_version)
    freeze = seal({**fields, 'git_commit': git_commit(args.root), 'note': args.note,
                   'created_at': datetime.now(timezone.utc).isoformat()})
    folder = args.root / 'reports' / 'protocol-freeze-v3' / freeze['freeze_sha256'][:12]
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / 'freeze.json').open('x', encoding='utf-8') as f:
        json.dump(freeze, f, indent=2, ensure_ascii=False)
    print(json.dumps({'freeze': str(folder / 'freeze.json'), 'freeze_sha256': freeze['freeze_sha256']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
