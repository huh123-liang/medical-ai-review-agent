"""v4 regression gate (offline, free). Run it before any change to v4 is merged.

  python scripts/regression_gate.py                        # mock gate: 480 testbed cases vs the stored baseline
  python scripts/regression_gate.py --replay RUN/records.jsonl [...]   # replay sealed real records
  python scripts/regression_gate.py --accept               # store a new baseline (only after a version bump)

Exit 0 = pass. Exit 1 = behaviour changed: with unchanged versions this is a silent change and is refused outright;
with bumped versions the new baseline must be accepted explicitly (--accept) and the change reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from medical_jury.domain import digest  # noqa: E402
from medical_jury.testbed.build import build_cases  # noqa: E402
from medical_jury.v4 import criteria as CR  # noqa: E402
from medical_jury.v4.regression import mock_digests, replay_record, same_versions, versions  # noqa: E402

BASELINE = ROOT / 'tests_v2' / 'fixtures' / 'v4-regression-baseline.json'
SEED, N_TRAIN = 20261112, 150


def mock_gate(baseline: Path, accept: bool) -> tuple[int, dict]:
    library = CR.load_library(ROOT)
    now = {'versions': versions(library), 'cases': mock_digests(ROOT, library, build_cases(N_TRAIN, SEED)[0])}
    now['digest'] = digest(now['cases'])
    if not baseline.exists():
        if accept:
            baseline.write_text(json.dumps(now, indent=1, sort_keys=True) + '\n', encoding='utf-8')
            return 0, {'gate': 'mock', 'result': 'baseline created', 'cases': len(now['cases'])}
        return 1, {'gate': 'mock', 'result': 'no baseline; run with --accept once'}
    old = json.loads(baseline.read_text(encoding='utf-8'))
    changed = sorted(k for k in set(old['cases']) | set(now['cases']) if old['cases'].get(k) != now['cases'].get(k))
    same = old['versions'] == now['versions']
    summary = {'gate': 'mock', 'cases': len(now['cases']), 'changed': len(changed), 'changed_examples': changed[:10],
               'versions_changed': {k: (old['versions'].get(k), v) for k, v in now['versions'].items()
                                    if old['versions'].get(k) != v}}
    if not changed and same:
        return 0, {**summary, 'result': 'pass'}
    if changed and same:
        return 1, {**summary, 'result': 'FAIL: behaviour changed but no version changed (bump the version of what '
                                        'you edited, then re-run with --accept)'}
    if accept:
        baseline.write_text(json.dumps(now, indent=1, sort_keys=True) + '\n', encoding='utf-8')
        return 0, {**summary, 'result': 'new baseline accepted'}
    return 1, {**summary, 'result': 'versions changed: review the change, then re-run with --accept'}


def replay_gate(paths: list[Path]) -> tuple[int, dict]:
    library = CR.load_library(ROOT)
    out, code = [], 0
    for p in paths:
        seal = json.loads(p.with_name(p.stem + '-seal.json').read_text(encoding='utf-8'))
        if hashlib.sha256(p.read_bytes()).hexdigest() != seal['sha256']:
            raise SystemExit(f'{p} does not match its seal')
        recs = [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
        res = [(r, replay_record(r, library)) for r in recs]
        counts = {}
        for _, x in res:
            counts[x['status']] = counts.get(x['status'], 0) + 1
        changed = [{'case_id': r['case_id'], **x} for r, x in res if x['status'] == 'changed']
        silent = [c for (r, x), c in zip([(r, x) for r, x in res if x['status'] == 'changed'], changed)
                  if same_versions(r, library)]
        if silent:
            code = 1
        out.append({'records': str(p), 'n': len(recs), 'counts': counts, 'changed': changed[:20],
                    'silent_changes': len(silent),
                    'decision_changes': sum(c['triage_before']['decision'] != c['triage_after']['decision']
                                            for c in changed)})
    return code, {'gate': 'replay', 'result': 'pass' if code == 0 else 'FAIL: real decisions changed with the '
                  'same versions', 'runs': out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--replay', type=Path, nargs='*', help='sealed records.jsonl files from real runs')
    ap.add_argument('--accept', action='store_true', help='store the current behaviour as the new mock baseline')
    ap.add_argument('--baseline', type=Path, default=BASELINE)
    a = ap.parse_args(argv)
    code, summary = replay_gate(a.replay) if a.replay else mock_gate(a.baseline, a.accept)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
