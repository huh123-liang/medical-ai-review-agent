"""Post-hoc strict audit of the testbed write-ups (testbed-verify-2) and the false-alarm rate restated on the clean
cases that pass it. Offline; reads renders, cases and sealed review records; writes a NEW report folder only.

    python scripts/audit_testbed.py --run-dir reports/review-v4-20261116T051941Z-6aa0cb
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from medical_jury.evaluation import v4_scoring as SC  # noqa: E402
from medical_jury.testbed import TESTBED_VERSION  # noqa: E402
from medical_jury.testbed.audit import AUDIT_VERSION, audit  # noqa: E402
from medical_jury.testbed.build import WRITER_PROMPT_VERSION  # noqa: E402


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]


def run(root: Path, run_dir: Path | None) -> dict:
    tb = root / 'data' / 'testbed' / TESTBED_VERSION
    cases = {c['case_id']: c for c in _jsonl(tb / 'cases.jsonl')}
    out = {'audit_version': AUDIT_VERSION, 'testbed': TESTBED_VERSION, 'styles': {},
           'note': 'Post-hoc check added after review; pre-registered results are unchanged and reported as primary.'}
    for path in sorted(tb.glob(f'renders-*-{WRITER_PROMPT_VERSION}.jsonl')):
        style = path.name[len('renders-'):-len(f'-{WRITER_PROMPT_VERSION}.jsonl')]
        renders = {r['case_id']: r for r in _jsonl(path) if r.get('status') == 'ok'}
        failing = {}
        for k, r in renders.items():
            injected = {g['entity'] for g in SC.golds(cases[k])}        # an omitted element may BE the injected error
            problems = [p for p in audit(cases[k]['plan'], r['text']) if p.split(':')[0] not in injected]
            if problems:
                failing[k] = problems
        st = {'renders_ok': len(renders), 'failing': len(failing),
              'by_problem': dict(collections.Counter(p.split(':')[0] for v in failing.values() for p in v)),
              'clean_failing': {s: sorted(k for k in renders if not SC.golds(cases[k]) and cases[k]['split'] == s
                                          and k in failing) for s in ('train', 'test')},
              'failing_cases': failing, 'arms': {}}
        if run_dir is not None:
            for rec_path in sorted(run_dir.glob(f'*-test-{style}/records.jsonl')):
                arm = rec_path.parent.name.split('-')[0]
                recs = {r['case_id']: r for r in _jsonl(rec_path)}
                clean = [k for k in recs if not SC.golds(cases[k])]
                fp = [k for k in clean if any(f['severity'] == 'severe' for f in recs[k].get('findings', []))]
                good = [k for k in clean if k not in failing]
                st['arms'][arm] = {'clean_severe_fp_preregistered': [len(fp), len(clean)],
                                   'clean_severe_fp_on_audit_passing': [sum(k in fp for k in good), len(good)],
                                   'audit_failing_clean_flagged_severe': [sum(k in fp for k in clean if k in failing),
                                                                          sum(k in failing for k in clean)]}
        out['styles'][style] = st
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--run-dir', type=Path)
    a = p.parse_args(argv)
    run_dir = None if a.run_dir is None else (a.run_dir if a.run_dir.is_absolute() else a.root / a.run_dir)
    res = run(a.root, run_dir)
    out = a.root / 'reports' / f'testbed-audit-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
    out.mkdir(parents=True)
    (out / 'audit.json').write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    for s, st in res['styles'].items():
        print(s, {k: v for k, v in st.items() if k not in ('failing_cases',)})
    print(f'Written: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
