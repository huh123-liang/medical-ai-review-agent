"""Offline acceptance replay of locator-3 on the sealed dev200 quotes (no model call, no label read).

    python scripts/replay_locator3_dev200.py \
        --predictions reports/dev-v2-deepseek-pro-20261025T103434Z/predictions.jsonl \
        --audit reports/dev200-review-20261025T105907Z/quote-audit-14.csv

Acceptance (dev200 review, P1-2): every locator-2 'exact' quote stays exact with identical spans; of the 14
unlocated quotes, the 3 with real changes (skipped words, rewritten punctuation, dropped safety qualifier) must
stay 'not_located'. Writes reports/locator3-replay-<UTC>-<rand>/summary.json with ids and tiers only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from medical_jury.judge_v2 import locate_quote  # noqa: E402
from medical_jury.v3.locator import LOCATOR_VERSION, locate_quote_v3  # noqa: E402

MUST_STAY_UNLOCATED = ('跳词', '标点', '摘要式改写')   # suspected_cause markers in the review's audit CSV


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    args = p.parse_args(argv)
    answers = {}
    with (args.root / 'data/prepared/dev.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            answers[row['case']['case_id']] = row['case']['candidate_output']
    preds = [json.loads(line) for line in args.predictions.read_text(encoding='utf-8').splitlines() if line.strip()]
    by_id = {p['example_id']: p for p in preds}
    old_t, new_t, transitions, exact_changed, rows = Counter(), Counter(), Counter(), [], []
    for pr in preds:
        answer = answers[pr['case_id']]
        for k, q in enumerate(pr.get('evidence') or []):
            o, n = locate_quote(answer, q), locate_quote_v3(answer, q)
            old_t[o['tier']] += 1
            new_t[n['tier']] += 1
            if o['tier'] == 'exact' and (n['tier'] != 'exact' or n['spans'] != o['spans']):
                exact_changed.append([pr['example_id'], k])
            if o['tier'] != 'exact':
                transitions[f"{o['tier']} -> {n['tier']}{' (cross_structure)' if n['cross_structure'] else ''}"] += 1
                rows.append({'example_id': pr['example_id'], 'quote_index': k, 'locator2': o['tier'],
                             'locator3': n['tier'], 'cross_structure': n['cross_structure']})
    must, must_ok = [], True
    with args.audit.open(encoding='utf-8-sig') as f:
        for a in csv.DictReader(f):
            if any(m in a['suspected_cause'] or m in a['semantic_change'] for m in MUST_STAY_UNLOCATED):
                pr = by_id[a['example_id']]
                tier = locate_quote_v3(answers[pr['case_id']], pr['evidence'][int(a['quote_index'])])['tier']
                must.append({'example_id': a['example_id'], 'quote_index': int(a['quote_index']), 'tier': tier})
                must_ok &= tier == 'not_located'
    summary = {'locator': LOCATOR_VERSION, 'quotes': sum(old_t.values()), 'locator2_tiers': dict(old_t),
               'locator3_tiers': dict(new_t), 'transitions': dict(transitions),
               'exact_changed': exact_changed, 'must_stay_unlocated': must,
               'acceptance': {'exact_unchanged': not exact_changed, 'real_changes_still_unlocated': must_ok and len(must) == 3},
               'non_exact_rows': rows, 'note': 'Diagnostic tiers are never counted as located; ids and tiers only.'}
    out = args.root / 'reports' / ('locator3-replay-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:6])
    out.mkdir(parents=True)
    with (out / 'summary.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: summary[k] for k in ('quotes', 'locator2_tiers', 'locator3_tiers', 'transitions', 'acceptance')},
                     indent=2, ensure_ascii=False))
    print('Written:', out)
    return 0 if all(summary['acceptance'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
