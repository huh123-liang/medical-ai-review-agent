"""Export DEV disagreements for human open coding (error taxonomy).

    python scripts/export_error_review.py reports/dev-v2-deepseek-<stamp> [--limit 100]
Writes <run>/error-review.csv (UTF-8 with BOM so Excel opens it) and error-review.md.
The error_type / notes columns are intentionally EMPTY: a person codes them. Never auto-fill
with a model, and never export validation/test runs (that would turn them into dev evidence).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from medical_jury.evaluation.data import load_prepared, references_by_id  # noqa: E402

ERROR_TYPES = ['judge_too_lenient', 'judge_too_strict', 'task_misread', 'criterion_ambiguous',
               'physicians_disagree', 'missing_medical_knowledge', 'negative_criterion_polarity', 'other']
COLUMNS = ['example_id', 'topic', 'criterion', 'judge_verdict', 'judge_confidence', 'judge_analysis',
           'physician_labels', 'physicians_disagree', 'answer_excerpt', 'error_type', 'notes']


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--seed', type=int, default=7)
    args = p.parse_args(argv)
    run = args.run if args.run.is_absolute() else args.root / args.run
    plan = json.loads((run / 'plan.json').read_text(encoding='utf-8'))
    manifest = json.loads((run / 'sample-manifest.json').read_text(encoding='utf-8'))
    if manifest.get('split') != 'dev':
        raise SystemExit('Refusing: error review export is for dev runs only (PROTOCOL-V2 §4).')
    preds = [json.loads(l) for l in (run / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    refs = references_by_id(args.root, {'dev'})
    rows = {r['example_id']: r for r in load_prepared(args.root, 'dev')}
    disagreements = []
    for pr in preds:
        if pr.get('status') != 'Completed':
            continue
        labels = refs[pr['example_id']]['labels']
        pred = pr['verdict'] == 'met'
        if all(l == pred for l in labels):
            continue
        row = rows[pr['example_id']]
        disagreements.append({
            'example_id': pr['example_id'], 'topic': pr.get('topic', ''), 'criterion': row['rubric']['text'],
            'judge_verdict': pr['verdict'], 'judge_confidence': pr.get('confidence'),
            'judge_analysis': pr.get('analysis', ''),
            'physician_labels': ' '.join('met' if l else 'not_met' for l in labels),
            'physicians_disagree': len(set(labels)) > 1,
            'answer_excerpt': row['case']['candidate_output'][:600], 'error_type': '', 'notes': ''})
    random.Random(args.seed).shuffle(disagreements)
    chosen = disagreements[:args.limit]
    with (run / 'error-review.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(chosen)
    md = [f"# Error review — {plan.get('prompt_version')} / {plan.get('model')}", '',
          f'{len(chosen)} of {len(disagreements)} disagreements (any physician differs from the judge).', '',
          'Allowed error_type values: ' + ', '.join(ERROR_TYPES), '',
          'Code each row yourself; physicians_disagree=True rows are usually not judge errors.', '']
    for d in chosen:
        md += [f"## {d['example_id']} · {d['topic']}", f"**Criterion:** {d['criterion']}",
               f"**Judge:** {d['judge_verdict']} ({d['judge_confidence']}) — {d['judge_analysis']}",
               f"**Physicians:** {d['physician_labels']}", '', '> ' + d['answer_excerpt'].replace('\n', '\n> '), '',
               'error_type: ____    notes: ____', '']
    (run / 'error-review.md').write_text('\n'.join(md), encoding='utf-8')
    print(f'Exported {len(chosen)} rows to {run / "error-review.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
