"""Freeze the evaluated system + validation/test samples BEFORE any held-out run (PROTOCOL-V2 §5).

Example (single DSPy judge with a dev-fitted threshold):
    python scripts/freeze_protocol.py --judges deepseek-pro \
        --program deepseek-pro=reports/dspy-opt-deepseek-pro-<t>/program.json --threshold 0.5
Example (three-judge jury with Dawid-Skene parameters fitted on dev votes):
    python scripts/freeze_protocol.py --judges deepseek-pro,qwen-local,third \
        --ds-params reports/ablation-jury-<t>/ds-params.json
Writes reports/protocol-freeze/<freeze_id>/freeze.json. Afterwards: commit, then
    git tag protocol-v2-<freeze_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from medical_jury.evaluation import freeze as F  # noqa: E402
from medical_jury.evaluation.data import load_prepared, load_rater_index  # noqa: E402
from medical_jury.evaluation.sampling import natural_sample, panel_sample  # noqa: E402
from medical_jury.jury_config import load_panel  # noqa: E402
from medical_jury.judge_v2 import PROMPT_VERSION_V2, SYSTEM_V2  # noqa: E402
from medical_jury.domain import digest  # noqa: E402

PROTOCOL = 'docs/project/PROTOCOL-V2.md'


def rel(root: Path, p: str) -> str:
    path = Path(p)
    path = path if path.is_absolute() else (root / path)
    return path.resolve().relative_to(root.resolve()).as_posix()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--judges', required=True, help='Comma-separated names from jury-v2.json.')
    p.add_argument('--program', action='append', default=[], help='NAME=path/to/program.json (DSPy judge).')
    p.add_argument('--threshold', type=float, default=None, help='Dev-fitted P(met) threshold (single judge only).')
    p.add_argument('--ds-params', default=None, help='Dawid-Skene params from ablation_jury.py (multi-judge).')
    p.add_argument('--natural-groups', type=int, default=600)
    p.add_argument('--natural-items-per-group', type=int, default=2,
                   help='2 x 600 groups ~ 1200 items: sized so a physician-level judge passes the 0.10 margin (gate_sanity_check.py).')
    p.add_argument('--panel', default='10x40x3', help='annotators x items x max-items-per-group')
    p.add_argument('--seed', type=int, default=2026102599)
    p.add_argument('--alt-epsilon', type=float, default=0.2)
    p.add_argument('--non-inferiority-margin', type=float, default=0.10)
    args = p.parse_args(argv)
    root = args.root
    names = [n.strip() for n in args.judges.split(',') if n.strip()]
    panel = {j['name']: j for j in load_panel(root)}
    missing = [n for n in names if n not in panel]
    if missing:
        raise SystemExit(f'Unknown judges: {missing}')
    if args.threshold is not None and len(names) != 1:
        raise SystemExit('A threshold can only be frozen for a single judge (juries use their aggregator).')
    if len(names) > 1 and not args.ds_params:
        print('Note: no --ds-params -> the jury will use majority vote (ties count as undecided).')
    programs = dict(item.split('=', 1) for item in args.program)
    if set(programs) - set(names):
        raise SystemExit('--program given for a judge that is not on the jury.')
    artifacts = {'protocol': PROTOCOL}
    artifacts.update({f'program:{n}': rel(root, path) for n, path in programs.items()})
    if args.ds_params:
        artifacts['ds_params'] = rel(root, args.ds_params)
        ds = json.loads((root / artifacts['ds_params']).read_text(encoding='utf-8'))
        if set(ds['judges']) != set(names):
            raise SystemExit('ds-params judges do not match the frozen jury.')
    used = F.previously_used_groups(root)
    manifests = {}
    k, n, g = (int(x) for x in args.panel.split('x'))
    for split in ('validation', 'test'):
        rows = load_prepared(root, split)
        _, natural = natural_sample(rows, min(args.natural_groups, len({r['case']['group_id'] for r in rows
                                                                        if r['case']['group_id'] not in used})),
                                    args.seed, split, exclude_groups=used,
                                    items_per_group=args.natural_items_per_group)
        ids, pan = panel_sample(load_rater_index(root, with_labels=False, splits={split}), k, n, args.seed, split,
                                exclude_groups=used, max_items_per_group=g)
        manifests[split] = {'natural': natural, 'panel': pan}
    freeze = {
        'protocol_version': 'PROTOCOL-V2', 'created_at': datetime.now(timezone.utc).isoformat(),
        'jury': [panel[nm] for nm in names], 'programs': programs,
        'prompt_version': PROMPT_VERSION_V2, 'system_prompt_sha256': digest(SYSTEM_V2),
        'decision': {'threshold': args.threshold, 'aggregator': 'dawid_skene_frozen' if args.ds_params else
                     ('single' if len(names) == 1 else 'majority')},
        'acceptance': {'non_inferiority_margin': args.non_inferiority_margin, 'alt_epsilon': args.alt_epsilon,
                       'alt_alpha': 0.05, 'alt_min_items_per_annotator': 30, 'max_failure_share': 0.05},
        'excluded_previously_used_groups': len(used),
        'manifests': manifests, 'artifacts': artifacts,
        'artifact_sha256': F.artifact_hashes(root, artifacts),
        'code_tree_sha256': F.code_tree_sha(root), 'git_commit': F.git_commit(root),
    }
    freeze = F.seal(freeze)
    freeze_id = freeze['freeze_sha256'][:12]
    out = root / 'reports/protocol-freeze' / freeze_id
    out.mkdir(parents=True, exist_ok=False)
    (out / 'freeze.json').write_text(json.dumps(freeze, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'freeze_id': freeze_id, 'jury': names, 'decision': freeze['decision'],
                      'validation_natural_items': len(manifests['validation']['natural']['example_ids']),
                      'validation_panel_items': len(manifests['validation']['panel']['example_ids']),
                      'excluded_groups': len(used), 'git_commit': freeze['git_commit']}, indent=2))
    print(f'Frozen -> {out.relative_to(root)}/freeze.json\nNext: git add/commit, then `git tag protocol-v2-{freeze_id}`.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
