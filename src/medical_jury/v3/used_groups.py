"""HealthBench dev groups that any earlier PAID run has already shown to a model or whose labels were read.

Fresh v3 experiments exclude all of them, so a new result never rests on material that was already inspected.
Sources (label-free; only group ids are read):
* dev-v2 runs that started with a real judge (``reports/dev-v2-*/sample-manifest.json``);
* the legacy database and iteration review packs (``freeze.previously_used_groups``);
* v3 experiments that write a ``sample-manifest.json`` with ``group_ids`` (rag ablation, selective evaluation,
  router v2 ...), whether or not they finished.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..evaluation.freeze import previously_used_groups

V3_EXPERIMENT_GLOBS = ('rag-ablation-v3-*', 'selective-v3-*', 'router-v3b-*')


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def used_dev_groups_v3(root: Path) -> tuple[set[str], list[str]]:
    groups: set[str] = set(previously_used_groups(root))
    sources = ['legacy database / iteration packs']
    reports = root / 'reports'
    for plan_file in sorted(reports.glob('dev-v2-*/plan.json')):
        plan, man = _read(plan_file), _read(plan_file.parent / 'sample-manifest.json')
        if not plan or not man or plan.get('backend') == 'mock':
            continue
        if not (plan_file.parent / 'predictions.partial.jsonl').exists():
            continue
        groups.update(man.get('group_ids', []))
        sources.append(str(plan_file.parent.relative_to(root)))
    for pattern in V3_EXPERIMENT_GLOBS:
        for man_file in sorted(reports.glob(f'{pattern}/sample-manifest.json')):
            man = _read(man_file)
            if man:
                groups.update(man.get('group_ids', []))
                sources.append(str(man_file.parent.relative_to(root)))
    return groups, sources
