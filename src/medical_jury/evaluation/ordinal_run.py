"""Shared logic for the ordinal validation runs of the v3 agent (PROTOCOL-V3-ORDINAL).

Order of operations is enforced by the scripts that use this module:
  load inputs (no labels) -> score every item -> seal predictions (sha256) -> only then join labels -> metrics.
Validation and test runs additionally need a freeze record whose fields equal the current system exactly,
and each (dataset, split) can be run once (a sentinel file is created before the first call).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..domain import CaseInput, digest
from ..v3.evaluator import (EVALUATOR_PROMPT_VERSION, GROUP_TOKENS_ANCHORS_ONLY, MAX_OUTPUT_TOKENS_V3,
                            DimensionEvaluator, system_prompt)
from ..v3.locator import LOCATOR_VERSION
from ..v3.schema import ScoringProfile, load_profile
from .freeze import code_tree_sha, file_sha

PROTOCOL = 'docs/project/PROTOCOL-V3-ORDINAL.md'
PROFILE_FILES = {'bionlp-safety-3level': 'kb/v3/profile-bionlp-safety-3level.json',
                 'medexpert-severity-5level': 'kb/v3/profile-medexpert-severity-5level.json'}
DATASETS = {'bionlp-trec': {'profile': 'bionlp-safety-3level', 'splits': ('dev', 'validation'),
                            'primary': ['missing_content', 'inappropriate_content', 'harm_extent', 'empathy']},
            'medexpert': {'profile': 'medexpert-severity-5level', 'splits': ('test',),
                          'primary': ['factuality', 'omission']}}
RUN_SCRIPTS = ('scripts/run_ordinal_v3.py', 'scripts/freeze_v3.py', 'scripts/prepare_ordinal_v3.py',
               'scripts/aggregate_ordinal_v3.py')
CALL_MODES = ('single', 'per_dimension')


def load_items(root: Path, dataset: str, split: str) -> list[dict]:
    if split not in DATASETS[dataset]['splits']:
        raise ValueError(f'{dataset} has no {split!r} split (protocol roles: {DATASETS[dataset]["splits"]}).')
    path = root / 'data' / 'prepared-v3' / dataset / f'{split}.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    for r in rows:
        if set(r) != {'item_id', 'group_id', 'dataset', 'strata', 'request', 'split'}:
            raise ValueError(f'{path.name}: unexpected fields {sorted(r)} (labels must never be in inputs).')
    return rows


def subset(rows: list[dict], n: int | None, seed: str, complement: bool = False) -> list[dict]:
    """Whole question groups in seeded hash order until at least n items (label-blind, reproducible).

    ``complement=True`` returns the other items instead (the groups NOT taken), so an already inspected dev
    subset (dev-A) and the unseen remainder (dev-B) are defined by the same n and seed."""
    if not n or n >= len(rows):
        if complement:
            raise ValueError('complement needs an n smaller than the split.')
        return rows
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r['group_id'], []).append(r)
    out: list[dict] = []
    for g in sorted(groups, key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest()):
        if len(out) >= n:
            break
        out += groups[g]
    if not complement:
        return out
    taken = {r['group_id'] for r in out}
    return [r for r in rows if r['group_id'] not in taken]


def part_of(rows: list[dict], dataset: str, split: str, part: int, parts: int) -> list[dict]:
    """Deterministic, label-blind partition of a held-out split into ``parts`` batches of whole groups
    (so a run that cannot fit one budget reservation can be split without choosing items)."""
    if parts < 1 or not 1 <= part <= parts:
        raise ValueError('need 1 <= part <= parts')
    if parts == 1:
        return rows
    from ..datasets.ordinal_v3 import assign_splits
    names = {f'part{k}': 1 / parts for k in range(1, parts + 1)}
    fix = 1 - sum(list(names.values())[:-1])
    names[f'part{parts}'] = fix
    assigned = assign_splits(rows, names, f'ordinal-v3-parts:{dataset}:{split}:{parts}')
    return [r for r in rows if assigned[r['group_id']] == f'part{part}']


def profile_for(root: Path, dataset: str) -> ScoringProfile:
    return load_profile(root / PROFILE_FILES[DATASETS[dataset]['profile']])


def max_tokens_for(call_mode: str, profile: ScoringProfile) -> int:
    return MAX_OUTPUT_TOKENS_V3 if call_mode == 'per_dimension' else \
        make_evaluator(None, None, None).group_max_tokens(len(profile.dimensions))


def make_evaluator(backend, store, stop, prompt_version: str = EVALUATOR_PROMPT_VERSION) -> DimensionEvaluator:
    """Evaluator for the anchor-only ordinal profiles (smaller grouped output allowance)."""
    import threading
    return DimensionEvaluator(backend, store, max_tokens=MAX_OUTPUT_TOKENS_V3, stop=stop or threading.Event(),
                              prompt_version=prompt_version, group_tokens_per_dimension=GROUP_TOKENS_ANCHORS_ONLY)


def system_fields(root: Path, judge_spec: dict, call_mode: str, prompt_version: str) -> dict:
    """Everything that defines the evaluated system and its data (protocol section 6)."""
    if call_mode not in CALL_MODES:
        raise ValueError(f'call_mode must be one of {CALL_MODES}')
    grouped = call_mode == 'single'
    return {
        'protocol': PROTOCOL, 'protocol_sha256': file_sha(root / PROTOCOL),
        'judge': {k: judge_spec.get(k) for k in ('name', 'kind', 'model', 'price_in', 'price_out')},
        'temperature': 0, 'call_mode': call_mode, 'prompt_version': prompt_version,
        'system_prompt_sha256': digest(system_prompt(prompt_version, grouped=grouped)),
        'max_output_tokens': {ds: max_tokens_for(call_mode, profile_for(root, ds)) for ds in DATASETS},
        'locator': LOCATOR_VERSION,
        'profiles_sha256': {pid: file_sha(root / rel) for pid, rel in PROFILE_FILES.items()},
        'manifests_sha256': {ds: file_sha(root / 'data' / 'prepared-v3' / ds / 'manifest.json') for ds in DATASETS},
        'inputs_sha256': {f'{ds}/{s}': file_sha(root / 'data' / 'prepared-v3' / ds / f'{s}.jsonl')
                          for ds, spec in DATASETS.items() for s in spec['splits']},
        'code_tree_sha256': code_tree_sha(root),
        'run_scripts_sha256': {s: file_sha(root / s) for s in RUN_SCRIPTS if (root / s).exists()},
    }


def freeze_mismatches(freeze: dict, current: dict) -> list[str]:
    return sorted(k for k in current if freeze.get(k) != current[k])


def score_item(evaluator: DimensionEvaluator, item: dict, profile: ScoringProfile, call_mode: str,
               job_id: str) -> dict:
    """One prediction record (no labels). Completed only if every dimension was scored."""
    case = CaseInput.model_validate(item['request']['case'])
    specs = [(d, [], []) for d in profile.dimensions]
    if call_mode == 'single':
        recs = evaluator.evaluate_group(case, specs, job_id)
    else:
        recs = [evaluator.evaluate(case, d, c, r, job_id) for d, c, r in specs]
    pred = {r['dimension']: (r.get('score') if r['status'] == 'Completed' else None) for r in recs}
    failed = [r for r in recs if r['status'] != 'Completed']
    status = 'Completed' if not failed else ('NotRun' if all(r['status'] == 'NotRun' for r in recs) else 'Failed')
    return {'item_id': item['item_id'], 'group_id': item['group_id'], 'status': status,
            'failure_code': failed[0].get('failure_code') if failed else None, 'pred': pred,
            'dimensions': [{k: r.get(k) for k in ('dimension', 'status', 'failure_code', 'model_score', 'score',
                                                  'confidence', 'usage', 'finish_reason', 'messages_sha256',
                                                  'prompt_version', 'call_mode', 'latency_s', 'raw_sha256',
                                                  'raw_chars')} for r in recs],
            'analysis': {r['dimension']: r.get('analysis') for r in recs}}


def join_labels(predictions: list[dict], labels_path: Path) -> list[dict]:
    labels = {}
    for line in labels_path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            labels[row['item_id']] = row
    out = []
    for p in predictions:
        lab = labels[p['item_id']]
        out.append({'item_id': p['item_id'], 'group_id': p['group_id'], 'status': p['status'],
                    'pred': p.get('pred') or {}, 'raters': lab['raters'], 'scores': lab['scores']})
    return out
