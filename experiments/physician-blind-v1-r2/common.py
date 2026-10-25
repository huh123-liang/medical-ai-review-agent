"""Frozen public physician meta-evaluation helpers; no label lookup or model calls here."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = ROOT / 'reports/iteration-15-physician-blind-r2'
PRIVATE = ROOT / 'data/private/iteration-15-physician-blind-r2'
ARM_A = 'A_current_scoped'
ARM_B = 'B_candidate_zero'
TOPICS = ('emergency_referrals', 'communication', 'context_seeking', 'health_data_tasks',
          'hedging', 'global_health', 'complex_responses')
SEED = 1520261025
PER_TOPIC = 2
ROUND_LIMIT = 8.0
MAX_CANDIDATE_OUTPUT = 2400


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def write(path, value, *, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    else:
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temp.replace(path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_module():
    return load('physician_trial_candidate', ROOT / 'experiments/prompt-candidate-v1/candidate.py')


def protected():
    sys.path.insert(0, str(ROOT / 'scripts'))
    from verify_iteration12_shadow import protected as old
    result = old()
    for folder in ['src', 'tests', 'docs', 'scripts', 'experiments', 'examples']:
        for p in sorted((ROOT / folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and not p.is_relative_to(HERE):
                result[str(p.relative_to(ROOT))] = sha(p)
    for name in ['iteration-13-methods-and-acceptance', 'iteration-14-candidate-prompts']:
        for p in sorted((ROOT / 'reports' / name).rglob('*')):
            if p.is_file():
                result[str(p.relative_to(ROOT))] = sha(p)
    for p in sorted((ROOT / 'reports/iteration-15-physician-blind').rglob('*')):
        if p.is_file():
            result[str(p.relative_to(ROOT))] = sha(p)
    for p in ROOT.iterdir():
        if p.is_file():
            result[str(p.relative_to(ROOT))] = sha(p)
    result['data/public/healthbench-meta.jsonl'] = sha(ROOT / 'data/public/healthbench-meta.jsonl')
    return result


def source_code():
    return {p.name: sha(p) for p in sorted(HERE.iterdir()) if p.is_file()}


def topic(row):
    return next((t for t in TOPICS if row['category'].startswith('cluster:' + t + '_')), None)


def select_rows(rows, excluded=()):
    """Outcome-blind: no references, labels, prior predictions or response-quality filtering."""
    pool = {t: [] for t in TOPICS}
    for row in rows:
        t = topic(row)
        if t and row['case']['split'] == 'validation' and row['case']['group_id'] not in excluded:
            pool[t].append(row)
    rng = random.Random(SEED)
    for t in TOPICS:
        pool[t].sort(key=lambda r: r['example_id'])
        rng.shuffle(pool[t])
    chosen, used = [], set(excluded)
    for _ in range(PER_TOPIC):
        for t in TOPICS:
            row = next((r for r in pool[t] if r['case']['group_id'] not in used), None)
            if row is None:
                raise ValueError('INSUFFICIENT_UNSEEN_VALIDATION_GROUPS')
            chosen.append(row)
            used.add(row['case']['group_id'])
    return chosen


def candidate_input(row):
    c = candidate_module()
    text = row['rubric']['text']
    # Preserve the WHOLE original rubric as one item. No LLM/researcher clinical decomposition.
    # The original text can itself be compound; this is NOT per-atomic-obligation validation.
    obligations = [{'id': 'O1', 'start_char': 0, 'end_char': len(text), 'quote': text}]
    return c.legacy.prepare_input(row['case']['context'], row['case']['candidate_output'], text, obligations,
                                   manifest_status='research_draft', source_texts=(), task_hint=None)


def candidate_messages(row):
    """Evaluation-only assembly of frozen B1 instructions/schema, explicitly validation split.

    Do NOT call old dev-only builder while lying about split. Do NOT modify its gate.
    This entry point has no examples, learning, memory writes or runtime API registration.
    """
    if row['case']['split'] != 'validation':
        raise ValueError('EVALUATION_ONLY_VALIDATION_SPLIT')
    c = candidate_module()
    inputs = candidate_input(row)
    c.validate_input(inputs)
    system = (ROOT / 'experiments/prompt-candidate-v1/system.md').read_text(encoding='utf-8')
    system += '\nCANDIDATE_SCHEMA\n' + c.canonical(c.SCHEMA)
    payload = {'protocol_version': c.VERSION, 'demonstrations': [], 'target': inputs}
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': c.canonical(payload)}]
    if len(c.canonical(messages).encode('utf-8')) > c.MAX_MESSAGE_BYTES:
        raise ValueError('PROMPT_BYTE_LIMIT')
    return messages


def check_freeze():
    f = read(REPORT / 'freeze.json')
    assert source_code() == f['new_code']
    assert protected() == f['protected_files']
    manifest = read(REPORT / 'manifest.json')
    expected = manifest.pop('manifest_hash')
    assert digest(manifest) == expected
    assert sha(REPORT / 'inputs.json') == manifest['inputs_sha256']
    assert sha(REPORT / 'schedule.json') == manifest['schedule_sha256']
    assert sha(PRIVATE / 'references.json') == manifest['sealed_reference_sha256']
    # Hash checks do not expose labels to a predictor.
    manifest['manifest_hash'] = expected
    return f, manifest
