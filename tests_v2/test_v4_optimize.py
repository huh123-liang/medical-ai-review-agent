"""v4 DSPy (GEPA) instruction optimisation: offline tests only."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import dspy
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from medical_jury.judge_v2 import BackendError, ScriptedBackend  # noqa: E402
from medical_jury.testbed.build import build_cases  # noqa: E402
from medical_jury.v4 import optimize as OP  # noqa: E402
from medical_jury.v4.criteria import load_library  # noqa: E402
from medical_jury.v4.mocks import keyword_check, oracle_responder, plain_document  # noqa: E402
from medical_jury.v4.pipeline import ReviewerV4  # noqa: E402

from test_v4_pipeline import _project  # noqa: E402

logging.getLogger('dspy').setLevel(logging.ERROR)


@pytest.fixture(scope='module')
def examples():
    cases = [c for c in build_cases(150, 20261112)[0] if c['split'] == 'train'][:14]
    docs = {(c['patient_summary'], plain_document(c)): c for c in cases}
    rev = ReviewerV4(ScriptedBackend(oracle_responder(docs)), load_library(ROOT))
    return [OP.make_example(c, rev.review(c['patient_summary'], plain_document(c), c['case_id']), plain_document(c))
            for c in cases]


def test_adapter_refuses_demos_and_keeps_pipeline_format():
    sig = OP.CheckSignature.with_instructions('INSTR')
    msgs = OP.V4Adapter().format(sig, [], {'payload': '{"a": 1}'})
    assert msgs == [{'role': 'system', 'content': 'INSTR'}, {'role': 'user', 'content': '{"a": 1}'}]
    with pytest.raises(ValueError):
        OP.V4Adapter().format(sig, [{'payload': 'x', 'review': 'y'}], {'payload': 'z'})


def test_score_review_feedback(examples):
    clean = next(e for e in examples if not e.golds)
    ok = json.dumps({'criteria': [{'id': i['id'], 'status': 'met'} for i in clean.criteria_items], 'other_findings': []})
    assert OP.score_review(clean, ok)[0] == 1.0
    bad = json.loads(ok)
    bad['other_findings'] = [{'category': 'dose_error', 'severity': 'severe', 'entity': 'x', 'issue': 'made up'}]
    assert OP.score_review(clean, json.dumps(bad))[0] == 0.0
    assert OP.score_review(clean, 'not json')[0] == 0.0


def test_capped_backend_refuses_before_sending():
    sent = []
    cb = OP.CappedBackend(ScriptedBackend(lambda m: sent.append(1) or '{}'), cap=2)
    cb.complete([], 10, 'j')
    cb.complete([], 10, 'j')
    with pytest.raises(BackendError):
        cb.complete([], 10, 'j')
    assert len(sent) == 2 and cb.refused == 1


def test_leak_check():
    doc = 'one two three four five six seven eight nine ten eleven twelve thirteen'
    assert OP.leak_check('Review carefully. ' + doc, [doc], baseline='x' * 1000)
    assert OP.leak_check('x' * 3000, [], baseline='x' * 1000)
    assert OP.leak_check(OP.SYSTEM_CHECK, [doc]) == []


def test_gepa_adopts_only_an_improving_instruction_within_caps(examples):
    seen = set()

    def task(messages):
        seen.add(messages[0]['content'][:12])
        payload = json.loads(messages[1]['content'])
        out = keyword_check(payload)
        if messages[0]['content'].startswith('STOP-AWARE') and 'stop your blood pressure' in payload['document'].lower():
            out['other_findings'].append({'category': 'unsafe_instruction', 'severity': 'moderate',
                                          'entity': 'stopping blood pressure medicine', 'issue': 'stop', 'quote': None})
        return json.dumps(out)
    reflect = ScriptedBackend(lambda m: json.dumps({'new_instruction': 'STOP-AWARE ' + OP.SYSTEM_CHECK}))
    res = OP.optimise_check(examples[:10], examples[10:], ScriptedBackend(task), reflect, 'J',
                            max_task_calls=70, max_reflection_calls=8)
    assert res['changed'] and res['instruction'].startswith('STOP-AWARE')
    assert res['calls']['task'] <= 70 and res['calls']['reflection'] <= 8
    assert res['guard_problems'] == []


def test_optimize_driver_with_mock_end_to_end(tmp_path, capsys):
    import optimize_v4 as OV
    import run_review_v4 as RV
    cases = build_cases(150, 20261112)[0]
    train = [c for c in cases if c['split'] == 'train']
    pick = {c['case_id']: c for c in train[:30] + [c for c in train if int(c['patient']['patient_id'][3:]) >= 35][:20]}
    root = _project(tmp_path, list(pick.values()))
    assert RV.main(['run', '--root', str(root), '--split', 'train', '--threads', '1']) == 0
    run_dir = next((root / 'reports').glob('review-v4-*'))
    assert OV.main(['--root', str(root), '--train-run-dir', str(run_dir), '--max-task-calls', '80',
                    '--max-reflection-calls', '5']) == 0
    res = json.loads(next((root / 'reports').glob('review-v4-dspy-*/optimised-check.json')).read_text(encoding='utf-8'))
    assert res['offline_mock'] is True and res['usable'] is False
    assert all(not c.startswith('TB-01') for c in res['train_cases'] + res['val_cases'])   # train split only
    with pytest.raises(SystemExit):                                       # an offline/unusable prompt is refused
        RV.main(['run', '--root', str(root), '--split', 'train', '--check-prompt-file',
                 str(next((root / 'reports').glob('review-v4-dspy-*/optimised-check.json')))])
