import json
import shutil
from pathlib import Path

import httpx
import pytest

from medical_jury.dspy_judge import DspyJudge, build_program, program_sha256
from medical_jury.evaluation import freeze as F
from medical_jury.judge_v2 import DeepSeekBackend, ScriptedBackend

from v2_fixtures import load_script, make_project

REPO = Path(__file__).resolve().parents[1]
DSPY_OK = json.dumps({'reasoning': 'Answer gives weight-based dosing.', 'verdict': 'met', 'confidence': 0.8})


def test_dspy_judge_reasoning_first_and_record_contract(case, rubric, store):
    seen = []

    def responder(messages):
        seen.append(messages)
        return DSPY_OK
    rec = DspyJudge(build_program(), ScriptedBackend(responder), store).judge(case, rubric, 'J')
    assert rec['status'] == 'Completed' and rec['verdict'] == 'met' and rec['p_met'] == 0.8
    system = seen[0][0]['content']
    assert system.index('`reasoning`') < system.index('`verdict`')
    assert rec['prompt_version'].startswith('jury-v2-dspy-cot-2:')


def test_dspy_judge_failure_is_a_record(case, rubric, store):
    rec = DspyJudge(build_program(), ScriptedBackend(lambda m: 'not json at all'), store).judge(case, rubric, 'J')
    assert rec['status'] == 'Failed'


def test_dspy_calls_go_through_cost_ledger(case, rubric, settings, store):
    def handler(request):
        return httpx.Response(200, json={'usage': {'prompt_tokens': 800, 'completion_tokens': 60},
                                         'choices': [{'finish_reason': 'stop', 'message': {'content': DSPY_OK}}]})
    backend = DeepSeekBackend(settings, store, 5.0, client=httpx.Client(transport=httpx.MockTransport(handler)),
                              ceiling_cny=30)
    DspyJudge(build_program(), backend, store).judge(case, rubric, 'J-led')
    with store.connect() as db:
        rows = db.execute("SELECT state FROM costs WHERE job_id='J-led'").fetchall()
    assert rows and all(r['state'] == 'settled' for r in rows)


def test_optimize_dspy_offline_bootstrap(tmp_path):
    root = make_project(tmp_path)
    opt = load_script('optimize_dspy')
    assert opt.main(['--root', str(root), '--judge', 'mock', '--optimizer', 'bootstrap',
                     '--n-train', '12', '--n-val', '6']) == 0
    out = next((root / 'reports').glob('dspy-opt-mock-*'))
    summary = json.loads((out / 'summary.json').read_text())
    prog = build_program()
    prog.load(str(out / 'program.json'))
    assert program_sha256(prog) == summary['program_sha256']
    assert summary['plan']['split'] == 'dev'
    assert (out / 'threshold.json').exists() and 'threshold_fit_on_dev_val' in summary


def test_optimize_dspy_paid_requires_authorization(tmp_path):
    root = make_project(tmp_path)
    (root / 'jury-v2.json').write_text(json.dumps({'judges': [{'name': 'ds', 'kind': 'deepseek'}]}))
    assert load_script('optimize_dspy').main(['--root', str(root), '--judge', 'ds', '--n-train', '10', '--n-val', '5']) == 0
    out = next((root / 'reports').glob('dspy-opt-ds-*'))
    assert (out / 'plan.json').exists() and not (out / 'program.json').exists()


def frozen_project(tmp_path):
    root = make_project(tmp_path, n_groups=40, splits=('dev', 'validation', 'test'))
    for rel in ('docs/project/PROTOCOL-V2.md', 'scripts/run_validation_once.py', 'scripts/freeze_protocol.py'):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, root / rel)
    shutil.copytree(REPO / 'src/medical_jury', root / 'src/medical_jury',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (root / 'jury-v2.json').write_text(json.dumps({'judges': [{'name': n, 'kind': 'mock'} for n in ('a', 'b', 'c')]}))
    load_script('freeze_protocol').main(['--root', str(root), '--judges', 'a,b,c', '--natural-groups', '30',
                                         '--panel', '3x10x3'])
    return root, next((root / 'reports/protocol-freeze').glob('*/freeze.json'))


def test_freeze_run_once_lock_and_test_gate(tmp_path):
    root, fz = frozen_project(tmp_path)
    run = load_script('run_validation_once')
    assert run.main([str(fz), '--root', str(root), '--kind', 'natural']) == 0
    with pytest.raises(SystemExit, match='one-shot'):
        run.main([str(fz), '--root', str(root), '--kind', 'natural'])
    with pytest.raises(SystemExit, match='--final'):
        run.main([str(fz), '--root', str(root), '--kind', 'natural'], split='test')
    with pytest.raises(SystemExit, match='Run both validation kinds'):
        run.main([str(fz), '--root', str(root), '--kind', 'natural', '--final'], split='test')
    assert run.main([str(fz), '--root', str(root), '--kind', 'panel']) == 0
    assert run.main([str(fz), '--root', str(root), '--kind', 'natural', '--final'], split='test') == 0
    summary = json.loads(next((root / 'reports/frozen-runs').glob('*-validation-natural/summary.json')).read_text())
    assert summary['scores']['planned_items'] == len(json.loads(fz.read_text())['manifests']['validation']['natural']['example_ids'])


def test_frozen_run_refuses_after_any_change(tmp_path):
    root, fz = frozen_project(tmp_path)
    protocol = root / 'docs/project/PROTOCOL-V2.md'
    protocol.write_text(protocol.read_text() + '\nchanged')
    with pytest.raises(SystemExit, match='protocol'):
        load_script('run_validation_once').main([str(fz), '--root', str(root), '--kind', 'natural'])
    protocol.write_text(protocol.read_text().removesuffix('\nchanged'))
    code = root / 'src/medical_jury/judge_v2.py'
    code.write_text(code.read_text() + '\n# edit\n')
    assert any('code tree' in p for p in F.verify(root, json.loads(fz.read_text())))


def test_freeze_self_hash_and_previously_used_groups(tmp_path):
    root = make_project(tmp_path)
    sealed = F.seal({'a': 1})
    assert F.verify.__name__ and sealed['freeze_sha256']
    folder = root / 'data/private/iteration-15-physician-blind'
    folder.mkdir(parents=True)
    (folder / 'refs.jsonl').write_text('{"group_id": "HBG-abc123", "x": 1}\n')
    assert 'HBG-abc123' in F.previously_used_groups(root)


def test_freeze_rejects_threshold_for_jury_and_mismatched_ds(tmp_path):
    root = make_project(tmp_path, splits=('dev', 'validation', 'test'))
    (root / 'docs/project').mkdir(parents=True)
    shutil.copy(REPO / 'docs/project/PROTOCOL-V2.md', root / 'docs/project/PROTOCOL-V2.md')
    (root / 'jury-v2.json').write_text(json.dumps({'judges': [{'name': n, 'kind': 'mock'} for n in ('a', 'b')]}))
    fp = load_script('freeze_protocol')
    with pytest.raises(SystemExit, match='threshold'):
        fp.main(['--root', str(root), '--judges', 'a,b', '--threshold', '0.5'])
    (root / 'ds.json').write_text(json.dumps({'prior_met': .7, 'judges': {'a': {}, 'z': {}}}))
    with pytest.raises(SystemExit, match='ds-params'):
        fp.main(['--root', str(root), '--judges', 'a,b', '--ds-params', 'ds.json'])


def test_gate_sanity_check_runs_offline(tmp_path):
    root = make_project(tmp_path, n_groups=40, raters=3, splits=('dev', 'validation'))
    assert load_script('gate_sanity_check').main(['--root', str(root), '--bootstrap', '50']) == 0
    summary = json.loads((root / 'reports/gate-sanity/summary.json').read_text())
    assert len(summary['dev']) == 3 and 'non_inferior' in summary['dev'][0]['checks']
