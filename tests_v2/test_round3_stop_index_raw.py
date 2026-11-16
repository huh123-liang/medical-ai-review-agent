"""Round-3 'A' fixes: F01 stop policy, F02 stop_reason, F03 index verification, F09 raw trace."""
import json

import pytest

from medical_jury.evaluation.data import verify_rater_index
from medical_jury.evaluation.runner import LiveWriter, StopController, run_planned
from medical_jury.judge_v2 import BackendError, Completion, JudgeV2

from v2_fixtures import load_script, make_project

OK = json.dumps({'analysis': 'a', 'verdict': 'met', 'confidence': 0.7, 'evidence': []})


class Seq:
    """Synthetic backend: plays a script of outcomes, then succeeds. No network, no clinical meaning."""
    name, model, billable = 'seq', 'SYNTHETIC', False

    def __init__(self, plan):
        self.plan, self.calls = list(plan), 0

    def complete(self, messages, max_tokens, job_id):
        kind = self.plan[self.calls] if self.calls < len(self.plan) else 'ok'
        self.calls += 1
        if kind.startswith('http'):
            raise BackendError('PROVIDER_HTTP_' + kind[4:])
        if kind == 'len':
            return Completion('{"analysis": "cut', 'length', {'completion_tokens': 800}, 'SYNTHETIC')
        if kind == 'bad':
            return Completion('{not json', 'stop', {'completion_tokens': 5}, 'SYNTHETIC')
        return Completion(OK, 'stop', {'completion_tokens': 40}, 'SYNTHETIC')


def run(case, rubric, tmp_path, plan, n=10, **ctl):
    controller = StopController(**ctl)
    backend = Seq(plan)
    judge = JudgeV2(backend, stop=controller.event)
    writer = LiveWriter(tmp_path / f'p-{"-".join(plan) or "none"}.jsonl')
    recs, _ = run_planned(list(range(n)), lambda _: judge.judge(case, rubric, 'J'), id_of=str,
                          writer=writer, controller=controller)
    writer.close()
    return recs, backend, controller


# ---- F01 ------------------------------------------------------------------------------------

@pytest.mark.parametrize('code,reason', [('401', 'auth_or_quota'), ('402', 'auth_or_quota'),
                                         ('403', 'auth_or_quota'), ('429', 'rate_limit')])
def test_auth_quota_and_rate_limit_stop_immediately(case, rubric, tmp_path, code, reason):
    recs, backend, controller = run(case, rubric, tmp_path, ['http' + code])
    assert backend.calls == 1, 'no further provider call after an auth/quota/rate-limit error'
    assert controller.stop_reason == reason
    assert [r['status'] for r in recs[1:]] == ['NotRun'] * 9
    assert all(r['failure_code'] == 'STOPPED_' + reason.upper() for r in recs[1:])


def test_three_consecutive_truncations_stop(case, rubric, tmp_path):
    recs, backend, controller = run(case, rubric, tmp_path, ['len', 'len', 'len'])
    assert backend.calls == 3 and controller.stop_reason == 'consecutive_failures'
    assert controller.reasons['consecutive_failures']['codes'] == ['INCOMPLETE_MODEL_OUTPUT'] * 3


def test_success_resets_the_streak(case, rubric, tmp_path):
    recs, backend, controller = run(case, rubric, tmp_path, ['bad', 'bad', 'ok', 'bad', 'bad', 'ok'])
    assert backend.calls == 10 and controller.stop_reason is None


def test_transient_server_error_counts_toward_streak_not_immediate(case, rubric, tmp_path):
    recs, backend, controller = run(case, rubric, tmp_path, ['http503', 'ok'])
    assert backend.calls == 10 and controller.stop_reason is None
    recs, backend, controller = run(case, rubric, tmp_path, ['http500', 'http502', 'http503'])
    assert backend.calls == 3 and controller.stop_reason == 'consecutive_failures'


def test_failure_rate_rule_still_applies(case, rubric, tmp_path):
    plan = ['bad', 'ok', 'bad', 'ok', 'bad', 'ok', 'bad', 'ok', 'bad', 'ok', 'bad']
    recs, backend, controller = run(case, rubric, tmp_path, plan, n=20, window=50, max_failures=5)
    assert controller.stop_reason == 'failure_rate_rule' and backend.calls == 11


def test_priority_when_several_reasons_fire():
    c = StopController()
    c.trigger('consecutive_failures', {})
    c.trigger('budget', {})
    assert c.stop_reason == 'budget' and set(c.summary()['stop_reasons']) == {'budget', 'consecutive_failures'}


# ---- F02 ------------------------------------------------------------------------------------

def test_non_budget_stop_is_not_reported_as_budget(case, rubric, tmp_path):
    recs, backend, controller = run(case, rubric, tmp_path, ['bad'] * 10)
    summary = controller.summary()
    assert summary['stop_reason'] == 'consecutive_failures' and summary['budget_stopped'] is False


def _patched_dev_run(monkeypatch, root, plan, extra=()):
    mod = load_script('run_dev_v2')
    backend = Seq(plan)
    monkeypatch.setattr(mod, 'make_backend', lambda args, settings, store: backend)
    (root / 'jury-v2.json').write_text(json.dumps({'judges': [{'name': 'synthetic', 'kind': 'mock'}]}))
    out = root / f"reports/dev-v2-synthetic-{len(list((root / 'reports').glob('*'))) if (root / 'reports').exists() else 0}"
    rc = mod.main(['--root', str(root), '--backend', 'panel', '--judge', 'synthetic', '--n-groups', '10',
                   '--out', str(out), *extra])
    return rc, out, backend


def test_dev_runner_reports_stop_reason_and_keeps_denominator(tmp_path, monkeypatch):
    root = make_project(tmp_path)
    rc, out, backend = _patched_dev_run(monkeypatch, root, ['http401'])
    summary = json.loads((out / 'summary.json').read_text())
    assert rc == 0 and backend.calls == 1
    assert summary['stop_reason'] == 'auth_or_quota' and summary['budget_stopped'] is False
    assert summary['status_counts'] == {'Failed:PROVIDER_HTTP_401': 1, 'NotRun:STOPPED_AUTH_OR_QUOTA': 9}
    assert len((out / 'predictions.jsonl').read_text().splitlines()) == 10


def test_deliberate_rerun_required_for_same_sample(tmp_path, monkeypatch):
    root = make_project(tmp_path)
    _patched_dev_run(monkeypatch, root, ['bad', 'bad', 'bad'])
    # same sample only when used groups are explicitly kept; then a rerun must be deliberate
    rc, out, backend = _patched_dev_run(monkeypatch, root, [], extra=['--include-used-groups'])
    assert rc == 2 and backend.calls == 0 and not (out / 'predictions.jsonl').exists()
    first = sorted((root / 'reports').glob('dev-v2-synthetic-*'))[0]
    rc, out, backend = _patched_dev_run(monkeypatch, root, [], extra=['--rerun-of', str(first)])
    assert rc == 0 and backend.calls == 10
    plan = json.loads((out / 'plan.json').read_text())
    assert plan['rerun_of'] and plan['previous_runs_on_this_sample'][0]['stop_reason'] == 'consecutive_failures'


# ---- F03 ------------------------------------------------------------------------------------

def test_index_verification_detects_tampering(tmp_path):
    root = make_project(tmp_path)
    assert verify_rater_index(root)['ok']
    meta_path = root / 'data/private/physician-rater-index.meta.json'
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({**meta, 'source_sha256': 'other'}))
    assert 'different public source' in ' '.join(verify_rater_index(root)['problems'])
    meta_path.write_text(json.dumps({**meta, 'index_sha256': '0' * 64}))
    assert 'content hash differs' in ' '.join(verify_rater_index(root)['problems'])
    meta_path.unlink()
    assert not verify_rater_index(root)['ok']


def test_index_verification_checks_planned_ids_and_split(tmp_path):
    root = make_project(tmp_path)
    assert verify_rater_index(root, ids=['HBX-0-0'], split='dev')['ok']
    res = verify_rater_index(root, ids=['HBX-0-0', 'nope'], split='dev')
    assert not res['ok'] and res['missing_ids'] == 1
    assert not verify_rater_index(root, ids=['HBX-0-0'], split='validation')['ok']


def test_dev_runner_refuses_tampered_index_before_any_call(tmp_path, monkeypatch):
    root = make_project(tmp_path)
    meta_path = root / 'data/private/physician-rater-index.meta.json'
    meta_path.write_text(json.dumps({**json.loads(meta_path.read_text()), 'index_sha256': 'x'}))
    rc, out, backend = _patched_dev_run(monkeypatch, root, [])
    assert rc == 2 and backend.calls == 0


def test_dev_summary_records_index_hash(tmp_path, monkeypatch):
    root = make_project(tmp_path)
    rc, out, _ = _patched_dev_run(monkeypatch, root, [])
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['scoring_index_sha256'] == verify_rater_index(root)['index_sha256']


# ---- F09 ------------------------------------------------------------------------------------

def test_raw_length_and_hash_recorded_on_every_branch_with_content(case, rubric):
    import hashlib
    ok = JudgeV2(Seq([])).judge(case, rubric, 'J')
    cut = JudgeV2(Seq(['len'])).judge(case, rubric, 'J')
    bad = JudgeV2(Seq(['bad'])).judge(case, rubric, 'J')
    for rec, text in ((ok, OK), (cut, '{"analysis": "cut'), (bad, '{not json')):
        assert rec['raw_chars'] == len(text)
        assert rec['raw_sha256'] == hashlib.sha256(text.encode()).hexdigest()
        assert 'raw_text' not in rec and 'content' not in rec
    assert cut['usage'] == {'completion_tokens': 800} and cut['finish_reason'] == 'length'
    assert ok['finish_reason'] == 'stop'
    failed_transport = JudgeV2(Seq(['http500'])).judge(case, rubric, 'J')
    assert 'raw_chars' not in failed_transport


def test_dev_summary_reports_output_lengths(tmp_path, monkeypatch):
    root = make_project(tmp_path)
    rc, out, _ = _patched_dev_run(monkeypatch, root, ['len'])
    lengths = json.loads((out / 'summary.json').read_text())['output_length']
    assert lengths['truncated_finish_length'] == 1 and lengths['completion_tokens']['max'] == 800
    assert lengths['finish_reasons'] == {'length': 1, 'stop': 9} and lengths['incomplete_non_stop'] == 1
    assert lengths['max_output_tokens'] == 800
