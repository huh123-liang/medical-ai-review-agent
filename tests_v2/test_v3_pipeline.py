"""Step 1 of the v3 build: the single review chain, offline (mock / scripted backends only)."""
import json
import shutil
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from medical_jury.domain import CaseInput
from medical_jury.judge_v2 import BackendError, Completion, ScriptedBackend
from medical_jury.store import Store
from medical_jury.v3.evaluator import (EvalFailure, DimensionEvaluator, apply_rules, build_messages_v3,
                                       mock_responder_v3, parse_v3)
from medical_jury.v3.graph import ReviewPipeline
from medical_jury.v3.retrieval import FilterRetriever, build_query
from medical_jury.v3.schema import Dimension, ReviewRequest, RubricItem, ScoringProfile, load_items, load_profile

from v2_fixtures import load_script

REPO = Path(__file__).resolve().parents[1]
EX = REPO / 'examples/v3'
SRC = {'kind': 'project_draft', 'citation': 'unit-test fixture'}


def profile():
    return load_profile(EX / 'profile-platform-6d.json')


def kb():
    return load_items(EX / 'seed-kb.jsonl')


def request(**case_changes):
    raw = json.loads((EX / 'demo-case.json').read_text())
    raw['case'].update(case_changes)
    return ReviewRequest.model_validate(raw)


def small_profile(lo=0, hi=2):
    anchors = {str(v): f'level {v}' for v in range(lo, hi + 1)}
    dims = [Dimension(key=k, label=k, description=k, scale_min=lo, scale_max=hi, anchors=anchors, source=SRC)
            for k in ('omission', 'harm')]
    return ScoringProfile(profile_id='public-ordinal-test', version='1', description='t', dimensions=dims)


def pipeline(tmp_path, backend=None, profiles=None, items=None, **kw):
    store = Store(tmp_path)
    profiles = profiles or {'platform-6d-1to5': profile()}
    return ReviewPipeline(profiles, FilterRetriever(items if items is not None else kb()),
                          DimensionEvaluator(backend or ScriptedBackend(mock_responder_v3, name='mock'), store),
                          store, tmp_path, checkpointer=MemorySaver(), **kw), store


def wire(checks, score=3, confidence=0.9, analysis='a'):
    return json.dumps({'analysis': analysis, 'checks': checks, 'score': score, 'confidence': confidence})


# ---- schema: provenance and the decisions of 2026-10-25 are enforced ---------------------------------------

def test_disease_items_are_reference_only_until_approved():
    base = dict(item_id='D-1', level='disease', dimension='safety', profiles=['p'], diseases=['x'], text='abc', source=SRC)
    with pytest.raises(ValueError, match='reference-only'):
        RubricItem(role='scored', **base)
    assert RubricItem(role='scored', approved_by='physician:Dr-A', **base).role == 'scored'
    assert RubricItem(role='reference', **base).role == 'reference'


@pytest.mark.parametrize('bad', [
    dict(item_id='R-1', level='general', role='reference', dimension='safety', profiles=['p'], text='abc', source=SRC, cap_on_fail=2),
    dict(item_id='C-1', level='case', role='scored', dimension='safety', profiles=['p'], text='abc', source=SRC),
    dict(item_id='G-1', level='general', role='scored', dimension='safety', profiles=['p'], text='abc', source=SRC, case_id='X'),
    dict(item_id='G-2', level='general', role='scored', dimension='safety', profiles=['p'], text='abc'),  # no source
])
def test_invalid_items_rejected(bad):
    with pytest.raises(ValueError):
        RubricItem(**bad)


def test_dimension_anchors_must_cover_the_whole_scale():
    with pytest.raises(ValueError, match='anchors'):
        Dimension(key='acc', label='a', description='d', scale_min=1, scale_max=5, anchors={'1': 'x', '5': 'y'}, source=SRC)


# ---- retrieval: applicability and the answer-leak guard ------------------------------------------------------

def test_retrieval_filters_by_profile_level_disease_and_case():
    ids = {r.item.item_id for r in FilterRetriever(kb()).retrieve(build_query(request().case, request().meta, profile())).items}
    assert 'D-T2DM-REF-1' in ids and 'D-HTN-REF-1' not in ids and 'C-DEMO-1' in ids
    other = request(case_id='OTHER-CASE')
    ids2 = {r.item.item_id for r in FilterRetriever(kb()).retrieve(build_query(other.case, other.meta, profile())).items}
    assert 'C-DEMO-1' not in ids2


def test_answer_never_influences_retrieval():
    a, b = request(), request(candidate_output='Completely different answer about hypertension and statins.')
    qa, qb = build_query(a.case, a.meta, profile()), build_query(b.case, b.meta, profile())
    assert qa == qb and a.case.candidate_output not in qa.text
    r = FilterRetriever(kb()).retrieve(qa)
    assert r.trace['answer_used_in_query'] is False


def test_mandatory_items_survive_the_per_dimension_cap():
    items = [RubricItem(item_id=f'G-S-{i}', level='general', role='scored', dimension='safety',
                        profiles=['platform-6d-1to5'], text=f'item {i}', source=SRC, mandatory=(i < 3)) for i in range(6)]
    got = FilterRetriever(items, max_per_dimension=2).retrieve(build_query(request().case, request().meta, profile()))
    ids = [r.item.item_id for r in got.items]
    assert {'G-S-0', 'G-S-1', 'G-S-2'} <= set(ids) and got.trace['dropped_by_cap'] == 3


# ---- evaluator: strict contract, rules flag/cap but never repair ---------------------------------------------

def _checks():
    return [i for i in kb() if i.dimension == 'safety' and i.role == 'scored']


def _dim():
    return profile().dimension('safety')


def test_parse_valid_and_quotes_located():
    case = request().case
    raw = wire([{'id': 'G-SAF-1', 'result': 'fail', 'quote': 'Increase metformin to 3 g daily'},
                {'id': 'G-SAF-2', 'result': 'pass', 'quote': 'Plan: Increase metformin'}], score=2)
    out = parse_v3(raw, case, _dim(), _checks())
    tiers = [c['quote_tier'] for c in out['checks']]
    assert out['model_score'] == 2 and tiers == ['exact', 'markup']          # locator-3 tier names


@pytest.mark.parametrize('raw,code', [
    (wire([{'id': 'G-SAF-1', 'result': 'pass', 'quote': ''}]), 'OUTPUT_CHECKS_INCOMPLETE'),
    (wire([{'id': 'G-SAF-1', 'result': 'pass', 'quote': ''}, {'id': 'G-SAF-2', 'result': 'pass', 'quote': ''},
           {'id': 'G-XXX', 'result': 'pass', 'quote': ''}]), 'OUTPUT_UNKNOWN_CHECK'),
    (wire([{'id': 'G-SAF-1', 'result': 'pass', 'quote': ''}, {'id': 'G-SAF-1', 'result': 'fail', 'quote': ''},
           {'id': 'G-SAF-2', 'result': 'pass', 'quote': ''}]), 'OUTPUT_DUPLICATE_CHECK'),
    (wire([{'id': 'G-SAF-1', 'result': 'pass', 'quote': ''}, {'id': 'G-SAF-2', 'result': 'pass', 'quote': ''}], score=6), 'OUTPUT_SCORE_OUT_OF_SCALE'),
    (wire([{'id': 'G-SAF-1', 'result': 'pass', 'quote': ''}, {'id': 'G-SAF-2', 'result': 'pass', 'quote': ''}], score=0), 'OUTPUT_SCORE_OUT_OF_SCALE'),
    (wire([{'id': 'G-SAF-1', 'result': 'pass', 'quote': ''}, {'id': 'G-SAF-2', 'result': 'pass', 'quote': ''}], score=3.5), 'OUTPUT_SCHEMA_INVALID'),
    (json.dumps({'score': 3, 'analysis': 'a', 'checks': [], 'confidence': 0.9}), 'OUTPUT_SCORE_BEFORE_ANALYSIS'),
    ('```json\n{}\n```', 'OUTPUT_MARKDOWN_FENCE'),
    (wire([{'id': 'G-SAF-1', 'result': 'maybe', 'quote': ''}, {'id': 'G-SAF-2', 'result': 'pass', 'quote': ''}]), 'OUTPUT_SCHEMA_INVALID'),
])
def test_parse_rejects_contract_violations(raw, code):
    with pytest.raises(EvalFailure) as e:
        parse_v3(raw, request().case, _dim(), _checks())
    assert e.value.code == code


def test_rules_cap_and_flag_conflicts():
    case = request().case
    parsed = parse_v3(wire([{'id': 'G-SAF-1', 'result': 'fail', 'quote': ''},
                            {'id': 'G-SAF-2', 'result': 'pass', 'quote': ''}], score=5), case, _dim(), _checks())
    out = apply_rules(parsed, _dim(), _checks())
    assert out['score'] == 2 and out['model_score'] == 5
    assert out['rule_adjustments'][0]['item'] == 'G-SAF-1' and 'MANDATORY_FAIL_BUT_MAX_SCORE' in out['score_check_conflicts']
    parsed = parse_v3(wire([{'id': 'G-SAF-1', 'result': 'pass', 'quote': ''},
                            {'id': 'G-SAF-2', 'result': 'not_applicable', 'quote': ''}], score=1), case, _dim(), _checks())
    assert apply_rules(parsed, _dim(), _checks())['score_check_conflicts'] == ['ALL_PASS_BUT_MIN_SCORE']


class Seq:
    name, model, billable = 'seq', 'SYNTHETIC', False

    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def complete(self, messages, max_tokens, job_id):
        self.calls += 1
        kind = self.outcomes.pop(0) if self.outcomes else 'mock'
        if kind.startswith('http'):
            raise BackendError('PROVIDER_HTTP_' + kind[4:])
        if kind == 'len':
            return Completion('{"analysis": "cut', 'length', {'completion_tokens': 1200}, 'SYNTHETIC')
        return Completion(mock_responder_v3(messages), 'stop', {'completion_tokens': 50}, 'SYNTHETIC')


def test_evaluator_failures_are_records_and_stop_prevents_calls():
    case, dim = request().case, _dim()
    ev = DimensionEvaluator(Seq(['http500', 'len']))
    assert ev.evaluate(case, dim, _checks(), [], 'J')['failure_code'] == 'PROVIDER_HTTP_500'
    assert ev.evaluate(case, dim, _checks(), [], 'J')['failure_code'] == 'INCOMPLETE_MODEL_OUTPUT'
    ev.stop.set()
    rec = ev.evaluate(case, dim, _checks(), [], 'J')
    assert rec['status'] == 'NotRun' and ev.backend.calls == 2


def test_reference_notes_are_shown_but_never_checks():
    msgs = build_messages_v3(request().case, _dim(), _checks(), [i for i in kb() if i.item_id == 'D-T2DM-REF-1'])
    payload = json.loads(msgs[1]['content'])
    assert [c['id'] for c in payload['checks']] == ['G-SAF-1', 'G-SAF-2']
    assert payload['reference_notes'][0]['id'] == 'D-T2DM-REF-1'
    assert 'never means that a\n     problem described in the item is present' in msgs[0]['content']


# ---- the whole chain -------------------------------------------------------------------------------------------

def test_chain_end_to_end_with_human_override(tmp_path):
    backend = Seq([])
    pipe, store = pipeline(tmp_path, backend)
    out = pipe.start('J1', request())
    assert out['status'] == 'AwaitingHumanReview' and backend.calls == 6   # one call per dimension
    queued = {i['dimension'] for i in out['review']['items']}
    assert 'context' in queued                                            # no check items -> anchors only -> review
    with pytest.raises(ValueError, match='note'):
        pipe.resume('J1', {'context': {'decision': 'override', 'score': 3, 'reviewer': 'LTF'}})


def test_chain_override_validation_and_final_report(tmp_path):
    pipe, _ = pipeline(tmp_path)
    out = pipe.start('J2', request())
    dims = [i['dimension'] for i in out['review']['items']]
    decisions = {d: {'decision': 'accept', 'reviewer': 'LTF'} for d in dims}
    decisions[dims[0]] = {'decision': 'override', 'score': 3, 'reviewer': 'LTF', 'note': 'reason'}
    done = pipe.resume('J2', decisions)
    rep = done['report']
    row = next(d for d in rep['dimensions'] if d['dimension'] == dims[0])
    assert row['final_status'] == 'human_overrode' and row['final_score'] == 3 and row['agent_score'] is not None
    assert (tmp_path / 'reports/v3-reviews/J2/report.json').exists()
    assert len(rep['snapshot']['snapshot_sha256']) == 64 and rep['clinical_validation'] == 'NOT_VALIDATED'


@pytest.mark.parametrize('bad', [
    {'context': {'decision': 'override', 'score': 9, 'reviewer': 'LTF', 'note': 'x'}},
    {'context': {'decision': 'accept', 'reviewer': ''}},
    {'nonexistent': {'decision': 'accept', 'reviewer': 'LTF'}},
    {'context': {'decision': 'approve', 'reviewer': 'LTF'}},
])
def test_invalid_human_decisions_rejected(tmp_path, bad):
    pipe, _ = pipeline(tmp_path)
    pipe.start('J3', request())
    with pytest.raises(ValueError):
        pipe.resume('J3', bad)


def test_failed_dimension_is_never_given_a_score(tmp_path):
    pipe, _ = pipeline(tmp_path, Seq(['http500']), require_human=False)
    rep = pipe.start('J4', request())['report']
    failed = [d for d in rep['dimensions'] if d['status'] == 'Failed']
    assert len(failed) == 1 and failed[0]['final_score'] is None and failed[0]['final_status'] == 'not_scored'
    assert 'DIMENSION_NOT_SCORED' in failed[0]['review_reasons']


def test_batch_mode_marks_flagged_dimensions_unreviewed(tmp_path):
    pipe, _ = pipeline(tmp_path, require_human=False)
    rep = pipe.start('J5', request())['report']
    ctx = next(d for d in rep['dimensions'] if d['dimension'] == 'context')
    assert ctx['final_status'] == 'agent_unreviewed' and ctx['final_score'] == ctx['agent_score']


def test_scale_comes_from_the_profile(tmp_path):
    prof = small_profile(0, 2)
    items = [RubricItem(item_id='G-OM-1', level='general', role='scored', dimension='omission',
                        profiles=[prof.profile_id], text='No clinically important omission.', source=SRC)]
    req = request()
    req = ReviewRequest(case=req.case, meta=req.meta, profile_id=prof.profile_id)
    pipe, _ = pipeline(tmp_path, profiles={prof.profile_id: prof}, items=items, require_human=False)
    rep = pipe.start('J6', req)['report']
    assert all(d['scale'] == [0, 2] and 0 <= d['agent_score'] <= 2 for d in rep['dimensions'])
    with pytest.raises(EvalFailure):
        parse_v3(wire([{'id': 'G-OM-1', 'result': 'pass', 'quote': ''}], score=3), req.case, prof.dimension('omission'), items)


def test_reports_are_never_overwritten_and_html_is_escaped(tmp_path):
    req = request(candidate_output='<script>alert(1)</script> Increase metformin.')
    pipe, _ = pipeline(tmp_path, require_human=False)
    pipe.start('J7', req)
    html = (tmp_path / 'reports/v3-reviews/J7/report.html').read_text()
    assert '<script>' not in html and '&lt;script&gt;' in html
    pipe2, _ = pipeline(tmp_path, require_human=False)
    with pytest.raises(FileExistsError):
        pipe2.start('J7', req)


# ---- CLI: offline run + resume in a NEW process-like call; paid path is plan-only without --authorized --------

STEP1 = ['--kb', 'examples/v3/seed-kb.jsonl', '--profile', 'examples/v3/profile-platform-6d.json', '--retriever', 'filter']


def _cli_root(tmp_path):
    shutil.copytree(EX, tmp_path / 'examples/v3')
    return tmp_path


def test_cli_offline_run_and_resume(tmp_path, capsys):
    root = _cli_root(tmp_path)
    cli = load_script('run_review_v3')
    assert cli.main(['--root', str(root), *STEP1, '--case', 'examples/v3/demo-case.json', '--job-id', 'CLI1']) == 0
    first = capsys.readouterr().out
    assert 'AwaitingHumanReview' in first
    (root / 'dec.json').write_text(json.dumps({'context': {'decision': 'accept', 'reviewer': 'LTF'}}))
    cli2 = load_script('run_review_v3')   # fresh module: state must come from the sqlite checkpoint
    assert cli2.main(['--root', str(root), *STEP1, '--resume', 'CLI1', '--decisions', str(root / 'dec.json')]) == 0
    assert (root / 'reports/v3-reviews/CLI1/report.html').exists()


def test_cli_paid_judge_is_plan_only_and_budgets_the_v3_output_limit(tmp_path, monkeypatch, capsys):
    root = _cli_root(tmp_path)
    (root / 'jury-v2.json').write_text(json.dumps({'judges': [{'name': 'deepseek-pro', 'kind': 'deepseek',
                                                               'model': 'deepseek-v4-pro', 'price_in': 13.2, 'price_out': 39.6}]}))
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key-not-real')
    monkeypatch.setenv('JURY_PRICING_VALID_UNTIL', '2099-01-01')
    cli = load_script('run_review_v3')
    backend, paid, _ = cli.make_backend('deepseek-pro', root, Store(root), 1.0)
    assert paid and backend.settings.max_output_tokens == 1200    # reservation matches the v3 call size
    backend.close()
    assert cli.main(['--root', str(root), *STEP1, '--case', 'examples/v3/demo-case.json', '--judge', 'deepseek-pro']) == 0
    plan = capsys.readouterr().out
    assert '"planned_calls": 6' in plan and 'Plan only' in plan
    assert not (root / 'reports/v3-reviews').exists()            # plan mode writes nothing


def _paid_root(tmp_path, monkeypatch, data_source='synthetic-demo'):
    root = _cli_root(tmp_path)
    (root / 'jury-v2.json').write_text(json.dumps({'judges': [{'name': 'deepseek-pro', 'kind': 'deepseek',
                                                               'model': 'deepseek-v4-pro', 'price_in': 13.2, 'price_out': 39.6}]}))
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key-not-real')
    monkeypatch.setenv('JURY_PRICING_VALID_UNTIL', '2099-01-01')
    case = json.loads((root / 'examples/v3/demo-case.json').read_text())
    case['meta']['data_source'] = data_source
    (root / 'case.json').write_text(json.dumps(case))
    return root


def test_paid_run_refuses_placeholder_rubrics_before_any_call(tmp_path, monkeypatch, capsys):
    root = _paid_root(tmp_path, monkeypatch)
    cli = load_script('run_review_v3')
    assert cli.main(['--root', str(root), *STEP1, '--case', 'case.json', '--judge', 'deepseek-pro', '--authorized']) == 2
    assert 'project_draft' in capsys.readouterr().out and not (root / 'reports/v3-reviews').exists()


def test_paid_run_on_non_public_data_needs_an_approval_reference(tmp_path, monkeypatch, capsys):
    root = _paid_root(tmp_path, monkeypatch, data_source='mimic-iv-derived')
    cli = load_script('run_review_v3')
    assert cli.main(['--root', str(root), *STEP1, '--case', 'case.json', '--judge', 'deepseek-pro', '--authorized',
                     '--allow-draft-rubrics']) == 2
    assert '--data-approval-ref' in capsys.readouterr().out and not (root / 'reports/v3-reviews').exists()


def test_immediate_stop_codes_stop_the_other_dimensions():
    case, dim = request().case, _dim()
    ev = DimensionEvaluator(Seq(['http401']))
    assert ev.evaluate(case, dim, _checks(), [], 'J')['failure_code'] == 'PROVIDER_HTTP_401'
    assert ev.stop.is_set() and ev.evaluate(case, dim, _checks(), [], 'J')['status'] == 'NotRun'
    ev5 = DimensionEvaluator(Seq(['http500']))
    ev5.evaluate(case, dim, _checks(), [], 'J')
    assert not ev5.stop.is_set()                                   # a single 5xx does not stop the run


def test_shared_prefix_comes_first_for_prompt_cache():
    c = request().case
    a = build_messages_v3(c, profile().dimension('safety'), _checks(), [])[1]['content']
    b = build_messages_v3(c, profile().dimension('accuracy'), [], [])[1]['content']
    prefix = json.dumps({'conversation': [m.model_dump() for m in c.context], 'answer_to_evaluate': c.candidate_output},
                        ensure_ascii=False, separators=(',', ':'))[:-1]
    assert a.startswith(prefix) and b.startswith(prefix)
