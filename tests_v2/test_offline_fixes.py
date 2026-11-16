"""Regression tests for the 2026-10-25 offline fixes (review items 1-9 plus the settle-twice bug)."""
import json
import threading

import httpx
import pytest

from medical_jury.budget import BudgetExceeded
from medical_jury.evaluation import agreement as A
from medical_jury.evaluation.runner import LiveWriter, StopController, budget_gate, ledger_status, run_planned
from medical_jury.judge_v2 import (PROMPT_VERSION_V2, SYSTEM_V2, BackendError, Completion, DeepSeekBackend,
                                   JudgeV2, LedgerBudget, ProtocolFailure, ScriptedBackend, met_probability,
                                   parse_v2)

from v2_fixtures import load_script, make_project

GOOD = {'analysis': 'a', 'verdict': 'met', 'confidence': 0.8, 'evidence': []}


class BudgetAfter:
    """Backend that succeeds n times, then the ledger refuses the reservation."""
    name, model, billable = 'budget-after', 'TEST', True

    def __init__(self, n, settle_exceeded=False):
        self.n, self.calls, self.settle_exceeded = n, 0, settle_exceeded

    def complete(self, messages, max_tokens, job_id):
        self.calls += 1
        if self.calls > self.n:
            raise BudgetExceeded('Actual usage exceeded reservation' if self.settle_exceeded
                                 else 'Call blocked before sending')
        return Completion(json.dumps(GOOD), 'stop', {}, 'TEST')


# ---- 1. budget block -> recorded failure, later items NotRun without calls ----------------

def test_budget_block_is_recorded_and_stops_further_calls(case, rubric):
    backend = BudgetAfter(2)
    judge = JudgeV2(backend)
    recs = [judge.judge(case, rubric, 'J') for _ in range(5)]
    assert [r['status'] for r in recs] == ['Completed', 'Completed', 'Failed', 'NotRun', 'NotRun']
    assert recs[2]['failure_code'] == 'BUDGET_BLOCKED'
    assert recs[3]['failure_code'] == 'STOPPED_BY_RUN_STOP'
    assert backend.calls == 3 and judge.halted


def test_settlement_exceeded_is_distinguished(case, rubric):
    rec = JudgeV2(BudgetAfter(0, settle_exceeded=True)).judge(case, rubric, 'J')
    assert rec['failure_code'] == 'BUDGET_SETTLEMENT_EXCEEDED'


def test_settle_exceeded_is_not_masked_by_double_settlement(case, rubric, settings, store):
    """Usage far above the reservation: settle raises BudgetExceeded; the finally-block must not
    settle again (which used to raise 'already settled' ValueError and hide the real cause)."""
    def handler(request):
        return httpx.Response(200, json={'usage': {'prompt_tokens': 10_000_000, 'completion_tokens': 1},
                                         'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(GOOD)}}]})
    backend = DeepSeekBackend(settings, store, 25.0, client=httpx.Client(transport=httpx.MockTransport(handler)),
                              ceiling_cny=30)
    judge = JudgeV2(backend, store)
    rec = judge.judge(case, rubric, 'J-x')
    assert rec['status'] == 'Failed' and rec['failure_code'] == 'BUDGET_SETTLEMENT_EXCEEDED' and judge.halted
    with store.connect() as db:
        assert db.execute("SELECT state FROM costs").fetchone()['state'] == 'reservation_exceeded'


def test_run_planned_keeps_full_denominator_and_writes_live(tmp_path, case, rubric):
    controller = StopController()
    judge = JudgeV2(BudgetAfter(3), stop=controller.event)
    items = [f'E{i}' for i in range(8)]
    writer = LiveWriter(tmp_path / 'p.jsonl')
    recs, interrupted = run_planned(items, lambda _: judge.judge(case, rubric, 'J'), id_of=lambda x: x,
                                    writer=writer, controller=controller, threads=1)
    writer.close()
    assert not interrupted and [r['example_id'] for r in recs] == items
    assert [r['status'] for r in recs].count('Completed') == 3
    assert recs[3]['failure_code'] == 'BUDGET_BLOCKED'
    assert all(r['status'] == 'NotRun' and r['failure_code'] == 'STOPPED_BUDGET' for r in recs[4:])
    assert controller.stop_reason == 'budget' and controller.summary()['budget_stopped']
    assert len((tmp_path / 'p.jsonl').read_text().splitlines()) == 8


def test_run_planned_turns_unexpected_exception_into_failed_record(tmp_path):
    def work(x):
        if x == 'b':
            raise RuntimeError('boom')
        return {'status': 'Completed'}
    writer = LiveWriter(tmp_path / 'p.jsonl')
    recs, _ = run_planned(['a', 'b', 'c'], work, id_of=lambda x: x, writer=writer, controller=StopController(), threads=2)
    writer.close()
    assert [r['status'] for r in recs] == ['Completed', 'Failed', 'Completed']
    assert recs[1]['failure_code'] == 'UNHANDLED_RuntimeError'


def test_live_writer_never_overwrites(tmp_path):
    (tmp_path / 'p.jsonl').write_text('old')
    with pytest.raises(FileExistsError):
        LiveWriter(tmp_path / 'p.jsonl')


def test_dev_runner_refuses_before_calls_without_scoring_index(tmp_path):
    root = make_project(tmp_path)
    (root / 'data/private/physician-rater-index.jsonl').unlink()
    out = root / 'reports/no-index'
    assert load_script('run_dev_v2').main(['--root', str(root), '--n-groups', '5', '--out', str(out)]) == 2
    assert not (out / 'predictions.jsonl').exists()


def test_budget_gate_and_read_only_ledger(tmp_path, settings, store):
    status = ledger_status(settings.root, ceiling_cny=30)
    assert status['ledger_accounted_cny'] == 0 and status['remaining_cny'] == 30
    plan = {'cost_upper_bound_cny': 12.0, 'run_limit_cny': 50, 'ledger': {'remaining_cny': 10.0}}
    assert 'remaining cumulative budget' in budget_gate(plan, True, False)
    assert budget_gate(plan, True, True) is None
    assert budget_gate(plan, False, False) is None
    assert 'run-limit' in budget_gate({**plan, 'run_limit_cny': 5}, True, True)


def test_explicit_ceiling_cannot_exceed_200(settings, store):
    # owner authorization 2026-10-26: cumulative CNY 200 (was 100, before that 30)
    for bad in (200.01, 500, 0, float('nan')):
        with pytest.raises(ValueError):
            LedgerBudget(settings, store, bad)
    assert LedgerBudget(settings, store, 200).ceiling == 200
    assert LedgerBudget(settings, store, 100).ceiling == 100
    assert LedgerBudget(settings, store, 30).ceiling == 30


# ---- 2. external precheck writes nothing under data/ (see test_external_medexpert) ---------
# ---- 3. human_ceiling never overwrites; --no-index writes nothing under data/ -------------

# test_human_ceiling_guard moved to test_round4_fixes.py (synthetic source; no skip without real data).


# ---- 4. confidence outside the 0.5-1.0 contract is flagged, raw value kept ----------------

def test_confidence_out_of_contract_is_flagged_not_hidden(case):
    out = parse_v2(json.dumps({**GOOD, 'confidence': 0.3}), case)
    assert out['confidence_raw'] == 0.3 and out['confidence_out_of_contract'] is True
    assert met_probability(out['verdict'], out['confidence']) == 0.5
    assert parse_v2(json.dumps(GOOD), case)['confidence_out_of_contract'] is False


# ---- 5. evidence is located in the ANSWER only, with occurrence counts ---------------------

def test_evidence_only_in_context_is_not_counted_as_answer_evidence(case):
    quote = 'My 2 year old has a fever'
    out = parse_v2(json.dumps({**GOOD, 'evidence': [quote]}), case)
    assert out['evidence_located'] == [False] and out['evidence_trace'][0]['in_context_only']
    assert out['evidence_status'] == 'some_unlocated_in_answer'
    assert 'NOT a check' in out['evidence_note']


def test_ambiguous_and_overlong_quotes(case):
    dup = case.model_copy(update={'candidate_output': 'Rest well. Drink water. Rest well.'})
    assert parse_v2(json.dumps({**GOOD, 'evidence': ['Rest well.']}), dup)['evidence_status'] == 'located_ambiguous'
    with pytest.raises(ProtocolFailure) as e:
        parse_v2(json.dumps({**GOOD, 'evidence': ['x' * 700]}), case)
    assert e.value.code == 'OUTPUT_EVIDENCE_QUOTE_TOO_LONG'


def test_quote_spanning_answer_context_boundary_is_not_located(case):
    boundary = case.candidate_output[-10:] + '\n' + case.context[0].content[:10]
    assert parse_v2(json.dumps({**GOOD, 'evidence': [boundary]}), case)['evidence_located'] == [False]


# ---- 6. criterion logic follows the criterion (AND / OR / conditions) ----------------------

def test_prompt_follows_criterion_logic_and_is_versioned():
    assert PROMPT_VERSION_V2 == 'jury-v2-reason-first-3'
    assert '"or" need only one' in SYSTEM_V2 and 'conditions' in SYSTEM_V2
    assert 'If it has several required parts, all must hold.' not in SYSTEM_V2
    assert 'too little to decide' in SYSTEM_V2


# ---- 7. permutation at the item level -------------------------------------------------------

def test_permutation_is_item_level():
    import random
    rng = random.Random(4)
    items = []
    for _ in range(40):
        truth = rng.random() < 0.5
        pred = truth if rng.random() < 0.6 else not truth
        items.append((pred, [truth] * 6))  # 6 identical labels: pair-level shuffling would overstate evidence
    item_level = A.permutation_test_kappa(items, 2000, 1)
    pair_level = A.permutation_test_kappa([(p, l) for p, ls in items for l in ls], 2000, 1)
    assert item_level['unit'] == 'item'
    assert item_level['observed_kappa'] == pytest.approx(pair_level['observed_kappa'])
    assert item_level['p_value'] > pair_level['p_value']


# ---- 8. DSPy: one capped LM, budget/cap stops recorded, no label leakage --------------------

def test_dspy_budget_block_and_call_cap_are_recorded(case, rubric):
    from medical_jury.dspy_judge import DspyJudge, build_program
    judge = DspyJudge(build_program(), BudgetAfter(0))
    first, second = judge.judge(case, rubric, 'J'), judge.judge(case, rubric, 'J')
    assert first['failure_code'] == 'BUDGET_BLOCKED' and second['status'] == 'NotRun' and judge.halted

    opt = load_script('optimize_dspy')
    from medical_jury.dspy_judge import LedgerEngine
    import dspy
    responder = lambda m: json.dumps({'reasoning': 'r', 'verdict': 'met', 'confidence': 0.7})  # noqa: E731
    engine = opt.CallCap(LedgerEngine(ScriptedBackend(responder), 'J'), max_calls=1)
    lm = dspy.LM('ledger/x', engine=engine, cache=False, num_retries=0)
    lm.ledger_engine = engine
    capped = DspyJudge(build_program(), ScriptedBackend(responder), lm=lm)
    assert capped.judge(case, rubric, 'J')['status'] == 'Completed'
    blocked = capped.judge(case, rubric, 'J')
    assert blocked['failure_code'] == 'CALL_CAP_REACHED' and capped.halted


def test_optimizer_disables_data_aware_proposer_and_checks_leaks():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / 'scripts/optimize_dspy.py').read_text()
    assert 'data_aware_proposer=False' in text
    assert "refusing to save it" in text and 'lm=lm' in text


def test_optimizer_aborts_cleanly_when_cap_reached(tmp_path):
    root = make_project(tmp_path)
    rc = load_script('optimize_dspy').main(['--root', str(root), '--judge', 'mock', '--optimizer', 'bootstrap',
                                            '--n-train', '6', '--n-val', '4', '--max-calls', '2'])
    assert rc == 3
    out = next((root / 'reports').glob('dspy-opt-mock-*'))
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['status'] == 'ABORTED' and not (out / 'program.json').exists()


# ---- 9. wording ----------------------------------------------------------------------------

def test_reference_wording():
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    protocol = (repo / 'docs/project/PROTOCOL-V2.md').read_text(encoding='utf-8')
    assert '人类上限' not in protocol and '专家上限' not in protocol
    assert '不是数学上不可超越的上限' in protocol
