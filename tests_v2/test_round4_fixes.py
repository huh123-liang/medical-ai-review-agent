"""Round-4 fixes: locator-2, prompt v3 + verbatim prompt archive, group B (F04-F09), group permutation,
validation stop-code propagation, length statistics, rerun/used-group handling. Offline, synthetic only."""
import hashlib
import json
import threading

import pytest

from medical_jury.domain import CaseInput, Rubric
from medical_jury.evaluation import agreement as A
from medical_jury.judge_v2 import (JUDGING_RULES, PROMPT_SHA256, PROMPT_VERSION_V2, SYSTEM_V2, SYSTEM_V2_PREVIOUS,
                                   JudgeV2, build_messages_v2, locate_quote, parse_v2, system_prompt)

from v2_fixtures import load_script, make_project

GOOD = {'analysis': 'a', 'verdict': 'met', 'confidence': 0.8}
MD_ANSWER = ('Overview\n\n**Adverse Effects:**  \n- Gastrointestinal: Nausea (most common).\n'
             'Survival is approximately **40-50%**.\n**No**—the evidence isn’t there.\n'
             'Dose 2*3 tablets. Repeat. Repeat. xyzxyzxyz')


def md_case():
    return CaseInput(case_id='C', group_id='G', task='t', split='dev',
                     context=[{'role': 'user', 'content': 'Is it proven? Tell me the survival.'}],
                     candidate_output=MD_ANSWER)


def tiers(quotes):
    out = parse_v2(json.dumps({**GOOD, 'evidence': quotes}), md_case())
    return [t['tier'] for t in out['evidence_trace']], out


# ---- locator-2 --------------------------------------------------------------------------------

def test_exact_quote_records_offsets_into_original_answer():
    tier, out = tiers(['Survival is approximately **40-50%**.'])
    assert tier == ['exact'] and out['evidence_located'] == [True]
    (a, b), = out['evidence_trace'][0]['spans']
    assert MD_ANSWER[a:b] == 'Survival is approximately **40-50%**.'
    assert out['evidence_answer_sha256'] == hashlib.sha256(MD_ANSWER.encode()).hexdigest()
    assert out['evidence_locator'] == 'locator-2' and out['evidence_status'] == 'all_located_in_answer'


def test_markup_only_is_diagnostic_never_exact():
    tier, out = tiers(['Survival is approximately 40-50%.', 'No—the evidence isn’t there.'])
    assert tier == ['markup_only', 'markup_only']
    assert out['evidence_located'] == [False, False] and out['evidence_status'] == 'some_unlocated_in_answer'
    (a, b), = out['evidence_trace'][0]['spans']
    assert MD_ANSWER[a:b] == 'Survival is approximately **40-50%**.'   # span covers the ** in the original
    assert out['evidence_tiers'] == {'markup_only': 2}


def test_heading_joined_with_list_line_is_whitespace_tier():
    tier, _ = tiers(['Adverse Effects: - Gastrointestinal: Nausea (most common).'])
    assert tier == ['markup_or_whitespace_only']


@pytest.mark.parametrize('quote', [
    'Survival is approximately...',                 # ellipsis truncation
    'Gastrointestinal: Nausea.',                     # omitted parenthetical
    'survival is approximately 40-50%.',             # case change
    'the evidence is there.',                        # negation removed (after markup strip)
    'Survival is approximately 40-60%.',             # number changed
    'Dose 23 tablets.',                              # 2*3 must never collapse into 23
    'Is it proven?',                                 # context, not answer
])
def test_content_changes_are_never_located(quote):
    tier, out = tiers([quote])
    assert tier == ['not_located'] and out['evidence_located'] == [False]


def test_context_only_quote_is_flagged():
    _, out = tiers(['Is it proven?'])
    assert out['evidence_trace'][0]['in_context_only'] is True


@pytest.mark.parametrize('quote', [' ', '.', 'No', '**', '— —'])
def test_trivial_quotes_are_unverifiable_not_located(quote):
    tier, out = tiers([quote])
    assert tier == ['too_short_unverifiable'] and out['evidence_located'] == [False]


def test_repeated_and_overlapping_occurrences_are_all_reported():
    _, out = tiers(['Repeat.', 'xyzxyz'])
    assert [len(t['spans']) for t in out['evidence_trace']] == [2, 2]       # 'xyzxyz' in 'xyzxyzxyz' overlaps twice
    assert out['evidence_status'] == 'located_ambiguous'


def test_locate_quote_flags_ellipsis():
    assert locate_quote(MD_ANSWER, 'Survival is approximately...')['has_ellipsis'] is True


# ---- prompt v3 + verbatim archive (F08) ------------------------------------------------------------

def test_prompt_archive_is_verbatim_and_pinned():
    assert PROMPT_VERSION_V2 == 'jury-v2-reason-first-3'
    texts = {**SYSTEM_V2_PREVIOUS, PROMPT_VERSION_V2: SYSTEM_V2}
    assert set(PROMPT_SHA256) == set(texts)
    for version, text in texts.items():
        assert hashlib.sha256(text.encode('utf-8')).hexdigest() == PROMPT_SHA256[version], version
    assert 'If it has several required parts, all must hold.' in SYSTEM_V2_PREVIOUS['jury-v2-reason-first-1']
    assert 'Keep quotes under 25 words each.' in SYSTEM_V2_PREVIOUS['jury-v2-reason-first-2']


def test_old_messages_can_be_rebuilt_exactly(case, rubric):
    old = build_messages_v2(case, rubric, version='jury-v2-reason-first-2')
    assert old[0]['content'] == SYSTEM_V2_PREVIOUS['jury-v2-reason-first-2']
    assert old[1] == build_messages_v2(case, rubric)[1]           # user payload format unchanged
    with pytest.raises(KeyError):
        system_prompt('jury-v2-unknown')


def test_v3_prompt_demands_character_exact_quotes():
    for phrase in ('character for character', 'Markdown symbols', 'never use "..."', 'contiguous span',
                   'describe the omission in the'):
        assert phrase in SYSTEM_V2, phrase
    assert JUDGING_RULES in SYSTEM_V2


# ---- F05: DSPy uses the same rule text ------------------------------------------------------------

def test_dspy_signature_uses_shared_rules():
    from medical_jury.dspy_judge import JudgeCriterion
    instructions = JudgeCriterion.instructions
    assert JUDGING_RULES in instructions
    assert 'all must hold' not in instructions.replace('parts joined by "and" must all hold', '')


# ---- F06: atomic call cap covering stream ---------------------------------------------------------

def test_call_cap_is_atomic_and_covers_stream():
    from medical_jury.dspy_judge import LedgerEngine, LedgerEngineError
    from medical_jury.judge_v2 import ScriptedBackend
    opt = load_script('optimize_dspy')
    backend = ScriptedBackend(lambda m: '{}', name='mock')
    cap = opt.CallCap(LedgerEngine(backend, 'J'), 25)

    class Req:
        system, messages = 'sys', []
    errors = []

    def hammer(use_stream):
        for _ in range(20):
            try:
                cap.stream(Req()) if use_stream else cap.complete(Req())
            except LedgerEngineError as error:
                errors.append(str(error))
    threads = [threading.Thread(target=hammer, args=(i % 2 == 0,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert backend.calls == 25 and cap.issued == 25
    assert errors and set(errors) == {'CALL_CAP_REACHED'} and len(errors) == 8 * 20 - 25


# ---- F04: single-judge decision score is the judge's own P(met) ------------------------------------

class FixedBackend:
    name, model, billable = 'fixed', 'SYNTHETIC', False

    def __init__(self, verdict='met', confidence=0.6, fail=None):
        self.verdict, self.confidence, self.fail, self.calls = verdict, confidence, fail, 0

    def complete(self, messages, max_tokens, job_id):
        from medical_jury.judge_v2 import BackendError, Completion
        self.calls += 1
        if self.fail:
            raise BackendError(self.fail)
        return Completion(json.dumps({'analysis': 'a', 'verdict': self.verdict, 'confidence': self.confidence,
                                      'evidence': []}), 'stop', {'completion_tokens': 10}, 'SYNTHETIC')


def test_single_judge_p_met_is_judge_probability_not_vote_share(tmp_path, case, rubric, store):
    from medical_jury.workflow_v2 import JuryEngine
    engine = JuryEngine({'pro': JudgeV2(FixedBackend('met', 0.6), store)}, store, tmp_path,
                        require_human=False, persist_reports=False)
    c = engine.start('J1', case, [rubric])['report']['criteria'][0]
    assert c['p_met'] == pytest.approx(0.6) and c['judge_p_met'] == pytest.approx(0.6)
    assert c['vote_share'] == 1.0 and c['ds_posterior'] is None
    assert 'LOW_CONFIDENCE' in c['review_reasons']          # band (0.25, 0.75) now actually fires


def test_jury_records_judge_failure_codes(tmp_path, case, rubric, store):
    from medical_jury.workflow_v2 import JuryEngine
    judges = {'a': JudgeV2(FixedBackend(fail='PROVIDER_HTTP_401'), store), 'b': JudgeV2(FixedBackend(), store)}
    engine = JuryEngine(judges, store, tmp_path, require_human=False, persist_reports=False)
    c = engine.start('J2', case, [rubric])['report']['criteria'][0]
    assert c['judge_failure_codes'] == {'a': 'PROVIDER_HTTP_401'} and c['failed_judges'] == ['a']


# ---- validation propagates immediate stop codes -----------------------------------------------------

def test_validation_stops_on_judge_level_401(tmp_path, monkeypatch):
    from test_p3_dspy_freeze import frozen_project
    root, fz = frozen_project(tmp_path)
    run = load_script('run_validation_once')
    backends = {}

    def fake_build(spec, settings, store, run_limit):
        backends[spec['name']] = FixedBackend(fail='PROVIDER_HTTP_401' if spec['name'] == 'a' else None)
        return backends[spec['name']]
    monkeypatch.setattr(run, 'build_backend', fake_build)
    assert run.main([str(fz), '--root', str(root), '--kind', 'natural', '--threads', '1']) == 0
    summary = json.loads(next((root / 'reports/frozen-runs').glob('*-validation-natural/summary.json')).read_text())
    assert summary['stop_reason'] == 'auth_or_quota'
    assert backends['a'].calls == 1                        # stopped after the first item
    counts = summary['status_counts']
    assert sum(v for k, v in counts.items() if k.startswith('NotRun:STOPPED_AUTH_OR_QUOTA')) == \
        summary['scores']['planned_items'] - 1


# ---- F07: MedExpert inputs never carry labels -------------------------------------------------------

def test_medexpert_inputs_are_label_free(tmp_path):
    from medical_jury.datasets import medexpert as MX
    from test_external_medexpert import raw_rows
    rows = raw_rows()
    with pytest.raises(ValueError, match='label-free'):
        MX.build_inputs(rows)
    path = tmp_path / 'raw.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    slim = MX.load_raw(path, keep=MX.INPUT_FIELDS)
    assert not any(k in r for r in slim for k in MX.LABEL_FIELDS)
    public, manifest = MX.build_inputs(slim)
    assert manifest['labels_read'] is False
    full_public, private, _ = MX.build(rows)
    assert public == full_public and [p['example_id'] for p in private] == [p['example_id'] for p in public]


# ---- group-level permutation ---------------------------------------------------------------------

def test_group_permutation_is_more_conservative_for_correlated_groups():
    import random
    rng = random.Random(7)
    items, groups = [], []
    for g in range(30):
        truth = rng.random() < 0.5
        noisy = truth if rng.random() < 0.7 else not truth
        for _ in range(4):                       # 4 identical criteria per dialogue: perfectly correlated
            items.append((noisy, [truth, truth]))
            groups.append(f'G{g}')
    item = A.permutation_test_kappa(items, 1000, 1)
    group = A.permutation_test_kappa(items, 1000, 1, groups=groups)
    assert group['unit'] == 'group' and group['n_groups'] == 30 and group['group_sizes'] == {'4': 30}
    assert group['observed_kappa'] == item['observed_kappa']
    assert group['p_value'] >= item['p_value']


def test_group_permutation_requires_one_group_per_item():
    with pytest.raises(ValueError):
        A.permutation_test_kappa([(True, [True])], 10, 0, groups=['a', 'b'])


# ---- dev runner: median, used groups, rerun-of, F09 -------------------------------------------------

def test_conventional_median():
    mod = load_script('run_dev_v2')
    assert mod._median(list(range(10, 101, 10))) == 55
    assert mod._median([3, 1, 2]) == 2


class Seq:
    name, model, billable = 'seq', 'SYNTHETIC', False

    def __init__(self, plan):
        self.plan, self.calls = list(plan), 0

    def complete(self, messages, max_tokens, job_id):
        from medical_jury.judge_v2 import BackendError, Completion
        kind = self.plan[self.calls] if self.calls < len(self.plan) else 'ok'
        self.calls += 1
        if kind.startswith('http'):
            raise BackendError('PROVIDER_HTTP_' + kind[4:])
        if kind == 'ctrl_c':
            raise KeyboardInterrupt  # synthetic Ctrl+C during a call
        return Completion(json.dumps({**GOOD, 'evidence': []}), 'stop', {'completion_tokens': 40}, 'SYNTHETIC')


def dev(monkeypatch, root, plan, extra=(), tag='x'):
    mod = load_script('run_dev_v2')
    backend = Seq(plan)
    monkeypatch.setattr(mod, 'make_backend', lambda args, settings, store: backend)
    (root / 'jury-v2.json').write_text(json.dumps({'judges': [{'name': 'synthetic', 'kind': 'mock'}]}))
    out = root / f'reports/dev-v2-synthetic-{tag}'
    rc = mod.main(['--root', str(root), '--backend', 'panel', '--judge', 'synthetic', '--n-groups', '10',
                   '--out', str(out), *extra])
    return rc, out, backend


def test_fresh_dev_run_excludes_groups_already_sent(tmp_path, monkeypatch):
    root = make_project(tmp_path)
    dev(monkeypatch, root, [], tag='a')
    rc, out, _ = dev(monkeypatch, root, [], tag='b')
    first = set(json.loads((root / 'reports/dev-v2-synthetic-a/sample-manifest.json').read_text())['group_ids'])
    second = json.loads((out / 'sample-manifest.json').read_text())
    plan = json.loads((out / 'plan.json').read_text())
    assert rc == 0 and not first & set(second['group_ids'])
    assert second['excluded_groups'] == 10 and plan['excluded_used_groups']['n_groups'] == 10


def test_bogus_rerun_of_is_refused(tmp_path, monkeypatch):
    root = make_project(tmp_path)
    rc, _, backend = dev(monkeypatch, root, [], extra=['--rerun-of', str(root / 'reports/nowhere')], tag='c')
    assert rc == 2 and backend.calls == 0
    dev(monkeypatch, root, [], tag='d')
    rc, _, backend = dev(monkeypatch, root, [], extra=['--rerun-of', str(root / 'reports/nowhere')], tag='e')
    assert rc == 2 and backend.calls == 0


def test_metamorphic_and_repeats_are_written_live_and_stop(tmp_path, monkeypatch):
    root = make_project(tmp_path)
    # 10 main items succeed, then the first metamorphic call returns 401: everything after is NotRun.
    rc, out, backend = dev(monkeypatch, root, ['ok'] * 10 + ['http401'],
                           extra=['--metamorphic-per-kind', '1', '--repeats', '2'], tag='f')
    summary = json.loads((out / 'summary.json').read_text())
    mm = [json.loads(x) for x in (out / 'metamorphic.partial.jsonl').read_text().splitlines()]
    rep = [json.loads(x) for x in (out / 'repeats.partial.jsonl').read_text().splitlines()]
    assert summary['stop_reason'] == 'auth_or_quota' and backend.calls == 11
    assert mm[0]['failure_code'] == 'PROVIDER_HTTP_401'
    assert all(r['status'] == 'NotRun' for r in mm[1:]) and all(r['status'] == 'NotRun' for r in rep)
    assert len(rep) == 10 * 2 and (out / 'metamorphic.jsonl').exists() and (out / 'repeats.jsonl').exists()


def _lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def _assert_complete_audit(out, n_main=10, n_mm=None, n_rep=20):
    main, mm, rep = (_lines(out / f) for f in ('predictions.partial.jsonl', 'metamorphic.partial.jsonl',
                                                'repeats.partial.jsonl'))
    for recs in (main, mm, rep):                         # every planned item exactly once, no duplicates
        ids = [r['example_id'] for r in recs]
        assert len(ids) == len(set(ids))
    assert len(main) == n_main and len(rep) == n_rep and (n_mm is None or len(mm) == n_mm)
    assert len(_lines(out / 'predictions.jsonl')) == n_main
    return main, mm, rep


MM_REP = ['--metamorphic-per-kind', '1', '--repeats', '2']


def test_ctrl_c_in_main_batch_records_every_later_item_as_not_run(tmp_path, monkeypatch):
    root = make_project(tmp_path)
    rc, out, backend = dev(monkeypatch, root, ['ok', 'ok', 'ctrl_c'], extra=MM_REP, tag='g')
    main, mm, rep = _assert_complete_audit(out)
    summary = json.loads((out / 'summary.json').read_text())
    assert rc == 130 and backend.calls == 3 and summary['stop_reason'] == 'interrupted'
    assert sum(r['status'] == 'Completed' for r in main) == 2
    assert mm and all(r['status'] == 'NotRun' and r['failure_code'] == 'STOPPED_INTERRUPTED' for r in mm + rep)


def test_ctrl_c_in_metamorphic_batch_keeps_repeats_audited(tmp_path, monkeypatch):
    root = make_project(tmp_path)
    rc, out, backend = dev(monkeypatch, root, ['ok'] * 11 + ['ctrl_c'], extra=MM_REP, tag='h')
    main, mm, rep = _assert_complete_audit(out)
    assert rc == 130 and backend.calls == 12
    assert mm[0]['status'] == 'Completed' and all(r['status'] == 'NotRun' for r in mm[1:])
    assert all(r['failure_code'] == 'STOPPED_INTERRUPTED' for r in rep)


def test_ctrl_c_between_batches_is_completed_as_not_run(tmp_path, monkeypatch):
    root = make_project(tmp_path)
    mod = load_script('run_dev_v2')
    real, calls = mod.run_planned, []

    def interrupt_second_batch(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt  # Ctrl+C after the main batch returned, before metamorphic started
        return real(*args, **kwargs)
    monkeypatch.setattr(mod, 'run_planned', interrupt_second_batch)
    backend = Seq([])
    monkeypatch.setattr(mod, 'make_backend', lambda args, settings, store: backend)
    (root / 'jury-v2.json').write_text(json.dumps({'judges': [{'name': 'synthetic', 'kind': 'mock'}]}))
    out = root / 'reports/dev-v2-synthetic-i'
    rc = mod.main(['--root', str(root), '--backend', 'panel', '--judge', 'synthetic', '--n-groups', '10',
                   '--out', str(out), *MM_REP])
    main, mm, rep = _assert_complete_audit(out)
    assert rc == 130 and backend.calls == 10 and all(r['status'] == 'Completed' for r in main)
    assert mm and all(r['status'] == 'NotRun' and r['failure_code'] == 'STOPPED_INTERRUPTED' for r in mm + rep)
    assert len(calls) == 2      # repeats batch never started; it was completed as NotRun without any call


# ---- F08: human_ceiling guard on a synthetic raw file (no real data needed) --------------------------

def synthetic_healthbench(root):
    raw = root / 'data/public/healthbench-meta.jsonl'
    raw.parent.mkdir(parents=True, exist_ok=True)
    import random
    rng = random.Random(3)
    lines = []
    for g in range(40):
        for i in range(3):
            n = 3 if g % 2 == 0 else 2
            lines.append(json.dumps({
                'prompt': [{'role': 'user', 'content': f'synthetic question {g}'}],
                'completion': f'synthetic answer {g}-{i}', 'rubric': f'criterion {i}',
                'category': 'cluster:hedging_x' if g % 2 else 'cluster:context_seeking_x',
                'binary_labels': [rng.random() < 0.7 for _ in range(n)],
                'anonymized_physician_ids': [f'P{(g + k) % 5}' for k in range(n)]}))
    raw.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    (root / 'data/public/source-manifest.json').write_text(json.dumps({'sha256': sha}), encoding='utf-8')
    return raw


def test_human_ceiling_guard_on_synthetic_source(tmp_path):
    hc = load_script('human_ceiling')
    synthetic_healthbench(tmp_path)
    assert hc.main(['--root', str(tmp_path), '--no-index', '--bootstrap', '5']) == 0
    assert not (tmp_path / 'data/private').exists()
    (tmp_path / 'data/private').mkdir()
    (tmp_path / 'data/private/physician-rater-index.jsonl').write_text('{}\n')
    with pytest.raises(SystemExit, match='without'):
        hc.main(['--root', str(tmp_path), '--bootstrap', '5'])
