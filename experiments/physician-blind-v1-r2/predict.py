"""Predictor: reads frozen label-free messages only. NO physician label deserialization.

Real mode has an explicit flag; one pre-reserved DeepSeek call per planned arm/item.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace

import httpx

from medical_jury.budget import Budget, BudgetExceeded
from medical_jury.config import Settings
from medical_jury.domain import EvaluationRequest
from medical_jury.locking import RunnerLock
from medical_jury.store import Store, timestamp
from medical_jury.workflow import Engine

from common import (ARM_A, ARM_B, REPORT, ROOT, ROUND_LIMIT, candidate_input, candidate_module,
                    canonical, check_freeze, digest, read, sha, write)


def run_candidate(item, row, settings, store):
    """Independent evaluation adapter, NOT a production wire/API replacement."""
    s = replace(settings, max_output_tokens=item['max_output_tokens'])
    s.validate_paid()
    budget = Budget(s, store)
    request = EvaluationRequest(case=row['case'], rubrics=[row['rubric']], provider='deepseek', max_run_cost_cny=1.0)
    job_id = store.create_job(request)
    store.update(job_id, status='Running', stage='research_candidate_predicting', snapshot_json=canonical({
        'protocol': 'prompt-evidence-research-v1', 'experiment': 'iteration-15-physician-blind',
        'arm': ARM_B, 'model': s.model, 'split': row['case']['split'],
        'messages_sha256': item['messages_sha256'], 'input_sha256': item['input_sha256'],
        'max_output_tokens': s.max_output_tokens, 'reference_scores_provided': False,
        'manifest_status': 'research_draft', 'external_sources_provided': 0}))
    reservation = None
    settled = False
    usage = None
    result = {'job_id': job_id, 'status': 'Failed', 'decision': None, 'error_code': None,
              'reason': None, 'reported_model': None}
    try:
        estimate = budget.estimate(item['messages'])
        assert abs(estimate - item['reservation_upper_cny']) < 1e-6
        reservation = budget.reserve(job_id, estimate, 1.0)
        store.audit(job_id, 'research_provider_attempt', {'reservation': reservation, 'model': s.model,
                    'arm': ARM_B, 'messages_sha256': item['messages_sha256'], 'attempt': 1})
        with httpx.Client(timeout=s.timeout_seconds, follow_redirects=False, trust_env=False) as client:
            response = client.post(s.provider_url, headers={'Authorization': 'Bearer ' + s.api_key},
                json={'model': s.model, 'messages': item['messages'], 'temperature': 0,
                      'thinking': {'type': 'disabled'}, 'stream': False, 'max_tokens': s.max_output_tokens,
                      'response_format': {'type': 'json_object'}})
        if response.status_code != 200:
            budget.settle(reservation, None, False)
            settled = True
            result['error_code'] = 'PROVIDER_HTTP_' + str(response.status_code)
            return result
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError('PROVIDER_ENVELOPE_INVALID')
        usage = body.get('usage')
        budget.settle(reservation, usage, True)
        settled = True
        reported = body.get('model')
        result['reported_model'] = reported[:100] if isinstance(reported, str) else None
        choice = body['choices'][0]
        raw = choice['message']['content']
        if isinstance(raw, str):
            write(REPORT / 'raw-candidate' / (job_id + '.json'), {
                'origin': 'ACTUAL_PUBLIC_BENCHMARK_MODEL_OUTPUT', 'raw': raw,
                'raw_sha256': digest(raw), 'usage': usage, 'finish_reason': choice.get('finish_reason'),
                'requested_model': s.model, 'reported_model': result['reported_model']}, exclusive=True)
        if choice.get('finish_reason') != 'stop':
            result['error_code'] = 'INCOMPLETE_MODEL_OUTPUT'
            return result
        if not isinstance(raw, str) or not raw.strip():
            result['error_code'] = 'EMPTY_MODEL_OUTPUT'
            return result
        check = candidate_module().check_candidate(raw, candidate_input(row))
        write(REPORT / 'candidate-checks' / (job_id + '.json'), check, exclusive=True)
        if check['protocol_status'] != 'valid':
            result.update(error_code='CANDIDATE_PROTOCOL_INVALID', diagnostics=check['errors'])
            return result
        parsed = json.loads(raw)
        result.update(status='Completed', decision=parsed['criterion_decision'],
                      reason=parsed['decision_reason'], review_required=True, check_flags=check['flags'])
        report = {'experiment': 'iteration-15-physician-blind', 'arm': ARM_B,
                  'criterion_decision': result['decision'], 'reason': result['reason'],
                  'candidate_payload': parsed, 'inspection': check, 'clinical_release': False,
                  'score': None, 'physician_labels_seen': False, 'usage': usage,
                  'clinical_validation': 'PUBLIC_BENCHMARK_RESEARCH_NOT_CLINICAL_CLEARANCE'}
        store.update(job_id, report_json=canonical(report))
        return result
    except BudgetExceeded:
        result['error_code'] = 'BUDGET_BLOCKED'
        return result
    except httpx.HTTPError:
        result['error_code'] = 'PROVIDER_TRANSPORT_ERROR'
        return result
    except (ValueError, KeyError, IndexError, TypeError):
        result['error_code'] = 'PROVIDER_ENVELOPE_INVALID'
        return result
    finally:
        if reservation and not settled:
            with store.connect() as db:
                state = db.execute('SELECT state FROM costs WHERE id=?', (reservation,)).fetchone()
            if state and state[0] == 'reserved':
                budget.settle(reservation, None, False)
        store.update(job_id, status=result['status'], stage='research_completed' if result['status'] == 'Completed'
                     else 'research_failed', error_code=result['error_code'])
        store.audit(job_id, 'research_prediction_finished', {'arm': ARM_B, 'status': result['status'],
                    'decision': result['decision'], 'error_code': result['error_code'],
                    'requested_model': s.model, 'reported_model': result['reported_model']})


def run_baseline(item, row, settings, store):
    request = EvaluationRequest(case=row['case'], rubrics=[row['rubric']], provider='deepseek', max_run_cost_cny=1.0)
    job = Engine(settings, store).run(request)
    result = {'job_id': job['id'], 'status': job['status'], 'decision': None, 'error_code': job['error_code'], 'reason': None}
    if job['status'] == 'Completed':
        judgment = job['report']['judgments'][0]
        result.update(decision=judgment['judgment'], reason=judgment['reason'],
                      review_required=job['report']['review_required'])
    return result


def main(real):
    freeze, manifest = check_freeze()
    assert sha(REPORT / 'protocol.json') == freeze['protocol_sha256']
    protocol = read(REPORT / 'protocol.json')
    rows = {r['example_id']: r for r in read(REPORT / 'inputs.json')}
    schedule = read(REPORT / 'schedule.json')
    for item in schedule:
        assert item['messages_sha256'] == digest(item['messages'])
        assert item['input_sha256'] == digest(rows[item['example_id']])
    if not real:
        print(json.dumps({'status': 'PREFLIGHT_ONLY_NO_PAID_CALLS', 'planned_calls': len(schedule),
                          'reservation_upper_cny': protocol['sum_full_reservations_cny']}))
        return
    if not (REPORT / 'offline-test-gate.json').is_file():
        raise RuntimeError('Offline tests not verified; no paid calls allowed yet.')
    assert read(REPORT / 'offline-test-gate.json')['passed'] is True
    if (REPORT / 'predictions-state.json').exists() or (REPORT / 'prediction-seal.json').exists():
        raise FileExistsError('Never rerun or overwrite a paid first pass.')
    settings = Settings.load()
    settings.validate_paid()
    assert settings.model == protocol['model'] and settings.max_attempts == 1
    assert settings.judge_profile == protocol['baseline_profile'] and settings.max_output_tokens == protocol['baseline_max_output_tokens']
    with RunnerLock(ROOT):
        store = Store(ROOT)
        budget = Budget(settings, store)
        before = budget.total()['accounted_cny']
        assert before == protocol['cumulative_before_cny']
        assert protocol['sum_full_reservations_cny'] <= min(ROUND_LIMIT, budget.total()['remaining_cny'])
        state = {'state': 'Running', 'started_at': timestamp(), 'manifest_hash': manifest['manifest_hash'],
                 'predictions': [{k: v for k, v in p.items() if k != 'messages'} |
                                 {'status': 'NotRun', 'decision': None, 'job_id': None, 'error_code': None}
                                 for p in schedule], 'labels_deserialized_by_predictor': False}
        write(REPORT / 'predictions-state.json', state, exclusive=True)
        consecutive_transport = 0
        candidate_protocol_failures = 0
        stop_candidate = False
        halt = None
        with ThreadPoolExecutor(max_workers=2) as pool:
            for start in range(0, len(schedule), 2):
                if halt:
                    break
                batch = []
                planned_batch = []
                for index in range(start, min(start + 2, len(schedule))):
                    item = schedule[index]
                    if stop_candidate and item['arm'] == ARM_B:
                        state['predictions'][index]['error_code'] = 'CANDIDATE_PROTOCOL_STOP_RULE'
                        continue
                    fn = run_baseline if item['arm'] == ARM_A else run_candidate
                    planned_batch.append((index, item, fn))
                used = budget.total()['accounted_cny'] - before
                if used + sum(item['reservation_upper_cny'] for _, item, _ in planned_batch) > ROUND_LIMIT:
                    halt = 'ROUND_BUDGET_CAP'
                    break
                for index, item, fn in planned_batch:
                    future = pool.submit(fn, item, rows[item['example_id']], settings, store)
                    batch.append((future, index, item))
                lookup = {f: (index, item) for f, index, item in batch}
                for future in as_completed(lookup):
                    index, item = lookup[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        # Do not log arbitrary exception bodies (credentials/request data).
                        result = {'status': 'Failed', 'decision': None, 'job_id': None,
                                  'error_code': 'UNEXPECTED_' + type(exc).__name__}
                        halt = result['error_code']
                    state['predictions'][index].update(result)
                    code = result.get('error_code')
                    consecutive_transport = consecutive_transport + 1 if code == 'PROVIDER_TRANSPORT_ERROR' else 0
                    if code in {'PROVIDER_HTTP_401', 'PROVIDER_HTTP_402', 'PROVIDER_HTTP_403', 'BUDGET_BLOCKED'} or consecutive_transport >= 3:
                        halt = code
                    if item['arm'] == ARM_B:
                        candidate_protocol_failures = candidate_protocol_failures + 1 if code in {
                            'CANDIDATE_PROTOCOL_INVALID', 'INCOMPLETE_MODEL_OUTPUT', 'PROVIDER_ENVELOPE_INVALID'} else 0
                        if candidate_protocol_failures >= 3:
                            stop_candidate = True
                    write(REPORT / 'predictions-state.json', state)
                    print(f"item={index + 1}/{len(schedule)} arm={item['arm']} status={result['status']} decision={result.get('decision')} error={code or '-'}", flush=True)
        state.update(state='Finished', finished_at=timestamp(), halt_reason=halt, candidate_stop_rule=stop_candidate,
                     round_accounted_cny=round(budget.total()['accounted_cny'] - before, 6), budget=budget.total())
        assert state['budget']['accounted_cny'] <= 30 and state['round_accounted_cny'] <= ROUND_LIMIT
        write(REPORT / 'predictions-state.json', state)
        write(REPORT / 'predictions-frozen.json', state, exclusive=True)
        write(REPORT / 'prediction-seal.json', {'sealed_at': timestamp(),
              'predictions_sha256': sha(REPORT / 'predictions-frozen.json'),
              'manifest_hash': manifest['manifest_hash'], 'labels_deserialized_by_predictor': False,
              'scoring_not_yet_run': True, 'goal_architecture': 'PASS_SINGLE_JURY_FROZEN_INPUTS_NO_PLATFORM_CHANGE'}, exclusive=True)
        check_freeze()
        print(json.dumps({'status': 'PREDICTIONS_SEALED_BEFORE_LABEL_SCORING', 'budget': state['budget'],
                          'round_cost': state['round_accounted_cny'], 'halt': halt}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real', action='store_true', help='Explicit user-authorized paid run, never a test default.')
    main(p.parse_args().real)
