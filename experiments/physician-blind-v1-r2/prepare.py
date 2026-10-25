"""Coordinator only: outcome-blind sample first, freeze separate physician labels second."""
from __future__ import annotations

import json
import math
import random
import sqlite3
from collections import Counter
from dataclasses import replace

from medical_jury.benchmark import convert_row
from medical_jury.config import Settings
from medical_jury.domain import CaseInput, Rubric
from medical_jury.provider import build_messages, effective_profile
from medical_jury.store import timestamp

from common import (ARM_A, ARM_B, HERE, MAX_CANDIDATE_OUTPUT, PER_TOPIC, PRIVATE, REPORT, ROOT,
                    ROUND_LIMIT, SEED, TOPICS, candidate_messages, canonical, digest, protected,
                    read, select_rows, sha, source_code, topic, write)


def upper(messages, settings):
    n = len(canonical(messages).encode('utf-8')) + 2048
    return math.ceil((n * settings.input_rate + settings.max_output_tokens * settings.output_rate)) / 1_000_000


def main():
    if REPORT.exists() or PRIVATE.exists():
        raise FileExistsError('Do not overwrite a frozen experiment or its separate references.')
    settings = Settings.load()
    settings.validate_paid()
    assert settings.model == 'deepseek-v4-pro' and settings.max_attempts == 1
    assert settings.judge_profile == 'scoped-review-v1' and settings.max_output_tokens == 1200
    with sqlite3.connect(f'file:{(ROOT / "data/jury.sqlite").as_posix()}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        old_jobs = [dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY id')]
        old_costs = [dict(r) for r in db.execute('SELECT * FROM costs ORDER BY id')]
        excluded = {json.loads(r['request_json'])['case']['group_id'] for r in old_jobs}
        excluded.update(r[0] for r in db.execute('SELECT DISTINCT group_id FROM rubrics'))
    accounted = sum(r['accounted'] for r in old_costs)
    assert accounted + ROUND_LIMIT <= 30 and not any(r['state'] == 'reserved' for r in old_costs)
    rows = [json.loads(line) for line in (ROOT / 'data/prepared/validation.jsonl').read_text(encoding='utf-8').splitlines()]
    selected = select_rows(rows, excluded)
    assert len(selected) == len({r['case']['group_id'] for r in selected}) == len(TOPICS) * PER_TOPIC
    by_id = {r['example_id']: r for r in selected}
    schedule = []
    rng = random.Random(SEED)
    for row in selected:
        arms = [ARM_A, ARM_B]
        rng.shuffle(arms)
        for arm in arms:
            messages = build_messages(CaseInput(**row['case']), Rubric(**row['rubric']), True,
                                      settings.judge_profile, ROOT) if arm == ARM_A else candidate_messages(row)
            s = settings if arm == ARM_A else replace(settings, max_output_tokens=MAX_CANDIDATE_OUTPUT)
            schedule.append({'example_id': row['example_id'], 'group_id': row['case']['group_id'],
                'topic': topic(row), 'category': row['category'], 'arm': arm, 'messages': messages,
                'messages_sha256': digest(messages), 'input_sha256': digest(row),
                'max_output_tokens': s.max_output_tokens, 'reservation_upper_cny': upper(messages, s),
                'effective_profile': effective_profile(CaseInput(**row['case']), s.judge_profile) if arm == ARM_A
                    else 'prompt-evidence-research-v1-original-rubric-single-item-research-draft'})
    total_upper = round(sum(r['reservation_upper_cny'] for r in schedule), 6)
    if total_upper > ROUND_LIMIT:
        raise ValueError(f'Entire prespecified batch upper bound {total_upper} exceeds {ROUND_LIMIT}; no calls or silent shrinking.')
    # Only after selection is immutable do we inspect scores, solely to freeze the scorer's references.
    raw = ROOT / 'data/public/healthbench-meta.jsonl'
    source = read(ROOT / 'data/public/source-manifest.json')
    assert sha(raw) == source['sha256']
    references = {}
    duplicates = 0
    with raw.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            original = json.loads(line)
            public, labels = convert_row(original)
            eid = public['example_id']
            if eid not in by_id:
                continue
            assert public == by_id[eid]
            record = {**labels, 'anonymized_physician_ids': original['anonymized_physician_ids'],
                      'source_line': line_no, 'original_row_sha256': digest(original)}
            if eid in references:
                duplicates += 1
                assert record['binary_labels'] == references[eid]['binary_labels']
                assert record['anonymized_physician_ids'] == references[eid]['anonymized_physician_ids']
            else:
                references[eid] = record
    assert set(references) == set(by_id)
    original_private = {r['example_id']: r for r in map(json.loads,
        (ROOT / 'data/private/physician-labels.jsonl').read_text(encoding='utf-8').splitlines())}
    assert all(references[e]['binary_labels'] == original_private[e]['binary_labels'] for e in references)
    selected_groups = {r['case']['group_id'] for r in selected}
    assert not selected_groups & excluded
    for split in ['dev', 'test']:
        other_groups = {json.loads(line)['case']['group_id'] for line in
                        (ROOT / f'data/prepared/{split}.jsonl').read_text(encoding='utf-8').splitlines()}
        assert not selected_groups & other_groups
    REPORT.mkdir()
    PRIVATE.mkdir()
    write(REPORT / 'inputs.json', selected, exclusive=True)
    write(REPORT / 'schedule.json', schedule, exclusive=True)
    write(PRIVATE / 'references.json', {'status': 'FROZEN_PUBLIC_PHYSICIAN_LABELS_SCORER_ONLY',
          'source': source, 'labels': references}, exclusive=True)
    protocol = {'version': 'physician-blind-paired-v1', 'created_at': timestamp(), 'split': 'validation',
        'planned_cases': len(selected), 'planned_calls': len(schedule), 'topics': TOPICS,
        'sampling': 'Fixed seed, outcome-blind two distinct conversation groups per official theme; no label balancing.',
        'seed': SEED, 'one_criterion_per_group': True, 'primary_arm': ARM_A, 'exploratory_arm': ARM_B,
        'model': settings.model, 'temperature': 0, 'thinking': 'disabled', 'attempts': 1,
        'baseline_profile': settings.judge_profile, 'baseline_evidence_policy': settings.evidence_policy,
        'baseline_max_output_tokens': settings.max_output_tokens, 'candidate_max_output_tokens': MAX_CANDIDATE_OUTPUT,
        'candidate_scope': 'Frozen B1 system/schema. Whole original rubric preserved as O1 research_draft; no invented atomic clinical checklist, no external fact sources.',
        'candidate_expected_limitation': 'Draft manifest / unavailable medical evidence may produce many unable_to_determine responses; report coverage, not only successful decisions.',
        'not_single_factor_prompt_ablation': 'Output contracts and token caps differ; paired scores are descriptive, not proof of causal effect of one instruction.',
        'candidate_validation_gate': 'Separate evaluation-only message assembly; old dev-only builder/API remain unchanged; no examples or optimization on validation.',
        'physician_protocol': 'Compare each binary prediction to each ORIGINAL physician label; majority only as secondary descriptive unanimous/tie analysis.',
        'failure_protocol': 'Keep all planned cases; primary binary metrics withheld if an arm has invalid/unknown/not-run cases. Conditional diagnostics labelled.',
        'stop_rules': ['HTTP401/402/403 or budget/reservation issue: stop all', '3 consecutive transport failures: stop all',
                       '3 consecutive candidate format/protocol failures: stop candidate arm, keep baseline', 'No retries or prompt edits after labels are unsealed'],
        'round_limit_cny': ROUND_LIMIT, 'sum_full_reservations_cny': total_upper,
        'cumulative_before_cny': round(accounted, 6), 'cumulative_limit_cny': 30,
        'price_review': {'source': 'https://api-docs.deepseek.com/zh-cn/quick_start/pricing/',
                         'checked_date': '2026-10-25', 'published_peak_input_cny_per_million': 9.0,
                         'published_peak_output_cny_per_million': 27.0,
                         'used_conservative_input_ceiling': settings.input_rate,
                         'used_conservative_output_ceiling': settings.output_rate,
                         'no_cache_or_off_peak_discount_assumed': True},
        'bootstrap': '3000 within-theme conversation-group resamples; exploratory with only 2 groups/theme.',
        'limits': ['Not full-paper replication or natural clinic accuracy; fixed themed sample is small.',
                   'Public data may have appeared in foundation-model training; semantic near-duplicates not fully excluded.',
                   'Physician labels are reference opinions, not infallible medical truth.',
                   'Oracle given-rubric test, not RAG retrieval validation. No clinical release.'],
        'gate_goal_architecture': 'PASS: single Jury, labels separate, frozen answers/rubrics, bounded paid calls, no original platform access.'}
    write(REPORT / 'protocol.json', protocol, exclusive=True)
    manifest = {'inputs_sha256': sha(REPORT / 'inputs.json'), 'schedule_sha256': sha(REPORT / 'schedule.json'),
        'protocol_sha256': sha(REPORT / 'protocol.json'), 'sealed_reference_sha256': sha(PRIVATE / 'references.json'),
        'source_sha256': source['sha256'], 'validation_sha256': sha(ROOT / 'data/prepared/validation.jsonl'),
        'entries': [{'example_id': r['example_id'], 'group_id': r['case']['group_id'], 'input_sha256': digest(r),
                     'topic': topic(r)} for r in selected],
        'excluded_prior_group_count': len(excluded), 'topic_counts': dict(Counter(topic(r) for r in selected)),
        'selection_read_labels': False, 'duplicate_selected_raw_rows_checked': duplicates}
    manifest['manifest_hash'] = digest(manifest)
    write(REPORT / 'manifest.json', manifest, exclusive=True)
    old_state = {'jobs_sha256': digest(old_jobs), 'costs_sha256': digest(old_costs),
                 'job_ids': [r['id'] for r in old_jobs], 'cost_ids': [r['id'] for r in old_costs]}
    write(REPORT / 'old-ledger-state.json', old_state, exclusive=True)
    write(REPORT / 'freeze.json', {'created_at': timestamp(), 'protected_files': protected(),
          'new_code': source_code(), 'protocol_sha256': sha(REPORT / 'protocol.json'),
          'manifest_hash': manifest['manifest_hash'], 'old_ledger_sha256': sha(REPORT / 'old-ledger-state.json')}, exclusive=True)
    print(json.dumps({'status': 'FROZEN_BEFORE_ANY_MODEL_CALL', 'groups': len(selected),
          'planned_calls': len(schedule), 'reservation_upper_cny': total_upper,
          'cumulative_before': round(accounted, 6), 'round_cap': ROUND_LIMIT,
          'goal_architecture': 'PASS', 'labels_in_messages': False}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
