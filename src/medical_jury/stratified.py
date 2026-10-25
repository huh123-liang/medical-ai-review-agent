"""Frozen development-set selection; label-aware balancing stays outside judge input."""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .domain import digest


TOPICS = ('emergency_referrals','communication','context_seeking','health_data_tasks',
          'hedging','global_health','complex_responses')
LABEL_QUOTAS = {'unanimous_met':4, 'unanimous_not_met':4, 'disputed':2}
SAMPLING_VERSION = 'topic-label-domain-dev-v1'
SCENARIO_PATTERNS = (
    ('chronic_metabolic_renal', r'\b(diabet\w*|hypertens\w*|hba1c|insulin|chronic kidney|kidney disease|cholesterol|copd)\b'),
    ('mental_health', r'\b(suicid\w*|self.harm|depress\w*|anxiety|panic attack|bipolar|schizophren\w*)\b'),
    ('women_children', r'\b(pregnan\w*|breastfeed\w*|postpartum|toddler|infant|pediatric|baby|newborn|child)\b'),
    ('medication', r'\b(medication\w*|drug\w*|dose|dosage|side effect\w*|prescri\w*|antibiotic\w*)\b'),
    ('cardiorespiratory', r'\b(chest|cardiac|heart|breath\w*|palpitation\w*|asthma|oxygen)\b'),
    ('infection_skin', r'\b(fever|cough\w*|infect\w*|rash|itch\w*|wound|swollen|covid|flu)\b'),
)


def scenario_tag(case):
    # Transparent heuristic for coverage diagnostics, NOT physician diagnoses or gold topic labels.
    text = '\n'.join(m['content'] for m in case['context']).lower()
    return next((name for name,pattern in SCENARIO_PATTERNS if re.search(pattern,text)), 'other_general')


def label_pattern(labels):
    return 'unanimous_met' if all(labels) else 'unanimous_not_met' if not any(labels) else 'disputed'


def select_rows(rows, references, *, seed=20261024, excluded_groups=()):
    excluded = set(excluded_groups)
    pools = defaultdict(list)
    for row in rows:
        # Topic names themselves contain underscores: match known prefixes, longest first.
        topic = next((t for t in TOPICS if row['category'].startswith('cluster:'+t+'_')), None)
        if topic is None or row['case']['split'] != 'dev' or row['case']['group_id'] in excluded:
            continue
        gold = references[row['example_id']]['binary_labels']
        if not gold or any(type(x) is not bool for x in gold):
            raise ValueError('Invalid physician labels; cannot balance a valid benchmark.')
        pools[(topic,label_pattern(gold))].append(row)
    rng = random.Random(seed)
    # Sort first to make the sample invariant to source iteration order.
    for key in sorted(pools):
        values = pools[key]
        values.sort(key=lambda r:r['example_id'])
        rng.shuffle(values)
    selected, used, counts = [], set(excluded), Counter()
    for topic in TOPICS:
        for pattern, quota in LABEL_QUOTAS.items():
            for _ in range(quota):
                candidates = [r for r in pools[(topic,pattern)] if r['case']['group_id'] not in used]
                if not candidates:
                    raise ValueError(f'Insufficient independent groups for {topic}/{pattern}; no silent backfill.')
                # Prefer underrepresented heuristic scenario and rubric category within the topic.
                row = min(candidates,key=lambda r:(counts[(topic,scenario_tag(r['case']))], counts[(topic,r['category'])]))
                selected.append(row)
                used.add(row['case']['group_id'])
                counts[(topic,scenario_tag(row['case']))] += 1
                counts[(topic,row['category'])] += 1
    return selected


def prepare_plan(root:Path, output:Path):
    dev = root/'data/prepared/dev.jsonl'
    reference_file = root/'data/private/physician-labels.jsonl'
    rows = list(map(json.loads,dev.read_text(encoding='utf-8').splitlines()))
    references = {r['example_id']:r for r in map(json.loads,reference_file.read_text(encoding='utf-8').splitlines())}
    prior_file = root/'reports/pilot-first-pass-20261024-193334/analysis.json'
    prior = json.loads(prior_file.read_text(encoding='utf-8'))
    excluded = {r['group_id'] for r in prior['items']}
    chosen = select_rows(rows,references,excluded_groups=excluded)
    entries = [{'example_id':r['example_id'],'group_id':r['case']['group_id'],'input_hash':digest(r),
                'topic':next(t for t in TOPICS if r['category'].startswith('cluster:'+t+'_')),
                'category':r['category'],'heuristic_scenario':scenario_tag(r['case'])} for r in chosen]
    plan = {'sampling_version':SAMPLING_VERSION,'seed':20261024,'split':'dev','planned_examples':len(chosen),
            'dev_file_sha256':hashlib.sha256(dev.read_bytes()).hexdigest(),
            'labels_file_sha256':hashlib.sha256(reference_file.read_bytes()).hexdigest(),
            'excluded_prior_groups':sorted(excluded),'topic_counts':dict(Counter(r['topic'] for r in entries)),
            'label_quotas_per_topic':LABEL_QUOTAS,'scenario_tag_policy':'keyword-based context coverage only, not clinical diagnosis',
            'heuristic_scenario_counts':dict(Counter(r['heuristic_scenario'] for r in entries)),
            'category_counts':dict(Counter(r['category'] for r in entries)),'entries':entries,
            'limitations':['Deliberately balanced diagnostic sample, not natural-prevalence estimate.',
                           'One criterion per distinct conversation. Public data may be in model pretraining.',
                           'Gold labels used only by sampler/evaluator, never by judge. Oracle given-rubric test.']}
    plan['manifest_hash'] = digest(plan)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as file:
        file.write(json.dumps(plan,ensure_ascii=False,indent=2))
    return plan


def load_plan(root, path):
    plan = json.loads(Path(path).read_text(encoding='utf-8'))
    check = dict(plan)
    expected = check.pop('manifest_hash')
    if digest(check) != expected or plan['split'] != 'dev':
        raise ValueError('Frozen manifest integrity/split mismatch.')
    dev = root/'data/prepared/dev.jsonl'
    if hashlib.sha256(dev.read_bytes()).hexdigest() != plan['dev_file_sha256']:
        raise ValueError('Prepared development data changed.')
    if hashlib.sha256((root/'data/private/physician-labels.jsonl').read_bytes()).hexdigest() != plan['labels_file_sha256']:
        raise ValueError('Evaluation labels changed after sampling.')
    by_id = {r['example_id']:r for r in map(json.loads,dev.read_text(encoding='utf-8').splitlines())}
    selected = [by_id[e['example_id']] for e in plan['entries']]
    if any(digest(row)!=e['input_hash'] for row,e in zip(selected,plan['entries'])):
        raise ValueError('Frozen input changed.')
    if len({r['case']['group_id'] for r in selected}) != len(selected):
        raise ValueError('Repeated conversation in stratified selection.')
    return plan, selected
