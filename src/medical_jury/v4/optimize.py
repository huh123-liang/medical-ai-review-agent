"""Step 5 of PLAN-V4: DSPy optimisation of the CHECK instruction only (no demonstrations), on the train split only.

* The optimiser is DSPy GEPA (reflective instruction search). Every model call goes through the project's budgeted
  backend (LedgerEngine), with a hard cap on metric calls (one metric call = one check call).
* A custom adapter sends exactly the message format the pipeline uses at test time (system = instruction,
  user = the same JSON payload), so the optimised instruction is evaluated in the format it will run in.
* The extraction step is not optimised here: each training case reuses the extraction and criteria list of a sealed
  train-split run, so only the check call is repeated.
* Training signal: testbed error labels of the TRAIN split (synthetic; never physician labels).
* Guards on the result: no demos, length cap, and no verbatim copying of training documents into the instruction.
"""
from __future__ import annotations

import json
import re

import dspy

from ..domain import digest
from ..dspy_judge import make_lm
from ..evaluation import v4_scoring as SC
from ..judge_v2 import BackendError
from .check import SYSTEM_CHECK, CheckFailure, parse_check
from .rules import SEVERE

MAX_INSTRUCTION_RATIO = 2.5          # optimised instruction may be at most 2.5x the hand-written one
LEAK_WORDS = 12                      # no run of 12+ consecutive words copied from a training document


class CappedBackend:
    """Hard call cap in front of the budgeted backend: call cap+1 is refused before anything is sent."""

    def __init__(self, inner, cap: int):
        self.inner, self.cap, self.calls, self.refused = inner, cap, 0, 0
        self.name, self.model = getattr(inner, 'name', '?'), getattr(inner, 'model', '?')
        self.billable = getattr(inner, 'billable', True)

    def complete(self, messages, max_tokens, job_id):
        if self.calls >= self.cap:
            self.refused += 1
            raise BackendError('OPTIMISER_CALL_CAP')
        self.calls += 1
        return self.inner.complete(messages, max_tokens, job_id)


class V4Adapter(dspy.Adapter):
    """system = signature instructions; user = the JSON payload unchanged; the whole completion is the output."""

    def format(self, signature, demos, inputs):
        if demos:
            raise ValueError('v4 optimisation is instruction-only: demonstrations are not allowed')
        return [{'role': 'system', 'content': signature.instructions},
                {'role': 'user', 'content': inputs['payload']}]

    def parse(self, signature, completion):
        return {'review': completion}


class CheckSignature(dspy.Signature):
    """placeholder"""
    payload: str = dspy.InputField()
    review: str = dspy.OutputField()


class CheckProgram(dspy.Module):
    def __init__(self, instruction: str = SYSTEM_CHECK):
        super().__init__()
        self.check = dspy.Predict(CheckSignature.with_instructions(instruction))

    def forward(self, payload: str):
        with dspy.context(adapter=V4Adapter()):      # only the check call uses the pipeline's own format
            return self.check(payload=payload)


def instruction_of(program: CheckProgram) -> str:
    return program.check.signature.instructions


def make_example(case: dict, record: dict, document: str) -> dspy.Example:
    """One training example from a testbed case and its sealed full-arm record (extraction + criteria + code)."""
    from .check import check_messages, code_checks_of
    rule_findings = [f for f in record['findings'] + record.get('uncertain', []) if f['source'] == 'rule']
    payload = check_messages(case['patient_summary'], document, record['items'],
                             code_checks=code_checks_of(rule_findings, record['extraction']['medications']))[1]['content']
    coded = [f for f in record['findings'] if f['source'] in ('rule', 'criteria')]
    return dspy.Example(payload=payload, criteria_items=record['items'], coded=coded, golds=SC.golds(case),
                        case_id=case['case_id'], document=document).with_inputs('payload')


def score_review(example, raw: str) -> tuple[float, str]:
    """Score in [0, 1] plus textual feedback for the reflection model."""
    items = [{'id': i['id']} for i in example.criteria_items]
    meta = {i['id']: i for i in example.criteria_items}
    try:
        parsed = parse_check(raw, items)
    except CheckFailure as e:
        return 0.0, f'Output rejected ({e.code}): return valid JSON with every criterion id.'
    findings = list(example.coded)
    for cid, res in parsed['criteria'].items():
        if res['status'] == 'not_met':
            findings.append({'category': meta[cid].get('category', ''), 'entity': meta[cid].get('entity', ''),
                             'issue': meta[cid]['text'], 'severity': meta[cid].get('severity_if_not_met', 'moderate')})
    findings += parsed['other_findings']
    rec = {'findings': findings}
    if not example.golds:
        severe = [f for f in findings if f['severity'] == SEVERE]
        if severe:
            return 0.0, 'This document has no error, but a severe problem was reported: ' + \
                '; '.join(f.get('issue', '')[:120] for f in severe[:3])
        return 1.0, 'Correct: no severe problem reported on an error-free document.'
    hit = [SC.found(g, rec) for g in example.golds]
    missed = [g['description'] for g, h in zip(example.golds, hit) if not h]
    score = sum(hit) / len(hit)
    return score, ('All errors found.' if not missed else 'Missed: ' + '; '.join(missed))


def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    score, feedback = score_review(example, prediction.review)
    return dspy.Prediction(score=score, feedback=feedback)


def leak_check(instruction: str, documents: list[str], baseline: str = SYSTEM_CHECK) -> list[str]:
    problems = []
    if len(instruction) > MAX_INSTRUCTION_RATIO * len(baseline):
        problems.append(f'instruction too long ({len(instruction)} chars > {MAX_INSTRUCTION_RATIO}x baseline)')
    words = re.findall(r"[a-z0-9']+", instruction.lower())
    grams = {' '.join(words[i:i + LEAK_WORDS]) for i in range(max(0, len(words) - LEAK_WORDS + 1))}
    for d in documents:
        dw = re.findall(r"[a-z0-9']+", d.lower())
        if any(' '.join(dw[i:i + LEAK_WORDS]) in grams for i in range(max(0, len(dw) - LEAK_WORDS + 1))):
            problems.append('instruction copies a training document verbatim')
            break
    return problems


def optimise_check(train: list[dspy.Example], val: list[dspy.Example], backend, reflection_backend, job_id: str,
                   max_task_calls: int, max_reflection_calls: int, seed: int = 0) -> dict:
    """GEPA stops by its own budget (max_metric_calls) a little before the hard cap; the hard caps guarantee that
    no more than max_task_calls + max_reflection_calls calls are ever sent."""
    task = CappedBackend(backend, max_task_calls)
    reflect = CappedBackend(reflection_backend, max_reflection_calls)
    budget = max_task_calls - len(val) - 6
    if budget < 2 * len(val):
        raise ValueError('max_task_calls too small for this validation set')
    student = CheckProgram()
    task_lm = make_lm(task, f'{job_id}:task')
    reflection_lm = make_lm(reflect, f'{job_id}:reflect')
    with dspy.context(lm=task_lm, adapter=dspy.JSONAdapter()):   # reflection prompts ask for JSON (JSON-mode API)
        optimiser = dspy.GEPA(metric=metric, max_metric_calls=budget, reflection_lm=reflection_lm,
                              reflection_minibatch_size=3, candidate_selection_strategy='pareto', use_merge=False,
                              num_threads=1, seed=seed, skip_perfect_score=True, track_stats=False)
        best = optimiser.compile(student, trainset=train, valset=val)
    instruction = instruction_of(best)
    problems = leak_check(instruction, [e.document for e in train + val])
    if task.refused or reflect.refused:
        problems.append(f'call cap reached (task refused {task.refused}, reflection refused {reflect.refused})')
    return {'instruction': instruction, 'sha256': digest(instruction), 'baseline_sha256': digest(SYSTEM_CHECK),
            'changed': instruction != SYSTEM_CHECK, 'guard_problems': problems,
            'calls': {'task': task.calls, 'reflection': reflect.calls},
            'caps': {'task': max_task_calls, 'reflection': max_reflection_calls, 'gepa_max_metric_calls': budget},
            'train_cases': [e.case_id for e in train], 'val_cases': [e.case_id for e in val],
            'optimiser': 'dspy.GEPA', 'dspy_version': dspy.__version__}
