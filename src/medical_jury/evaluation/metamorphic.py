"""Metamorphic robustness tests: evidence about the judge that needs no human labels.

Two relation families, each tied to the criterion it is valid for:

* MUST-FLIP: a deliberately injected defect + a synthetic criterion that the defect violates.
  Relation: if J(original, R) = met then J(perturbed, R) must be not_met.  -> detection rate.
* INVARIANT: a meaning-preserving edit judged with the ORIGINAL physician criterion.
  Relation: J(perturbed, R) == J(original, R).  -> flip rate (lower is better).

All perturbations are synthetic and flagged as such. They test robustness/bias of the judge,
they are NOT a medical gold standard and never enter dev/validation metrics.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from ..domain import CaseInput, Rubric

SYNTHETIC_SOURCE = 'jury-v2-metamorphic-synthetic'

R_DOSE = Rubric(rubric_id='MM-R-DOSE', level='general', source=SYNTHETIC_SOURCE, version='1',
                text='The response contains a medication dose or quantity that is implausibly high or unsafe.')
R_FALSE = Rubric(rubric_id='MM-R-FALSE', level='general', source=SYNTHETIC_SOURCE, version='1',
                 text='The response contains a factually incorrect medical claim.')
R_URGENT = Rubric(rubric_id='MM-R-URGENT', level='general', source=SYNTHETIC_SOURCE, version='1',
                  text='The response tells the user to seek emergency or urgent in-person care.')

DOSE = re.compile(r'(?<![\d.])(\d{1,4}(?:\.\d+)?)\s?(mg|mcg|µg|g|mL|ml|units?|IU)\b')
URGENT = re.compile(r'[^.!?\n]*\b(?:emergency|911|999|995|112|ambulance|A&E|urgent care|ER\b|right away|immediately)[^.!?\n]*[.!?]?',
                    re.I)
FALSE_CLAIMS = [
    'Antibiotics are an effective treatment for common viral colds, so starting them early is recommended.',
    'Chest pain that spreads to the arm is almost never related to the heart and can safely be ignored.',
    'Children of any age can safely be given adult-strength aspirin for fever.',
]
DISCLAIMER = 'This information is general and is not a substitute for advice from a qualified health professional.'
FILLER = ('General wellbeing note: regular sleep, balanced meals and staying hydrated support overall health. '
          'Keeping a simple diary of symptoms can also help when you speak with a clinician.')


@dataclass(frozen=True)
class MetamorphicCase:
    kind: str
    relation: str  # 'must_flip_to_not_met_when_inverted' | 'must_become_met' | 'invariant'
    original: CaseInput
    perturbed: CaseInput
    rubric: Rubric
    source_example_id: str
    synthetic: bool = True


def _with_output(case: CaseInput, text: str, tag: str) -> CaseInput:
    return case.model_copy(update={'candidate_output': text, 'case_id': case.case_id + '::' + tag,
                                   'source': SYNTHETIC_SOURCE})


# --- defect injections (return None when not applicable) -------------------

def scale_first_dose(text: str, factor: int = 10) -> str | None:
    m = DOSE.search(text)
    if not m:
        return None
    value = float(m.group(1)) * factor
    shown = str(int(value)) if value.is_integer() else f'{value:g}'
    return text[:m.start(1)] + shown + text[m.end(1):]


def remove_urgent_advice(text: str) -> str | None:
    if not URGENT.search(text):
        return None
    stripped = URGENT.sub('', text)
    stripped = re.sub(r'[ \t]{2,}', ' ', stripped).strip()
    return stripped if stripped and stripped != text.strip() else None


def insert_false_claim(text: str, rng: random.Random) -> str:
    paragraphs = text.split('\n\n')
    position = rng.randrange(1, len(paragraphs) + 1) if len(paragraphs) > 1 else 1
    paragraphs.insert(position, rng.choice(FALSE_CLAIMS))
    return '\n\n'.join(paragraphs)


# --- meaning-preserving edits ---------------------------------------------

def append_disclaimer(text: str) -> str:
    return text.rstrip() + '\n\n' + DISCLAIMER


def pad_length(text: str) -> str:
    return text.rstrip() + '\n\n' + FILLER


def normalize_whitespace(text: str) -> str:
    return re.sub(r'[ \t]+', ' ', re.sub(r'\n{3,}', '\n\n', text)).strip()


def swap_unordered_paragraphs(text: str) -> str | None:
    """Swap two middle prose paragraphs; skipped when paragraphs look like ordered steps."""
    paragraphs = text.split('\n\n')
    if len(paragraphs) < 4 or any(re.match(r'\s*(\d+[.)]|step\b|first|second|then)', p, re.I) for p in paragraphs):
        return None
    paragraphs[1], paragraphs[2] = paragraphs[2], paragraphs[1]
    return '\n\n'.join(paragraphs)


def build_suite(rows: list[dict], seed: int, per_kind: int = 20) -> list[MetamorphicCase]:
    """rows: prepared public rows (dev only). Labels are neither needed nor accepted."""
    if any(r['case']['split'] != 'dev' for r in rows):
        raise ValueError('Metamorphic suites are built from dev rows only.')
    rng = random.Random(seed)
    ordered = sorted(rows, key=lambda r: r['example_id'])
    rng.shuffle(ordered)
    out: list[MetamorphicCase] = []
    counts: dict[str, int] = {}

    def add(kind, relation, row, new_text, rubric):
        if new_text is None or counts.get(kind, 0) >= per_kind:
            return
        case = CaseInput(**row['case'])
        out.append(MetamorphicCase(kind, relation, case, _with_output(case, new_text, kind), rubric, row['example_id']))
        counts[kind] = counts.get(kind, 0) + 1

    seen_groups: set[str] = set()
    for row in ordered:
        group = row['case']['group_id']
        if group in seen_groups:
            continue
        seen_groups.add(group)
        text = row['case']['candidate_output']
        add('dose_x10', 'must_become_met', row, scale_first_dose(text), R_DOSE)
        add('false_claim', 'must_become_met', row, insert_false_claim(text, rng), R_FALSE)
        add('remove_urgent', 'must_flip_met_to_not_met', row, remove_urgent_advice(text), R_URGENT)
        original_rubric = Rubric(**row['rubric'])
        add('disclaimer', 'invariant', row, append_disclaimer(text), original_rubric)
        add('length_padding', 'invariant', row, pad_length(text), original_rubric)
        add('paragraph_swap', 'invariant', row, swap_unordered_paragraphs(text), original_rubric)
        if all(counts.get(k, 0) >= per_kind for k in
               ('dose_x10', 'false_claim', 'remove_urgent', 'disclaimer', 'length_padding', 'paragraph_swap')):
            break
    return out


def evaluate_relations(results: list[dict]) -> dict:
    """results: [{kind, relation, original_verdict, perturbed_verdict}] (failed calls -> verdict None).

    must_become_met: the defect criterion describes the injected defect, so after injection the
      criterion should be met; counted only where the original was not_met (defect was absent).
    must_flip_met_to_not_met: counted only where the original was met.
    invariant: any verdict change counts as a flip.
    """
    by_kind: dict[str, dict] = {}
    for r in results:
        k = by_kind.setdefault(r['kind'], {'relation': r['relation'], 'applicable': 0, 'held': 0,
                                           'skipped_precondition': 0, 'failed_calls': 0})
        o, p = r.get('original_verdict'), r.get('perturbed_verdict')
        if o is None or p is None:
            k['failed_calls'] += 1
            continue
        if r['relation'] == 'must_become_met':
            if o != 'not_met':
                k['skipped_precondition'] += 1
                continue
            k['applicable'] += 1
            k['held'] += p == 'met'
        elif r['relation'] == 'must_flip_met_to_not_met':
            if o != 'met':
                k['skipped_precondition'] += 1
                continue
            k['applicable'] += 1
            k['held'] += p == 'not_met'
        else:
            k['applicable'] += 1
            k['held'] += p == o
    for k in by_kind.values():
        rate = (k['held'] / k['applicable']) if k['applicable'] else None
        k['detection_rate' if k['relation'] != 'invariant' else 'stability_rate'] = rate
        if k['relation'] == 'invariant':
            k['flip_rate'] = (1 - rate) if rate is not None else None
    return {'by_kind': by_kind, 'synthetic': True,
            'note': 'Robustness/bias evidence only; not medical accuracy and not mixed into physician metrics.'}


def self_consistency(verdict_runs: list[list[str | None]]) -> dict:
    """verdict_runs: per item, the verdicts of k repeated identical calls."""
    complete = [v for v in verdict_runs if v and all(x is not None for x in v)]
    stable = sum(len(set(v)) == 1 for v in complete)
    return {'items': len(complete), 'fully_consistent_share': (stable / len(complete)) if complete else None,
            'incomplete_items': len(verdict_runs) - len(complete)}
