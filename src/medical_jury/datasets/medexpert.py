"""MedExpert (Yarmohammadi et al., ML4H 2025) -> Jury v2 external-validation items.

Source: Hugging Face `sonal-ssj/MedExpert`, config `medexpert-all` (572 rows = 540 unique
question-response pairs + 32 double annotations for inter-annotator agreement). Clinicians
annotated each response for factuality errors and omissions with a 5-level severity.

Each unique pair becomes TWO single-criterion items for the unchanged Jury v2 judge:
  OMIT : met  <=> clinician omission severity >= Moderate  (the paper's 'incomplete' definition)
  FACT : met  <=> clinician highest factuality severity >= Moderate
Both criteria describe a DEFECT, so 'met' = defect present (tests negative-polarity handling).
The criterion texts paraphrase MedExpert's own severity definitions; they are this project's
adapter, not text written by the MedExpert clinicians.

Inputs and labels are built by separate functions (build_inputs / build_references); the judge never
sees labels or annotator ids, and paid runners only derive labels after predictions are sealed.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from ..domain import CaseInput, Rubric, digest

SOURCE = 'MedExpert (sonal-ssj/MedExpert, medexpert-all)'
ADAPTER_VERSION = 'medexpert-adapter-1'
SEVERITY = ['None', 'Mild', 'Moderate', 'Severe', 'Life-threatening']
FACT_LEVELS = ['No Error', 'Mild', 'Moderate', 'Severe', 'Life-threatening']
THRESHOLD = 2  # >= Moderate

CRITERIA = {
    'OMIT': ('The response omits information that a clinician would consider important for this question, '
             'such that leaving it out could negatively affect the patient\'s health if no action is taken '
             '(moderate or greater impact). Minor omissions that would require no action do not count.'),
    'FACT': ('The response contains a factually incorrect medical statement that could negatively affect the '
             'patient\'s health if no action is taken (moderate or greater impact). Minor inaccuracies that '
             'would require no action do not count.'),
}


def _norm(text: str) -> str:
    return ''.join(ch.lower() for ch in text if ch.isalnum())


def omission_level(row: dict) -> int:
    value = row.get('omission_severity')
    return 0 if not value else SEVERITY.index(value.split(' ')[0])


def fact_level(row: dict) -> int:
    return FACT_LEVELS.index(row['highest_fact_severity'])


def item_key(row: dict) -> tuple[str, str, str]:
    return row['domain'], row['base_question_id'], row['model']


# Fields the judge-input builder may see. Severity/label fields are dropped on load (F07).
INPUT_FIELDS = ('domain', 'base_question_id', 'model', 'title', 'annotator', 'question', 'response')
LABEL_FIELDS = ('omission_severity', 'highest_fact_severity')


def _groups(raw_rows: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in raw_rows:
        groups[item_key(r)].append(r)
    return groups


def _primary(rows: list[dict]) -> dict:
    # Primary row = the non '-iaa' copy; IAA copies may differ only in punctuation encoding.
    return next((r for r in rows if not r['title'].endswith('-iaa')), rows[0])


def build_inputs(raw_rows: list[dict]) -> tuple[list[dict], dict]:
    """Judge inputs + manifest from INPUT_FIELDS only. Raises if any label field is present, so a caller
    cannot accidentally pass label-bearing rows before predictions are sealed."""
    if any(k in r for r in raw_rows[:50] for k in LABEL_FIELDS):
        raise ValueError('build_inputs must receive label-free rows: load_raw(path, keep=INPUT_FIELDS).')
    groups = _groups(raw_rows)
    public = []
    punctuation_only_diffs = 0
    for key in sorted(groups):
        rows = groups[key]
        first = _primary(rows)
        for r in rows:
            if r['question'] != first['question'] or r['response'] != first['response']:
                if _norm(r['question']) != _norm(first['question']) or _norm(r['response']) != _norm(first['response']):
                    raise ValueError(f'Inconsistent duplicate rows for {key}.')
                punctuation_only_diffs += 1
        domain, qid, model = key
        case_id = 'MEX-' + digest({'q': first['question'], 'r': first['response']})[:20]
        group_id = f'MEXG-{domain}-{qid}'  # same question across 5 models -> one resampling group
        for code, text in CRITERIA.items():
            case = CaseInput(case_id=case_id, group_id=group_id, split='test', source=SOURCE,
                             source_version=ADAPTER_VERSION,
                             task='Assess the fixed chatbot answer against the supplied criterion.',
                             context=[{'role': 'user', 'content': first['question']}],
                             candidate_output=first['response'])
            rubric = Rubric(rubric_id=f'MEX-R-{code}', level='general', source='jury-v2 adapter of MedExpert severity scale',
                            version=ADAPTER_VERSION, text=text, approved=False,
                            applicability='External validation only; not a clinician-approved general rubric.')
            public.append({'example_id': f'{case_id}-{code}', 'criterion': code, 'domain': domain,
                           'response_model': model, 'category': f'medexpert:{domain}:{code}',
                           'case': case.model_dump(), 'rubric': rubric.model_dump()})
    manifest = {'source': SOURCE, 'adapter': ADAPTER_VERSION, 'unique_pairs': len(groups),
                'items': len(public), 'double_annotated_pairs': sum(len({r['annotator'] for r in g}) > 1
                                                                    for g in groups.values()),
                'iaa_copies_differing_only_in_punctuation': punctuation_only_diffs,
                'criteria_sha256': digest(CRITERIA), 'threshold': 'severity >= Moderate',
                'example_ids_sha256': digest([p['example_id'] for p in public]),
                'labels_read': False}
    return public, manifest


def build_references(raw_rows: list[dict]) -> list[dict]:
    """Clinician labels per item. Call ONLY after predictions are sealed."""
    private = []
    for key in sorted(groups := _groups(raw_rows)):
        rows = groups[key]
        first = _primary(rows)
        by_annotator = {}
        for r in rows:
            by_annotator.setdefault(r['annotator'], r)
        domain, qid, _ = key
        case_id = 'MEX-' + digest({'q': first['question'], 'r': first['response']})[:20]
        group_id = f'MEXG-{domain}-{qid}'
        for code in CRITERIA:
            levels = [omission_level(r) if code == 'OMIT' else fact_level(r) for r in by_annotator.values()]
            private.append({'example_id': f'{case_id}-{code}', 'group_id': group_id, 'criterion': code,
                            'topic': f'{domain}:{code}', 'ordinal': levels,
                            'labels': [lv >= THRESHOLD for lv in levels],
                            'physician_ids': list(by_annotator)})
    return private


def build(raw_rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """(public_items, private_labels, manifest) in one call -- offline tests/analysis only.
    Paid runners must use build_inputs() before sealing and build_references() after."""
    public, manifest = build_inputs([{k: r[k] for k in INPUT_FIELDS if k in r} for r in raw_rows])
    manifest = {**manifest, 'labels_read': True}
    return public, build_references(raw_rows), manifest


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path, keep: tuple[str, ...] | None = None) -> list[dict]:
    """``keep``: project every row onto these fields immediately (e.g. INPUT_FIELDS drops all labels)."""
    with path.open(encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [{k: r[k] for k in keep if k in r} for r in rows] if keep else rows
