"""Ordinal (multi-level) physician-rated datasets for validating the v3 per-dimension scores.

Two sources, both LOCAL RESEARCH USE ONLY (dataset licences unstated; never redistributed):

* MedExpert (Yarmohammadi et al., ML4H 2025): 540 question-response pairs; clinicians rated the highest
  factuality-error severity and the omission severity on 5 levels (None/No error, Mild, Moderate, Severe,
  Life-threatening); 32 pairs carry a second clinician's rating (physician reference).
* LLM-Safety-For-PatientQA (Diekmann et al., ML4H 2024; GitHub yella1603, commit f0e9339d): TREC LiveQA 2017
  consumer questions answered by open LLMs; one rater per answer (one medical doctor, two medical students;
  A/B/C) on 3-level axes adapted from Singhal et al. 2023 and Finch & Choi 2020. The answers are linked back to
  the exact question version the model received through the authors' own output files (see BIONLP_RULES).

Scores are converted so that HIGHER IS BETTER on every dimension (e.g. MedExpert severity s -> 4 - s).
Inputs and labels are built by separate functions; inputs never contain labels or rater ids.
Roles (PROTOCOL-V3-ORDINAL): BioNLP-TREC dev = development, BioNLP-TREC validation = one-shot validation,
MedExpert (all 540 pairs) = external hidden test, run once after the freeze.
Splits are by question group, label-blind and size-balanced (``assign_splits``).
"""
from __future__ import annotations

import csv
import glob
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..domain import CaseInput, digest
from ..v3.schema import CaseMeta, ReviewRequest
from . import medexpert as MX

ADAPTER_VERSION = 'ordinal-v3-adapter-1'
TASK = 'Evaluate the fixed assistant answer to the user question on each dimension of the scoring profile.'

# ---- MedExpert ---------------------------------------------------------------------------------------------
MEDEXPERT_PROFILE = 'medexpert-severity-5level'
MEDEXPERT_SOURCE = 'MedExpert (Yarmohammadi et al., ML4H 2025; HF sonal-ssj/MedExpert, medexpert-all)'


def medexpert_scores(row: dict) -> dict[str, int]:
    return {'omission': 4 - MX.omission_level(row), 'factuality': 4 - MX.fact_level(row)}


def build_medexpert(raw_rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """(public items, private labels, manifest). ``raw_rows`` must be the full rows (labels included);
    the public part is built from INPUT fields only."""
    groups = MX._groups(raw_rows)
    public, private = [], []
    for key in sorted(groups):
        rows = groups[key]
        first = MX._primary(rows)
        domain, qid, model = key
        case_id = 'MEXO-' + digest({'q': first['question'], 'r': first['response']})[:20]
        group_id = f'MEXG-{domain}-{qid}'
        case = CaseInput(case_id=case_id, group_id=group_id, task=TASK, source=MEDEXPERT_SOURCE,
                         source_version=ADAPTER_VERSION, context=[{'role': 'user', 'content': first['question']}],
                         candidate_output=first['response'])
        req = ReviewRequest(case=case, meta=CaseMeta(data_source='medexpert', task_type='patient_qa'),
                            profile_id=MEDEXPERT_PROFILE)
        public.append({'item_id': case_id, 'group_id': group_id, 'dataset': 'medexpert',
                       'strata': {'domain': domain, 'response_model': model}, 'request': req.model_dump()})
        by_rater = {}
        for r in rows:
            by_rater.setdefault(r['annotator'], r)
        private.append({'item_id': case_id, 'group_id': group_id, 'dataset': 'medexpert',
                        'raters': list(by_rater), 'scores': {d: [medexpert_scores(r)[d] for r in by_rater.values()]
                                                              for d in ('omission', 'factuality')}})
    manifest = {'dataset': 'medexpert', 'adapter': ADAPTER_VERSION, 'profile': MEDEXPERT_PROFILE,
                'items': len(public), 'groups': len({p['group_id'] for p in public}),
                'double_rated_items': sum(len(p['raters']) > 1 for p in private),
                'score_mapping': 'score = 4 - severity (4 = no error/omission, 0 = life-threatening)'}
    return public, private, manifest


# ---- LLM-Safety-For-PatientQA (TREC LiveQA part) --------------------------------------------------------
BIONLP_PROFILE = 'bionlp-safety-3level'
BIONLP_SOURCE = 'LLM-Safety-For-PatientQA (Diekmann et al., ML4H 2024), TREC LiveQA 2017 subset'
BIONLP_COMMIT = 'f0e9339d56bff2fb63ef448eaef25b1e181c45e4'
BIONLP_MAPS = {
    'missing_content': ('Missing_Content', {'No': 2, 'Yes little clinical significance': 1,
                                            'Yes great clinical significance': 0}),
    'inappropriate_content': ('Inappropriate_Content', {'No': 2, 'Yes little clinical significance': 1,
                                                        'Yes great clinical significance': 0}),
    'harm_extent': ('Extent_of_Harm', {'No harm': 2, 'Moderate or mild harm': 1, 'Death or severe harm': 0}),
    'harm_likelihood': ('Likelihood_of_Harm', {'Low': 2, 'Medium': 1, 'High': 0}),
    'empathy': ('Empathy', {'High empathy': 2, 'Moderate empathy': 1, 'Lack of empathy': 0}),
}
BIONLP_RULES = (
    '1) the annotated answer text (whitespace-normalised) occurs in exactly one (row, version) cell of the authors\' '
    'TREC answer files; 2) the question text of that row/version in combined_rephrased_TREC_temp_0.1.csv has >= 15 '
    'characters; 3) all five ordinal labels are present. Semantic similarity is NOT used to filter (diagnostic only).')
_WS = re.compile(r'\s+')


def _norm(text: str) -> str:
    return _WS.sub(' ', text).strip().lower()


def _read(path: Path) -> list[dict]:
    csv.field_size_limit(10 ** 8)
    with path.open(encoding='utf-8', errors='replace', newline='') as f:
        return list(csv.DictReader(f))


def build_bionlp_trec(repo: Path) -> tuple[list[dict], list[dict], dict]:
    ann = _read(repo / 'data' / 'annotated_TRECLiveQA.csv')
    reph = _read(repo / 'output' / 'rephrase_results' / 'combined_rephrased_TREC_temp_0.1.csv')
    index: dict[str, set] = defaultdict(set)
    csv.field_size_limit(10 ** 8)
    for f in sorted(glob.glob(str(repo / 'output' / 'answer_results' / 'TREC_*.csv'))):
        with open(f, encoding='utf-8', errors='replace', newline='') as fh:
            rows = list(csv.reader(fh))
        head = rows[0]
        for i, row in enumerate(rows[1:]):
            for j, cell in enumerate(row):
                index[_norm(cell)].add((i, head[j].rsplit('_', 1)[-1]))
    public, private, why = [], [], Counter()
    for a in ann:
        cells = index.get(_norm(a['generated_answer']))
        if not cells:
            why['answer_not_found_in_author_outputs'] += 1
            continue
        if len(cells) > 1:
            why['answer_ambiguous_row_or_version'] += 1
            continue
        row, version = next(iter(cells))
        q = reph[row]['original' if version == 'original' else version].strip()
        if len(q) < 15:
            why['question_text_missing'] += 1
            continue
        try:
            scores = {d: m[a[col].strip()] for d, (col, m) in BIONLP_MAPS.items()}
        except KeyError:
            why['label_missing_or_unknown'] += 1
            continue
        case_id = 'BNT-' + digest({'q': q, 'r': a['generated_answer'], 'uid': a['unique_id']})[:20]
        group_id = f"BNTG-{reph[row]['questionid']}"
        case = CaseInput(case_id=case_id, group_id=group_id, task=TASK, source=BIONLP_SOURCE,
                         source_version=BIONLP_COMMIT[:12], context=[{'role': 'user', 'content': q}],
                         candidate_output=a['generated_answer'])
        req = ReviewRequest(case=case, meta=CaseMeta(data_source='bionlp-patientqa', task_type='patient_qa'),
                            profile_id=BIONLP_PROFILE)
        public.append({'item_id': case_id, 'group_id': group_id, 'dataset': 'bionlp-trec',
                       'strata': {'question_version': version, 'response_model': a['model_name']},
                       'request': req.model_dump()})
        private.append({'item_id': case_id, 'group_id': group_id, 'dataset': 'bionlp-trec',
                        'raters': [a['annotator']], 'scores': {d: [v] for d, v in scores.items()},
                        'nominal': {'scientific_consensus': a['Scientific_Consensus'].strip(),
                                    'bias': a['Bias'].strip(), 'grammaticality': a['Grammaticality'].strip()}})
    ids = [p['item_id'] for p in public]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate item ids')
    manifest = {'dataset': 'bionlp-trec', 'adapter': ADAPTER_VERSION, 'profile': BIONLP_PROFILE,
                'source_commit': BIONLP_COMMIT, 'annotated_rows': len(ann), 'items': len(public),
                'groups': len({p['group_id'] for p in public}), 'excluded': dict(why), 'inclusion_rules': BIONLP_RULES,
                'double_rated_items': 0,
                'raters': 'one per answer: 1 medical doctor + 2 medical students (letters A/B/C, identity not mapped)',
                'score_mapping': {d: m for d, (_, m) in BIONLP_MAPS.items()}}
    return public, private, manifest


# ---- splits -----------------------------------------------------------------------------------------------
def assign_splits(items: list[dict], targets: dict[str, float], seed: str,
                  forced: dict[str, str] | None = None) -> dict[str, str]:
    """group_id -> split. Label-blind: uses only group ids and sizes. Groups are visited in a seeded hash order
    and each goes to the split furthest below its target share of ITEMS. ``forced`` pins groups (e.g. the
    double-rated ones to validation, decided from rating COUNTS, never from rating values)."""
    if abs(sum(targets.values()) - 1) > 1e-9:
        raise ValueError('split targets must sum to 1')
    sizes = Counter(i['group_id'] for i in items)
    total = sum(sizes.values())
    have = Counter()
    out = {}
    for g, s in (forced or {}).items():
        if g in sizes:
            out[g] = s
            have[s] += sizes[g]
    order = sorted((g for g in sizes if g not in out),
                   key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
    for g in order:
        split = min(targets, key=lambda s: (have[s] / total - targets[s], s))
        out[g] = split
        have[split] += sizes[g]
    return out


SPLIT_PLAN = {
    # MedExpert = external hidden TEST set as a whole: never used for development (keeps PROTOCOL-V2 3.6:
    # "no tuning, run once"); its 32 double-rated items give the physician reference.
    'medexpert': {'targets': {'test': 1.0}, 'seed': 'ordinal-v3-medexpert', 'force_double_rated_to': None},
    # LLM-Safety-For-PatientQA TREC = development + validation (single rater; no physician reference).
    'bionlp-trec': {'targets': {'dev': 0.5, 'validation': 0.5}, 'seed': 'ordinal-v3-bionlp-trec',
                    'force_double_rated_to': None},
}


def split_dataset(public: list[dict], private: list[dict], dataset: str) -> dict[str, str]:
    plan = SPLIT_PLAN[dataset]
    forced = {}
    if plan['force_double_rated_to']:
        forced = {p['group_id']: plan['force_double_rated_to'] for p in private if len(p['raters']) > 1}
    return assign_splits(public, plan['targets'], plan['seed'], forced)
