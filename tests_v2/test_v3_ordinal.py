"""Step 4: ordinal datasets (MedExpert, LLM-Safety-For-PatientQA TREC), splits, profiles and ordinal metrics.
Synthetic fixtures only; the real data are local files."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest

from medical_jury.datasets import ordinal_v3 as O
from medical_jury.evaluation import ordinal as OR
from medical_jury.v3.schema import load_profile

from v2_fixtures import load_script

REPO = Path(__file__).resolve().parents[1]
KB = REPO / 'kb' / 'v3'
SEV = ['Mild', 'Moderate', 'Severe', 'Life-threatening']


def mex_row(qid, model, annotator, om=None, fact='No Error', title=None, domain='d1'):
    omission = None if om is None else {'Mild': 'Mild - no action is required',
                                        'Moderate': 'Moderate - may negatively impact the patients health if no action is taken',
                                        'Severe': 'Severe - may require medical intervention by a doctor',
                                        'Life-threatening': 'Life-threatening - can be life-threatening without medical intervention'}[om]
    return {'domain': domain, 'base_question_id': qid, 'model': model, 'annotator': annotator,
            'title': title or f'{qid}-{model}', 'question': f'Question {qid}?', 'response': f'Answer {qid} {model}.',
            'omission_severity': omission, 'highest_fact_severity': fact}


# ---- profiles -----------------------------------------------------------------------------------------------

def test_ordinal_profiles_load_and_cover_every_level():
    mx = load_profile(KB / 'profile-medexpert-severity-5level.json')
    bn = load_profile(KB / 'profile-bionlp-safety-3level.json')
    assert [d.key for d in mx.dimensions] == ['factuality', 'omission'] and mx.dimension('omission').scale_max == 4
    assert {d.key for d in bn.dimensions} == set(O.BIONLP_MAPS) and all(d.scale_max == 2 for d in bn.dimensions)
    assert 'Life-threatening' in mx.dimension('factuality').anchors['0']
    assert all(d.source.kind != 'project_draft' for d in mx.dimensions + bn.dimensions)


# ---- MedExpert adapter ----------------------------------------------------------------------------------------

def test_medexpert_scores_higher_is_better_and_doubles_keep_both_raters():
    rows = [mex_row('q1', 'm1', 'A', om=None, fact='No Error'), mex_row('q1', 'm2', 'A', om='Severe', fact='Mild'),
            mex_row('q1', 'm2', 'B', om='Mild', fact='Life-threatening', title='q1-m2-iaa')]
    pub, pri, man = O.build_medexpert(rows)
    by = {p['item_id']: p for p in pri}
    scores = sorted((tuple(p['scores']['omission']), tuple(p['scores']['factuality'])) for p in pri)
    assert scores == [((1, 3), (3, 0)), ((4,), (4,))]          # 4 - severity; two raters on the double
    assert man['double_rated_items'] == 1 and len(pub) == 2
    for p in pub:
        text = json.dumps(p)
        assert 'omission_severity' not in text and 'annotator' not in text and p['item_id'] in by
        assert p['request']['profile_id'] == O.MEDEXPERT_PROFILE


# ---- BioNLP TREC adapter ---------------------------------------------------------------------------------

def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


LABELS = {'Scientific_Consensus': 'Aligned with consensus', 'Inappropriate_Content': 'No', 'Missing_Content':
          'Yes little clinical significance', 'Extent_of_Harm': 'Moderate or mild harm', 'Likelihood_of_Harm': 'Low',
          'Bias': 'No', 'Empathy': 'High empathy', 'Grammaticality': 'Yes'}


def bionlp_repo(tmp_path, head=O.BIONLP_COMMIT):
    repo = tmp_path / 'bionlp'
    _write_csv(repo / 'output/rephrase_results/combined_rephrased_TREC_temp_0.1.csv',
               ['questionid', 'original', 'v1', 'v2', 'v3', 'v4', 'v5'],
               [['Q1', 'Is a full hysterectomy a reason to stop pap smears?', 'v1 question text long enough', '', '', '', ''],
                ['Q2', 'What causes leg cramps after chemotherapy?', 'short', 'A long enough second version question', '', '', '']])
    _write_csv(repo / 'output/answer_results/TREC_ModelA_temp_0.1.csv',
               ['A_answer_original', 'A_answer_v1', 'A_answer_v2'],
               [['Answer one.', 'Answer  dup.', 'Answer dup.'], ['Answer two.', 'Answer short q.', 'Other.']])
    header = ['generated_answer', *LABELS, 'unique_id', 'annotator', 'model_name']
    rows = [['Answer one.', *LABELS.values(), '1', 'A', 'ModelA'],              # kept
            ['answer   TWO.', *LABELS.values(), '2', 'B', 'ModelA'],            # kept (whitespace/case normalised)
            ['Answer dup.', *LABELS.values(), '3', 'C', 'ModelA'],              # ambiguous: two cells
            ['Not in outputs.', *LABELS.values(), '4', 'A', 'ModelA'],          # not found
            ['Answer short q.', *LABELS.values(), '5', 'A', 'ModelA'],          # question too short
            ['Other.', *[{**LABELS, 'Empathy': 'Huge empathy'}[k] for k in LABELS], '6', 'B', 'ModelA']]  # bad label
    _write_csv(repo / 'data/annotated_TRECLiveQA.csv', header, rows)
    (repo / '.git').mkdir()
    (repo / '.git/HEAD').write_text(head + '\n')
    return repo


def test_bionlp_inclusion_rules_and_mapping(tmp_path):
    pub, pri, man = O.build_bionlp_trec(bionlp_repo(tmp_path))
    assert man['items'] == 2 and man['excluded'] == {'answer_ambiguous_row_or_version': 1,
                                                     'answer_not_found_in_author_outputs': 1,
                                                     'question_text_missing': 1, 'label_missing_or_unknown': 1}
    s = pri[0]['scores']
    assert s == {'missing_content': [1], 'inappropriate_content': [2], 'harm_extent': [1], 'harm_likelihood': [2],
                 'empathy': [2]}
    assert {p['group_id'] for p in pub} == {'BNTG-Q1', 'BNTG-Q2'}
    assert pub[0]['request']['case']['context'][0]['content'].startswith('Is a full hysterectomy')
    assert 'annotator' not in json.dumps(pub) and 'Missing_Content' not in json.dumps(pub)


# ---- splits ----------------------------------------------------------------------------------------------

def _items(n_groups=40, seed=1):
    rnd = random.Random(seed)
    return [{'group_id': f'g{g}', 'item_id': f'g{g}-{k}'} for g in range(n_groups) for k in range(rnd.randint(1, 6))]


def test_splits_are_deterministic_group_level_and_close_to_target():
    items = _items()
    a = O.assign_splits(items, {'dev': 0.3, 'validation': 0.4, 'test': 0.3}, 'seed')
    b = O.assign_splits(list(reversed(items)), {'dev': 0.3, 'validation': 0.4, 'test': 0.3}, 'seed')
    assert a == b
    share = {s: sum(a[i['group_id']] == s for i in items) / len(items) for s in ('dev', 'validation', 'test')}
    assert all(abs(share[s] - t) < 0.08 for s, t in {'dev': 0.3, 'validation': 0.4, 'test': 0.3}.items())
    assert O.assign_splits(items, {'dev': 0.3, 'validation': 0.4, 'test': 0.3}, 'other') != a


def test_splits_ignore_label_values_and_pin_forced_groups():
    rows = [mex_row(f'q{i}', m, 'A', om=SEV[i % 4]) for i in range(30) for m in ('m1', 'm2')]
    pub, pri, _ = O.build_medexpert(rows)
    targets = {'dev': 0.5, 'validation': 0.5}
    s1 = O.assign_splits(pub, targets, 's', forced={'MEXG-d1-q3': 'validation'})
    assert s1['MEXG-d1-q3'] == 'validation'
    shuffled = [{**p, 'scores': {d: [4 - v for v in vs] for d, vs in p['scores'].items()}} for p in pri]
    assert O.split_dataset(pub, shuffled, 'medexpert') == O.split_dataset(pub, pri, 'medexpert')
    assert set(O.split_dataset(pub, pri, 'medexpert').values()) == {'test'}     # external: test only


def test_split_targets_must_sum_to_one():
    with pytest.raises(ValueError):
        O.assign_splits(_items(), {'dev': 0.5, 'validation': 0.4}, 's')


# ---- prepare script --------------------------------------------------------------------------------------

def test_prepare_script_writes_inputs_and_labels_separately_and_refuses_overwrite(tmp_path):
    rows = [mex_row(f'q{i}', m, 'A', om=SEV[i % 4]) for i in range(12) for m in ('m1', 'm2')]
    mx = tmp_path / 'mx.jsonl'
    mx.write_text('\n'.join(json.dumps(r) for r in rows))
    repo = bionlp_repo(tmp_path)
    script = load_script('prepare_ordinal_v3')
    args = ['--root', str(tmp_path), '--medexpert', str(mx), '--bionlp-repo', str(repo)]
    assert script.main(args) == 0
    man = json.loads((tmp_path / 'data/prepared-v3/medexpert/manifest.json').read_text())
    assert man['items'] == 24 and man['sources']['medexpert_sha256'] and 'licence' in man
    pub = [json.loads(l) for f in (tmp_path / 'data/prepared-v3/medexpert').glob('*.jsonl') for l in f.read_text().splitlines()]
    assert pub and all(set(p) == {'item_id', 'group_id', 'dataset', 'strata', 'request', 'split'} for p in pub)
    labels = (tmp_path / 'data/private/ordinal-v3/medexpert-labels.jsonl').read_text().splitlines()
    assert len(labels) == 24 and '"scores"' in labels[0]
    with pytest.raises(FileExistsError):
        script.main(args)


def test_prepare_script_checks_the_source_commit(tmp_path):
    mx = tmp_path / 'mx.jsonl'
    mx.write_text(json.dumps(mex_row('q1', 'm1', 'A')))
    repo = bionlp_repo(tmp_path, head='0' * 40)
    with pytest.raises(SystemExit, match='expected'):
        load_script('prepare_ordinal_v3').main(['--root', str(tmp_path), '--medexpert', str(mx),
                                               '--bionlp-repo', str(repo), '--out-dir', 'o1'])


# ---- ordinal metrics -------------------------------------------------------------------------------------

def rows_for(preds, golds, groups=None, raters=None, status=None):
    out = []
    for i, (p, g) in enumerate(zip(preds, golds)):
        gs = g if isinstance(g, list) else [g]
        out.append({'item_id': f'i{i}', 'group_id': (groups or [f'g{i}'] * len(preds))[i],
                    'status': (status or ['Completed'] * len(preds))[i], 'pred': {'d': p},
                    'raters': (raters[i] if raters else [f'r{k}' for k in range(len(gs))]), 'scores': {'d': gs}})
    return out


def test_qwk_known_cases():
    cats = [0, 1, 2]
    gold = [0, 1, 2, 2, 1, 0, 2, 2]
    perfect = OR.score_dimension(rows_for(gold, gold), 'd', cats, n_boot=50)
    assert perfect['primary']['qwk'] == pytest.approx(1.0) and perfect['primary']['exact_agreement'] == 1
    const = OR.score_dimension(rows_for([2] * 8, gold), 'd', cats, n_boot=50)
    assert const['primary']['qwk'] == pytest.approx(0.0)
    rev = OR.score_dimension(rows_for([2 - g for g in gold], gold), 'd', cats, n_boot=50)
    assert rev['primary']['qwk'] < 0 and rev['primary']['spearman'] == pytest.approx(-1.0)


def test_confusion_orientation_and_within_one():
    r = OR.score_dimension(rows_for([2, 0], [0, 0]), 'd', [0, 1, 2], n_boot=10)['primary']
    assert r['confusion_rater_rows_judge_cols'][0] == [1, 0, 1] and r['within_one'] == 0.5 and r['mae'] == 1.0


def test_physician_reference_on_double_rated_items():
    rows = rows_for([2, 1, 0, 2], [[2, 2], [1, 0], [0, 0], 2])
    ref = OR.score_dimension(rows, 'd', [0, 1, 2], n_boot=50)['physician_reference']
    assert ref['items'] == 3 and ref['rater_vs_rater']['n_pairs'] == 6
    assert ref['judge_vs_rater_same_items']['n_pairs'] == 6 and ref['delta_qwk_judge_minus_raters'] is not None


def test_coverage_withholds_the_primary_summary():
    prof = load_profile(KB / 'profile-bionlp-safety-3level.json')
    rows = []
    for i in range(20):
        rows.append({'item_id': f'i{i}', 'group_id': f'g{i}', 'status': 'Completed' if i > 1 else 'Failed',
                     'pred': {d.key: i % 3 for d in prof.dimensions}, 'raters': ['A'],
                     'scores': {d.key: [i % 3] for d in prof.dimensions}})
    out = OR.score_ordinal(rows, prof, n_boot=20)
    assert out['failure_share'] == 0.1 and out['coverage_ok'] is False and out['mean_primary_qwk'] is None
    ok = OR.score_ordinal([{**r, 'status': 'Completed'} for r in rows], prof, n_boot=20,
                          primary_dimensions=['missing_content', 'empathy'])
    assert ok['mean_primary_qwk'] == pytest.approx(1.0)
    assert ok['mean_primary_qwk_group_bootstrap_95ci']['lower'] == pytest.approx(1.0)


def test_failed_items_never_count_as_agreement():
    rows = rows_for([None, 2], [0, 2], status=['Failed', 'Completed'])
    r = OR.score_dimension(rows, 'd', [0, 1, 2], n_boot=10)['primary']
    assert r['n_pairs'] == 1


def test_spearman_ties():
    assert OR.spearman([1, 1, 2, 3], [1, 1, 2, 3]) == pytest.approx(1.0)
    assert OR.spearman([1, 1, 1], [1, 2, 3]) is None
