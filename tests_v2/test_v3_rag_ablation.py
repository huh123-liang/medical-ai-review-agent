"""Step 6: rubric-selection ablation (single agent / router + evaluator / oracle). Synthetic data, mock judges."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from medical_jury.evaluation.ablation_scoring import score_ablation
from medical_jury.v3 import ablation as AB
from medical_jury.v3.evaluator import step_numbers
from medical_jury.v3.schema import load_kb

from v2_fixtures import load_script

REPO = Path(__file__).resolve().parents[1]
KB = load_kb(REPO / 'kb' / 'v3')


def all_clusters():
    return sorted(AB.cluster_index(KB))


def test_every_healthbench_cluster_resolves_to_its_kb_criterion():
    clusters = all_clusters()
    assert len(clusters) == 34
    for c in clusters:
        t = AB.resolve(c, KB)
        assert t.gold_situation.startswith(f'HB:{t.theme}:') and t.group == f'HB:{t.theme}'
        assert AB.item_for(KB, t, t.gold_situation) is AB.cluster_index(KB)[c]
    t = AB.resolve('cluster:context_seeking_not-enough-context_helpful_safe', KB)
    assert (t.gold_situation, t.axis) == ('HB:context_seeking:not-enough-context', 'helpful_safe')
    with pytest.raises(ValueError):
        AB.resolve('cluster:unknown_theme_axis', KB)


def test_sibling_without_this_aspect_has_no_criterion():
    t = AB.resolve('cluster:context_seeking_enough-context_precise', KB)
    assert AB.item_for(KB, t, 'HB:context_seeking:not-enough-context') is None
    route = {'status': 'Completed', 'choices': [{'group': 'HB:context_seeking',
                                                 'situation': 'HB:context_seeking:not-enough-context', 'confidence': 0.9}]}
    assert AB.check_for_arm('rag', KB, t, route) is None


def test_rag_with_the_gold_situation_sends_exactly_the_oracle_check():
    t = AB.resolve('cluster:emergency_referrals_emergent_emergency_behavior', KB)
    gold = {'status': 'Completed', 'choices': [{'group': t.group, 'situation': t.gold_situation, 'confidence': 0.8}]}
    assert AB.check_for_arm('rag', KB, t, gold) == AB.check_for_arm('oracle', KB, t)
    wrong = {'status': 'Completed', 'choices': [{'group': t.group, 'situation': 'HB:emergency_referrals:non-emergent',
                                                 'confidence': 0.8}]}
    assert AB.check_for_arm('rag', KB, t, wrong).text != AB.check_for_arm('oracle', KB, t).text
    for route in (None, {'status': 'Failed'}, {'status': 'Completed', 'choices': [
            {'group': t.group, 'situation': 'none', 'confidence': 0.7}]}):
        assert AB.rag_selection(route, t) is None and AB.check_for_arm('rag', KB, t, route) is None


def test_single_agent_check_holds_the_whole_theme_rubric_without_merged_steps():
    for c in all_clusters():
        t = AB.resolve(c, KB)
        item = AB.single_agent_check(KB, t)
        for sit in AB.siblings(KB, t):
            assert sit.applies_when in item.text
        assert step_numbers(item.text) == []          # sibling procedures are not merged into one step list
        unprefixed = '\n'.join(line[2:] if line.startswith('| ') else line for line in item.text.splitlines())
        assert AB.check_for_arm('oracle', KB, t).text.strip() in unprefixed


def test_verdict_mapping():
    rec = lambda r: {'status': 'Completed', 'checks': [{'result': r}]}  # noqa: E731
    assert AB.verdict(rec('pass')) == 'met' and AB.verdict(rec('not_applicable')) == 'met'
    assert AB.verdict(rec('fail')) == 'not_met'
    assert AB.verdict({'status': 'Failed'}) is None and AB.verdict(None) is None


def _rows(n=40):
    rows = []
    for k in range(n):
        label = k % 3 != 0
        rows.append({'example_id': f'e{k}', 'group_id': f'g{k}', 'topic': 'hedging', 'labels': [label, label],
                     'physician_ids': ['P1', 'P2'],
                     'verdicts': {'single': 'met', 'rag': 'met' if label else 'not_met',
                                  'oracle': 'met' if label else 'not_met'},
                     'rag': {'router_status': 'Completed', 'selected': 'S1' if k % 2 else 'S2', 'gold': 'S1',
                             'assessed': True}})
    return rows


def test_score_ablation_paired_comparison_and_router_metrics():
    m = score_ablation(_rows(), n_boot=200)
    primary = m['comparisons']['rag_minus_single']
    assert primary['kappa_b'] == pytest.approx(1.0) and primary['delta_kappa'] > 0.5
    assert m['primary_reading'].startswith('rag better')
    assert m['router']['accuracy_on_completed'] == pytest.approx(0.5)
    assert m['comparisons']['oracle_minus_rag']['delta_kappa'] == pytest.approx(0.0)
    rows = _rows()
    for r in rows[:5]:
        r['verdicts']['rag'] = None                    # failed evaluator calls
    m2 = score_ablation(rows, n_boot=50)
    assert m2['arms']['rag']['coverage_ok'] is False and m2['primary_reading'].startswith('coverage rule failed')
    assert m2['comparisons']['rag_minus_single']['items'] == 35


# ------------------------------------------------------------- end-to-end with mock judges

def make_root(tmp_path, n_groups=24) -> Path:
    root = tmp_path / 'proj'
    shutil.copytree(REPO / 'kb' / 'v3', root / 'kb' / 'v3')
    (root / 'data/prepared').mkdir(parents=True)
    (root / 'data/private').mkdir(parents=True)
    clusters = all_clusters()
    with (root / 'data/prepared/dev.jsonl').open('w') as fp, \
            (root / 'data/private/physician-rater-index.jsonl').open('w') as fr:
        for g in range(n_groups):
            for i in range(2):
                eid, gid, cid = f'HBX-{g}-{i}', f'HBG-{g}', f'HBC-{g}-{i}'
                cat = clusters[(g + i) % len(clusters)]
                fp.write(json.dumps({'example_id': eid, 'category': cat, 'case': {
                    'case_id': cid, 'group_id': gid, 'task': 't', 'split': 'dev',
                    'context': [{'role': 'user', 'content': f'I am a nurse. Is this emergent? question {g}'}],
                    'candidate_output': f'Answer {g}-{i}.'}}) + '\n')
                fr.write(json.dumps({'example_id': eid, 'group_id': gid, 'split': 'dev', 'category': cat,
                                     'topic': cat.split(':')[1].split('_')[0], 'labels': [(g + i) % 3 != 0] * 2,
                                     'physician_ids': ['P1', 'P2']}) + '\n')
    return root


def run(*argv):
    return load_script('run_rag_ablation_v3').main(list(argv))


def test_three_stages_end_to_end_with_mock_judges(tmp_path, capsys):
    root = make_root(tmp_path)
    labels = root / 'data/private/physician-rater-index.jsonl'
    hidden = labels.with_suffix('.hidden')
    labels.rename(hidden)                              # route/evaluate must work without any label file
    assert run('route', '--root', str(root), '--n-groups', '12') == 0
    run_dir = next((root / 'reports').glob('rag-ablation-v3-*'))
    assert (run_dir / 'routes-seal.json').exists()
    manifest = json.loads((run_dir / 'sample-manifest.json').read_text())
    assert len(manifest['example_ids']) == 12 and manifest['outcome_blind'] is True
    for k in (1, 2):
        assert run('evaluate', '--root', str(root), '--run-dir', str(run_dir), '--parts', '2', '--part', str(k)) == 0
    with pytest.raises(SystemExit, match='already run'):
        run('evaluate', '--root', str(root), '--run-dir', str(run_dir), '--parts', '2', '--part', '1')
    with pytest.raises(SystemExit, match='re-partitioned'):
        run('evaluate', '--root', str(root), '--run-dir', str(run_dir), '--parts', '3', '--part', '1')
    hidden.rename(labels)
    capsys.readouterr()
    assert run('score', '--root', str(root), '--run-dir', str(run_dir), '--bootstrap', '50') == 0
    out = json.loads(capsys.readouterr().out)
    assert out['primary'] == 'rag_minus_single'
    metrics = json.loads((next(run_dir.glob('score-*')) / 'metrics.json').read_text())
    assert metrics['planned_items'] == 12 and set(metrics['arms']) == {'single', 'rag', 'oracle'}
    items = [json.loads(l) for l in (next(run_dir.glob('score-*')) / 'items.jsonl').read_text().splitlines()]
    assert all(set(i['verdicts']) == {'single', 'rag', 'oracle'} for i in items)


def test_score_refuses_missing_part_or_tampered_predictions(tmp_path):
    root = make_root(tmp_path)
    assert run('route', '--root', str(root), '--n-groups', '10') == 0
    run_dir = next((root / 'reports').glob('rag-ablation-v3-*'))
    assert run('evaluate', '--root', str(root), '--run-dir', str(run_dir), '--parts', '2', '--part', '1') == 0
    with pytest.raises(SystemExit, match='Part 2/2'):
        run('score', '--root', str(root), '--run-dir', str(run_dir))
    assert run('evaluate', '--root', str(root), '--run-dir', str(run_dir), '--parts', '2', '--part', '2') == 0
    with (run_dir / 'evaluate-part2of2' / 'predictions.jsonl').open('a') as f:
        f.write('\n')
    with pytest.raises(SystemExit, match='seal'):
        run('score', '--root', str(root), '--run-dir', str(run_dir))


def test_routes_are_sealed_before_evaluation(tmp_path):
    root = make_root(tmp_path)
    assert run('route', '--root', str(root), '--n-groups', '6') == 0
    run_dir = next((root / 'reports').glob('rag-ablation-v3-*'))
    with (run_dir / 'routes.jsonl').open('a') as f:
        f.write('\n')
    with pytest.raises(SystemExit, match='seal'):
        run('evaluate', '--root', str(root), '--run-dir', str(run_dir))


def test_later_stages_need_the_run_dir(tmp_path):
    with pytest.raises(SystemExit, match='--run-dir'):
        run('evaluate', '--root', str(make_root(tmp_path)))
