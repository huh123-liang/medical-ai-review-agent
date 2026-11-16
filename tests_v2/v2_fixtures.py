"""Shared synthetic project builder for tests_v2 (not a conftest, to avoid name clashes)."""
import importlib.util
import json
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_project(root: Path, n_groups=60, per_group=3, raters=2, splits=('dev',)):
    """Tiny prepared split files + rater index with synthetic labels for script-level tests."""
    (root / 'data/prepared').mkdir(parents=True, exist_ok=True)
    (root / 'data/private').mkdir(parents=True, exist_ok=True)
    topics = ['hedging', 'communication', 'emergency_referrals', 'context_seeking']
    with (root / 'data/private/physician-rater-index.jsonl').open('w') as fr:
        for split in splits:
            prefix = '' if split == 'dev' else split[0].upper()
            with (root / f'data/prepared/{split}.jsonl').open('w') as fp:
                for g in range(n_groups):
                    for i in range(per_group):
                        eid, gid, cid = f'HBX-{prefix}{g}-{i}', f'HBG-{prefix}{g}', f'HBC-{prefix}{g}-{i}'
                        category = f'cluster:{topics[g % 4]}_x'
                        row = {'example_id': eid, 'category': category,
                               'case': {'case_id': cid, 'group_id': gid, 'task': 't', 'split': split,
                                        'context': [{'role': 'user', 'content': f'question {g}'}],
                                        'candidate_output': f'Answer {g}-{i}. Take 500 mg twice daily.\n\nCall 995 if worse.'},
                               'rubric': {'rubric_id': f'R-{prefix}{g}-{i}', 'text': 'criterion', 'level': 'case',
                                          'case_id': cid, 'group_id': gid, 'split': split, 'source': 's', 'version': '1'}}
                        fp.write(json.dumps(row) + '\n')
                        n = raters + (1 if g % 5 == 0 else 0)
                        labels = [((g + i + k) % 4 != 0) for k in range(n)]
                        fr.write(json.dumps({'example_id': eid, 'group_id': gid, 'split': split, 'category': category,
                                             'topic': topics[g % 4], 'labels': labels,
                                             'physician_ids': [f'P{(g + k) % 7}' for k in range(n)]}) + '\n')
    write_index_meta(root)
    return root


def write_index_meta(root: Path, source_sha: str = 'synthetic-source'):
    """Synthetic provenance so verify_rater_index() accepts the synthetic index."""
    from medical_jury.evaluation.data import file_sha256_lf
    (root / 'data/public').mkdir(parents=True, exist_ok=True)
    manifest = root / 'data/public/source-manifest.json'
    if not manifest.exists():
        manifest.write_text(json.dumps({'sha256': source_sha}), encoding='utf-8')
    index = root / 'data/private/physician-rater-index.jsonl'
    meta = {'source_sha256': json.loads(manifest.read_text())['sha256'], 'index_sha256': file_sha256_lf(index)}
    (root / 'data/private/physician-rater-index.meta.json').write_text(json.dumps(meta), encoding='utf-8')
