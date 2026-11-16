"""Prepare the ordinal validation datasets for the v3 agent (offline, no model call).

    python scripts/prepare_ordinal_v3.py \
        --medexpert data/external/medexpert/medexpert-all.jsonl \
        --bionlp-repo data/external/llm-safety-for-patientqa      # git clone of yella1603/LLM-Safety-For-PatientQA

Writes (never overwrites):
    data/prepared-v3/<dataset>/{dev,validation,test}.jsonl   judge inputs only (question, answer, profile id)
    data/prepared-v3/<dataset>/manifest.json                 counts, source hashes, split rule, exclusions
    data/private/ordinal-v3/<dataset>-labels.jsonl           rater scores + split (read only after sealing)
Both sources are local-research-use only (licences unstated); do not commit or redistribute these files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from medical_jury.datasets import medexpert as MX  # noqa: E402
from medical_jury.datasets import ordinal_v3 as O  # noqa: E402
from medical_jury.domain import digest  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(repo: Path) -> str | None:
    """Read the checked-out commit from .git without running git inside the downloaded tree."""
    head = repo / '.git' / 'HEAD'
    if not head.exists():
        return None
    ref = head.read_text().strip()
    if not ref.startswith('ref:'):
        return ref
    name = ref.split(' ', 1)[1]
    loose = repo / '.git' / name
    if loose.exists():
        return loose.read_text().strip()
    packed = repo / '.git' / 'packed-refs'
    if packed.exists():
        for line in packed.read_text().splitlines():
            if line.endswith(' ' + name):
                return line.split(' ')[0]
    return None


def write(dataset: str, public: list[dict], private: list[dict], manifest: dict, sources: dict,
          out_dir: Path, labels_dir: Path) -> dict:
    split = O.split_dataset(public, private, dataset)
    folder = out_dir / dataset
    folder.mkdir(parents=True, exist_ok=False)          # never overwrite a prepared dataset
    labels_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    by_split: dict[str, list] = {}
    for p in public:
        s = split[p['group_id']]
        by_split.setdefault(s, []).append(p)
        counts[s] += 1
    for s, rows in sorted(by_split.items()):
        with (folder / f'{s}.jsonl').open('x', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps({**r, 'split': s}, ensure_ascii=False) + '\n')
    with (labels_dir / f'{dataset}-labels.jsonl').open('x', encoding='utf-8') as f:
        for r in private:
            f.write(json.dumps({**r, 'split': split[r['group_id']]}, ensure_ascii=False) + '\n')
    plan = O.SPLIT_PLAN[dataset]
    full = {**manifest, 'sources': sources, 'created_at': datetime.now(timezone.utc).isoformat(),
            'split_rule': {**plan, 'method': 'label-blind; groups in sha256(seed:group) order, each to the split '
                                             'furthest below its item-share target; double-rated groups pinned '
                                             'by rating COUNT only'},
            'items_per_split': dict(sorted(counts.items())),
            'groups_per_split': dict(sorted(Counter(split.values()).items())),
            'split_groups_sha256': {s: digest(sorted(g for g, v in split.items() if v == s)) for s in plan['targets']},
            'item_ids_sha256': digest(sorted(p['item_id'] for p in public)),
            'labels_file': f'data/private/ordinal-v3/{dataset}-labels.jsonl',
            'licence': 'unstated by the dataset authors: local research use only, do not redistribute'}
    with (folder / 'manifest.json').open('x', encoding='utf-8') as f:
        json.dump(full, f, indent=2, ensure_ascii=False)
    return full


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--medexpert', type=Path, default=Path('data/external/medexpert/medexpert-all.jsonl'))
    p.add_argument('--bionlp-repo', type=Path, default=None)
    p.add_argument('--out-dir', type=Path, default=Path('data/prepared-v3'))
    p.add_argument('--labels-dir', type=Path, default=Path('data/private/ordinal-v3'))
    args = p.parse_args(argv)
    root = args.root
    resolve = lambda x: x if x.is_absolute() else root / x  # noqa: E731
    out_dir, labels_dir = resolve(args.out_dir), resolve(args.labels_dir)
    summary = {}
    mx_path = resolve(args.medexpert)
    raw = MX.load_raw(mx_path)
    pub, pri, man = O.build_medexpert(raw)
    summary['medexpert'] = write('medexpert', pub, pri, man, {'medexpert_file': str(args.medexpert),
                                                             'medexpert_sha256': sha(mx_path)}, out_dir, labels_dir)
    if args.bionlp_repo:
        repo = resolve(args.bionlp_repo)
        head = git_head(repo)
        if head != O.BIONLP_COMMIT:
            raise SystemExit(f'LLM-Safety-For-PatientQA checkout is {head}, expected {O.BIONLP_COMMIT}.')
        files = ['data/annotated_TRECLiveQA.csv', 'output/rephrase_results/combined_rephrased_TREC_temp_0.1.csv']
        files += sorted(str(x.relative_to(repo)).replace('\\', '/')
                        for x in (repo / 'output' / 'answer_results').glob('TREC_*.csv'))
        pub, pri, man = O.build_bionlp_trec(repo)
        summary['bionlp-trec'] = write('bionlp-trec', pub, pri, man,
                                       {'commit': head, 'files_sha256': {f: sha(repo / f) for f in files}},
                                       out_dir, labels_dir)
    print(json.dumps({k: {x: v[x] for x in ('items', 'groups', 'items_per_split', 'double_rated_items')}
                      for k, v in summary.items()}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
