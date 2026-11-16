"""Measure physician-physician agreement (the REFERENCE for judge agreement) on HealthBench meta-eval.

Run from the project root (no network, no model calls):
    python scripts/human_ceiling.py --no-index     # compute in memory; writes ONLY reports/human-ceiling/
    python scripts/human_ceiling.py                # also builds the scoring index (needs owner approval)
Reads   data/public/healthbench-meta.jsonl (verified against source-manifest.json SHA256)
Writes  reports/human-ceiling/summary.json + summary.md
        and, without --no-index, data/private/physician-rater-index.jsonl (+ .meta.json): labels and
        anonymized physician ids derived from the public file; never committed. An existing index is
        NEVER overwritten: it is verified against its .meta.json (source hash + index hash) and reused,
        or the script refuses.

Physician-physician agreement is a reference point, not a mathematical upper bound and not medical
truth. The primary reference for 'judge vs physician majority' is 'one physician vs the majority of the
other physicians' on items with >= 3 raters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from medical_jury.benchmark import convert_row  # noqa: E402
from medical_jury.evaluation import agreement as A  # noqa: E402
from medical_jury.evaluation.sampling import topic_of  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


GENERATOR_FILES = ('scripts/human_ceiling.py', 'src/medical_jury/benchmark.py')


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk.replace(b'\r\n', b'\n'))
    return h.hexdigest()


def build_index(root: Path, write: bool = True) -> list[dict]:
    raw = root / 'data/public/healthbench-meta.jsonl'
    manifest = json.loads((root / 'data/public/source-manifest.json').read_text(encoding='utf-8'))
    if sha256(raw) != manifest['sha256']:
        raise SystemExit('Raw public file does not match source-manifest SHA256; refusing to continue.')
    out, seen, rejected = [], set(), Counter()
    with raw.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            try:
                public, private = convert_row(row)
            except (ValueError, KeyError, TypeError):
                rejected['invalid_or_out_of_input_bounds'] += 1
                continue
            if public['example_id'] in seen:
                rejected['duplicate_example'] += 1
                continue
            seen.add(public['example_id'])
            out.append({'example_id': public['example_id'], 'group_id': private['group_id'],
                        'split': public['case']['split'], 'category': row['category'],
                        'topic': topic_of(row['category']), 'labels': private['binary_labels'],
                        'physician_ids': row['anonymized_physician_ids']})
    target = root / 'data/private/physician-rater-index.jsonl'
    meta_path = target.with_suffix('.meta.json')
    content = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in out)
    content_sha = hashlib.sha256(content.encode('utf-8')).hexdigest()
    if not write:
        print(f'index rows={len(out)} rejected={dict(rejected)} (in memory only; nothing written to data/)')
        return out
    if target.exists():
        if not meta_path.exists():
            raise SystemExit(f'{target} exists without {meta_path.name}; refusing to overwrite or trust it. '
                             'Ask the owner how to proceed.')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('source_sha256') != manifest['sha256'] or meta.get('index_sha256') != _file_sha(target):
            raise SystemExit(f'{target} does not match its meta (source or content changed); refusing to overwrite.')
        if meta['index_sha256'] != content_sha:
            raise SystemExit('Existing index differs from what this code would derive; refusing to overwrite. '
                             'Report the generator difference to the owner.')
        print(f'index verified and reused: rows={len(out)} -> {target.relative_to(root)}')
        return out
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.jsonl.tmp')
    with tmp.open('x', encoding='utf-8', newline='\n') as f:
        f.write(content)
    tmp.rename(target)
    meta = {'source': 'data/public/healthbench-meta.jsonl', 'source_sha256': manifest['sha256'],
            'index_sha256': content_sha, 'rows': len(out), 'rejected': dict(rejected),
            'generator_sha256': {g: _file_sha(root / g) for g in GENERATOR_FILES if (root / g).exists()},
            'created_at': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            'note': 'Derived from the public file; original labels/splits unchanged. Do not commit.'}
    with meta_path.open('x', encoding='utf-8') as f:
        f.write(json.dumps(meta, indent=2))
    print(f'index rows={len(out)} rejected={dict(rejected)} -> {target.relative_to(root)} (+ {meta_path.name})')
    return out


def ceiling(rows: list[dict], n_boot: int, seed: int) -> dict:
    label_sets = [r['labels'] for r in rows]
    pairwise = A.pairwise_human_pairs(label_sets)
    loo = A.leave_one_out_majority_pairs(label_sets, min_raters=3)
    groups_pw, groups_loo = defaultdict(list), defaultdict(list)
    for r in rows:
        groups_pw[r['group_id']].extend(A.pairwise_human_pairs([r['labels']]))
        groups_loo[r['group_id']].extend(A.leave_one_out_majority_pairs([r['labels']], 3))
    kappa = lambda ps: A.cohen_kappa([p for p, _ in ps], [g for _, g in ps]) if ps else None  # noqa: E731
    labels = [v for r in rows for v in r['labels']]
    per_phys = defaultdict(list)
    for r in rows:
        for pid, v in zip(r['physician_ids'], r['labels']):
            per_phys[pid].append(v)
    leniency = sorted(sum(v) / len(v) for v in per_phys.values() if len(v) >= 30)
    loo_summary = A.binary_summary(loo) if loo else None
    planning = None
    if loo_summary:
        po = loo_summary['agreement']
        p1 = loo_summary['reference_met_rate']
        pe = p1 * p1 + (1 - p1) * (1 - p1)
        planning = {str(n): round(1.96 * A.kappa_standard_error(po, pe, n), 3) for n in (150, 300, 600, 1000)}
    return {
        'n_items': len(rows), 'n_groups': len(groups_pw),
        'raters_per_item': dict(sorted(Counter(len(r['labels']) for r in rows).items())),
        'label_met_rate': sum(labels) / len(labels) if labels else None,
        'unanimous_item_share': (sum(len(set(r['labels'])) == 1 for r in rows) / len(rows)) if rows else None,
        'krippendorff_alpha_nominal': A.krippendorff_alpha_nominal(label_sets),
        'physician_pairwise': {**A.binary_summary(pairwise),
                               'kappa_group_bootstrap_95ci': A.group_bootstrap_ci(groups_pw, kappa, n_boot, seed)},
        'physician_vs_majority_of_others_min3': (
            {**loo_summary, 'kappa_group_bootstrap_95ci': A.group_bootstrap_ci(
                {k: v for k, v in groups_loo.items() if v}, kappa, n_boot, seed)} if loo_summary else None),
        'physicians_with_30plus_items': len(leniency),
        'physician_met_rate_quartiles': ([leniency[int((len(leniency) - 1) * q)] for q in (0, .25, .5, .75, 1)]
                                         if leniency else None),
        'kappa_95ci_halfwidth_by_n_planning': planning,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--bootstrap', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20261025)
    parser.add_argument('--no-index', action='store_true', help='Compute in memory only; write nothing under data/.')
    args = parser.parse_args(argv)
    rows = build_index(args.root, write=not args.no_index)
    result = {'source': 'HealthBench meta-eval public physician labels (anonymized ids)',
              'unit': 'one (completion, criterion) item; physicians compared on identical items',
              'overall': ceiling(rows, args.bootstrap, args.seed),
              'by_split': {s: ceiling([r for r in rows if r['split'] == s], args.bootstrap, args.seed)
                           for s in ('dev', 'validation', 'test')},
              'by_topic_pairwise_kappa': {}}
    for t in sorted({r['topic'] for r in rows}):
        sub = [r for r in rows if r['topic'] == t]
        pw = A.pairwise_human_pairs([r['labels'] for r in sub])
        lo = A.leave_one_out_majority_pairs([r['labels'] for r in sub], 3)
        result['by_topic_pairwise_kappa'][t] = {
            'n_items': len(sub), 'pairwise': A.binary_summary(pw),
            'vs_majority_min3_kappa': A.binary_summary(lo)['cohen_kappa'] if lo else None,
            'vs_majority_min3_n': len(lo)}
    result['interpretation'] = (
        'Primary judge reference = physician_vs_majority_of_others_min3. Physician-physician agreement is a '
        'reference point for interpreting judge agreement, not an upper bound and not clinical truth; '
        'physicians are reference opinions.')
    out = args.root / 'reports/human-ceiling'  # path kept for compatibility with earlier commands
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    o = result['overall']
    pw, lo = o['physician_pairwise'], o['physician_vs_majority_of_others_min3']
    lines = ['# Physician agreement reference (HealthBench meta-eval)', '',
             f"Items {o['n_items']} / groups {o['n_groups']}; met rate {o['label_met_rate']:.3f}; "
             f"Krippendorff alpha {o['krippendorff_alpha_nominal']:.3f}", '',
             '| Comparison | n | agreement | Cohen kappa (95% group CI) | Gwet AC1 | Macro-F1 |', '|---|---|---|---|---|---|',
             f"| physician vs physician | {pw['n']} | {pw['agreement']:.3f} | {pw['cohen_kappa']:.3f} "
             f"({pw['kappa_group_bootstrap_95ci']['lower']:.3f}-{pw['kappa_group_bootstrap_95ci']['upper']:.3f}) | "
             f"{pw['gwet_ac1']:.3f} | {pw['macro_f1']:.3f} |"]
    if lo:
        lines.append(f"| physician vs majority of others (>=3 raters) | {lo['n']} | {lo['agreement']:.3f} | "
                     f"{lo['cohen_kappa']:.3f} ({lo['kappa_group_bootstrap_95ci']['lower']:.3f}-"
                     f"{lo['kappa_group_bootstrap_95ci']['upper']:.3f}) | {lo['gwet_ac1']:.3f} | {lo['macro_f1']:.3f} |")
    lines += ['', '## Per split', '', '| split | items | met rate | pairwise kappa | vs-majority kappa (n) |', '|---|---|---|---|---|']
    for s, v in result['by_split'].items():
        m = v['physician_vs_majority_of_others_min3']
        lines.append(f"| {s} | {v['n_items']} | {v['label_met_rate']:.3f} | {v['physician_pairwise']['cohen_kappa']:.3f} | "
                     + (f"{m['cohen_kappa']:.3f} ({m['n']})" if m else 'n/a') + ' |')
    lines += ['', '## Per topic', '', '| topic | items | pairwise kappa | vs-majority kappa (n) |', '|---|---|---|---|']
    for t, v in result['by_topic_pairwise_kappa'].items():
        vm = v['vs_majority_min3_kappa']
        lines.append(f"| {t} | {v['n_items']} | {v['pairwise']['cohen_kappa']:.3f} | "
                     + (f"{vm:.3f} ({v['vs_majority_min3_n']})" if vm is not None else 'n/a') + ' |')
    lines += ['', f"Sample-size planning (approx. 95% CI half-width of kappa at the vs-majority operating point): "
              f"{o['kappa_95ci_halfwidth_by_n_planning']}", '', result['interpretation']]
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
