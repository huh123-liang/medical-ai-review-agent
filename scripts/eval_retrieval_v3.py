"""Step-2 retrieval evaluation: conversation -> applicable HealthBench situation (BM25 vs dense vs hybrid).

Offline and free. Inputs: kb/v3, data/prepared/dev.jsonl (public conversations + HealthBench category only;
physician labels are not read), and the local embedding model in .cache/fastembed.

Protocol
* Unit = one conversation group (exact-normalised duplicates share a group). Gold = the HealthBench
  theme x situation assigned to that conversation (exactly one per group; others are reported and skipped).
* dev groups are split by hash into dev-A (60 %, configuration choice) and dev-B (40 %, reported once).
  Validation and test splits are not touched.
* For each mode the configuration with the best dev-A top-1 accuracy (tie: MRR) is frozen, then dev-B is scored.
* Selection used by the agent = top-1 situation (one per exclusive group); item metrics are computed on it.

    python scripts/eval_retrieval_v3.py [--limit N]
Writes reports/retrieval-v3-<UTC>-<rand>/ (REPORT.md, metrics.json, per-query.csv, manifest.json); never overwrites.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from medical_jury.domain import CaseInput, digest  # noqa: E402
from medical_jury.v3.hybrid import CachedEmbedder, FastEmbedEmbedder, HybridRetriever, open_index, query_text  # noqa: E402
from medical_jury.v3.retrieval import build_query  # noqa: E402
from medical_jury.v3.schema import CaseMeta, load_kb, load_profile  # noqa: E402

BGE_PREFIX = 'Represent this sentence for searching relevant passages: '
MODEL = 'BAAI/bge-small-en-v1.5'


def load_queries(path: Path, limit: int | None):
    from importlib import util
    spec = util.spec_from_file_location('kbgen', ROOT / 'scripts' / 'build_kb_v3_healthbench.py')
    kbgen = util.module_from_spec(spec)
    spec.loader.exec_module(kbgen)
    groups: dict[str, dict] = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            g = row['case']['group_id']
            theme, sit, _ = kbgen.parse(row['category'])
            entry = groups.setdefault(g, {'case': row['case'], 'situations': set()})
            entry['situations'].add(f'HB:{theme}:{sit}')
    ok, skipped = [], Counter()
    for g, e in sorted(groups.items()):
        if len(e['situations']) != 1:
            skipped['more_than_one_situation'] += 1
            continue
        ok.append({'group_id': g, 'case': CaseInput.model_validate(e['case']), 'gold': next(iter(e['situations']))})
    return (ok[:limit] if limit else ok), dict(skipped)


def split_of(group_id: str) -> str:
    return 'A' if int(hashlib.sha256(('retrieval-v3:' + group_id).encode()).hexdigest()[:8], 16) % 100 < 60 else 'B'


def ascii_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isascii() for c in letters) / len(letters) if letters else 1.0


def score_query(retriever: HybridRetriever, q, gold: str, kb_links: dict):
    ranked = retriever.rank(q)
    order = [r['situation_id'] for r in ranked if r['score'] > 0]
    selected = retriever.select(ranked)
    rank = order.index(gold) + 1 if gold in order else None
    gold_items = kb_links[gold]
    chosen_items = set().union(*(kb_links[s] for s in selected)) if selected else set()
    hit = len(gold_items & chosen_items)
    group = gold.rsplit(':', 1)[0]
    return {'top1': selected[:1] == [gold], 'theme_top1': bool(selected) and selected[0].rsplit(':', 1)[0] == group,
            'recall3': rank is not None and rank <= 3, 'rr': 1.0 / rank if rank else 0.0, 'rank': rank,
            'selected': selected[0] if selected else '', 'empty': not selected,
            'item_recall': hit / len(gold_items), 'item_precision': (hit / len(chosen_items)) if chosen_items else None,
            'item_exact': chosen_items == gold_items}


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    prec = [r['item_precision'] for r in rows if r['item_precision'] is not None]
    return {'n': n, 'top1_acc': sum(r['top1'] for r in rows) / n, 'theme_top1_acc': sum(r['theme_top1'] for r in rows) / n,
            'recall_at_3': sum(r['recall3'] for r in rows) / n, 'mrr': sum(r['rr'] for r in rows) / n,
            'empty_rate': sum(r['empty'] for r in rows) / n, 'item_recall': sum(r['item_recall'] for r in rows) / n,
            'item_precision': (sum(prec) / len(prec)) if prec else None,
            'item_exact_set': sum(r['item_exact'] for r in rows) / n}


def bootstrap(values: list[float], reps: int, seed: int) -> list[float]:
    rnd, n = random.Random(seed), len(values)
    stats = sorted(sum(values[rnd.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return [round(stats[int(0.025 * reps)], 4), round(stats[int(0.975 * reps) - 1], 4)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--limit', type=int, default=None, help='Debug: only the first N groups.')
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args(argv)
    root = args.root
    kb, profile = load_kb(root / 'kb/v3'), load_profile(root / 'kb/v3/profile-platform-6d.json')
    links = defaultdict(set)
    for i in kb.items:
        for s in i.situations:
            links[s].add(i.item_id)
    queries, skipped = load_queries(root / 'data/prepared/dev.jsonl', args.limit)
    for q in queries:
        q['split'] = split_of(q['group_id'])
        q['query'] = build_query(q['case'], CaseMeta(), profile)
    dev_a = [q for q in queries if q['split'] == 'A']
    dev_b = [q for q in queries if q['split'] == 'B']

    embedders = {}
    for prefix_name, prefix in [('none', ''), ('bge', BGE_PREFIX)]:
        e = CachedEmbedder(FastEmbedEmbedder(MODEL, root / '.cache/fastembed', query_prefix=prefix))
        e.warm([query_text(q['query'], m) for q in queries for m in ('full', 'last_user')])
        embedders[prefix_name] = e
    index_dir = root / 'data' / 'v3-index'
    grid = []
    for mode in ('bm25', 'dense', 'hybrid'):
        for fields, qmode in itertools.product(('applies_when', 'applies_when+criteria'), ('full', 'last_user')):
            prefixes = ['none'] if mode == 'bm25' else ['none', 'bge']
            ks = [60] if mode != 'hybrid' else [10, 60]
            for pref, k in itertools.product(prefixes, ks):
                grid.append({'mode': mode, 'doc_fields': fields, 'query_mode': qmode, 'query_prefix': pref, 'rrf_k': k})

    def make(cfg):
        emb = embedders[cfg['query_prefix']]
        index = open_index(kb, emb, index_dir, fields=cfg['doc_fields'])
        return HybridRetriever(kb, index, emb, mode=cfg['mode'], top_situations=1, rrf_k=cfg['rrf_k'],
                               query_mode=cfg['query_mode'])

    tuning = []
    for cfg in grid:
        r = make(cfg)
        tuning.append({**cfg, **summarise([score_query(r, q['query'], q['gold'], links) for q in dev_a])})
    chosen = {}
    for mode in ('bm25', 'dense', 'hybrid'):
        best = max((t for t in tuning if t['mode'] == mode), key=lambda t: (t['top1_acc'], t['mrr']))
        chosen[mode] = {k: best[k] for k in ('mode', 'doc_fields', 'query_mode', 'query_prefix', 'rrf_k')}

    per_query, final = [], {}
    rows_b = {}
    for mode, cfg in chosen.items():
        r = make(cfg)
        rows_b[mode] = [score_query(r, q['query'], q['gold'], links) for q in dev_b]
        s = summarise(rows_b[mode])
        s['top1_acc_ci95'] = bootstrap([float(x['top1']) for x in rows_b[mode]], args.bootstrap, 1)
        s['item_recall_ci95'] = bootstrap([x['item_recall'] for x in rows_b[mode]], args.bootstrap, 2)
        final[mode] = {'config': cfg, 'dev_b': s}
    paired = {}
    for other in ('bm25', 'dense'):
        diff = [float(a['top1']) - float(b['top1']) for a, b in zip(rows_b['hybrid'], rows_b[other])]
        paired[f'hybrid_minus_{other}_top1'] = {'mean': round(sum(diff) / len(diff), 4),
                                                'ci95': bootstrap(diff, args.bootstrap, 3)}
    majority = Counter(q['gold'] for q in dev_a).most_common(1)[0][0]
    baselines = {'majority_situation': majority,
                 'majority_top1_acc_dev_b': sum(q['gold'] == majority for q in dev_b) / len(dev_b),
                 'uniform_random_top1_acc': 1 / len(kb.situations)}
    # breakdowns on dev-B for the hybrid choice
    by_theme, by_lang, confusion = defaultdict(list), defaultdict(list), Counter()
    for q, r in zip(dev_b, rows_b['hybrid']):
        theme = q['gold'].split(':')[1]
        by_theme[theme].append(r['top1'])
        lang = 'mostly_ascii' if ascii_share(q['query'].text) >= 0.8 else 'non_ascii'
        by_lang[lang].append(r['top1'])
        confusion[(theme, r['selected'].split(':')[1] if r['selected'] else '(none)')] += 1
    breakdown = {'top1_by_gold_theme': {k: [len(v), round(sum(v) / len(v), 4)] for k, v in sorted(by_theme.items())},
                 'top1_by_script': {k: [len(v), round(sum(v) / len(v), 4)] for k, v in sorted(by_lang.items())},
                 'theme_confusion_hybrid': {f'{a} -> {b}': n for (a, b), n in sorted(confusion.items())}}
    for i, q in enumerate(dev_b):
        per_query.append({'group_id': q['group_id'], 'gold': q['gold'],
                          **{f'{m}_selected': rows_b[m][i]['selected'] for m in rows_b},
                          **{f'{m}_rank': rows_b[m][i]['rank'] for m in rows_b}})

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = root / 'reports' / f'retrieval-v3-{stamp}-{uuid4().hex[:6]}'
    out.mkdir(parents=True)
    data_file = root / 'data/prepared/dev.jsonl'
    manifest = {'kb_sha256': kb.sha256(), 'profile_sha256': profile.sha256(),
                'embedding_model': embedders['none'].model_id, 'dev_file_sha256': hashlib.sha256(data_file.read_bytes()).hexdigest(),
                'split_rule': 'sha256("retrieval-v3:"+group_id)[:8] % 100 < 60 -> dev-A',
                'dev_a_groups_sha256': digest(sorted(q['group_id'] for q in dev_a)),
                'dev_b_groups_sha256': digest(sorted(q['group_id'] for q in dev_b)),
                'skipped_groups': skipped, 'limit': args.limit, 'labels_read': False,
                'created_at': datetime.now(timezone.utc).isoformat()}
    metrics = {'n_dev_a': len(dev_a), 'n_dev_b': len(dev_b), 'baselines': baselines, 'chosen': final,
               'paired': paired, 'breakdown_dev_b_hybrid': breakdown, 'tuning_dev_a': tuning}
    for name, obj in [('manifest.json', manifest), ('metrics.json', metrics)]:
        with (out / name).open('x', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(per_query[0]))
    w.writeheader()
    w.writerows(per_query)
    with (out / 'per-query.csv').open('x', encoding='utf-8', newline='') as f:
        f.write(buf.getvalue())
    print(json.dumps({'out': str(out), 'baselines': baselines,
                      'dev_b': {m: {k: v for k, v in final[m]['dev_b'].items()} for m in final},
                      'chosen': chosen, 'paired': paired}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
