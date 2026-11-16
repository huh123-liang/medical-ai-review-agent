"""DIAGNOSTIC ONLY (not part of the agent): how well can a router LEARNED from dev-A situation assignments
pick the HealthBench situation, compared with similarity retrieval? Uses local bge embeddings + numpy
logistic regression / kNN chosen by 5-fold CV on dev-A; dev-B is scored once. Physician met/not-met labels
are not read; the HealthBench situation (category) of dev-A conversations is used as the training target.

    python scripts/diag_situation_router_v3.py .     (CPU: ~10 min for embeddings on 2 cores)

Result 2026-10-25 is stored in reports/retrieval-v3-20261025T133001Z-ce3be8/router-diagnostic.txt.
"""
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO / 'src')); sys.path.insert(0, str(REPO / 'scripts'))
import eval_retrieval_v3 as ev
from medical_jury.v3.hybrid import FastEmbedEmbedder, query_text
from medical_jury.v3.retrieval import build_query
from medical_jury.v3.schema import CaseMeta, load_profile

prof = load_profile(REPO / 'kb/v3/profile-platform-6d.json')
qs, skipped = ev.load_queries(REPO / 'data/prepared/dev.jsonl', None)
for q in qs:
    q['split'] = ev.split_of(q['group_id']); q['query'] = build_query(q['case'], CaseMeta(), prof)
emb = FastEmbedEmbedder('BAAI/bge-small-en-v1.5', REPO / '.cache/fastembed')
t = time.time()
texts = [query_text(q['query'], 'full') for q in qs]
X = np.array([v for i in range(0, len(texts), 32) for v in emb.embed(texts[i:i + 32], 'passage')], dtype=np.float64)
print('embedded', X.shape, round(time.time() - t, 1), 's')
labels = sorted({q['gold'] for q in qs}); li = {l: i for i, l in enumerate(labels)}
y = np.array([li[q['gold']] for q in qs]); A = np.array([q['split'] == 'A' for q in qs])
theme = np.array([l.rsplit(':', 1)[0] for l in labels])

def logreg(X, y, lam, iters=600, lr=0.5):
    n, d = X.shape; k = len(labels)
    Xb = np.hstack([X, np.ones((n, 1))]); W = np.zeros((d + 1, k)); Y = np.eye(k)[y]
    for _ in range(iters):
        Z = Xb @ W; Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = Xb.T @ (P - Y) / n; G[:-1] += lam * W[:-1]; W -= lr * G
    return W

def proba(W, X):
    Z = np.hstack([X, np.ones((len(X), 1))]) @ W; Z -= Z.max(1, keepdims=True); P = np.exp(Z); return P / P.sum(1, keepdims=True)

def knn(Xtr, ytr, X, k):
    S = X @ Xtr.T; idx = np.argsort(-S, 1)[:, :k]; P = np.zeros((len(X), len(labels)))
    for i, row in enumerate(idx):
        for j in row: P[i, ytr[j]] += 1
    return P / k

def metrics(P, y):
    order = np.argsort(-P, 1); top1 = order[:, 0] == y
    r3 = (order[:, :3] == y[:, None]).any(1); th = theme[order[:, 0]] == theme[y]
    rank = np.argmax(order == y[:, None], 1) + 1
    return {'top1': round(top1.mean(), 4), 'theme_top1': round(th.mean(), 4), 'recall3': round(r3.mean(), 4), 'mrr': round((1 / rank).mean(), 4)}

XA, yA = X[A], y[A]
rng = np.random.default_rng(0); folds = rng.integers(0, 5, len(yA))
best = None
for name, fn in [*[(f'logreg lam={l}', (lambda l: lambda Xt, yt, Xv: proba(logreg(Xt, yt, l), Xv))(l)) for l in (1e-4, 1e-3, 1e-2)],
                 *[(f'knn k={k}', (lambda k: lambda Xt, yt, Xv: knn(Xt, yt, Xv, k))(k)) for k in (5, 15, 30)]]:
    cv = []
    for f in range(5):
        P = fn(XA[folds != f], yA[folds != f], XA[folds == f]); cv.append(metrics(P, yA[folds == f])['top1'])
    m = float(np.mean(cv)); print(f'{name:20s} dev-A 5-fold CV top1 {m:.4f}')
    if best is None or m > best[0]: best = (m, name, fn)
P = best[2](XA, yA, X[~A]); mB = metrics(P, y[~A])
order = np.argsort(-P, 1); conf = P.max(1); correct = order[:, 0] == y[~A]
cover = {}
for th in (0.3, 0.4, 0.5, 0.6):
    keep = conf >= th
    cover[th] = {'coverage': round(keep.mean(), 4), 'precision': round(correct[keep].mean(), 4) if keep.any() else None}
print(json.dumps({'chosen': best[1], 'dev_b': mB, 'n_b': int((~A).sum()), 'abstain_curve': cover,
                  'majority_b': round(float((y[~A] == np.bincount(yA).argmax()).mean()), 4)}, indent=2))
