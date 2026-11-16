"""Local hybrid retrieval for the v3 chain: SQLite FTS5 (BM25) + Chroma (dense vectors) + reciprocal rank fusion.

What is ranked: *situations* (when a group of criteria applies). Which situations are APPLIED depends on
``situation_policy``:
* ``metadata`` (default): only situations named in the case metadata are applied; the hybrid ranking is
  recorded in the trace as ``suggested_situations`` (shadow mode) and does not affect scoring. Reason: on the
  HealthBench dev-B split, similarity ranking picked the right situation for ~12 % of conversations, no better
  than always predicting the most common one, and a wrong situation can invert a verdict (emergent vs not).
* ``llm``: if the metadata names no situation, an LLM ``SituationRouter`` (router.py) chooses from all
  candidate situations (conversation only). Its choice is applied; the hybrid ranking stays a shadow record.
* ``retrieved`` (experimental, explicit opt-in): the top-ranked situations are applied. Disease and case items are still chosen
by metadata filters, never by similarity, because similarity is not evidence of applicability.

Guarantees (tested):
* The query is the conversation only (``build_query``); the answer under review is never used.
* Mandatory items are always included and never displaced by retrieved items.
* At most one situation per ``exclusive_group`` (e.g. emergent vs non-emergent) is selected.
* Everything runs locally: the embedding model is loaded with ``local_files_only`` (no download at run time),
  and the index lives under ``data/v3-index`` (git-ignored) with a manifest binding it to the KB hash,
  the embedding model files and the tokenizer. A mismatching index is refused, not silently reused.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from ..domain import digest
from .retrieval import RetrievalQuery, RetrievalResult, RetrievedItem, applicable
from .schema import KnowledgeBase, RubricItem, Situation

INDEX_SCHEMA = 'v3-hybrid-1'
Mode = Literal['bm25', 'dense', 'hybrid']
DocFields = Literal['applies_when', 'applies_when+criteria']
QueryMode = Literal['full', 'last_user', 'user_turns']

STOPWORDS = frozenset('''a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has have having he
her here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now
of off on once only or other our ours ourselves out over own same she should so some such than that the their
theirs them themselves then there these they this those through to too under until up very was we were what when
where which while who whom why will with would you your yours yourself yourselves user assistant'''.split())
_TOKEN = re.compile(r'\w+', re.UNICODE)


# ---- embedders ---------------------------------------------------------------------------------------------------

class Embedder(Protocol):
    model_id: str

    def embed(self, texts: list[str], kind: Literal['query', 'passage']) -> list[list[float]]: ...


class FastEmbedEmbedder:
    """A fastembed ONNX model loaded from a local cache only. ``model_id`` includes a hash of the model files."""

    def __init__(self, model_name: str, cache_dir: Path, query_prefix: str = '', threads: int | None = None):
        from fastembed import TextEmbedding
        spec = next((m for m in TextEmbedding.list_supported_models() if m['model'] == model_name), None)
        if spec is None:
            raise ValueError(f'Unknown fastembed model {model_name!r}.')
        model_dir = Path(cache_dir) / ('models--' + spec['sources']['hf'].replace('/', '--'))
        files = sorted(p for p in (model_dir / 'snapshots').rglob('*') if p.is_file()) if model_dir.exists() else []
        if not files:
            raise FileNotFoundError(f'Embedding model not found in {model_dir}. It must be present locally '
                                    '(this runtime never downloads models).')
        h = hashlib.sha256()
        for f in files:
            h.update(f.name.encode())
            h.update(hashlib.sha256(f.read_bytes()).digest())
        self.files_sha256 = h.hexdigest()
        self.model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir), local_files_only=True,
                                   threads=threads)
        self.query_prefix = query_prefix
        self.model_id = f'{model_name}@{self.files_sha256[:16]}' + (f'+qp:{digest(query_prefix)[:8]}' if query_prefix else '')

    def embed(self, texts, kind):
        texts = [self.query_prefix + t for t in texts] if kind == 'query' else list(texts)
        # small batches: long texts at fastembed's default batch (256) need several GB of RAM
        return [[float(x) for x in v] for v in self.model.embed(texts, batch_size=16)]


class HashEmbedder:
    """Deterministic bag-of-words hashing vectors: offline tests only, not a semantic model."""

    def __init__(self, dim: int = 256):
        self.dim, self.model_id = dim, f'hash-bow-{dim}'

    def embed(self, texts, kind):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in tokens(t):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / n for x in v])
        return out


class CachedEmbedder:
    """Wraps an embedder with precomputed query vectors (batch evaluation); misses fall through."""

    def __init__(self, base: Embedder):
        self.base, self.model_id, self.cache = base, base.model_id, {}

    def warm(self, texts: list[str], batch: int = 64):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            self.cache.update(zip(chunk, self.base.embed(chunk, 'query')))

    def embed(self, texts, kind):
        if kind != 'query':
            return self.base.embed(texts, kind)
        missing = [t for t in texts if t not in self.cache]
        if missing:
            self.warm(missing)
        return [self.cache[t] for t in texts]


def tokens(text: str) -> list[str]:
    return [t for t in (m.group(0).lower() for m in _TOKEN.finditer(text)) if len(t) > 1 and t not in STOPWORDS]


# ---- index --------------------------------------------------------------------------------------------------------

def situation_document(s: Situation, items: list[RubricItem], fields: DocFields) -> str:
    body = f'{s.label}. {s.applies_when}'
    if fields == 'applies_when+criteria':
        body += '\n' + '\n'.join(i.text for i in items if s.situation_id in i.situations)
    return body


@dataclass
class HybridIndex:
    directory: Path
    manifest: dict
    lexical: sqlite3.Connection
    collection: object

    @property
    def manifest_sha256(self) -> str:
        return digest(self.manifest)


def _manifest(kb: KnowledgeBase, embedder: Embedder, fields: DocFields, tokenizer: str) -> dict:
    return {'schema': INDEX_SCHEMA, 'kb_sha256': kb.sha256(), 'embedding_model': embedder.model_id,
            'doc_fields': fields, 'fts_tokenizer': tokenizer, 'documents': len(kb.situations)}


def open_index(kb: KnowledgeBase, embedder: Embedder, base_dir: Path, fields: DocFields = 'applies_when',
               tokenizer: str = 'porter unicode61') -> HybridIndex:
    """Open the index for exactly this KB/model/config, building it on first use. Never reuses a stale index."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    if not kb.situations:
        raise ValueError('The knowledge base has no situations to index.')
    expected = _manifest(kb, embedder, fields, tokenizer)
    directory = Path(base_dir) / f"{expected['kb_sha256'][:12]}-{digest(expected)[:12]}"
    manifest_path = directory / 'manifest.json'
    if directory.exists():
        if not manifest_path.exists():
            raise RuntimeError(f'{directory} exists without a manifest (interrupted build?). Move it aside and retry.')
        found = json.loads(manifest_path.read_text(encoding='utf-8'))
        if found != expected:
            raise RuntimeError(f'Index manifest mismatch in {directory}; refusing to reuse it.')
        fresh = False
    else:
        directory.mkdir(parents=True)
        fresh = True
    lexical = sqlite3.connect(str(directory / 'lexical.sqlite'), check_same_thread=False)
    client = chromadb.PersistentClient(path=str(directory / 'chroma'), settings=ChromaSettings(anonymized_telemetry=False))
    if fresh:
        try:
            lexical.execute(f"CREATE VIRTUAL TABLE docs USING fts5(doc_id UNINDEXED, body, tokenize='{tokenizer}')")
        except sqlite3.OperationalError as error:
            raise RuntimeError(f'SQLite FTS5 unavailable or tokenizer unsupported: {error}') from None
        docs = [(s.situation_id, situation_document(s, kb.items, fields)) for s in kb.situations]
        lexical.executemany('INSERT INTO docs(doc_id, body) VALUES (?, ?)', docs)
        lexical.commit()
        collection = client.create_collection('situations', embedding_function=None,
                                              metadata={'hnsw:space': 'cosine', 'schema': INDEX_SCHEMA})
        vectors = embedder.embed([d for _, d in docs], 'passage')
        collection.add(ids=[i for i, _ in docs], embeddings=vectors, documents=[d for _, d in docs],
                       metadatas=[{'doc_id': i} for i, _ in docs])
        manifest_path.write_text(json.dumps(expected, indent=2, sort_keys=True), encoding='utf-8')  # written last
    else:
        collection = client.get_collection('situations')
        if collection.count() != expected['documents']:
            raise RuntimeError(f'{directory}: vector collection is incomplete.')
    return HybridIndex(directory, expected, lexical, collection)


# ---- retrieval ----------------------------------------------------------------------------------------------------

def query_text(q: RetrievalQuery, mode: QueryMode) -> str:
    if mode == 'full' or not q.messages:
        return q.text
    users = [c for r, c in q.messages if r == 'user']
    if not users:
        return q.text
    return users[-1] if mode == 'last_user' else '\n'.join(users)


class HybridRetriever:
    name = 'hybrid-v1'

    def __init__(self, kb: KnowledgeBase, index: HybridIndex, embedder: Embedder, mode: Mode = 'hybrid',
                 top_situations: int = 1, rrf_k: int = 60, weights: tuple[float, float] = (1.0, 1.0),
                 query_mode: QueryMode = 'full', max_per_dimension: int = 12, max_query_terms: int = 128,
                 situation_policy: Literal['metadata', 'retrieved', 'llm'] = 'metadata', router=None):
        if index.manifest['kb_sha256'] != kb.sha256() or index.manifest['embedding_model'] != embedder.model_id:
            raise ValueError('Index was built for a different KB or embedding model.')
        self.kb, self.index, self.embedder = kb, index, embedder
        self.mode, self.top_situations, self.rrf_k, self.weights = mode, top_situations, rrf_k, weights
        self.query_mode, self.max_per_dimension, self.max_query_terms = query_mode, max_per_dimension, max_query_terms
        self.situations = {s.situation_id: s for s in kb.situations}
        if situation_policy not in ('metadata', 'retrieved', 'llm'):
            raise ValueError('situation_policy must be metadata, retrieved or llm')
        if situation_policy == 'llm' and router is None:
            raise ValueError('situation_policy=llm needs a SituationRouter')
        self.situation_policy, self.router = situation_policy, router

    # rankings over situation ids, best first; ids never matched are absent
    def _bm25(self, text: str, candidates: list[str]) -> list[str]:
        terms = list(dict.fromkeys(tokens(text)))[:self.max_query_terms]
        if not terms or not candidates:
            return []
        match = ' OR '.join('"' + t.replace('"', '') + '"' for t in terms)
        marks = ','.join('?' * len(candidates))
        rows = self.index.lexical.execute(
            f'SELECT doc_id FROM docs WHERE docs MATCH ? AND doc_id IN ({marks}) ORDER BY bm25(docs), doc_id',
            [match, *candidates]).fetchall()
        return [r[0] for r in rows]

    def _dense(self, text: str, candidates: list[str]) -> list[str]:
        if not candidates:
            return []
        vector = self.embedder.embed([text], 'query')[0]
        where = {'doc_id': {'$in': candidates}} if len(candidates) > 1 else {'doc_id': candidates[0]}
        res = self.index.collection.query(query_embeddings=[vector], n_results=len(candidates), where=where,
                                          include=['distances'])
        return list(res['ids'][0])

    def rank(self, q: RetrievalQuery) -> list[dict]:
        """All candidate situations, best first, with the per-ranker ranks and the fused score."""
        candidates = sorted(sid for sid in self.situations
                            if any(sid in i.situations and applicable(i, q) for i in self.kb.items))
        text = query_text(q, self.query_mode)
        lex = self._bm25(text, candidates) if self.mode in ('bm25', 'hybrid') else []
        den = self._dense(text, candidates) if self.mode in ('dense', 'hybrid') else []
        lr = {sid: n + 1 for n, sid in enumerate(lex)}
        dr = {sid: n + 1 for n, sid in enumerate(den)}
        rows = []
        for sid in candidates:
            if self.mode == 'bm25':
                score = 1.0 / lr[sid] if sid in lr else 0.0
            elif self.mode == 'dense':
                score = 1.0 / dr[sid] if sid in dr else 0.0
            else:
                score = (self.weights[0] / (self.rrf_k + lr[sid]) if sid in lr else 0.0) + \
                        (self.weights[1] / (self.rrf_k + dr[sid]) if sid in dr else 0.0)
            rows.append({'situation_id': sid, 'exclusive_group': self.situations[sid].exclusive_group,
                         'bm25_rank': lr.get(sid), 'dense_rank': dr.get(sid), 'score': round(score, 8)})
        rows.sort(key=lambda r: (-r['score'], r['situation_id']))
        return rows

    def select(self, ranked: list[dict]) -> list[str]:
        chosen, groups = [], set()
        for r in ranked:
            if len(chosen) >= self.top_situations:
                break
            if r['score'] <= 0 or r['exclusive_group'] in groups:
                continue
            chosen.append(r['situation_id'])
            groups.add(r['exclusive_group'])
        return chosen

    def candidate_situations(self, q: RetrievalQuery) -> list[Situation]:
        return [self.situations[sid] for sid in sorted(self.situations)
                if any(sid in i.situations and applicable(i, q) for i in self.kb.items)]

    def retrieve(self, q: RetrievalQuery, job_id: str | None = None) -> RetrievalResult:
        unknown = set(q.situations) - set(self.situations)
        if unknown:
            raise ValueError(f'Case metadata names unknown situations: {sorted(unknown)}')
        ranked = self.rank(q)
        suggested = self.select(ranked)
        router_rec, source = None, 'metadata'
        if q.situations:
            selected = list(q.situations)
        elif self.situation_policy == 'retrieved':
            selected, source = suggested, 'retrieval (experimental)'
        elif self.situation_policy == 'llm':
            router_rec = self.router.route(q.messages, self.candidate_situations(q), job_id or 'unassigned')
            selected = list(router_rec.get('selected', []))
            source = 'llm_router' if router_rec['status'] == 'Completed' else 'llm_router_failed'
        else:
            selected = []
        order = {'case': 0, 'disease': 1, 'general': 2}
        chosen, dropped = [], 0
        for key in q.dimensions:
            pool = [i for i in self.kb.items if i.dimension == key and applicable(i, q)
                    and (not i.situational or set(i.situations) & set(selected))]
            pool.sort(key=lambda i: (not i.mandatory, i.situational, order[i.level], i.item_id))
            mandatory = [i for i in pool if i.mandatory]
            rest = [i for i in pool if not i.mandatory][:max(0, self.max_per_dimension - len(mandatory))]
            dropped += len(pool) - len(mandatory) - len(rest)
            for i in mandatory + rest:
                why = {'mandatory': i.mandatory, 'level': i.level}
                if i.situational:
                    why['situations'] = sorted(set(i.situations) & set(selected))
                chosen.append(RetrievedItem(i, why))
        return RetrievalResult(chosen, {
            'retriever': self.name, 'mode': self.mode, 'kb_sha256': self.kb.sha256(),
            'index_manifest_sha256': self.index.manifest_sha256, 'embedding_model': self.embedder.model_id,
            'query_mode': self.query_mode, 'query_sha256': digest(query_text(q, self.query_mode)),
            'top_situations': self.top_situations, 'rrf_k': self.rrf_k, 'situation_policy': self.situation_policy,
            'applied_situations': selected, 'suggested_situations': suggested,
            'situation_source': source,
            'situation_router': ({k: v for k, v in router_rec.items() if k not in ('raw_chars',)}
                                 if router_rec else None),
            'ranking': ranked[:5], 'dropped_by_cap': dropped, 'answer_used_in_query': False})
