from __future__ import annotations

import math

import numpy as np

from .config import Settings
from .domain import CaseInput, Rubric, canonical, digest
from .store import Store


class LocalEmbedder:
    def __init__(self, settings: Settings):
        from fastembed import TextEmbedding
        self.model_name = settings.embedding_model
        self.model = TextEmbedding(model_name=self.model_name, cache_dir=str(settings.root / ".cache/fastembed"), threads=2)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [np.asarray(vector, dtype=float).tolist() for vector in self.model.embed(texts)]


class Knowledge:
    def __init__(self, settings: Settings, store: Store, embedder=None):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        self.settings, self.store = settings, store
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(settings.root / "data/vector"),
                                               settings=ChromaSettings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection("rubrics_v1", embedding_function=None,
                    metadata={"hnsw:space": "cosine", "schema": "rubrics-v1", "embedding_model": settings.embedding_model})
        if self.collection.metadata.get("embedding_model") != settings.embedding_model:
            raise ValueError("Index embedding model mismatch; use a separately versioned index.")

    def embeddings(self, texts):
        if self.embedder is None:
            self.embedder = LocalEmbedder(self.settings)
        return self.embedder(texts)

    def upsert(self, rubrics: list[Rubric], allow_case_reference: bool = False):
        if not rubrics:
            return
        for r in rubrics:
            if not r.approved:
                raise ValueError("Unapproved rubric cannot enter the index.")
            if r.split != "dev":
                raise ValueError("Validation/test rubrics must not enter the development retrieval index.")
            if r.level == "case" and not allow_case_reference:
                raise ValueError("Explicit opt-in required to index development case rubrics; they remain case-bound.")
            with self.store.connect() as db:
                old = db.execute("SELECT data_json FROM rubrics WHERE id=?", (r.rubric_id,)).fetchone()
            if old and old[0] != canonical(r.model_dump()):
                raise ValueError("Rubric IDs are immutable. Create a new version/ID rather than overwrite a used rule.")
        vectors = self.embeddings([r.text for r in rubrics])
        for start in range(0, len(rubrics), 100):
            batch, vecs = rubrics[start:start + 100], vectors[start:start + 100]
            self.collection.upsert(ids=[r.rubric_id for r in batch], embeddings=vecs,
                documents=[r.text for r in batch], metadatas=[{"level": r.level, "task_type": r.task_type,
                "case_id": r.case_id, "split": r.split, "version": r.version} for r in batch])
            with self.store.connect() as db:
                for r in batch:
                    db.execute("INSERT OR REPLACE INTO rubrics VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (r.rubric_id, r.text, r.level, r.task_type, r.case_id, r.group_id, r.split,
                         int(r.approved), int(r.mandatory), canonical(r.model_dump())))

    @staticmethod
    def applicable(r: Rubric, case: CaseInput) -> bool:
        return r.approved and (r.level != "case" or r.case_id == case.case_id) and r.task_type in (case.task_type, "all")

    def retrieve(self, case: CaseInput, k: int) -> tuple[list[Rubric], dict]:
        import json
        with self.store.connect() as db:
            rows = db.execute("SELECT data_json FROM rubrics WHERE approved=1 AND split='dev'").fetchall()
        candidates = [Rubric.model_validate(json.loads(row[0])) for row in rows]
        candidates = [r for r in candidates if self.applicable(r, case)]
        if not candidates:
            raise ValueError("No applicable approved rubrics. No fallback to other patients or generic invented criteria.")
        required = [r for r in candidates if r.mandatory]
        optional = [r for r in candidates if not r.mandatory]
        selected = []
        if optional:
            # Deliberately exclude candidate_output: it must not choose its own grading criteria.
            query = case.task + "\n" + "\n".join(m.content for m in case.context)
            where = {"$and": [{"split": "dev"}, {"task_type": {"$in": [case.task_type, "all"]}},
                     {"$or": [{"level": {"$in": ["general", "task"]}}, {"case_id": case.case_id}]}]}
            result = self.collection.query(query_embeddings=self.embeddings([query]),
                n_results=min(len(candidates), k + len(required)), where=where, include=["distances"])
            by_id = {r.rubric_id: r for r in optional}
            selected = [by_id[item] for item in result["ids"][0] if item in by_id][:k]
        merged = {r.rubric_id: r for r in required + selected}
        ordered = sorted(merged.values(), key=lambda r: r.rubric_id)
        return ordered, {"mandatory_count": len(required), "retrieved_count": len(selected),
            "index_hash": digest(sorted([r.model_dump() for r in candidates], key=lambda r: r["rubric_id"])),
            "embedding_model": self.settings.embedding_model,
            "retrieval_claim": "Developer-selected rubrics only; not proof of unseen-case criterion correctness."}
