from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

from .domain import CaseInput, Message, Rubric, canonical, digest


META_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_meta_eval.jsonl"
DATASET_VERSION = "healthbench-meta-2025-05-07"


def download_meta(root: Path) -> dict:
    """Download official public data only. No provider calls, no input from local EHR."""
    target = root / "data/public/healthbench-meta.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(".partial")
        total = 0
        try:
            with urllib.request.urlopen(META_URL, timeout=45) as response, partial.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 256 * 1024 * 1024:
                        raise ValueError("Public data download exceeds 256 MiB limit.")
                    out.write(chunk)
            partial.replace(target)
        except BaseException:
            # Keep bounded partial file for diagnosis; next attempt explicitly overwrites it.
            raise
    hasher = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    manifest = {"source_url": META_URL, "sha256": hasher.hexdigest(), "bytes": target.stat().st_size,
                "version": DATASET_VERSION, "paper": "https://arxiv.org/abs/2505.08775",
                "upstream_code": "https://github.com/openai/simple-evals/blob/main/healthbench_meta_eval.py",
                "redistribution": "Local research download. Check upstream terms before redistribution; preserve canary in raw data."}
    (target.parent / "source-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def convert_row(row: dict) -> tuple[dict, dict]:
    labels = row["binary_labels"]
    if not labels or not all(type(label) is bool for label in labels):
        raise ValueError("Missing or non-boolean physician labels; cannot fabricate them.")
    if len(labels) != len(row["anonymized_physician_ids"]):
        raise ValueError("Physician/label cardinality mismatch.")
    context = [Message.model_validate(message) for message in row["prompt"]]
    # Group all completions/criteria of identical normalized conversation together.
    normalized = re.sub(r"\s+", " ", canonical([m.model_dump() for m in context])).strip().casefold()
    group_id = "HBG-" + hashlib.sha256(normalized.encode()).hexdigest()[:24]
    split_number = int(digest({"group": group_id, "seed": "jury-v1"})[:8], 16) % 100
    split = "dev" if split_number < 60 else "validation" if split_number < 80 else "test"
    case_id = "HBC-" + digest({"context": [m.model_dump() for m in context], "completion": row["completion"]})[:24]
    example_id = "HBX-" + digest({"case": case_id, "rubric": row["rubric"], "category": row["category"]})[:24]
    case = CaseInput(case_id=case_id, group_id=group_id, task="Assess the fixed assistant completion against the supplied physician-authored criterion.",
                     context=context, candidate_output=row["completion"], source="HealthBench meta-evaluation",
                     source_version=DATASET_VERSION, split=split)
    rubric = Rubric(rubric_id="HBR-" + digest({"case": case_id, "text": row["rubric"]})[:24],
                    text=row["rubric"], level="case", case_id=case_id, group_id=group_id, split=split,
                    source="HealthBench physician-authored consensus criterion", version=DATASET_VERSION,
                    approved=True, mandatory=False,
                    applicability="Bound to this original conversation/answer; not reviewed for cross-case reuse.")
    public = {"example_id": example_id, "case": case.model_dump(), "rubric": rubric.model_dump(), "category": row["category"]}
    private = {"example_id": example_id, "group_id": group_id, "binary_labels": labels,
               "label_source": "public physician annotations", "protocol": "pairwise each physician, not majority collapse"}
    return public, private


def prepare_dataset(root: Path) -> dict:
    raw = root / "data/public/healthbench-meta.jsonl"
    out = root / "data/prepared"
    private_dir = root / "data/private"
    out.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    counts = defaultdict(int)
    groups = defaultdict(set)
    rejected = defaultdict(int)
    handles = {name: (out / f"{name}.jsonl").open("w", encoding="utf-8") for name in ("dev", "validation", "test")}
    try:
        with raw.open(encoding="utf-8") as source, (private_dir / "physician-labels.jsonl").open("w", encoding="utf-8") as labels:
            for line_no, line in enumerate(source, 1):
                try:
                    public, reference = convert_row(json.loads(line))
                    if public["example_id"] in seen:
                        rejected["duplicate_example"] += 1
                        continue
                    seen.add(public["example_id"])
                    split = public["case"]["split"]
                    handles[split].write(canonical(public) + "\n")
                    labels.write(canonical(reference) + "\n")
                    counts[split] += 1
                    groups[split].add(public["case"]["group_id"])
                except (ValueError, KeyError, TypeError):
                    rejected["invalid_or_out_of_input_bounds"] += 1
    finally:
        for handle in handles.values():
            handle.close()
    if any(groups[a] & groups[b] for a, b in (("dev", "validation"), ("dev", "test"), ("validation", "test"))):
        raise AssertionError("Conversation-group leakage.")
    report = {"dataset_version": DATASET_VERSION, "examples": dict(counts), "groups": {k: len(v) for k, v in groups.items()},
              "rejected": dict(rejected), "group_overlap": 0,
              "warning": "Project-held-out splits of a PUBLIC benchmark, not proof of pretraining contamination absence. Exact-normalized duplicates grouped; semantic near-duplicates require future audit."}
    (out / "preparation-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def examples(root: Path, split: str, limit: int, max_per_group: int = 2) -> list[dict]:
    if split not in {"dev", "validation", "test"} or not 1 <= limit <= 1000:
        raise ValueError("Invalid split or bounded sample count.")
    selected, counts = [], defaultdict(int)
    with (root / f"data/prepared/{split}.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            group = item["case"]["group_id"]
            if counts[group] >= max_per_group:
                continue
            counts[group] += 1
            selected.append(item)
            if len(selected) >= limit:
                break
    return selected


def binary_metrics(pairs: list[tuple[bool, bool]]) -> dict:
    if not pairs:
        return {"macro_f1": None, "agreement": None, "cohen_kappa": None, "n_physician_comparisons": 0}
    tp = sum(pred and gold for pred, gold in pairs)
    tn = sum(not pred and not gold for pred, gold in pairs)
    fp = sum(pred and not gold for pred, gold in pairs)
    fn = sum(not pred and gold for pred, gold in pairs)
    n = len(pairs)
    pos_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    neg_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else None
    agreement = (tp + tn) / n
    expected = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    return {"macro_f1": (pos_f1 + neg_f1) / 2 if pos_f1 is not None and neg_f1 is not None else None,
            "agreement": agreement, "cohen_kappa": (agreement - expected) / (1 - expected) if expected < 1 else None,
            "positive_precision": tp / (tp + fp) if tp + fp else None,
            "positive_recall": tp / (tp + fn) if tp + fn else None,
            "negative_precision": tn / (tn + fn) if tn + fn else None,
            "negative_recall": tn / (tn + fp) if tn + fp else None,
            "n_physician_comparisons": n}


def score_run(root: Path, predictions: list[dict], provider: str, bootstrap: int = 200) -> dict:
    if provider == "mock":
        return {"validation_status": "OFFLINE_PIPELINE_ONLY_NOT_CLINICAL", "metrics": None,
                "reason": "Mock predictions are never reported as agreement with doctors."}
    references = {r["example_id"]: r for r in map(json.loads, (root / "data/private/physician-labels.jsonl").read_text(encoding="utf-8").splitlines())}
    grouped = defaultdict(list)
    failed = []
    for row in predictions:
        ref = references[row["example_id"]]
        if row.get("judgment") not in {"met", "not_met"}:
            failed.append(row["example_id"])
            continue
        pred = row["judgment"] == "met"
        grouped[ref["group_id"]].extend((pred, gold) for gold in ref["binary_labels"])
    # No cherry-picking successful predictions: primary metric withheld on incomplete runs.
    if failed:
        return {"validation_status": "INCOMPLETE_NO_PRIMARY_METRIC", "metrics": None,
                "coverage": (len(predictions) - len(failed)) / len(predictions), "failed_examples": failed}
    pairs = [pair for values in grouped.values() for pair in values]
    metric = binary_metrics(pairs)
    rng, boots = random.Random(42), []
    keys = list(grouped)
    for _ in range(bootstrap if len(keys) > 1 else 0):
        sample = [pair for key in rng.choices(keys, k=len(keys)) for pair in grouped[key]]
        value = binary_metrics(sample)["macro_f1"]
        if value is not None:
            boots.append(value)
    boots.sort()
    ci = [boots[int((len(boots)-1)*0.025)], boots[int((len(boots)-1)*0.975)]] if boots else None
    return {"validation_status": "PUBLIC_BENCHMARK_SUBSET", "metrics": metric, "macro_f1_group_bootstrap_95ci": ci,
            "n_examples": len(predictions), "n_conversation_groups": len(keys), "coverage": 1.0,
            "protocol": "Binary decision compared to each original physician annotation; group-level resampling.",
            "limitations": "Not a full-paper replication; public data may be in model pretraining. No 1–5 MAE/Spearman conversion."}
