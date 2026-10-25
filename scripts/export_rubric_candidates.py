"""Catalog public development criteria for later human review; do not auto-approve reuse."""
import json
from collections import defaultdict

from medical_jury.config import ROOT
from medical_jury.domain import canonical, digest

candidates = {}
source_cases = defaultdict(set)
with (ROOT / "data/prepared/dev.jsonl").open(encoding="utf-8") as stream:
    for line in stream:
        row = json.loads(line)
        text = row["rubric"]["text"]
        key = digest(text)[:24]
        source_cases[key].add(row["case"]["group_id"])
        candidates[key] = {"candidate_id": "CAND-" + key, "text": text,
            "source": "HealthBench public meta-evaluation DEVELOPMENT criteria",
            "proposed_level": "task", "approved_for_cross_case_reuse": False,
            "review_required": "Confirm applicability, level, conditions and scoring before indexing as reusable.",
            "source_version": row["rubric"]["version"]}
file = ROOT / "data/rubric-candidates-for-review.jsonl"
with file.open("w", encoding="utf-8") as out:
    for key, row in sorted(candidates.items()):
        row["development_conversation_count"] = len(source_cases[key])
        out.write(canonical(row) + "\n")
print(json.dumps({"candidate_templates": len(candidates), "file": str(file),
                  "approved_for_cross_case_reuse": 0, "note": "This catalog is NOT a physician-approved reusable medical knowledge base."}, indent=2))
