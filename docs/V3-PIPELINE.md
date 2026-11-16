# v3 review pipeline (single agent) — status and operating notes

Pipeline version `review-pipeline-v3.0`, evaluator prompt `v3-dimension-2` (v1 kept verbatim for comparisons),
quote locator `locator-3`, situation router prompt `v3-situation-router-1`.
Step 1: the chain runs end to end with a mock judge. Step 2: sourced rubric KB (`kb/v3`) and local hybrid
retrieval (BM25 + Chroma + RRF); see `docs/V3-KNOWLEDGE-BASE.md`. Step 3: judge fixes from the dev200 review,
grouped calls and the LLM situation router; see `docs/V3-STEP3-JUDGE-FIXES.md`.
Step 4: ordinal validation data (LLM-Safety-For-PatientQA TREC dev/validation, MedExpert external test), the
two ordinal scoring profiles and the ordinal agreement statistics; pre-registered in
`docs/project/PROTOCOL-V3-ORDINAL.md`. Step 5 (offline part): `scripts/run_ordinal_v3.py` (plan -> score -> seal ->
labels -> metrics), `scripts/freeze_v3.py`, held-out batches + `scripts/aggregate_ordinal_v3.py`.
Step 5 dev-A pilot (2026-10-26, 49 items, 0 failures): the judge was stricter than the clinician raters on every
dimension. Added the anchor-only prompt `v3-ordinal-1` (refuses check items / reference notes), dev-B selection
(`--n 40 --complement`) and the paired comparison `scripts/compare_ordinal_runs_v3.py`; the adoption rule is
pre-registered in PROTOCOL-V3-ORDINAL section 9.
Step 5 results: BioNLP validation mean QWK 0.151, MedExpert factuality 0.143 / omission -0.014 (PROTOCOL-V3-ORDINAL
section 10). Step 6 (offline part): rubric-selection ablation on fresh HealthBench dev groups - single agent with the
whole theme rubric vs situation router + evaluator (RAG) vs oracle criterion - `src/medical_jury/v3/ablation.py`,
`scripts/run_rag_ablation_v3.py`, pre-registered in `docs/project/PROTOCOL-V3-RAG-ABLATION.md`.
Plan phase 0 (2026-10-26): structured data-approval record, cumulative budget CNY 200, `docs/project/INTENDED-USE.md`
(autonomy levels L0-L3). Plan phase 1 (offline part): selective evaluation with a finite-sample guarantee
(`src/medical_jury/evaluation/selective.py`, `scripts/run_selective_v3.py`, `docs/project/PROTOCOL-V3-SELECTIVE.md`);
the DeepSeek backend accepts an optional sampling temperature (default 0) for self-consistency confidence.
Plan phase 2 (offline part, 2026-10-27): router prompt `v3-situation-router-2` (decision principle + per-group decision
tests in `kb/router/decision-tests-v2.json`; v1 unchanged and still the default) and the paired v1 vs v2 comparison on
fresh dev groups (`scripts/run_router_v2.py`, `src/medical_jury/evaluation/router_scoring.py`), pre-registered in
`docs/project/PROTOCOL-V3-ROUTER-V2.md`.

## Chain (LangGraph StateGraph)

```
prepare -> rule_gate -> retrieve (hybrid ranking; situations from metadata or the LLM router) -> freeze
        -> fan_out --Send per call group--> evaluate (default: one call per dimension; --call-mode single: one call)
        -> aggregate -> audit -> human_review (interrupt) -> persist (report.json + report.html)
```

Review reasons (any one sends a dimension to the human queue): DIMENSION_NOT_SCORED, LOW_CONFIDENCE,
RULE_CAP_APPLIED, SCORE_CHECK_CONFLICT, CHECK_STEP_CONFLICT, QUOTE_NOT_LOCATED, NO_CHECK_ITEMS_ANCHORS_ONLY,
SAFETY_RULE_FLAG, SITUATION_ROUTER_FAILED. locator-3 diagnostic tiers (markup / whitespace / list / first-letter
case) are recorded but do not by themselves trigger review.

* **Rubric levels**: `general` / `disease` / `case` (`examples/v3/seed-kb.jsonl`). Disease-level items are
  `reference` (context only, never scored) until a named reviewer approves them (`approved_by`), as agreed 2026-10-25.
* **Scoring**: one LLM call per profile dimension. The model must answer every retrieved check item
  (pass / fail / not_applicable, exact quote) and then give an anchored score on the profile's own scale.
  Deterministic rules then apply: a failed `mandatory` item with `cap_on_fail` caps the score; score/check
  conflicts are flagged for human review, never silently "fixed".
* **Retrieval guard**: the query is built from the task + conversation only, never from the answer under review.
  Mandatory items are always kept.
* **Human review**: decisions `accept` / `override` (integer in scale + note) / `defer`; a reviewer name is required.
  The run is checkpointed (SQLite) and can be resumed from a new process.

## Safety gates for paid runs (`scripts/run_review_v3.py`)

| Gate | Behaviour |
|---|---|
| Plan first | Without `--authorized` the script prints calls, cost upper bound, ledger, data flow and **writes nothing**. |
| Budget | Shared ledger, cumulative cap CNY 200 (owner-approved 2026-10-26; was 100); `--run-limit` per run (default 1.0). |
| Placeholder rubrics | Refused if any dimension anchors or scored items are `project_draft`, unless `--allow-draft-rubrics`. |
| Non-public data | Refused unless `--data-approval-ref "approver=...;date=YYYY-MM-DD;provider=deepseek;scope=..."` names all four fields and the provider matches the run (`src/medical_jury/v3/data_approval.py`); the parsed record is stored in the audit plan. See `docs/project/INTENDED-USE.md`. |
| Provider errors | 401/402/403/429 or budget exhaustion stop all remaining dimension calls of the run (recorded as `NotRun`). A single 5xx does not. |
| Output tokens | Reservation uses the same `max_tokens` as the request (1200 per dimension; in single-call mode 900 per dimension + 400, or 500 per dimension + 400 for anchor-only ordinal profiles, capped at 6000; 600 for the router); a larger request is refused. |
| Router | Counted in the plan (one call) but never called while planning; the plan assumes all candidate situations' criteria are included, so the bound stays an upper bound. |

## Known limitations (to be handled in later steps)

1. **Pricing snapshot expires 2026-10-31** (`JURY_PRICING_VALID_UNTIL`). Recheck the official DeepSeek price page and
   update `jury-v2.json` before any paid step after that date; the ledger assumes no cache discount.
2. **Checkpoint database holds case text** (`reports/v3-reviews/checkpoints.sqlite`). `reports/` and `*.sqlite*` are
   git-ignored. Do not copy it off the machine; delete finished jobs' checkpoints when the study ends.
3. **Human decisions are not ground truth.** The reviewer sees the agent's score (anchoring). Use them as
   confirmations/overrides only; validation uses independent physician scores collected blind.
4. **LOW_CONFIDENCE rarely fires**: the dev200 run showed overconfidence (mean confidence 0.917 vs agreement 0.821).
   Calibration is part of step 5.
5. **Anchors**: `kb/v3/profile-platform-6d.json` (v0.2) has paper-adapted anchors (not clinically reviewed). The
   step-1 placeholders remain only in `examples/v3` for tests, and the paid gate still refuses them.
6. Prompt ordering puts conversation + answer first so a provider prefix cache can reuse them across the six
   dimension calls; the budget still charges full price.
7. **Situation selection**: similarity retrieval is not reliable for HealthBench situations (dev-B top-1 0.117 vs
   majority 0.112). Situational criteria are applied only when the case metadata names the situation.
8. **Embedding model is English-only** (`bge-small-en-v1.5`) and the FTS tokenizer does not segment Chinese.
9. **Unmeasured**: prompt v3-dimension-2, the router and single-call mode are implemented and tested offline only.
   Their effect must be measured on unseen dev groups (step 5); the dev200 groups that motivated the fixes cannot
   prove them.
