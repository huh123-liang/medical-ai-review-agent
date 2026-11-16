# Step 3 — judge fixes, grouped calls, LLM situation router (offline)

Source of the fixes: `reports/dev200-review-20261025T105907Z/REVIEW.md` §7 (written for the v2 per-criterion
judge; ported here to the v3 per-dimension evaluator). No paid call was made in step 3.

## What was implemented

| Review item | Implementation (v3) | Offline evidence |
|---|---|---|
| P0-1 polarity | Prompt `v3-dimension-2`: "pass" = the item's requirement as written is satisfied, never "the described problem is present"; items with their own procedure are followed literally | tests: prompt rules |
| P0-1 steps | Items whose text defines numbered steps are marked `"steps": N`; the model must report every step; a verdict that contradicts its steps is flagged `CHECK_STEP_CONFLICT` and sent to human review, **never re-decided**; missing steps = protocol failure | tests: conflict, consistent, not-applicable, missing |
| P1-1 generic rules | Examples count; "or" = any alternative; an offer of further help is not seeking context; "every claim correct" items fail on any doubtful claim; length alone is not excess detail; user/audience and rewrite fidelity checked first | tests: prompt rules |
| P1-2 locator-3 | Symmetric Markdown stripping (single `*`/`_` only when not between letters/digits), whitespace, list/heading markers (+ `cross_structure`), first-letter case; never upgraded to exact. Only `not_located` triggers review | replay on the sealed dev200 quotes: 306/306 exact unchanged; the 3 quotes with real changes stay not_located (`reports/locator3-replay-*/summary.json`) |
| P1-3 criterion defects | The hedging/no-uncertainty step-2 polarity is corrected in the KB with the change recorded in `source.note` (step 2) | KB tests |
| P1-4 alt-test wording | `alt_test()['status']` = `not_evaluable` / `passed` / `failed`; acceptance logic unchanged | test |
| Call merging | `--call-mode single`: one call scores all dimensions (shared conversation/answer once); envelope errors fail all dimensions, a defect inside one dimension fails only that one; reservation must cover the request | tests; plan: 1 call instead of 6 |
| Situation selection | `SituationRouter` (prompt `v3-situation-router-1`): conversation only, all candidate situations in fixed order, one choice or "none" per group; program selects (max 1 by default); failure applies nothing and flags every dimension `SITUATION_ROUTER_FAILED`; metadata situations skip the call | tests; mock end-to-end run |

The step-1 prompt `v3-dimension-1` is kept byte-for-byte (`--prompt-version v3-dimension-1`) for comparisons.

## Not done here (by design)

* P1-5 confidence calibration — needs paid dev runs (step 5).
* P2-1 quote-to-step linking, P2-2 fact-check routing — deferred; P2-2 overlaps with the step-6 RAG agent.
* No claim that any fix works: every change above came from SEEN dev200 labels, so its effect must be shown on new,
  unseen dev groups. The dev200 replay can only show that nothing regressed in the program logic.

## Measurements prepared for step 5 (each needs owner approval before running)

| Run | Command (plan first; add `--authorized` after review) | Calls |
|---|---|---|
| Router accuracy, dev-A (inspection) | `python scripts/eval_situation_router_v3.py --split A --n 60 --judge deepseek-pro` | 60 |
| Router accuracy, dev-B (report once) | `python scripts/eval_situation_router_v3.py --split B --n 150 --judge deepseek-pro` | 150 |
| Prompt v2 vs v1, per-dimension vs single call | via the step-4/5 ordinal validation harness on unseen dev groups | to be planned |

Router decision rule (pre-registered here, before any run): enable `--situation-policy llm` by default only if on
dev-B its top-1 accuracy is clearly above both similarity retrieval and the majority baseline on the same rows
(lower 95 % bound above 0.20) and its within-theme accuracy (emergent vs non-emergent etc.) is at least 0.60.
Otherwise situations stay metadata-only.
