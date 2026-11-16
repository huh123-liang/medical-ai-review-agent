"""v4: error-detection review of chronic-disease management advice (PLAN-V4-ERROR-DETECTION).

Two model calls per case (extract, check); everything else is deterministic code:
extract -> verify extraction -> agent rule layer -> applicable criteria -> check -> aggregate -> triage.
Rules, templates and severities are project-written from cited sources and NOT physician-reviewed.
The agent's rule layer is independent of the testbed's answer key (src/medical_jury/testbed/rules.py)."""
PIPELINE_VERSION = 'review-v4.1'   # 4.1: step 7 reconciles open-review claims with code checks
