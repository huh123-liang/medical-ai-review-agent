"""Review pipeline v3: one LangGraph chain

    prepare -> rule_gate -> retrieve (three-level rubric knowledge base) -> freeze
            -> [one evaluator call per scoring dimension] -> aggregate -> audit
            -> human_review (interrupt, resumable) -> persist (JSON + HTML report)

Scoring dimensions and their scales come from a ScoringProfile, never hard-coded, so the same chain
can be validated on platform 1-5 ratings and on public datasets with their own ordinal scales.
Legacy pipelines (workflow.py, workflow_v2.py) are untouched.
"""
PIPELINE_VERSION = 'review-pipeline-v3.0'
