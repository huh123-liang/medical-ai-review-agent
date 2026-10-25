"""V2 candidate: strict protocol, preserved research verdict, separate review policy."""
from __future__ import annotations
from .domain import canonical
from .evidence_ids import payload_sources,resolve_ids
from .fact_sources import retrieve_facts,FactSourceError
from .complex_scope import scope_kind
from .task_spec_v3 import task_spec_v3,assess_task,TASK_POLICY_V3

PROFILE='grounded-review-v2'
VERSION='review-policy-v2'


class FactBindingError(FactSourceError):
    def __init__(self,subcode):
        super().__init__('Invalid fact source binding.')
        self.issue={'kind':subcode}


def messages(case,rubric,binary,root):
    from .structured_review import SYSTEM_IDS
    system=SYSTEM_IDS+TASK_POLICY_V3+"""
Separate a research met/not_met verdict from evidence sufficiency. Missing retrieval is not
evidence of falsehood, and an unresolved claim is not automatically a failed criterion.
Keep the original rubric, conditions, numerical units and negations. A previous assistant's
medical statement is not an independently verified source. A source-card hit or selected ID
does not prove the whole answer. Use fact_checks only for actually covered selected claims;
never invent citations. For facts outside supplied coverage, explain the limitation and
request review; do not claim a guideline or bibliography was checked from memory.
The application independently computes review policy, retaining your verdict and review flag
separately. Invalid JSON, unknown IDs or fabricated source bindings remain protocol errors.
"""
    if not binary:system+='\nNon-benchmark: abstain is allowed with review_required=true when genuinely unjudgeable.'
    payload={'task':case.task,**payload_sources(case),'rubric':rubric.text,'rubric_id':rubric.rubric_id,
             'task_spec':task_spec_v3(case),'fact_review':retrieve_facts(case,root)}
    return [{'role':'system','content':system},{'role':'user','content':canonical(payload)}]


def decode_with_sources(parsed,case,rubric,facts):
    """facts must be trusted application retrieval or a verified historical snapshot.

    This function never loads labels, edits the verdict/reason or fetches external material.
    A caller replaying old output must label this as parser/policy replay, not a fresh model run.
    """
    from .structured_review import WireStructured
    wire=WireStructured.model_validate(parsed)
    judgment,trace=resolve_ids(wire,case)  # Provenance/structure remains strictly checked.
    cards=facts['cards'];lookup={f['id']:f for f in cards}
    if len(lookup)!=len(cards):raise FactBindingError('duplicate_supplied_source_id')
    pairs=[(c.claim_id,c.fact_id) for c in wire.fact_checks]
    if len(pairs)!=len(set(pairs)):raise FactBindingError('duplicate_claim_source_binding')
    for c in wire.fact_checks:
        if c.fact_id not in lookup:raise FactBindingError('source_id_not_provided')
        if c.claim_id not in wire.answer_ids:raise FactBindingError('claim_not_selected_answer')
    task=assess_task(case,wire.task_kind)
    reasons=list(task['review_reasons'])
    if wire.review_required:reasons.append('MODEL_REVIEW_REQUESTED')
    if not wire.fact_checks:
        evidence_status='insufficient_evidence'
        reasons.append('NO_BOUND_EXTERNAL_EVIDENCE')
    elif any(c.relation=='uncertain' for c in wire.fact_checks):
        evidence_status='selected_claims_unresolved'
        reasons.append('FACT_RELATION_UNCERTAIN')
    else:
        evidence_status='selected_claims_assessed_not_whole_answer'
    # There is not yet an independently measured full-answer claim inventory/coverage.
    # Even one supported claim cannot release the entire answer as medically verified.
    reasons.append('WHOLE_ANSWER_MEDICAL_COVERAGE_NOT_ESTABLISHED')
    effective_review=wire.review_required or bool(reasons)
    judgment=judgment.model_copy(update={'review_required':effective_review})
    review={'wire_protocol':'structured-ids-v1','review_policy_version':VERSION,
            'protocol_status':'valid','model_verdict':wire.judgment,'model_reason':wire.reason,
            'model_review_required':wire.review_required,'task_kind':wire.task_kind,
            'task_spec':task,'task_status':'conflict' if task['conflict'] else
                         'mixed_intent' if task['mixed_intent'] else 'heuristic_assessment',
            'evidence_status':evidence_status,'fact_status':evidence_status,
            'policy_status':'NeedsReview','review_reasons':reasons,
            'review_required':effective_review,'criterion_scope':scope_kind(rubric.text),
            'fact_checks':[c.model_dump() for c in wire.fact_checks],
            'fact_evidence':[lookup[f] for f in dict.fromkeys(c.fact_id for c in wire.fact_checks)],
            'fact_retrieval':facts,'medical_claims_fully_verified':False,
            'automatic_verdict_changes':False,
            'limits':'Research verdict preserved, not medically cleared. Missing evidence is not falsehood; '
                     'a valid selected binding is not independent entailment verification. '
                     'Review policy is intentionally conservative, not a calibrated selective classifier.'}
    return judgment,trace,review
