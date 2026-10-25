"""Offline audit AND logic, not a clinical grader or replacement for physician labels."""
from __future__ import annotations


def audit_conjunction(obligations, *, strict_factuality=False, full_claim_coverage=False):
    """Aggregate explicitly supplied research observations without guessing missing clauses.

    A not_met obligation must have an applicable, identified counterexample. Lack of
    retrieval cannot itself be a counterexample. This validates audit logic, not truth
    of the human/researcher's evidence assessment. No model or label access occurs.
    """
    if not obligations or len(obligations)>6:raise ValueError('Expected 1-6 bounded obligations.')
    ids=[o['id'] for o in obligations]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate obligation.')
    errors=[];states=[]
    for o in obligations:
        state=o['status']
        if state not in {'met','not_met','unresolved'}:raise ValueError('Unknown obligation status.')
        applicability=o.get('applicability','unresolved')
        if applicability not in {'in_scope','out_of_scope','unresolved'}:raise ValueError('Unknown scope.')
        if state=='not_met':
            if applicability!='in_scope':
                errors.append({'id':o['id'],'kind':'failure_scope_not_established'})
                state='unresolved'
            if not o.get('counterexample_ids') or o.get('basis') in {'no_source','could_add_more','length_only','disclaimer_present'}:
                errors.append({'id':o['id'],'kind':'unsupported_failure_basis'})
                state='unresolved'
        elif state=='met' and applicability!='in_scope':
            errors.append({'id':o['id'],'kind':'pass_scope_not_established'});state='unresolved'
        states.append(state)
    status='not_met' if 'not_met' in states else 'unresolved' if 'unresolved' in states else 'met'
    if strict_factuality and status=='met' and not full_claim_coverage:
        status='unresolved';errors.append({'id':'whole-answer','kind':'full_claim_coverage_not_established'})
    return {'audit_suggestion':status,'basis_warnings':errors,
            'clinical_release':False,'original_model_verdict_changed':False,
            'authority':'Non-blind research audit, not independent physician adjudication.'}
