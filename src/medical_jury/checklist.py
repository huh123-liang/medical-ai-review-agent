"""Deterministic rubric clause IDs and validation; no doctor labels or disease facts."""
import re

VERSION='criterion-checklist-v1'


class ChecklistError(ValueError):
    pass


def required_checks(rubric):
    steps=re.findall(r'^Step\s+\d+:\s*(.+)$',rubric.text,re.M|re.I)
    if len(steps)>6: raise ChecklistError('Too many explicit criterion steps.')
    texts=steps or [rubric.text]
    return [{'check_id':f'C{i}','criterion_text':text} for i,text in enumerate(texts,1)]


def validate_checklist(wire,rubric):
    expected=[r['check_id'] for r in required_checks(rubric)]
    actual=[c.check_id for c in wire.criterion_checks]
    if sorted(actual)!=sorted(expected) or len(set(actual))!=len(actual):
        raise ChecklistError('Checklist coverage mismatch.')
    statuses=[c.status for c in wire.criterion_checks]
    if 'not_met' in statuses and wire.judgment!='not_met':
        raise ChecklistError('Failed conjunctive clause cannot produce met.')
    if all(s=='met' for s in statuses) and wire.judgment!='met':
        raise ChecklistError('All clauses met contradict final verdict.')
    if 'uncertain' in statuses and not wire.review_required:
        raise ChecklistError('Uncertain check must request human review.')
    return {'version':VERSION,'task_kind':wire.task_kind,'required_checks':required_checks(rubric),
            'checks':[c.model_dump() for c in wire.criterion_checks],
            'note':'Checks are model judgments, not externally verified facts or hidden reasoning.'}
