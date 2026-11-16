"""Jury v2 LangGraph workflow (opt-in; the legacy workflow.py stays the default).

    prepare -> rule_gate -> [Send fan-out: every judge x every criterion, in parallel]
            -> aggregate (majority or frozen Dawid-Skene) -> audit (route uncertain items)
            -> human_review (LangGraph interrupt; resumable from the checkpoint) -> persist

Least privilege: judges receive only conversation/answer/criterion text and have no tools;
state never contains physician labels; the rule gate only flags, it never decides; human
decisions are stored next to (never over) the jury decision.
"""
from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from .domain import CaseInput, Rubric, digest
from .evaluation.aggregation import majority, posterior
from .judge_v2 import JudgeV2
from .safety import screen_case
from .store import Store, timestamp

WORKFLOW_VERSION = 'jury-v2-workflow-2'  # -2: p_met = judge P(met) / DS posterior, not vote share
HUMAN_DECISIONS = {'met', 'not_met', 'defer'}


class JuryState(TypedDict, total=False):
    job_id: str
    case: dict
    rubrics: list[dict]
    rule_screen: dict
    votes: Annotated[list[dict], operator.add]
    decisions: list[dict]
    review_queue: list[dict]
    human_decisions: dict
    report: dict


class JuryEngine:
    def __init__(self, judges: dict[str, JudgeV2], store: Store, root: Path, *, ds_params: dict | None = None,
                 review_band: tuple[float, float] = (0.25, 0.75), require_human: bool = True, checkpointer=None,
                 persist_reports: bool = True):
        if not judges:
            raise ValueError('A jury needs at least one judge.')
        self.judges, self.store, self.root = judges, store, root
        self.ds_params, self.review_band, self.require_human = ds_params, review_band, require_human
        self.persist_reports = persist_reports
        g = StateGraph(JuryState)
        for name in ('prepare', 'rule_gate', 'judge_one', 'aggregate', 'audit', 'human_review', 'persist'):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, 'prepare')
        g.add_edge('prepare', 'rule_gate')
        g.add_conditional_edges('rule_gate', self.fan_out, ['judge_one'])
        g.add_edge('judge_one', 'aggregate')
        g.add_edge('aggregate', 'audit')
        g.add_edge('audit', 'human_review')
        g.add_edge('human_review', 'persist')
        g.add_edge('persist', END)
        self.graph = g.compile(checkpointer=checkpointer)

    # -- nodes ---------------------------------------------------------------
    def prepare(self, state: JuryState):
        case = CaseInput.model_validate(state['case'])
        rubrics = [Rubric.model_validate(r) for r in state['rubrics']]
        if not rubrics or len(rubrics) > 60:
            raise ValueError('Need 1..60 criteria for one bounded jury run.')
        for r in rubrics:
            if r.level == 'case' and r.case_id != case.case_id:
                raise ValueError('Case-specific criterion bound to another case.')
        self.store.audit(state['job_id'], 'jury_v2_prepare', {
            'workflow': WORKFLOW_VERSION, 'judges': sorted(self.judges), 'n_criteria': len(rubrics),
            'case_sha256': digest(state['case']), 'aggregator': 'dawid_skene' if self.ds_params else 'majority'})
        return {}

    def rule_gate(self, state: JuryState):
        screen = screen_case(CaseInput.model_validate(state['case']))
        self.store.audit(state['job_id'], 'jury_v2_rule_gate', {'review_required': screen['review_required'],
                                                                'n_findings': len(screen['findings'])})
        return {'rule_screen': {'review_required': screen['review_required'], 'version': screen['version'],
                                'findings': [{'rule_id': f['rule_id'], 'finding': f['finding']} for f in screen['findings']]}}

    def fan_out(self, state: JuryState):
        return [Send('judge_one', {'job_id': state['job_id'], 'case': state['case'], 'rubric': r, 'judge_name': name})
                for r in state['rubrics'] for name in sorted(self.judges)]

    def judge_one(self, task: dict):
        judge = self.judges[task['judge_name']]
        rec = judge.judge(CaseInput.model_validate(task['case']), Rubric.model_validate(task['rubric']), task['job_id'])
        rec['judge_name'] = task['judge_name']
        return {'votes': [rec]}

    def aggregate(self, state: JuryState):
        """Three separate numbers (F04), never conflated:
        * ``vote_share``   -- fraction of completed judges voting met (0/1 for a single judge);
        * ``judge_p_met``  -- mean of the completed judges' own P(met) (verdict + confidence);
        * ``ds_posterior`` -- frozen Dawid-Skene posterior, only when DS parameters are frozen.
        ``p_met`` (the decision score used by the frozen threshold and the LOW_CONFIDENCE band) is the DS
        posterior when frozen, otherwise ``judge_p_met``. With one judge it is that judge's own P(met),
        so a dev-fitted threshold and the review band keep their meaning."""
        decisions = []
        for r in state['rubrics']:
            recs = {v['judge_name']: v for v in state['votes'] if v['rubric_id'] == r['rubric_id']}
            votes = {n: (recs[n]['verdict'] == 'met') if recs.get(n, {}).get('status') == 'Completed' else None
                     for n in sorted(self.judges)}
            valid = [v for v in votes.values() if v is not None]
            judge_ps = [recs[n]['p_met'] for n in sorted(self.judges)
                        if votes[n] is not None and isinstance(recs[n].get('p_met'), (int, float))]
            vote_share = (sum(valid) / len(valid)) if valid else None
            judge_p = (sum(judge_ps) / len(judge_ps)) if judge_ps else None
            ds_p = None
            if self.ds_params and valid:
                ds_p = posterior(votes, self.ds_params['judges'], self.ds_params['prior_met'])
                decision, p, method = ds_p >= 0.5, ds_p, 'dawid_skene_frozen'
            else:
                decision, p, method = majority(votes), judge_p, 'majority'
            failure_codes = {n: recs.get(n, {}).get('failure_code') or recs.get(n, {}).get('status', 'MISSING')
                             for n, v in votes.items() if v is None}
            decisions.append({'rubric_id': r['rubric_id'], 'votes': {k: (None if v is None else ('met' if v else 'not_met'))
                                                                     for k, v in votes.items()},
                              'failed_judges': [n for n, v in votes.items() if v is None],
                              'judge_failure_codes': failure_codes,
                              'jury_decision': None if decision is None else ('met' if decision else 'not_met'),
                              'p_met': p, 'vote_share': vote_share, 'judge_p_met': judge_p, 'ds_posterior': ds_p,
                              'method': method, 'weight': r.get('weight', 1.0)})
        return {'decisions': decisions}

    def audit(self, state: JuryState):
        lo, hi = self.review_band
        queue = []
        for d in state['decisions']:
            reasons = []
            if d['jury_decision'] is None:
                reasons.append('UNDECIDED_OR_TIE')
            if d['failed_judges']:
                reasons.append('JUDGE_FAILURE')
            if len({v for v in d['votes'].values() if v is not None}) > 1:
                reasons.append('JUDGES_DISAGREE')
            if d['p_met'] is not None and lo < d['p_met'] < hi:
                reasons.append('LOW_CONFIDENCE')
            if state['rule_screen']['review_required']:
                reasons.append('SAFETY_RULE_FLAG')
            d['review_reasons'] = reasons
            if reasons:
                queue.append({'rubric_id': d['rubric_id'], 'jury_decision': d['jury_decision'],
                              'p_met': d['p_met'], 'votes': d['votes'], 'reasons': reasons})
        self.store.audit(state['job_id'], 'jury_v2_audit', {'n_review': len(queue), 'n_total': len(state['decisions'])})
        return {'decisions': state['decisions'], 'review_queue': queue}

    def human_review(self, state: JuryState):
        queue = state.get('review_queue', [])
        if not queue or not self.require_human:
            return {'human_decisions': {}}
        answer = interrupt({'job_id': state['job_id'], 'items': queue,
                            'instructions': "Reply {rubric_id: {'decision': 'met'|'not_met'|'defer', 'reviewer': str, 'note': str}}"})
        if not isinstance(answer, dict):
            raise ValueError('Human review payload must be a dict.')
        clean = {}
        allowed = {q['rubric_id'] for q in queue}
        for rid, item in answer.items():
            if rid not in allowed or not isinstance(item, dict) or item.get('decision') not in HUMAN_DECISIONS:
                raise ValueError(f'Invalid human decision for {rid!r}.')
            if not str(item.get('reviewer', '')).strip():
                raise ValueError('Every human decision needs a reviewer id.')
            clean[rid] = {'decision': item['decision'], 'reviewer': str(item['reviewer'])[:80],
                          'note': str(item.get('note', ''))[:1000], 'at': timestamp()}
        self.store.audit(state['job_id'], 'jury_v2_human_review', {'n': len(clean), 'reviewers': sorted({v['reviewer'] for v in clean.values()})})
        return {'human_decisions': clean}

    def persist(self, state: JuryState):
        human = state.get('human_decisions', {})
        rows = []
        for d in state['decisions']:
            h = human.get(d['rubric_id'])
            if not d['review_reasons']:
                status, final = 'jury_auto', d['jury_decision']
            elif h is None:
                status, final = 'pending_human_review', None
            elif h['decision'] == 'defer':
                status, final = 'deferred', None
            else:
                status = 'human_confirmed' if h['decision'] == d['jury_decision'] else 'human_overrode'
                final = h['decision']
            rows.append({**d, 'human': h, 'final_status': status, 'final_decision': final})
        report = {'job_id': state['job_id'], 'workflow': WORKFLOW_VERSION, 'created_at': timestamp(),
                  'case_id': state['case']['case_id'], 'judges': sorted(self.judges), 'criteria': rows,
                  'rule_screen': state['rule_screen'],
                  'auto_share': sum(r['final_status'] == 'jury_auto' for r in rows) / len(rows),
                  'clinical_validation': 'NOT_VALIDATED',
                  'note': 'Jury and human decisions are stored separately; humans never overwrite the jury record.'}
        if self.persist_reports:
            folder = self.root / 'reports' / state['job_id']
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'jury-v2-report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        self.store.audit(state['job_id'], 'jury_v2_completed', {'report_sha256': digest(report)})
        return {'report': report}

    # -- driver ----------------------------------------------------------------
    def start(self, job_id: str, case: CaseInput, rubrics: list[Rubric]) -> dict:
        payload = {'job_id': job_id, 'case': case.model_dump(), 'rubrics': [r.model_dump() for r in rubrics]}
        return self._result(self.graph.invoke(payload, self._config(job_id)))

    def resume(self, job_id: str, human_decisions: dict) -> dict:
        return self._result(self.graph.invoke(Command(resume=human_decisions), self._config(job_id)))

    @staticmethod
    def _config(job_id):
        return {'configurable': {'thread_id': job_id}, 'max_concurrency': 8}

    @staticmethod
    def _result(out: dict) -> dict:
        if out.get('__interrupt__'):
            return {'status': 'AwaitingHumanReview', 'review': out['__interrupt__'][0].value}
        return {'status': 'Completed', 'report': out['report']}
