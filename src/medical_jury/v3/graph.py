"""The single v3 review chain (LangGraph).

    prepare -> rule_gate -> retrieve (+ optional LLM situation router) -> freeze
            -> [Send: evaluate one call group each, in parallel; default one dimension per call]
            -> aggregate -> audit -> human_review (interrupt) -> persist

Least privilege: the model gets text only, no tools; the state never contains physician labels;
rules flag and cap, they do not invent scores; human decisions are stored NEXT TO the agent's scores.
"""
from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from ..domain import CaseInput, digest
from ..safety import screen_case
from ..store import timestamp
from . import PIPELINE_VERSION
from .evaluator import DimensionEvaluator
from .locator import LOCATOR_VERSION
from .report import render_html
from .retrieval import Retriever, build_query
from .schema import ReviewRequest, RubricItem, ScoringProfile

HUMAN_DECISIONS = {'accept', 'override', 'defer'}


class ReviewState(TypedDict, total=False):
    job_id: str
    request: dict
    rule_screen: dict
    retrieval: dict
    snapshot: dict
    dimension_results: Annotated[list[dict], operator.add]
    scores: list[dict]
    review_queue: list[dict]
    human_decisions: dict
    report: dict


class ReviewPipeline:
    def __init__(self, profiles: dict[str, ScoringProfile], retriever: Retriever, evaluator: DimensionEvaluator,
                 store, root: Path, *, low_confidence: float = 0.7, require_human: bool = True,
                 persist_reports: bool = True, checkpointer=None, max_concurrency: int = 6,
                 call_mode: str = 'per_dimension'):
        if call_mode not in ('per_dimension', 'single'):
            raise ValueError('call_mode must be per_dimension or single')
        self.call_mode = call_mode
        self.profiles, self.retriever, self.evaluator, self.store, self.root = profiles, retriever, evaluator, store, root
        self.low_confidence, self.require_human, self.persist_reports = low_confidence, require_human, persist_reports
        self.max_concurrency = max_concurrency
        g = StateGraph(ReviewState)
        for name in ('prepare', 'rule_gate', 'retrieve', 'freeze', 'evaluate_dimension', 'aggregate', 'audit',
                     'human_review', 'persist'):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, 'prepare')
        g.add_edge('prepare', 'rule_gate')
        g.add_edge('rule_gate', 'retrieve')
        g.add_edge('retrieve', 'freeze')
        g.add_conditional_edges('freeze', self.fan_out, ['evaluate_dimension'])
        g.add_edge('evaluate_dimension', 'aggregate')
        g.add_edge('aggregate', 'audit')
        g.add_edge('audit', 'human_review')
        g.add_edge('human_review', 'persist')
        g.add_edge('persist', END)
        self.graph = g.compile(checkpointer=checkpointer)

    # ---- helpers -------------------------------------------------------------------------------
    def _profile(self, state) -> ScoringProfile:
        return self.profiles[state['request']['profile_id']]

    @staticmethod
    def _request(state) -> ReviewRequest:
        return ReviewRequest.model_validate(state['request'])

    # ---- nodes ---------------------------------------------------------------------------------
    def prepare(self, state: ReviewState):
        req = self._request(state)
        if req.profile_id not in self.profiles:
            raise ValueError(f'Unknown scoring profile {req.profile_id!r}.')
        self.store.audit(state['job_id'], 'v3_prepare', {'pipeline': PIPELINE_VERSION, 'profile': req.profile_id,
                                                        'case_sha256': digest(state['request']['case']),
                                                        'data_source': req.meta.data_source})
        return {}

    def rule_gate(self, state: ReviewState):
        screen = screen_case(self._request(state).case)
        return {'rule_screen': {'version': screen['version'], 'review_required': screen['review_required'],
                                'findings': [{'rule_id': f['rule_id'], 'finding': f['finding']} for f in screen['findings']]}}

    def retrieve(self, state: ReviewState):
        req, profile = self._request(state), self._profile(state)
        result = self.retriever.retrieve(build_query(req.case, req.meta, profile), job_id=state['job_id'])
        return {'retrieval': {'items': [{'item': r.item.model_dump(), 'why': r.why} for r in result.items],
                              'trace': result.trace}}

    def freeze(self, state: ReviewState):
        profile = self._profile(state)
        snap = {'pipeline': PIPELINE_VERSION, 'evaluator_prompt': self.evaluator.prompt_version,
                'locator': LOCATOR_VERSION, 'call_mode': self.call_mode, 'call_groups': self._groups(profile),
                'judge': getattr(self.evaluator.backend, 'name', '?'), 'model': getattr(self.evaluator.backend, 'model', '?'),
                'profile_id': profile.profile_id, 'profile_sha256': profile.sha256(),
                'case_sha256': digest(state['request']['case']),
                'retrieved_ids': [r['item']['item_id'] for r in state['retrieval']['items']],
                'retrieval_trace': state['retrieval']['trace'], 'frozen_at': timestamp()}
        snap['snapshot_sha256'] = digest({k: v for k, v in snap.items() if k != 'frozen_at'})
        self.store.audit(state['job_id'], 'v3_snapshot', snap)
        return {'snapshot': snap}

    def _groups(self, profile: ScoringProfile) -> list[list[str]]:
        keys = [d.key for d in profile.dimensions]
        return [keys] if self.call_mode == 'single' else [[k] for k in keys]

    def fan_out(self, state: ReviewState):
        return [Send('evaluate_dimension', {'job_id': state['job_id'], 'request': state['request'],
                                            'retrieval': state['retrieval'], 'dimensions': group})
                for group in self._groups(self._profile(state))]

    def evaluate_dimension(self, task: dict):
        profile = self.profiles[task['request']['profile_id']]
        items = [RubricItem.model_validate(r['item']) for r in task['retrieval']['items']]
        case = CaseInput.model_validate(task['request']['case'])
        specs = []
        for key in task['dimensions']:
            dim = profile.dimension(key)
            specs.append((dim, [i for i in items if i.dimension == key and i.role == 'scored'],
                          [i for i in items if i.dimension == key and i.role == 'reference']))
        return {'dimension_results': self.evaluator.evaluate_group(case, specs, task['job_id'])}

    def aggregate(self, state: ReviewState):
        profile = self._profile(state)
        by_dim = {r['dimension']: r for r in state['dimension_results']}
        scores = []
        for d in profile.dimensions:
            r = by_dim.get(d.key, {'status': 'NotRun', 'failure_code': 'MISSING_RESULT'})
            scores.append({'dimension': d.key, 'label': d.label, 'scale': [d.scale_min, d.scale_max],
                           'status': r['status'], 'failure_code': r.get('failure_code'),
                           'agent_score': r.get('score'), 'model_score': r.get('model_score'),
                           'confidence': r.get('confidence'), 'rule_adjustments': r.get('rule_adjustments', []),
                           'score_check_conflicts': r.get('score_check_conflicts', []),
                           'checks': r.get('checks', []), 'analysis': r.get('analysis'),
                           'reference_ids': r.get('reference_ids', [])})
        return {'scores': scores}

    def audit(self, state: ReviewState):
        queue = []
        trace = state['retrieval']['trace']
        router_failed = trace.get('situation_source') == 'llm_router_failed'
        for s in state['scores']:
            reasons = []
            if s['status'] != 'Completed':
                reasons.append('DIMENSION_NOT_SCORED')
            else:
                if s['confidence'] is not None and s['confidence'] < self.low_confidence:
                    reasons.append('LOW_CONFIDENCE')
                if s['rule_adjustments']:
                    reasons.append('RULE_CAP_APPLIED')
                if any(c.startswith('CHECK_STEP_CONFLICT') for c in s['score_check_conflicts']):
                    reasons.append('CHECK_STEP_CONFLICT')
                if any(not c.startswith('CHECK_STEP_CONFLICT') for c in s['score_check_conflicts']):
                    reasons.append('SCORE_CHECK_CONFLICT')
                # locator-3 diagnostic tiers (markup/whitespace/list/case) are recorded, not escalated
                if any(c['quote'] and c['quote_tier'] == 'not_located' for c in s['checks']):
                    reasons.append('QUOTE_NOT_LOCATED')
                if not s['checks']:
                    reasons.append('NO_CHECK_ITEMS_ANCHORS_ONLY')
            if state['rule_screen']['review_required']:
                reasons.append('SAFETY_RULE_FLAG')
            if router_failed:
                reasons.append('SITUATION_ROUTER_FAILED')
            s['review_reasons'] = reasons
            if reasons:
                queue.append({'dimension': s['dimension'], 'agent_score': s['agent_score'], 'scale': s['scale'],
                              'reasons': reasons})
        self.store.audit(state['job_id'], 'v3_audit', {'n_review': len(queue), 'n_dimensions': len(state['scores'])})
        return {'scores': state['scores'], 'review_queue': queue}

    def human_review(self, state: ReviewState):
        queue = state.get('review_queue', [])
        if not queue or not self.require_human:
            return {'human_decisions': {}}
        answer = interrupt({'job_id': state['job_id'], 'items': queue,
                            'instructions': "Reply {dimension: {'decision': 'accept'|'override'|'defer', "
                                            "'score': int (override only), 'reviewer': str, 'note': str}}"})
        if not isinstance(answer, dict):
            raise ValueError('Human review payload must be a dict.')
        allowed = {q['dimension']: q for q in queue}
        clean = {}
        for key, item in answer.items():
            if key not in allowed or not isinstance(item, dict) or item.get('decision') not in HUMAN_DECISIONS:
                raise ValueError(f'Invalid human decision for {key!r}.')
            if not str(item.get('reviewer', '')).strip():
                raise ValueError('Every human decision needs a reviewer id.')
            row = {'decision': item['decision'], 'reviewer': str(item['reviewer'])[:80],
                   'note': str(item.get('note', ''))[:1000], 'at': timestamp()}
            if item['decision'] == 'override':
                lo, hi = allowed[key]['scale']
                if type(item.get('score')) is not int or not lo <= item['score'] <= hi:
                    raise ValueError(f'Override score for {key!r} must be an integer in [{lo}, {hi}].')
                if not row['note'].strip():
                    raise ValueError('An override needs a note explaining why.')
                row['score'] = item['score']
            clean[key] = row
        self.store.audit(state['job_id'], 'v3_human_review',
                         {'n': len(clean), 'reviewers': sorted({v['reviewer'] for v in clean.values()})})
        return {'human_decisions': clean}

    def persist(self, state: ReviewState):
        human = state.get('human_decisions', {})
        req = self._request(state)
        rows = []
        for s in state['scores']:
            h = human.get(s['dimension'])
            if s['status'] != 'Completed' and (h is None or h['decision'] != 'override'):
                status, final = ('pending_human_review' if self.require_human and h is None else
                                 ('deferred' if h else 'not_scored')), None
            elif not s['review_reasons']:
                status, final = 'agent_auto', s['agent_score']
            elif h is None:
                status, final = ('pending_human_review' if self.require_human else 'agent_unreviewed'), \
                                (None if self.require_human else s['agent_score'])
            elif h['decision'] == 'defer':
                status, final = 'deferred', None
            elif h['decision'] == 'accept':
                status, final = 'human_confirmed', s['agent_score']
            else:
                status, final = 'human_overrode', h['score']
            rows.append({**s, 'human': h, 'final_status': status, 'final_score': final})
        report = {'job_id': state['job_id'], 'pipeline': PIPELINE_VERSION, 'created_at': timestamp(),
                  'case_id': req.case.case_id, 'profile_id': req.profile_id, 'data_source': req.meta.data_source,
                  'snapshot': state['snapshot'], 'rule_screen': state['rule_screen'],
                  'retrieved_items': state['retrieval']['items'], 'dimensions': rows,
                  'auto_share': sum(r['final_status'] == 'agent_auto' for r in rows) / len(rows),
                  'clinical_validation': 'NOT_VALIDATED',
                  'notes': ['Agent scores and human decisions are stored separately; humans never overwrite the agent record.',
                            'Quote location proves only that text exists in the answer, not that it is medically correct.',
                            'Reference notes (e.g. unreviewed disease guidance) informed the judge but were not scored.',
                            f"Situational criteria source: {state['retrieval']['trace'].get('situation_source', 'metadata')}."]}
        report['report_sha256'] = digest({k: v for k, v in report.items() if k != 'created_at'})
        if self.persist_reports:
            folder = self.root / 'reports' / 'v3-reviews' / state['job_id']
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / 'report.json').open('x', encoding='utf-8') as f:
                f.write(json.dumps(report, indent=2, ensure_ascii=False))
            with (folder / 'report.html').open('x', encoding='utf-8') as f:
                f.write(render_html(report, req.case))
        self.store.audit(state['job_id'], 'v3_completed', {'report_sha256': report['report_sha256']})
        return {'report': report}

    # ---- driver ----------------------------------------------------------------------------------
    def start(self, job_id: str, request: ReviewRequest) -> dict:
        payload = {'job_id': job_id, 'request': request.model_dump()}
        return self._result(self.graph.invoke(payload, self._config(job_id)))

    def resume(self, job_id: str, decisions: dict) -> dict:
        return self._result(self.graph.invoke(Command(resume=decisions), self._config(job_id)))

    def _config(self, job_id):
        return {'configurable': {'thread_id': job_id}, 'max_concurrency': self.max_concurrency}

    @staticmethod
    def _result(out: dict) -> dict:
        if out.get('__interrupt__'):
            return {'status': 'AwaitingHumanReview', 'review': out['__interrupt__'][0].value}
        return {'status': 'Completed', 'report': out['report']}
