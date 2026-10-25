from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .budget import Budget, BudgetExceeded
from .config import Settings
from .domain import EvaluationRequest, canonical, digest, validate_judgment
from .provider import DeepSeekJudge, MockJudge, PROMPT_VERSION, PROFILE_VERSIONS, SYSTEM, build_messages, ProviderError,effective_profile
from .store import Cancelled, Store, timestamp
from .safety import screen_case, release_gate, VERSION as SAFETY_VERSION
from .task_scope import task_spec
from .task_spec_v2 import task_spec_v2
from .evidence_ids import catalog
from .structured_review import PROFILES as ID_PROFILES, fact_review, profile_task_spec


class State(TypedDict, total=False):
    job_id: str
    request: EvaluationRequest
    rubrics: list
    retrieval: dict
    snapshot: dict
    judgments: list
    report: dict
    safety_screen: dict


class Engine:
    def __init__(self, settings: Settings, store: Store, knowledge=None, judge_factory=None):
        self.settings, self.store, self.knowledge = settings, store, knowledge
        self.judge_factory = judge_factory
        graph = StateGraph(State)
        for name in ("prepare", "retrieve", "freeze", "judge", "validate", "aggregate", "persist"):
            graph.add_node(name, getattr(self, name))
        nodes = [START, "prepare", "retrieve", "freeze", "judge", "validate", "aggregate", "persist", END]
        for first, second in zip(nodes, nodes[1:]):
            graph.add_edge(first, second)
        self.graph = graph.compile()

    def stage(self, state: State, name: str):
        self.store.check_cancelled(state["job_id"])
        self.store.update(state["job_id"], status="Running", stage=name)
        self.store.audit(state["job_id"], name, {})

    def prepare(self, state):
        self.stage(state, "preparing_input")
        safety=screen_case(state['request'].case)
        self.store.audit(state['job_id'],'safety_screen',safety)
        return {'safety_screen':safety}

    def retrieve(self, state):
        self.stage(state, "retrieving_rubrics")
        request = state["request"]
        if request.mode == "oracle":
            rubrics, info = request.rubrics, {"mode": "oracle", "claim": "Provided criteria; NOT retrieval generalization."}
        else:
            if self.knowledge is None:
                from .knowledge import Knowledge
                self.knowledge = Knowledge(self.settings, self.store)
            rubrics, info = self.knowledge.retrieve(request.case, request.top_k)
        if not rubrics or len(rubrics) > 60:
            raise ValueError("Missing applicable rubrics or too many for one bounded run.")
        return {"rubrics": rubrics, "retrieval": info}

    def freeze(self, state):
        self.stage(state, "freezing_snapshot")
        request = state["request"]
        snapshot = {"schema_version": "jury-v1", "case": request.case.model_dump(),
                    "rubrics": [r.model_dump() for r in state["rubrics"]], "retrieval": state["retrieval"],
                    "provider": request.provider, "model": self.settings.model if request.provider == "deepseek" else MockJudge.model,
                    "prompt_version": PROFILE_VERSIONS[self.settings.judge_profile],
                    "judge_profile":self.settings.judge_profile,"evidence_policy":self.settings.evidence_policy,
                    "effective_judge_profile":effective_profile(request.case,self.settings.judge_profile),
                    "effective_prompt_version":PROFILE_VERSIONS[effective_profile(request.case,self.settings.judge_profile)],
                    "safety_screen_version":SAFETY_VERSION,"safety_screen":state['safety_screen'],
                    "system_prompt_hash":digest(build_messages(request.case,state['rubrics'][0],request.benchmark_binary,self.settings.judge_profile,self.settings.root)[0]['content']),
                    "protocol": "binary" if request.benchmark_binary else "abstention_allowed",
                    "temperature": 0, "thinking": "disabled", "max_output_tokens": self.settings.max_output_tokens}
        if self.settings.judge_profile == 'task-scope-v1':
            snapshot['task_scope'] = task_spec(request.case)
        if self.settings.judge_profile in ID_PROFILES:
            snapshot['source_catalog_hash']=digest(catalog(request.case))
            snapshot['task_scope']=profile_task_spec(request.case,self.settings.judge_profile)
            snapshot['fact_review']=fact_review(request.case,self.settings.judge_profile,self.settings.root)
            snapshot['message_hashes']=[digest(build_messages(request.case,r,request.benchmark_binary,self.settings.judge_profile,self.settings.root)) for r in state['rubrics']]
        snapshot["snapshot_hash"] = digest(snapshot)
        self.store.update(state["job_id"], snapshot_json=canonical(snapshot))
        return {"snapshot": snapshot}

    def judge(self, state):
        self.stage(state, "judging_criteria")
        request, job_id = state["request"], state["job_id"]
        judge = self.judge_factory(request.provider) if self.judge_factory else (
            MockJudge() if request.provider == "mock" else DeepSeekJudge(self.settings, self.store))
        judgments = []
        try:
            for rubric in state["rubrics"]:
                self.store.check_cancelled(job_id)
                judgment = judge.judge(request.case, rubric, job_id, request.max_run_cost_cny, request.benchmark_binary)
                self.store.check_cancelled(job_id)
                judgments.append({"rubric_id": rubric.rubric_id, "weight": rubric.weight, **judgment.model_dump()})
                self.store.audit(job_id, "criterion_judged", {"rubric_id": rubric.rubric_id, **judgment.model_dump(),
                                                           "note": "Partial/unvalidated until job completion."})
        finally:
            if hasattr(judge, "close"):
                judge.close()
        return {"judgments": judgments}

    def validate(self, state):
        from .domain import Judgment
        self.stage(state, "validating_result")
        expected = {r.rubric_id for r in state["rubrics"]}
        actual = [r["rubric_id"] for r in state["judgments"]]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError("Criterion coverage mismatch.")
        for item in state["judgments"]:
            obj = Judgment.model_validate({key: value for key, value in item.items() if key not in ("rubric_id", "weight")})
            validate_judgment(obj, state["request"].case, state["request"].benchmark_binary)
        return {}

    def aggregate(self, state):
        self.stage(state, "aggregating")
        request, judgments = state["request"], state["judgments"]
        abstained = sum(row["judgment"] == "abstain" for row in judgments)
        positive_weight = sum(max(0, row["weight"]) for row in judgments)
        earned = sum(row["weight"] for row in judgments if row["judgment"] == "met")
        score = max(0, min(1, earned / positive_weight)) if positive_weight and not abstained else None
        traces=[json.loads(e['detail_json']) for e in self.store.events(state['job_id']) if e['event']=='evidence_resolution']
        gate=release_gate(score,state['safety_screen'],traces)
        structured=[json.loads(e['detail_json']) for e in self.store.events(state['job_id']) if e['event']=='structured_review']
        if structured and (any(r['review_required'] for r in structured) or self.settings.judge_profile=='grounded-review-v1'):
            gate['weighted_score']=None
            gate['release_status']='NeedsReview'
            gate['release_block_reasons'].append('FACT_OR_TASK_VERIFICATION_PARTIAL_RESEARCH_ONLY')
        for review in structured:
            if review.get('review_policy_version')=='review-policy-v2':
                gate['weighted_score']=None
                gate['release_status']='NeedsReview'
                gate['release_block_reasons'].extend(review['review_reasons'])
        gate['release_block_reasons']=list(dict.fromkeys(gate['release_block_reasons']))
        report = {"job_id": state["job_id"], "created_at": timestamp(), "case_id": request.case.case_id,
                  "group_id": request.case.group_id, "split": request.case.split,
                  "schema_version": "jury-v1", "snapshot_hash": state["snapshot"]["snapshot_hash"],
                  "provider": request.provider, "model": state["snapshot"]["model"], "mode": request.mode,
                  "judgments": judgments, "criterion_count": len(judgments), "coverage": 1 - abstained / len(judgments),
                  **gate,"safety_screen":state['safety_screen'],
                  "score_protocol": "Research score is weighted met sum / positive-weight maximum; weighted_score withheld on abstention, safety alerts or ambiguous evidence. Never a clinical clearance or 1–5 doctor score.",
                  "clinical_validation": "NOT_VALIDATED" if request.provider != "mock" else "OFFLINE_MOCK_NOT_CLINICAL",
                  "review_required": bool(gate['release_block_reasons']) or any(row["review_required"] or row["judgment"] != "met" for row in judgments),
                  "retrieval": state["retrieval"], "budget": Budget(self.settings, self.store).total(),
                  "evidence_policy":self.settings.evidence_policy,
                  "evidence_resolution":traces,
                  "criterion_checklists":[json.loads(e['detail_json']) for e in self.store.events(state['job_id']) if e['event']=='criterion_checklist'],
                  "task_scope": state['snapshot'].get('task_scope'),
                  "structured_reviews": structured,
                  "fact_review":state['snapshot'].get('fact_review'),
                  "limitations": ["Exact evidence locations are checked, not full semantic medical truth.",
                                  "This is research quality review, not diagnosis, prescribing or clinical certification."]}
        return {"report": report}

    def persist(self, state):
        self.stage(state, "saving")
        self.store.update(state["job_id"], status="Completed", stage="completed", report_json=canonical(state["report"]))
        self.store.audit(state["job_id"], "completed", {"snapshot_hash": state["snapshot"]["snapshot_hash"]})
        folder = self.settings.root / "reports" / state["job_id"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "report.json").write_text(json.dumps(state["report"], ensure_ascii=False, indent=2), encoding="utf-8")
        lines = ["# Medical AI Jury — Research Report", "", f"Status: **{state['report']['clinical_validation']}**", "",
                 f"Case: `{state['report']['case_id']}`", f"\nSnapshot: `{state['report']['snapshot_hash']}`", "",
                 "This report is NOT a physician assessment or clinical advice.", ""]
        report=state['report']
        lines += [f"Release status: **{report['release_status']}**",f"Human review required: {report['review_required']}",
                  f"Research score (not clearance): {report['research_weighted_score']}",
                  f"Usable weighted score (withheld when blocked): {report['weighted_score']}",
                  f"Blocking reasons: {', '.join(report['release_block_reasons']) or 'none; research only'}",'']
        for finding in report['safety_screen']['findings']:
            lines += [f"## Safety review — {finding['rule_id']}",finding['finding'],
                      f"Source: [{finding['source']['title']}]({finding['source']['url']})",finding['evidence'],'']
        ambiguous=sum(t.get('ambiguous_count',0) for t in report['evidence_resolution'])
        if ambiguous:lines += [f'**Evidence has {ambiguous} ambiguous quote(s). All candidate positions are in report.json; none was selected silently.**','']
        for row in state["judgments"]:
            lines += [f"## {row['rubric_id']} — {row['judgment']}", "", row["reason"], ""]
        (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")
        return {}

    def run(self, request: EvaluationRequest, job_id: str | None = None) -> dict:
        job_id = job_id or self.store.create_job(request)
        try:
            if request.provider == "deepseek":
                self.settings.validate_paid()
            self.graph.invoke({"job_id": job_id, "request": request})
        except Cancelled:
            self.store.update(job_id, status="Cancelled", stage="cancelled", error_code="CANCELLED")
            self.store.audit(job_id, "cancelled", {})
        except Exception as error:
            if isinstance(error, BudgetExceeded):
                code = "BUDGET_BLOCKED"
            elif isinstance(error, ProviderError):
                code = error.code
            else:
                code = "VALIDATION_OR_CONFIGURATION_ERROR"
            self.store.update(job_id, status="Failed", stage="failed", error_code=code)
            self.store.audit(job_id, "failed", {"error_code": code, "exception_type": type(error).__name__})
        return self.store.get_job(job_id)
