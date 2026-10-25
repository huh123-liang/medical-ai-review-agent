"""Generate a truthful local readiness summary; never calls a paid model."""
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone

from medical_jury.budget import Budget
from medical_jury.config import Settings
from medical_jury.store import Store

s = Settings.load()
s.prepare()
store = Store(s.root)
prepared = s.root / "data/prepared/preparation-report.json"
experiments = sorted((s.root / 'reports').glob('stratified-*/analysis.json'))
latest_analysis = json.loads(experiments[-1].read_text(encoding='utf-8')) if experiments else None
paired_path=s.root/'reports/iteration-4/paired-analysis.json'
if not paired_path.exists():paired_path=s.root/'reports/iteration-3/paired-analysis.json'
paired=json.loads(paired_path.read_text(encoding='utf-8')) if paired_path.exists() else None
iteration4_path=s.root/'reports/iteration-4/completion-summary.json'
iteration4=json.loads(iteration4_path.read_text(encoding='utf-8')) if iteration4_path.exists() else None
iteration5_path=s.root/'reports/iteration-5/completion-summary.json'
iteration5=json.loads(iteration5_path.read_text(encoding='utf-8')) if iteration5_path.exists() else None
with store.connect() as db:
    statuses = [dict(row) for row in db.execute("SELECT provider,status,mode,COUNT(*) AS n FROM jobs GROUP BY provider,status,mode")]
    rubric_count = db.execute("SELECT COUNT(*) FROM rubrics").fetchone()[0]
summary = {
    "verified_at": datetime.now(timezone.utc).isoformat(), "root": str(s.root),
    "framework_versions": {name: importlib.metadata.version(name) for name in ("langgraph", "chromadb", "fastembed", "fastapi", "httpx", "pydantic")},
    "public_data": json.loads(prepared.read_text()) if prepared.exists() else None,
    "indexed_case_bound_development_rubrics": rubric_count,
    "job_summary": statuses, "budget": Budget(s, store).total(),
    "local_credential_configured": bool(s.api_key),
    "default_judge_profile":s.judge_profile,"default_evidence_policy":s.evidence_policy,
    "latest_paired_development_test":({k:paired[k] for k in ('planned_per_arm','common_valid_examples','status_counts',
         'common_valid_metrics','paired_delta_95ci','batch_accounted_cny')} if paired else None),
    "latest_paired_development_source":str(paired_path.relative_to(s.root)) if paired else None,
    "actual_doctor_agreement_metrics": ("STRATIFIED_DEV_INCOMPLETE_PRIMARY_METRICS_WITHHELD" if latest_analysis and latest_analysis['failed'] else "REAL_DEV_SUBSET_METRICS" if latest_analysis else "FIRST_PILOT_INCOMPLETE_PRIMARY_METRICS_WITHHELD" if any(r['provider']=='deepseek' for r in statuses) else "NOT_MEASURED_UNTIL_REAL_MODEL_META_EVALUATION"),
    "latest_stratified_test": ({'planned':latest_analysis['planned'],'completed':latest_analysis['completed'],
                               'failed':latest_analysis['failed'],'primary_metrics':latest_analysis['primary_metrics'],
                               'report':str(experiments[-1].relative_to(s.root))} if latest_analysis else None),
    "implemented": ["LangGraph single-agent workflow", "Chroma + local BGE embeddings", "Admin-authenticated local API", "official public data adapter and split isolation", "mocked/offline tests", "conservative persistent cost reservation"],
    "live_deepseek_attempted": any(r['provider']=='deepseek' for r in statuses),
    "not_implemented_or_not_validated": ["Complete representative real-model clinical baseline", "DeepEval adapter", "DSPy/TextGrad optimization", "Prometheus deployment", "multi-agent collaboration", "existing-platform integration", "generalizable cross-case clinical rubric selection", "semantic medical fact validation"],
}
if paired:
    summary['actual_doctor_agreement_metrics']='PAIRED_DEV_A_COMPLETE_B_INCOMPLETE_NO_VALIDATED_IMPROVEMENT'
if iteration4:
    summary['latest_reliability_iteration']=iteration4
    summary['actual_doctor_agreement_metrics']='ITER4_ROUTED_SMOKE_PASSED_GLOBAL_V4_NOT_PROMOTED_NO_INDEPENDENT_VALIDATION'
if iteration5:
    latest_path=s.root/'reports/iteration-5/candidate-v2/new/analysis.json'
    latest=json.loads(latest_path.read_text(encoding='utf-8'))
    summary['latest_complex_boundary_iteration']=iteration5
    summary['actual_doctor_agreement_metrics']='NEW80_DEV_INCOMPLETE_PRIMARY_WITHHELD_QUALITY_GATE_NOT_PASSED'
    summary['latest_paired_development_source']=str(latest_path.relative_to(s.root))
    summary['latest_paired_development_test']={'planned_per_arm':latest['planned'],
        'common_valid_examples':latest['scores']['common_valid_examples'],'status_counts':latest['status_counts'],
        'common_valid_metrics':latest['scores']['common_valid_metrics'],'uncertainty':latest['uncertainty'],
        'phase_accounted_cny':latest['phase_accounted_cny']}
(s.root / "reports/project-verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
