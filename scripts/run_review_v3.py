"""Run ONE case through the v3 review chain (retrieve -> evaluate -> score -> audit -> human review -> report).

Offline demo (no network, hash-derived mock judge):
    python scripts/run_review_v3.py --case examples/v3/demo-case.json
Resume after human review:
    python scripts/run_review_v3.py --resume <job_id> --decisions decisions.json
Paid judge (plan first, then --authorized):
    python scripts/run_review_v3.py --case <file> --judge deepseek-pro
    python scripts/run_review_v3.py --case <file> --judge deepseek-pro --authorized --run-limit 1

Writes reports/v3-reviews/<job_id>/report.json + report.html (never overwritten) and a LangGraph checkpoint
database reports/v3-reviews/checkpoints.sqlite so a run waiting for human review can be resumed later.

Retrieval (step 2): by default the local hybrid retriever (SQLite FTS5 BM25 + Chroma + RRF) over kb/v3, using
the embedding model already present in .cache/fastembed (never downloaded at run time). Its index is a cache
under data/v3-index, bound to the KB hash and model files. ``--retriever filter`` uses the metadata filter only.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from medical_jury.budget import Budget  # noqa: E402
from medical_jury.config import Settings  # noqa: E402
from medical_jury.evaluation.runner import budget_gate, ledger_status  # noqa: E402
from medical_jury.judge_v2 import DeepSeekBackend, ScriptedBackend  # noqa: E402
from medical_jury.jury_config import load_panel  # noqa: E402
from medical_jury.store import Store  # noqa: E402
from medical_jury.v3.evaluator import (EVALUATOR_PROMPT_VERSION, MAX_OUTPUT_TOKENS_V3, PROMPT_VERSIONS,  # noqa: E402
                                       DimensionEvaluator, build_messages_group, build_messages_v3, mock_responder_v3)
from medical_jury.v3.router import (ROUTER_MAX_OUTPUT_TOKENS, SituationRouter, build_router_messages,  # noqa: E402
                                    mock_router_responder)
from medical_jury.v3.graph import ReviewPipeline  # noqa: E402
from medical_jury.v3.hybrid import FastEmbedEmbedder, HashEmbedder, HybridRetriever, open_index  # noqa: E402
from medical_jury.v3.retrieval import FilterRetriever, RetrievalQuery, build_query  # noqa: E402
from medical_jury.v3.schema import ReviewRequest, load_kb, load_profile  # noqa: E402

DEFAULT_PROFILES = ['kb/v3/profile-platform-6d.json']
EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'
# Retrieval configuration frozen after the step-2 dev-A comparison (reports/retrieval-v3-20261025T133001Z-ce3be8).
# Situations are applied from case metadata; the hybrid ranking is recorded as a suggestion only (see docs).
HYBRID_CONFIG = {'mode': 'hybrid', 'top_situations': 1, 'rrf_k': 10, 'query_mode': 'full',
                 'doc_fields': 'applies_when+criteria'}
from medical_jury.v3.data_approval import PUBLIC_SOURCES, ApprovalError, check_run  # noqa: E402,F401
DEFAULT_KB = 'kb/v3'


def make_backend(judge: str, root: Path, store, run_limit: float, max_tokens: int = MAX_OUTPUT_TOKENS_V3,
                 responder=mock_responder_v3, temperature: float = 0.0):
    """The reservation is made with ``max_tokens`` output tokens, so it must equal the largest request."""
    if judge == 'mock':
        return ScriptedBackend(responder, name='mock'), False, None
    spec = next((j for j in load_panel(root) if j['name'] == judge), None)
    if spec is None:
        raise SystemExit(f'Judge {judge!r} not in jury-v2.json.')
    if spec['kind'] != 'deepseek':
        raise SystemExit('v3 supports the mock judge and DeepSeek only (output-token budget must match).')
    settings = Settings.load(root)
    settings = replace(settings, model=spec.get('model', settings.model),
                       input_rate=float(spec['price_in']), output_rate=float(spec['price_out']))
    backend = DeepSeekBackend(settings, store, run_limit, max_tokens=max_tokens, temperature=temperature)
    backend.name = spec['name']
    return backend, True, settings


def eval_max_tokens(call_mode: str, n_dimensions: int) -> int:
    return MAX_OUTPUT_TOKENS_V3 if call_mode == 'per_dimension' else \
        DimensionEvaluator(None).group_max_tokens(n_dimensions)


def make_retriever(kb, root: Path, kind: str, embedder_kind: str, situation_policy: str = 'metadata', router=None):
    if kind == 'filter':
        if situation_policy != 'metadata':
            raise SystemExit('--retriever filter supports only --situation-policy metadata.')
        return FilterRetriever(kb)
    if embedder_kind == 'hash':
        embedder = HashEmbedder()          # offline tests only: not a semantic model
    else:
        embedder = FastEmbedEmbedder(EMBEDDING_MODEL, root / '.cache' / 'fastembed')
    cfg = dict(HYBRID_CONFIG)
    index = open_index(kb, embedder, root / 'data' / 'v3-index', fields=cfg.pop('doc_fields'))
    return HybridRetriever(kb, index, embedder, situation_policy=situation_policy, router=router, **cfg)


def build_pipeline(root, profiles, retriever, evaluator, store, require_human=True, call_mode='per_dimension'):
    saver_dir = root / 'reports' / 'v3-reviews'
    saver_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(saver_dir / 'checkpoints.sqlite'), check_same_thread=False)
    pipeline = ReviewPipeline(profiles, retriever, evaluator, store, root, require_human=require_human,
                              checkpointer=SqliteSaver(conn), call_mode=call_mode)
    return pipeline, conn


def draft_content(profile, result) -> list[str]:
    """Placeholder (project_draft) anchors or scored items must not drive a paid evaluation."""
    bad = [f'dimension:{d.key}' for d in profile.dimensions if d.source.kind == 'project_draft']
    bad += [i.item.item_id for i in result.items if i.item.role == 'scored' and i.item.source.kind == 'project_draft']
    return bad


def plan_for(request, profile, retriever, settings, paid, run_limit, root, approval_ref=None,
             call_mode='per_dimension', prompt_version=EVALUATOR_PROMPT_VERSION, router_settings=None,
             approval=None):
    """Upper bound of calls and cost. Never calls the router: if it would run, every candidate situation's
    criteria are assumed to be included (a superset, so the bound stays an upper bound)."""
    q = build_query(request.case, request.meta, profile)
    router_call = None
    if getattr(retriever, 'situation_policy', 'metadata') == 'llm' and not q.situations:
        candidates = retriever.candidate_situations(q)
        if candidates:
            router_call = build_router_messages(q.messages, candidates)
            q = RetrievalQuery(**{**q.__dict__, 'situations': tuple(s.situation_id for s in candidates)})
    result = FilterRetriever(retriever.kb).retrieve(q) if hasattr(retriever, 'kb') else retriever.retrieve(q)
    specs = [(d, result.for_dimension(d.key, 'scored'), result.for_dimension(d.key, 'reference'))
             for d in profile.dimensions]
    if call_mode == 'single':
        messages = [build_messages_group(request.case, specs, prompt_version)]
    else:
        messages = [build_messages_v3(request.case, d, c, r, prompt_version) for d, c, r in specs]
    upper = 0.0
    if paid:
        budget = Budget(replace(settings, max_output_tokens=eval_max_tokens(call_mode, len(specs))), None)
        upper = sum(budget.estimate(m) for m in messages)
        if router_call:
            upper += Budget(replace(router_settings or settings, max_output_tokens=ROUTER_MAX_OUTPUT_TOKENS),
                            None).estimate(router_call)
        upper = round(upper, 4)
    return {'case_id': request.case.case_id, 'profile': profile.profile_id, 'call_mode': call_mode,
            'prompt_version': prompt_version, 'evaluation_calls': len(messages),
            'router_calls': 1 if router_call else 0, 'planned_calls': len(messages) + (1 if router_call else 0),
            'retrieved_items_upper_bound': len(result.items), 'cost_upper_bound_cny': upper,
            'run_limit_cny': run_limit, 'retriever': getattr(retriever, 'name', '?'),
            'situation_policy': getattr(retriever, 'situation_policy', 'metadata'),
            'metadata_situations': list(request.meta.situations),
            'ledger': ledger_status(root) if paid else None, 'data_source': request.meta.data_source,
            'data_approval_ref': approval_ref, 'data_approval': approval, 'draft_content': draft_content(profile, result),
            'data_flow': 'conversation + answer + rubric texts are sent to the judge provider (the router call sends '
                         'the conversation and situation descriptions only); no physician labels.'}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--case', type=Path)
    p.add_argument('--profile', type=Path, action='append')
    p.add_argument('--kb', type=Path, default=None, help='KB directory (or a single items .jsonl file).')
    p.add_argument('--retriever', choices=['hybrid', 'filter'], default='hybrid')
    p.add_argument('--situation-policy', choices=['metadata', 'llm', 'retrieved'], default='metadata',
                   help='llm = an LLM router picks situations when the metadata names none (one extra call); '
                        'retrieved = EXPERIMENTAL: apply the hybrid-ranked situation (dev-B top-1 ~0.12).')
    p.add_argument('--call-mode', choices=['per_dimension', 'single'], default='per_dimension',
                   help='single = one call scores all dimensions (cheaper; quality to be measured).')
    p.add_argument('--prompt-version', choices=list(PROMPT_VERSIONS), default=EVALUATOR_PROMPT_VERSION)
    p.add_argument('--embedder', choices=['fastembed', 'hash'], default='fastembed',
                   help='hash = deterministic offline test embedder (not semantic).')
    p.add_argument('--judge', default='mock')
    p.add_argument('--authorized', action='store_true')
    p.add_argument('--run-limit', type=float, default=1.0)
    p.add_argument('--no-human', action='store_true', help='Batch mode: do not pause for human review.')
    p.add_argument('--job-id', default=None)
    p.add_argument('--resume', default=None, help='Job id waiting for human review.')
    p.add_argument('--decisions', type=Path, default=None)
    p.add_argument('--data-approval-ref', default=None,
                   help='Required for a paid run on non-public data: '
                        '"approver=<name>;date=YYYY-MM-DD;provider=deepseek;scope=<data and use covered>".')
    p.add_argument('--allow-draft-rubrics', action='store_true',
                   help='Allow a paid run whose anchors/scored items are still project_draft placeholders.')
    args = p.parse_args(argv)
    root = args.root
    profiles = {pr.profile_id: pr for pr in (load_profile(root / f if not Path(f).is_absolute() else Path(f))
                                             for f in (args.profile or DEFAULT_PROFILES))}
    kb = load_kb(args.kb if args.kb and args.kb.is_absolute() else root / (args.kb or DEFAULT_KB))
    store = Store(root)
    n_dims = max(len(pr.dimensions) for pr in profiles.values())
    backend, paid, settings = make_backend(args.judge, root, store, args.run_limit,
                                           eval_max_tokens(args.call_mode, n_dims))
    router_backend = router = None
    if args.situation_policy == 'llm':
        router_backend, _, router_settings = make_backend(args.judge, root, store, args.run_limit,
                                                          ROUTER_MAX_OUTPUT_TOKENS, mock_router_responder)
    else:
        router_settings = None
    evaluator = DimensionEvaluator(backend, store, prompt_version=args.prompt_version)
    if router_backend is not None:
        router = SituationRouter(router_backend, store, stop=evaluator.stop)   # one stop event for the run
    retriever = make_retriever(kb, root, args.retriever, args.embedder, args.situation_policy, router)
    conn = None
    try:
        if args.resume:
            if not args.decisions:
                raise SystemExit('--resume needs --decisions <json file>.')
            pipeline, conn = build_pipeline(root, profiles, retriever, evaluator, store, require_human=not args.no_human,
                                            call_mode=args.call_mode)
            out = pipeline.resume(args.resume, json.loads(args.decisions.read_text(encoding='utf-8')))
        else:
            if not args.case:
                raise SystemExit('Give --case <file> (or --resume).')
            path = args.case if args.case.is_absolute() else root / args.case
            request = ReviewRequest.model_validate(json.loads(path.read_text(encoding='utf-8')))
            if request.profile_id not in profiles:
                raise SystemExit(f'Profile {request.profile_id!r} not loaded.')
            try:
                approval, approval_error = check_run(request.meta.data_source, args.data_approval_ref,
                                                     'deepseek' if paid else 'mock', paid), None
            except ApprovalError as error:
                approval, approval_error = {'error': str(error)}, str(error)
            plan = plan_for(request, profiles[request.profile_id], retriever, settings, paid, args.run_limit,
                            root, args.data_approval_ref, args.call_mode, args.prompt_version, router_settings,
                            approval=approval)
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            if paid and not args.authorized:
                print('Plan only; no call made and nothing written. Re-run with --authorized after review.')
                return 0
            if approval_error:
                print(f'Data source {request.meta.data_source!r}: {approval_error} No call made. '
                      '(--data-approval-ref needs approver, date, provider and scope.)')
                return 2
            if paid and plan['draft_content'] and not args.allow_draft_rubrics:
                print('Placeholder (project_draft) anchors/items would drive a paid evaluation: '
                      + ', '.join(plan['draft_content'][:8]) + '. No call made.')
                return 2
            refusal = budget_gate(plan, paid, False)
            if refusal:
                print(refusal)
                return 2
            pipeline, conn = build_pipeline(root, profiles, retriever, evaluator, store, require_human=not args.no_human,
                                            call_mode=args.call_mode)
            job = args.job_id or 'V3-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:6]
            store.audit(job, 'v3_run_plan', plan)
            out = pipeline.start(job, request)
            out['job_id'] = job
        if out['status'] == 'AwaitingHumanReview':
            print(json.dumps({'status': out['status'], 'job_id': out.get('job_id', args.resume),
                              'review_items': out['review']['items'],
                              'next': 'write decisions JSON, then --resume <job_id> --decisions <file>'}, indent=2,
                             ensure_ascii=False))
        else:
            r = out['report']
            print(json.dumps({'status': 'Completed', 'job_id': r['job_id'],
                              'scores': {d['dimension']: [d['agent_score'], d['final_score'], d['final_status']]
                                         for d in r['dimensions']},
                              'report': f"reports/v3-reviews/{r['job_id']}/report.html"}, indent=2))
        return 0
    finally:
        if conn is not None:
            conn.close()
        for b in (backend, router_backend):
            if b is not None and hasattr(b, 'close'):
                b.close()


if __name__ == '__main__':
    raise SystemExit(main())
