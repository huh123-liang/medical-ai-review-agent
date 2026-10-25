from __future__ import annotations

import argparse
import getpass
import json
from datetime import datetime
from pathlib import Path

from .benchmark import download_meta, examples, prepare_dataset, score_run
from .budget import Budget
from .config import Settings
from .domain import EvaluationRequest, Rubric, canonical
from .locking import RunnerLock
from .store import Store
from .workflow import Engine


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def configure(settings):
    key = getpass.getpass("DeepSeek API key (hidden, local project only): ").strip()
    if len(key) < 12 or any(c in key for c in "\r\n\"'"):
        raise ValueError("Credential is empty or malformed; nothing written.")
    file = settings.root / ".env.local"
    if file.exists() and input("Replace this project's local configuration? [y/N]: ").lower() != "y":
        print("Unchanged.")
        return
    content = (f"DEEPSEEK_API_KEY={key}\nJURY_MODEL=deepseek-flash\nJURY_INPUT_CNY_PER_MILLION=3\n"
               "JURY_OUTPUT_CNY_PER_MILLION=12\nJURY_PRICING_VALID_UNTIL=2026-10-31\nJURY_BUDGET_CNY=30\n"
               "JURY_MAX_OUTPUT_TOKENS=1200\nJURY_MAX_ATTEMPTS=1\nJURY_TIMEOUT_SECONDS=90\n")
    file.write_text(content, encoding="utf-8")
    print("Local configuration saved. No API request was sent and no key was printed.")


def batch(args, settings, store):
    if args.split == "test" and not args.allow_heldout:
        raise ValueError("Test split is held out. Freeze the experiment protocol and explicitly use --allow-heldout.")
    if args.provider == "deepseek":
        settings.validate_paid()
    rows = examples(settings.root, args.split, args.limit)
    if not rows:
        raise ValueError("No prepared examples in selected split.")
    start_cost = Budget(settings, store).total()["accounted_cny"]
    predictions, jobs = [], []
    engine = Engine(settings, store)
    for row in rows:
        spent = Budget(settings, store).total()["accounted_cny"] - start_cost
        remaining = args.max_batch_cost - spent
        if args.provider == "deepseek" and remaining <= 0:
            break
        request = EvaluationRequest(case=row["case"], rubrics=[row["rubric"]], provider=args.provider,
                      mode="oracle", max_run_cost_cny=min(args.max_run_cost, max(0.000001, remaining)), benchmark_binary=True)
        result = engine.run(request)
        report = result.get("report")
        judgment = report["judgments"][0]["judgment"] if result["status"] == "Completed" and report else None
        predictions.append({"example_id": row["example_id"], "job_id": result["id"], "status": result["status"], "judgment": judgment})
        jobs.append(result["id"])
        print(f"{len(jobs)}/{len(rows)} {result['id']} {result['status']}", flush=True)
        if result["status"] != "Completed":
            break
    # Keep intended denominator visible if a budget/network failure stops the batch.
    for row in rows[len(predictions):]:
        predictions.append({"example_id": row["example_id"], "status": "NotRun", "judgment": None})
    metrics = score_run(settings.root, predictions, args.provider)
    output = {"provider": args.provider, "model": settings.model if args.provider == "deepseek" else "OFFLINE-MOCK",
              "split": args.split, "planned_examples": len(rows), "executed_jobs": len(jobs),
              "predictions": predictions, "evaluation": metrics, "budget": Budget(settings, store).total(),
              "batch_accounted_cny": round(Budget(settings, store).total()["accounted_cny"] - start_cost, 6)}
    name = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + args.provider
    folder = settings.root / "reports" / ("benchmark-" + name)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "summary.md").write_text("# Public medical benchmark run\n\n" +
       f"Provider: **{args.provider}**\n\nValidation status: **{metrics['validation_status']}**\n\n" +
       "```json\n" + json.dumps(metrics, ensure_ascii=False, indent=2) + "\n```\n\n" +
       "Mock runs are plumbing tests, not doctor agreement. Public held-out data is not proof of absence from model pretraining.\n", encoding="utf-8")
    emit({"report": str(folder), "evaluation": metrics, "budget": output["budget"]})


def main():
    parser = argparse.ArgumentParser(description="Medical AI Jury — independent single-agent research evaluator")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "configure", "download", "prepare", "budget"):
        sub.add_parser(command)
    index = sub.add_parser("index")
    index.add_argument("--limit", type=int, default=100)
    query = sub.add_parser("query")
    query.add_argument("--top-k", type=int, default=4)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--provider", choices=("mock", "deepseek"), default="mock")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--provider", choices=("mock", "deepseek"), default="mock")
    bench.add_argument("--split", choices=("dev", "validation", "test"), default="dev")
    bench.add_argument("--limit", type=int, default=10)
    bench.add_argument("--max-run-cost", type=float, default=0.25)
    bench.add_argument("--max-batch-cost", type=float, default=0.50)
    bench.add_argument("--allow-heldout", action="store_true")
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=4205)
    args = parser.parse_args()
    settings = Settings.load()
    settings.prepare()
    if args.command == "configure":
        configure(settings)
        return
    if args.command == "download":
        emit(download_meta(settings.root))
        return
    if args.command == "prepare":
        emit(prepare_dataset(settings.root))
        return
    store = Store(settings.root)
    if args.command == "init":
        store.admin_token()
        emit({"root": str(settings.root), "clinical_validation": "NOT_VALIDATED", "budget": Budget(settings, store).total()})
        return
    if args.command == "budget":
        emit(Budget(settings, store).total())
        return
    with RunnerLock(settings.root):
        if args.command == "index":
            from .knowledge import Knowledge
            rows = examples(settings.root, "dev", args.limit)
            rubrics = {row["rubric"]["rubric_id"]: Rubric.model_validate(row["rubric"]) for row in rows}
            knowledge = Knowledge(settings, store)
            knowledge.upsert(list(rubrics.values()), allow_case_reference=True)
            emit({"indexed": len(rubrics), "embedding_model": settings.embedding_model, "source": "HealthBench public DEVELOPMENT case-bound rubrics",
                  "claim": "Known-case retrieval plumbing only. No unseen-case rubric-selection or clinical validation claim."})
        elif args.command == "query":
            from .domain import CaseInput
            from .knowledge import Knowledge
            row = examples(settings.root, "dev", 1)[0]
            selected, info = Knowledge(settings, store).retrieve(CaseInput.model_validate(row["case"]), args.top_k)
            emit({"rubric_ids": [r.rubric_id for r in selected], "retrieval": info})
        else:
            store.recover_interrupted()
            if args.command == "benchmark":
                if not 0 < args.max_batch_cost <= 30 or not 0 < args.max_run_cost <= 30:
                    raise ValueError("Batch/run cost bound must be in (0,30].")
                batch(args, settings, store)
            elif args.command == "evaluate":
                body = json.loads(args.input.read_text(encoding="utf-8-sig"))
                body["provider"] = args.provider
                request = EvaluationRequest.model_validate(body)
                emit(Engine(settings, store).run(request))
            elif args.command == "serve":
                import uvicorn
                from .api import create_app
                print("Local API: http://127.0.0.1:" + str(args.port))
                print("Admin token is in .runtime/admin-token (not printed). No browser UI; no paid calls on startup.")
                uvicorn.run(create_app(settings, store), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        # Keep failures actionable without leaking an API response or credentials.
        print(f"Stopped: {error}")
        raise SystemExit(1)
