"""Plan phase 3: build the chronic-disease known-answer testbed (PROTOCOL-V3-TESTBED). No paid call anywhere.

    # 1. deterministic cases (patients, reference plans, one injected error each) - instant, no model
    python scripts/build_testbed_v1.py cases
    # 2. let a LOCAL model write each plan up as prose, then verify the prose; resumable, run it again to continue
    python scripts/build_testbed_v1.py render --style mas_transcript --model qwen3:30b-a3b-instruct-2507-q4_K_M
    python scripts/build_testbed_v1.py render --style clinic_letter  --model gemma3:4b --split test
    # 3. how many cases were realised, by error type
    python scripts/build_testbed_v1.py summary
    # diagnose the local model (server up? model loaded? GPU/CPU split? speed on one real case)
    python scripts/build_testbed_v1.py check --model qwen3:30b-a3b-instruct-2507-q4_K_M

Everything stays on this computer: synthetic patients only, a localhost model only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from medical_jury.testbed import TESTBED_VERSION, local_model  # noqa: E402
from medical_jury.testbed.build import (STYLES, WRITER_PROMPT_VERSION, build_cases,  # noqa: E402
                                        patient_section_ok, writer_messages)
from medical_jury.testbed.errors import verify  # noqa: E402

DEFAULT_PATIENTS = 150
DEFAULT_SEED = 20261112
MAX_ATTEMPTS = 3


def _dir(root: Path) -> Path:
    return root / 'data' / 'testbed' / TESTBED_VERSION


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def stage_cases(args, root: Path) -> int:
    cases, manifest = build_cases(args.patients, args.seed)
    out = _dir(root)
    if (out / 'manifest.json').exists():
        old = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
        same = old['cases_sha256'] == manifest['cases_sha256']
        print(f'{out} already exists; ' + ('it matches this build.' if same else
                                           'it was built with other settings and is NOT overwritten.'))
        return 0 if same else 2
    out.mkdir(parents=True)
    with (out / 'cases.jsonl').open('x', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    with (out / 'manifest.json').open('x', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print(f'Written: {out}')
    return 0


def stage_check(args, root: Path) -> int:
    try:
        print('server version:', local_model.get('/api/version')['version'])
    except Exception as exc:
        print(f'The local Ollama server does not answer on localhost:11434 ({type(exc).__name__}: {exc}).')
        print('Start it (Start menu -> Ollama, or: E:\\Ollama\\ollama.exe serve) and run this check again.')
        return 2
    names = [m['name'] for m in local_model.get('/api/tags').get('models', [])]
    print('installed models:', ', '.join(names) or '(none)')
    if args.model not in names:
        print(f'{args.model} is not installed on this server (is OLLAMA_MODELS pointing at E:\\ollama-models?).')
        return 2
    def loaded():
        for m in local_model.get('/api/ps').get('models', []):
            vram, size = m.get('size_vram', 0), m.get('size', 0) or 1
            print(f"  loaded: {m['name']}  size {size / 1e9:.1f} GB, on GPU {vram / 1e9:.1f} GB ({100 * vram / size:.0f}%)")
    print('1) tiny request (loads the model; may take a minute or two the first time)...', flush=True)
    t = time.time()
    out = local_model.chat(args.model, [{'role': 'user', 'content': 'Reply with one word: ready'}], seed=1,
                           num_predict=10, timeout=args.timeout)
    print(f"   answered in {time.time() - t:.0f} s (model load {out['load_seconds']} s): {out['text'][:40]!r}")
    loaded()
    cases = _read_jsonl(_dir(root) / 'cases.jsonl') if (_dir(root) / 'cases.jsonl').exists() else build_cases(2, args.seed)[0]
    print('2) one real write-up, same settings as render...', flush=True)
    t = time.time()
    out = local_model.chat(args.model, writer_messages(cases[0], args.style), seed=1, timeout=args.timeout)
    rate = (out['eval_count'] or 0) / max(out['eval_seconds'], 0.1)
    print(f"   {time.time() - t:.0f} s total: prompt {out['prompt_tokens']} tokens in {out['prompt_seconds']} s, "
          f"wrote {out['eval_count']} tokens at {rate:.1f} tokens/s")
    print(f"   verify: {verify(cases[0]['plan'], cases[0]['gold'], out['text'])}")
    print('--- first 600 characters ---')
    print(out['text'][:600])
    return 0


def render_one(case: dict, style: str, model: str, chat=local_model.chat, timeout: float = 600.0) -> dict:
    """Up to MAX_ATTEMPTS writes; keep the first whose prose verifiably realises the plan (and the error)."""
    messages = writer_messages(case, style)
    rec = {'case_id': case['case_id'], 'style': style, 'model': model, 'writer_prompt': WRITER_PROMPT_VERSION,
           'attempts': [], 'status': 'unrealised'}
    for attempt in range(MAX_ATTEMPTS):
        seed = int(hashlib.sha256(f"{case['case_id']}:{style}:{attempt}".encode()).hexdigest()[:8], 16)
        try:
            out = chat(model, messages, seed=seed, timeout=timeout)
        except Exception as exc:                      # server down, timeout ...: stop the run, do not burn attempts
            rec['status'], rec['error'] = 'writer_error', f'{type(exc).__name__}: {exc}'[:300]
            return rec
        check = verify(case['plan'], case['gold'], out['text'])
        if not patient_section_ok(out['text'], case['plan'], style):
            check = {'ok': False, 'problems': check['problems'] + ['patient-facing section missing or not carrying the plan']}
        rec['attempts'].append({'seed': seed, 'problems': check['problems'], 'seconds': out.get('total_seconds'),
                                'tokens': out.get('eval_count')})
        if check['ok']:
            rec.update(status='ok', text=out['text'], text_sha256=hashlib.sha256(out['text'].encode()).hexdigest())
            return rec
        rec['last_text'] = out['text']
    return rec


def stage_render(args, root: Path, chat=local_model.chat) -> int:
    out = _dir(root)
    cases = _read_jsonl(out / 'cases.jsonl')
    path = out / f'renders-{args.style}-{WRITER_PROMPT_VERSION}.jsonl'
    done = {r['case_id'] for r in _read_jsonl(path)} if path.exists() else set()
    split = getattr(args, 'split', 'all')
    todo = [c for c in cases if c['case_id'] not in done and split in ('all', c['split'])][:args.limit]
    print(f'{len(done)} already written, {len(todo)} to write with {args.model} (style {args.style}).')
    started = time.time()
    with path.open('a', encoding='utf-8') as f:
        for n, case in enumerate(todo, 1):
            if n == 1:
                print(f"  writing {case['case_id']} (the first one also loads the model)...", flush=True)
            rec = render_one(case, args.style, args.model, chat, getattr(args, 'timeout', 600.0))
            if rec['status'] == 'writer_error':
                print(f"Stopped: the local model did not answer ({rec['error']}). Is Ollama running? "
                      'Run the same command again to continue.')
                return 2
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            f.flush()
            if n <= 3 or n % 10 == 0 or n == len(todo):
                rate = (time.time() - started) / n
                print(f'  {n}/{len(todo)}  last={rec["status"]}  about {rate * (len(todo) - n) / 60:.0f} min left', flush=True)
    return stage_summary(args, root)


def summarise(cases: list[dict], renders: list[dict]) -> dict:
    gold = {c['case_id']: 'clean' if not c['gold'] else 'two errors' if isinstance(c['gold'], list)
            else c['gold']['error_id'] for c in cases}
    rows: dict[str, dict] = {}
    for r in renders:
        row = rows.setdefault(gold[r['case_id']], {'ok': 0, 'unrealised': 0})
        row['ok' if r['status'] == 'ok' else 'unrealised'] += 1
    ok = sum(r['status'] == 'ok' for r in renders)
    return {'written': len(renders), 'of': len(cases), 'ok': ok, 'unrealised': len(renders) - ok,
            'first_attempt_ok': sum(r['status'] == 'ok' and len(r['attempts']) == 1 for r in renders),
            'by_error': dict(sorted(rows.items()))}


def stage_summary(args, root: Path) -> int:
    out = _dir(root)
    cases = _read_jsonl(out / 'cases.jsonl')
    for path in sorted(out.glob('renders-*.jsonl')):
        print(path.name)
        print(json.dumps(summarise(cases, _read_jsonl(path)), indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('cases', 'render', 'summary', 'check'))
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--patients', type=int, default=DEFAULT_PATIENTS)
    p.add_argument('--seed', type=int, default=DEFAULT_SEED)
    p.add_argument('--style', choices=sorted(STYLES), default='mas_transcript')
    p.add_argument('--model', default='qwen3:30b-a3b-instruct-2507-q4_K_M')
    p.add_argument('--limit', type=int, default=None, help='write at most this many cases in this run')
    p.add_argument('--split', choices=('all', 'train', 'test'), default='all', help='write only this split')
    p.add_argument('--timeout', type=float, default=600.0, help='seconds to wait for one write-up')
    args = p.parse_args(argv)
    return {'cases': stage_cases, 'render': stage_render, 'summary': stage_summary,
            'check': stage_check}[args.stage](args, args.root)


if __name__ == '__main__':
    raise SystemExit(main())
