"""Minimal client for a local Ollama server (standard library only). Local calls cost nothing and send nothing
off the machine; they are never routed through the paid budget ledger."""
from __future__ import annotations

import json
import os
import urllib.request

DEFAULT_HOST = 'http://localhost:11434'


def _host(host: str | None) -> str:
    host = (host or os.environ.get('OLLAMA_API_BASE') or DEFAULT_HOST).rstrip('/')
    if not host.startswith(('http://localhost', 'http://127.0.0.1')):
        raise ValueError('the testbed writer must be a local model (localhost only)')
    return host


def get(path: str, host: str | None = None, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(_host(host) + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def chat(model: str, messages: list[dict], seed: int, temperature: float = 0.3, num_predict: int = 900,
         num_ctx: int = 4096, timeout: float = 900.0, host: str | None = None) -> dict:
    host = _host(host)
    body = json.dumps({'model': model, 'messages': messages, 'stream': False, 'think': False,
                       'options': {'seed': seed, 'temperature': temperature, 'num_predict': num_predict,
                                   'num_ctx': num_ctx}}).encode()
    req = urllib.request.Request(host + '/api/chat', data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode('utf-8'))
    return {'text': out['message']['content'].strip(), 'eval_count': out.get('eval_count'),
            'total_seconds': round((out.get('total_duration') or 0) / 1e9, 1),
            'load_seconds': round((out.get('load_duration') or 0) / 1e9, 1),
            'prompt_tokens': out.get('prompt_eval_count'),
            'prompt_seconds': round((out.get('prompt_eval_duration') or 0) / 1e9, 1),
            'eval_seconds': round((out.get('eval_duration') or 0) / 1e9, 1)}
