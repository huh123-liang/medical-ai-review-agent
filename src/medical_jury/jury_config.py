"""Jury v2 panel configuration: which judges sit on the jury and how each is reached.

File: <root>/jury-v2.json (local; contains NO secrets — keys are read from env var names).
See examples/jury-v2.example.json. A heterogeneous panel (different model families) is the point:
same-family judges share blind spots (Verga et al. 2024, 'Replacing Judges with Juries').
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from .config import Settings, load_local_env
from .judge_v2 import DeepSeekBackend, OpenAICompatibleBackend, ScriptedBackend, mock_responder
from .store import Store

KINDS = {'deepseek', 'openai_compatible', 'mock'}


def load_panel(root: Path, path: Path | None = None) -> list[dict]:
    file = path or root / 'jury-v2.json'
    if not file.exists():
        raise FileNotFoundError(f'{file} missing; copy examples/jury-v2.example.json and edit it.')
    panel = json.loads(file.read_text(encoding='utf-8'))['judges']
    names = [j['name'] for j in panel]
    if len(names) != len(set(names)):
        raise ValueError('Judge names must be unique.')
    for j in panel:
        if j.get('kind') not in KINDS:
            raise ValueError(f"Unknown judge kind {j.get('kind')!r}.")
        if 'api_key' in j:
            raise ValueError('Never put keys in jury-v2.json; use api_key_env.')
    return panel


def build_backend(spec: dict, settings: Settings, store: Store | None, run_limit_cny: float):
    kind = spec['kind']
    if kind == 'mock':
        return ScriptedBackend(mock_responder, name=spec['name'])
    if kind == 'deepseek':
        s = settings
        if spec.get('model'):
            s = replace(s, model=spec['model'])
        if spec.get('price_in') is not None:
            s = replace(s, input_rate=float(spec['price_in']), output_rate=float(spec['price_out']))
        backend = DeepSeekBackend(s, store, run_limit_cny)
        backend.name = spec['name']
        return backend
    env = load_local_env(settings.root)
    key = env.get(spec['api_key_env'], '') if spec.get('api_key_env') else ''
    allowed = {h.strip() for h in env.get('JURY_V2_ALLOWED_HOSTS', '').split(',') if h.strip()}
    return OpenAICompatibleBackend(spec['name'], spec['base_url'], spec['model'], settings=settings, store=store,
                                   run_limit_cny=run_limit_cny, api_key=key,
                                   price_in=float(spec.get('price_in', 0)), price_out=float(spec.get('price_out', 0)),
                                   allowed_hosts=allowed, json_mode=spec.get('json_mode', True))
