"""Pre-registration freeze: hash everything that defines the evaluated system and its samples.

A frozen run refuses to start if any hashed artifact changed, and a split/kind can run once only.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

from ..domain import digest

CODE_GLOBS = ('src/medical_jury/**/*.py', 'scripts/run_validation_once.py', 'scripts/freeze_protocol.py')


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk.replace(b'\r\n', b'\n'))  # CRLF-insensitive (Windows checkouts)
    return h.hexdigest()


def code_tree_sha(root: Path) -> str:
    files = sorted({p for g in CODE_GLOBS for p in root.glob(g) if p.is_file() and '__pycache__' not in p.parts})
    return digest([[str(p.relative_to(root).as_posix()), file_sha(p)] for p in files])


def git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def previously_used_groups(root: Path) -> set[str]:
    """Groups already seen by any earlier run (legacy DB jobs/rubrics, iteration-15 blind packs)."""
    used: set[str] = set()
    db = root / 'data/jury.sqlite'
    if db.exists():
        with sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True) as con:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'jobs' in tables:
                for (raw,) in con.execute('SELECT request_json FROM jobs'):
                    try:
                        used.add(json.loads(raw)['case']['group_id'])
                    except (ValueError, KeyError, TypeError):
                        pass
            if 'rubrics' in tables:
                used.update(r[0] for r in con.execute('SELECT DISTINCT group_id FROM rubrics') if r[0])
    for folder in sorted((root / 'data/private').glob('iteration-*')):
        for f in folder.rglob('*.json*'):
            try:
                text = f.read_text(encoding='utf-8')
            except (OSError, UnicodeDecodeError):
                continue
            for token in text.replace('"', ' ').replace(',', ' ').split():
                if token.startswith('HBG-'):
                    used.add(token.strip('{}[]:'))
    return used


def artifact_hashes(root: Path, artifacts: dict[str, str]) -> dict[str, str]:
    out = {}
    for name, rel in artifacts.items():
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(f'Frozen artifact missing: {rel}')
        out[name] = file_sha(path)
    return out


def seal(freeze: dict) -> dict:
    body = {k: v for k, v in freeze.items() if k != 'freeze_sha256'}
    return {**body, 'freeze_sha256': digest(body)}


def verify(root: Path, freeze: dict) -> list[str]:
    """Return a list of problems (empty = intact)."""
    problems = []
    body = {k: v for k, v in freeze.items() if k != 'freeze_sha256'}
    if digest(body) != freeze.get('freeze_sha256'):
        problems.append('freeze.json content does not match its own hash')
    try:
        current = artifact_hashes(root, freeze['artifacts'])
    except FileNotFoundError as error:
        return problems + [str(error)]
    for name, value in freeze['artifact_sha256'].items():
        if current.get(name) != value:
            problems.append(f'artifact changed: {name}')
    if code_tree_sha(root) != freeze['code_tree_sha256']:
        problems.append('code tree changed since freeze')
    return problems
