"""Local data loaders for Jury v2 runs. Inputs and labels are read from separate files on purpose."""
from __future__ import annotations

import json
from pathlib import Path


def load_prepared(root: Path, split: str) -> list[dict]:
    """Label-free prepared rows {example_id, case, rubric, category}."""
    if split not in {'dev', 'validation', 'test'}:
        raise ValueError('Unknown split.')
    path = root / f'data/prepared/{split}.jsonl'
    if not path.exists():
        raise FileNotFoundError(f'{path} missing: run `python -m medical_jury.cli prepare` first.')
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_rater_index(root: Path, *, with_labels: bool, splits: set[str] | None = None) -> list[dict]:
    """Rows from data/private/physician-rater-index.jsonl (built by scripts/human_ceiling.py).

    with_labels=False strips labels so samplers stay outcome-blind.
    """
    path = root / 'data/private/physician-rater-index.jsonl'
    if not path.exists():
        raise FileNotFoundError(f'{path} missing: run `python scripts/human_ceiling.py` first.')
    out = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if splits and row['split'] not in splits:
                continue
            if not with_labels:
                row.pop('labels', None)
            out.append(row)
    return out


def references_by_id(root: Path, splits: set[str]) -> dict[str, dict]:
    return {r['example_id']: r for r in load_rater_index(root, with_labels=True, splits=splits)}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')


def file_sha256_lf(path: Path) -> str:
    """SHA-256 with CRLF normalised to LF (same convention as scripts/human_ceiling.py)."""
    import hashlib
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk.replace(b'\r\n', b'\n'))
    return h.hexdigest()


def verify_rater_index(root: Path, ids=None, split: str | None = None) -> dict:
    """Read-only integrity check of the scoring index BEFORE paid calls and again BEFORE scoring.

    ok only if: the index and its .meta.json exist; the index content hash equals the meta; the meta's
    source hash equals data/public/source-manifest.json; and (if given) every planned id is present
    for ``split``. Returns the index hash so runs can record exactly which index scored them.
    """
    index = root / 'data/private/physician-rater-index.jsonl'
    meta_path = index.with_suffix('.meta.json')
    manifest_path = root / 'data/public/source-manifest.json'
    problems: list[str] = []
    result = {'ok': False, 'index_sha256': None, 'problems': problems, 'missing_ids': None}
    if not index.exists():
        problems.append('index missing (data/private/physician-rater-index.jsonl)')
        return result
    if not meta_path.exists():
        problems.append('index has no .meta.json (provenance unknown)')
        return result
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
    except ValueError:
        problems.append('index .meta.json unreadable')
        return result
    actual = file_sha256_lf(index)
    result['index_sha256'] = actual
    if meta.get('index_sha256') != actual:
        problems.append('index content hash differs from its .meta.json')
    if not manifest_path.exists():
        problems.append('data/public/source-manifest.json missing')
    else:
        source_sha = json.loads(manifest_path.read_text(encoding='utf-8')).get('sha256')
        if meta.get('source_sha256') != source_sha:
            problems.append('index was built from a different public source file')
    if ids is not None:
        have = set()
        with index.open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                if split is None or row.get('split') == split:
                    have.add(row['example_id'])
        missing = [i for i in ids if i not in have]
        result['missing_ids'] = len(missing)
        if missing:
            problems.append(f'{len(missing)} planned ids absent from the index')
    result['ok'] = not problems
    return result
