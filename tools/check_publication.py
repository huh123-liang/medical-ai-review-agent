"""Check the public source package; never print potential secret values.

This is a preventive check, not a guarantee against every possible disclosure.
When in Git, inspect tracked AND nonignored new files. Runtime folders may exist
locally, but are never allowed in the Git index.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '.venv', '.cache', '.pytest_cache', '__pycache__', '.runtime', 'data', 'reports',
            '.ruff_cache', 'node_modules', 'github-release'}
PATTERNS = {
    'provider_key': re.compile(r'(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}'),
    'github_token': re.compile(r'(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'assigned_credential': re.compile(r'(?im)^\s*(?:DEEPSEEK_API_KEY|OPENAI_API_KEY|ADMIN_TOKEN|GITHUB_TOKEN)\s*=\s*[\x22\x27]?[A-Za-z0-9_-]{16,}'),
}
FORBIDDEN_SUFFIXES = {'.sqlite', '.sqlite3', '.db', '.db-wal', '.db-shm', '.zip', '.gz', '.pdf', '.jsonl', '.csv', '.parquet', '.log'}


def files() -> list[Path]:
    if (ROOT / '.git').exists():
        result = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                                check=True, capture_output=True)
        return sorted({ROOT / p.decode('utf-8') for p in result.stdout.split(b'\0') if p})
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(x in EXCLUDED for x in p.relative_to(ROOT).parts))


def main() -> int:
    issues = []
    candidates = files()
    for path in candidates:
        rel = path.relative_to(ROOT)
        if not path.exists():
            issues.append(f'{rel}: tracked file missing'); continue
        if any(x in EXCLUDED for x in rel.parts) or (path.name.startswith('.env') and path.name != '.env.example'):
            issues.append(f'{rel}: forbidden runtime/private file')
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(f'{rel}: data/archive/log file not in public source policy')
        if path.stat().st_size > 5_000_000:
            issues.append(f'{rel}: oversized public source file')
        try:
            content = path.read_text(encoding='utf-8-sig')
        except UnicodeError:
            issues.append(f'{rel}: unreviewed binary file'); continue
        for kind, pattern in PATTERNS.items():
            for match in pattern.finditer(content):
                line = content.count('\n', 0, match.start()) + 1
                issues.append(f'{rel}:{line}: possible {kind} (value suppressed)')
    # Validate current maintained documentation links, not archived runtime links.
    maintained = [ROOT / 'README.md', ROOT / 'SECURITY.md', ROOT / 'LICENSE-NOTICE.md']
    maintained += sorted((ROOT / 'docs/project').glob('*.md'))
    for path in maintained:
        if not path.is_file():
            issues.append(f'{path.name}: missing documentation'); continue
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', target) or target.startswith('#'):
                continue
            target = target.split('#', 1)[0]
            if target and not (path.parent / target).exists():
                issues.append(f'{path.relative_to(ROOT)}: missing linked file {target}')
    summary_path = ROOT / 'benchmarks/summaries/latest.json'
    if not summary_path.is_file():
        issues.append('Missing anonymous aggregate benchmark summary')
    else:
        report = json.loads(summary_path.read_text(encoding='utf-8'))
        base, candidate = report['arms']['A_current_scoped'], report['arms']['B_candidate_zero']
        checks = [report['n_groups'] == 14, report['n_original_physician_comparisons'] == 29,
                  base['primary_metrics'] is None, base['decided_cases'] == 13,
                  base['conditional_decided_metrics']['n'] == 27,
                  abs(base['conditional_decided_metrics']['agreement'] - 13 / 27) < 1e-12,
                  candidate['primary_metrics'] is None, candidate['decided_cases'] == 0,
                  report['candidate_diagnostics']['attempted'] == 3,
                  report['candidate_diagnostics']['not_run'] == 11]
        if not all(checks):
            issues.append('Aggregate benchmark assertions failed')
        forbidden_fields = {'binary_labels', 'physician_id', 'patient_id', 'context', 'candidate_output', 'raw_output', 'cases'}
        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in forbidden_fields:
                        issues.append(f'Aggregate summary contains forbidden field {key}')
                    walk(item)
            elif isinstance(value, list):
                for item in value: walk(item)
        walk(report)
    result = {'checked_files': len(candidates), 'issues': issues, 'passed': not issues,
              'scope': 'Public source, maintained doc links, credential patterns, aggregate-only report'}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if issues else 0


if __name__ == '__main__':
    raise SystemExit(main())
