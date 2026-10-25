"""Bounded local diagnostics: never store headers, keys or raw HTTP errors."""
from __future__ import annotations

import hashlib
import json
import re

from .store import timestamp


def redact(text: str, secret: str = '', limit: int = 1200) -> str:
    if secret:
        text = text.replace(secret, '[REDACTED]')
    text = re.sub(r'(?i)sk-[a-z0-9_-]{8,}', '[REDACTED_KEY]', text)
    text = re.sub(r'(?i)bearer\s+[^\s\"\x27,}]+', 'Bearer [REDACTED]', text)
    text = re.sub(r'(?i)(api[_ -]?key|password|secret|token)([\"\x27]?\s*[:=]\s*[\"\x27]?)[^\s\"\x27,}]+',
                  r'\1\2[REDACTED]', text)
    text = re.sub(r'[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[REDACTED_EMAIL]', text)
    return text[:limit]


def record_failure(settings, store, job_id, reservation, attempt, code, *, case, raw=None, details=None):
    def safe(value, depth=0):
        if depth > 5:
            return '<bounded>'
        if isinstance(value, dict):
            return {redact(str(k), settings.api_key, 80):safe(v,depth+1) for k,v in list(value.items())[:16]}
        if isinstance(value, list):
            return [safe(v,depth+1) for v in value[:16]]
        return redact(value,settings.api_key,500) if isinstance(value,str) else value
    record = {'diagnostic_version':'provider-diagnostic-v2', 'created_at':timestamp(),
              'job_id':job_id, 'reservation_id':reservation, 'attempt':attempt,
              'error_code':code, 'model':settings.model, 'details':safe(details or {})}
    if isinstance(raw, str):
        record.update(output_sha256=hashlib.sha256(raw.encode('utf-8')).hexdigest(), output_chars=len(raw))
        # Only the downloaded public benchmark gets a redacted, bounded model-output excerpt.
        # Ad-hoc/private cases retain error metadata and hashes only, not clinical text.
        if case.source == 'HealthBench meta-evaluation' and case.source_version == 'healthbench-meta-2025-05-07':
            record.update(redacted_output_excerpt=redact(raw, settings.api_key),
                          excerpt_truncated=len(raw)>1200, excerpt_policy='public-benchmark-only')
            if code in {'OUTPUT_EVIDENCE_OR_PROTOCOL_INVALID','OUTPUT_EVIDENCE_ANCHOR_INVALID'}:
                try:
                    parsed=json.loads(raw)
                    record['public_evidence_excerpts']={field:[redact(q,settings.api_key,300) for q in parsed.get(field,[])[:5]
                                      if isinstance(q,str)] for field in ('answer_evidence','context_evidence')}
                except (ValueError,TypeError,AttributeError):
                    pass
    folder = settings.root / 'data/private/provider-diagnostics' / job_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (reservation + '.json')
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    store.audit(job_id, 'provider_diagnostic', {'error_code':code, 'attempt':attempt,
                    'diagnostic_file':str(path.relative_to(settings.root)), 'details':record['details']})
    return record
