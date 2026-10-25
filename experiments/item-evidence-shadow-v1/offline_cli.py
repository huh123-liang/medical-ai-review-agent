"""Read supplied local files only; default disabled, no API/model entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shadow import ROOT, ShadowAdapter, ShadowStore


def local_file(value, limit):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('Input must be a bounded existing file inside this independent project.')
    return path.read_text(encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable-shadow', action='store_true', default=False)
    parser.add_argument('--primary-file')
    parser.add_argument('--snapshot-file')
    parser.add_argument('--candidate-file')
    parser.add_argument('--request-id')
    parser.add_argument('--store-root')
    parser.add_argument('--origin', choices=['synthetic_fixture', 'historical_original', 'historical_projection'],
                        default='synthetic_fixture')
    parser.add_argument('--case-reference')
    args = parser.parse_args()
    if not args.enable_shadow:
        print(json.dumps({'status': 'disabled', 'files_read': 0, 'records_written': 0}))
        return
    if not args.primary_file or not args.request_id:
        parser.error('Explicit enabled mode requires --primary-file and --request-id.')
    primary = json.loads(local_file(args.primary_file, 2_000_000))
    snapshot = json.loads(local_file(args.snapshot_file, 2_000_000)) if args.snapshot_file else None
    candidate = local_file(args.candidate_file, 96_000) if args.candidate_file else None
    adapter = ShadowAdapter(enabled=True, store=ShadowStore(args.store_root))
    print(json.dumps(adapter.observe(primary, request_id=args.request_id, snapshot=snapshot,
                                     candidate_json=candidate, origin=args.origin,
                                     case_reference=args.case_reference), ensure_ascii=False))


if __name__ == '__main__':
    main()
