"""Versioned wrapper; reuses the identical runner/budget protocol with a new candidate."""
import json
import hashlib
import sys
import run_iteration5 as runner

runner.OUT=runner.ROOT/'reports/iteration-5/candidate-v2'
runner.PROFILES={'A':'v2-strict','B':'complex-boundary-v2'}

if __name__=='__main__':
    runner.OUT.mkdir(exist_ok=True)
    if sys.argv[-1:] == ['new'] and sys.argv[1] in {'prepare','estimate','run'}:
        frozen=json.loads((runner.OUT/'calibration/preflight.json').read_text())
        for name,sha in frozen['code_sha256'].items():
            assert hashlib.sha256((runner.ROOT/'src/medical_jury'/name).read_bytes()).hexdigest()==sha, 'Candidate changed after old calibration; do not claim a frozen new-sample test.'
    gate=runner.OUT/'gate-1.json'
    if not gate.exists():
        parent=json.loads((runner.ROOT/'reports/iteration-5/gate-1.json').read_text())
        parent.update(revision_note='docs/ITERATION-5-REVISION-2.md',
            parent_boundary_gate='reports/iteration-5/gate-1.json',
            next='One further old-case regression; freeze before seeing new examples.')
        gate.write_text(json.dumps(parent,indent=2),encoding='utf-8')
    runner.main()
