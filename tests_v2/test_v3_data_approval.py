"""Plan phase 0.1: structured data-use approval record for paid runs on non-public data."""
from __future__ import annotations

import pytest

from medical_jury.v3.data_approval import ApprovalError, check_run, is_public, parse_approval

from test_v3_pipeline import STEP1, _paid_root
from v2_fixtures import load_script

OK = 'approver=A*STAR WP3 PI;date=2026-10-26;provider=DeepSeek;scope=MIMIC-IV chronic-disease cases, research use'


def test_valid_record_is_normalised():
    rec = parse_approval(OK)
    assert rec == {'approver': 'A*STAR WP3 PI', 'date': '2026-10-26', 'provider': 'deepseek',
                   'scope': 'MIMIC-IV chronic-disease cases, research use'}


@pytest.mark.parametrize('ref, message', [
    (None, 'No data-use approval'),
    ('see email', 'not key=value'),
    ('approver=X;date=2026-10-26;provider=deepseek', 'missing scope'),
    (OK + ';ticket=1', 'Unknown approval field'),
    (OK + ';approver=Y', 'given twice'),
    (OK.replace('2026-10-26', '26/10/2026'), 'YYYY-MM-DD'),
    (OK.replace('2026-10-26', '2999-01-01'), 'future'),
    ('approver=X;date=2026-10-26;provider=deepseek;scope=mimic', 'too short'),
])
def test_incomplete_or_malformed_records_are_refused(ref, message):
    with pytest.raises(ApprovalError, match=message):
        parse_approval(ref)


def test_check_run_rules():
    assert is_public('HealthBench') and not is_public('mimic-iv-derived')
    assert check_run('healthbench', None, 'deepseek', paid=True) is None          # public data needs nothing
    assert check_run('mimic-iv', None, 'mock', paid=False) is None                # offline runs send nothing
    assert check_run('mimic-iv', OK, 'deepseek', paid=True)['provider'] == 'deepseek'
    with pytest.raises(ApprovalError, match='No data-use approval'):
        check_run('mimic-iv', None, 'deepseek', paid=True)
    with pytest.raises(ApprovalError, match='covers provider'):
        check_run('mimic-iv', OK.replace('DeepSeek', 'azure-openai'), 'deepseek', paid=True)


def test_cli_refuses_an_unstructured_reference(tmp_path, monkeypatch, capsys):
    root = _paid_root(tmp_path, monkeypatch, data_source='mimic-iv-derived')
    cli = load_script('run_review_v3')
    assert cli.main(['--root', str(root), *STEP1, '--case', 'case.json', '--judge', 'deepseek-pro', '--authorized',
                     '--allow-draft-rubrics', '--data-approval-ref', 'email from PI']) == 2
    out = capsys.readouterr().out
    assert 'approver, date, provider and scope' in out and not (root / 'reports/v3-reviews').exists()


def test_cli_with_a_valid_record_passes_the_approval_gate_and_records_it(tmp_path, monkeypatch, capsys):
    root = _paid_root(tmp_path, monkeypatch, data_source='mimic-iv-derived')
    cli = load_script('run_review_v3')
    code = cli.main(['--root', str(root), *STEP1, '--case', 'case.json', '--judge', 'deepseek-pro', '--authorized',
                     '--allow-draft-rubrics', '--data-approval-ref', OK, '--run-limit', '0.000001'])
    out = capsys.readouterr().out
    assert code == 2 and 'run-limit' in out                 # stopped later, by the budget gate - not by approval
    assert '"approver": "A*STAR WP3 PI"' in out and '"provider": "deepseek"' in out
