import pytest

from medical_jury.judge_v2 import DEFAULT_V2_CEILING_CNY, LedgerBudget, V2_CEILING_AUTHORIZATION, v2_ceiling_from_env


# Owner authorization 2026-10-26 14:04: cumulative ceiling CNY 200 on the shared ledger (was CNY 100, before that 30).
def test_default_cumulative_budget_is_200(tmp_path, monkeypatch):
    monkeypatch.delenv('JURY_V2_BUDGET_CNY', raising=False)
    assert DEFAULT_V2_CEILING_CNY == 200
    assert v2_ceiling_from_env(tmp_path) == 200
    assert 'cumulative CNY 200' in V2_CEILING_AUTHORIZATION


@pytest.mark.parametrize('value', ['200', '100', '30', '5', '0.50'])
def test_budget_can_only_be_lowered(tmp_path, monkeypatch, value):
    monkeypatch.setenv('JURY_V2_BUDGET_CNY', value)
    assert v2_ceiling_from_env(tmp_path) == float(value)


@pytest.mark.parametrize('value', ['500', '200.01', '0', '-1', 'nan', 'inf'])
def test_budget_cannot_be_raised_or_invalid(tmp_path, monkeypatch, value):
    monkeypatch.setenv('JURY_V2_BUDGET_CNY', value)
    with pytest.raises(ValueError):
        v2_ceiling_from_env(tmp_path)


@pytest.mark.parametrize('ceiling', [200.01, 500, 0, -1, float('nan'), float('inf')])
def test_ledger_budget_rejects_ceiling_outside_authorization(tmp_path, ceiling):
    from medical_jury.config import Settings
    with pytest.raises(ValueError):
        LedgerBudget(Settings(root=tmp_path), None, ceiling)
