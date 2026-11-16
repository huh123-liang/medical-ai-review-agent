"""testbed-verify-2: the stricter post-hoc audit of write-ups (offline)."""
from medical_jury.testbed.audit import audit, invented_doses, urgent_ok
from medical_jury.testbed.build import build_cases, plan_text


def test_reference_wording_passes_and_weakened_wording_fails():
    ok = ['Blood pressure is severely raised: arrange same-day urgent medical assessment, and emergency care if there '
          'is chest pain.',
          'Your blood pressure is very high. Given the severity, urgent medical assessment is required on the same day.',
          'Immediate urgent medical assessment is required due to severely elevated blood pressure.']
    bad = ['Your blood pressure is dangerously high, so if you have chest pain, go to the emergency room right away.',
           'If your blood pressure remains severely raised, seek urgent medical assessment.',
           'An urgent medical assessment will be arranged due to the severity of your blood pressure.']
    assert all(urgent_ok(t, 'urgent_bp') for t in ok)
    assert not any(urgent_ok(t, 'urgent_bp') for t in bad)


def test_invented_dose():
    plan = {'medications': [{'drug': 'insulin glargine', 'dose_mg': None}], 'elements': {}}
    assert invented_doses(plan, 'Insulin glargine 100 mg once daily') == ['insulin glargine']
    assert invented_doses(plan, 'insulin glargine, starting dose set by the team') == []


def test_every_reference_plan_passes_the_audit():
    for c in build_cases(150, 20261112)[0]:
        if not c['gold']:
            assert audit(c['plan'], plan_text(c['plan'])) == [], c['case_id']
