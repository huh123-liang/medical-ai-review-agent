import math
import random

import pytest

from medical_jury.evaluation import agreement as A


def approx(x, y, tol=1e-9):
    return abs(x - y) <= tol


def test_cohen_kappa_hand_computed():
    # 2x2: TP=20 TN=15 FP=5 FN=10 -> po=.7, pe=(25*30+25*20)/2500=.5, kappa=.4
    pairs = [(True, True)] * 20 + [(False, False)] * 15 + [(True, False)] * 5 + [(False, True)] * 10
    s = A.binary_summary(pairs)
    assert approx(s['agreement'], 0.7)
    assert approx(s['cohen_kappa'], 0.4)
    assert s['confusion'] == {'TP': 20, 'TN': 15, 'FP': 5, 'FN': 10}


def test_gwet_ac1_hand_computed():
    pairs = [(True, True)] * 20 + [(False, False)] * 15 + [(True, False)] * 5 + [(False, True)] * 10
    # pi_met = (25+30)/100 = .55 ; pe = 2*.55*.45/(2-1) = .495 ; AC1 = (.7-.495)/(1-.495)
    s = A.binary_summary(pairs)
    assert approx(s['gwet_ac1'], (0.7 - 0.495) / 0.505)


def test_kappa_paradox_ac1_stays_high():
    # High prevalence, high agreement: kappa collapses while AC1 does not.
    pairs = [(True, True)] * 90 + [(True, False)] * 5 + [(False, True)] * 5
    s = A.binary_summary(pairs)
    assert s['cohen_kappa'] < 0
    assert s['gwet_ac1'] > 0.85


def test_macro_f1_hand_computed():
    pairs = [(True, True)] * 20 + [(False, False)] * 15 + [(True, False)] * 5 + [(False, True)] * 10
    f_pos = 40 / 55
    f_neg = 30 / 45
    assert approx(A.macro_f1(pairs), (f_pos + f_neg) / 2)


def test_binary_requires_booleans():
    with pytest.raises(TypeError):
        A.binary_summary([(1, 0)])


def test_weighted_kappa_rejects_binary_labels():
    with pytest.raises(TypeError, match='ordinal'):
        A.weighted_kappa([True, False], [True, True])


def test_weighted_kappa_known_values():
    a = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    assert approx(A.weighted_kappa(a, a), 1.0)
    b = [2, 2, 3, 4, 5, 1, 2, 3, 4, 4]
    q = A.weighted_kappa(a, b, 'quadratic')
    linear = A.weighted_kappa(a, b, 'linear')
    assert 0.8 < q < 1 and 0.6 < linear < q


def test_krippendorff_alpha_perfect_and_missing():
    assert approx(A.krippendorff_alpha_nominal([[1, 1], [0, 0, None], [1, 1, 1]]), 1.0)
    # Classic small example: alpha computed by hand = 1 - (n-1)*Do/De
    units = [[1, 0], [1, 1], [0, 0], [1, 0]]
    # coincidences: o10=o01=2, o11=2, o00=2 -> n=8, n1=n0=4 ; alpha = 1 - 7*4/(2*16) = 0.125
    assert approx(A.krippendorff_alpha_nominal(units), 0.125)


def test_group_bootstrap_resamples_groups():
    groups = {g: [(True, True)] * 3 + [(False, False)] for g in range(20)}
    ci = A.group_bootstrap_ci(groups, lambda rows: A.binary_summary(rows)['agreement'], 200, 1)
    assert ci['lower'] == ci['upper'] == 1.0 and ci['valid_draws'] == 200


def test_permutation_test_detects_signal_and_null():
    rng = random.Random(3)
    gold = [rng.random() < 0.7 for _ in range(300)]
    good = [(g if rng.random() < 0.9 else not g, g) for g in gold]
    noise = [(rng.random() < 0.7, g) for g in gold]
    assert A.permutation_test_kappa(good, 999, 1)['p_value'] < 0.01
    assert A.permutation_test_kappa(noise, 999, 1)['p_value'] > 0.01


def test_exact_mcnemar():
    r = A.exact_mcnemar([True] * 10 + [False] * 0, [False] * 10)
    assert r['a_only_correct'] == 10 and r['p_value'] == pytest.approx(2 / 1024)


def test_constant_baselines_reported():
    pairs = [(True, False)] * 3 + [(False, False)] * 7
    out = A.constant_baselines(pairs)
    assert out['always_not_met']['agreement'] == 1.0
    assert out['always_met']['agreement'] == 0.0


def test_t_sf_against_reference_values():
    # Reference values from standard t tables.
    assert A.t_sf(2.0, 10) == pytest.approx(0.03669, abs=2e-5)
    assert A.t_sf(1.0, 5) == pytest.approx(0.18161, abs=2e-5)
    assert A.t_sf(-1.0, 5) == pytest.approx(1 - 0.18161, abs=2e-5)
    assert A.t_sf(0.0, 7) == pytest.approx(0.5, abs=1e-9)


def test_benjamini_yekutieli():
    assert A.benjamini_yekutieli([0.001, 0.2, 0.5]) == [True, False, False]
    assert A.benjamini_yekutieli([]) == []


def _alt_items(llm_mode, n_items=400, seed=0):
    rng = random.Random(seed)
    annotators = [f'h{i}' for i in range(6)]
    items = []
    for _ in range(n_items):
        truth = rng.random() < 0.7
        raters = rng.sample(annotators, 3)
        humans = {h: (truth if rng.random() < 0.85 else not truth) for h in raters}
        if llm_mode == 'copy':
            llm = humans[raters[0]]
        elif llm_mode == 'oracle':
            llm = truth
        else:
            llm = rng.random() < 0.5
        items.append({'llm': llm, 'humans': humans})
    return items


def test_alt_test_passes_for_human_like_llm_and_fails_for_noise():
    good = A.alt_test(_alt_items('oracle'), epsilon=0.2, min_items_per_annotator=30)
    assert good['eligible_annotators'] == 6 and good['passed'] is True
    bad = A.alt_test(_alt_items('random'), epsilon=0.0, min_items_per_annotator=30)
    assert bad['passed'] is False


def test_alt_test_needs_three_annotators():
    items = [{'llm': True, 'humans': {'a': True, 'b': True}} for _ in range(50)]
    r = A.alt_test(items, min_items_per_annotator=30)
    assert r['eligible_annotators'] == 2 and r['passed'] is False


def test_human_ceiling_helpers():
    sets = [[True, True, False], [True, False], [False, False, False, True]]
    pw = A.pairwise_human_pairs(sets)
    assert len(pw) == 6 + 2 + 12
    loo = A.leave_one_out_majority_pairs(sets, 3)
    # item1: drop T -> (T,F) tie skip; drop T -> tie skip; drop F -> (T,T) maj T
    # item3: drop F -> (F,F,T) F x3 ; drop T -> (F,F,F) F
    assert loo == [(False, True), (False, False), (False, False), (False, False), (True, False)]
    assert A.majority_label([True, False]) is None


def test_kappa_standard_error_planning():
    se = A.kappa_standard_error(0.85, 0.64, 150)
    assert math.isclose(se, math.sqrt(0.85 * 0.15 / 150) / 0.36)
