from __future__ import annotations

import math

import pytest

import stats as ss

NAN = float("nan")


def approx(value, tol=1e-6):
    return pytest.approx(value, abs=tol)


# - percentiles

@pytest.mark.parametrize("q,expected", [
    (25, 25.75),
    (75, 75.25),
    (10, 10.9),
    (99, 99.01),
    (50, 50.5),
])
def test_percentile_r7(q, expected):
    assert ss.percentile(list(range(1, 101)), q) == approx(expected)


def test_percentile_empty():
    assert math.isnan(ss.percentile([], 50))


# - summarize

def test_summarize_seed_reference():
    import datastore as ds

    values = list(ds.SEED_MULTIPLIERS)
    s = ss.summarize(values)
    assert s["n"] == 60
    assert s["min"] == approx(1.00)
    assert s["max"] == approx(83.73)
    assert s["mean"] == approx(5.954, tol=5e-3)
    assert s["median"] == approx(1.935, tol=5e-3)
    assert s["sd"] == approx(15.126, tol=5e-3)


def test_summarize_single():
    s = ss.summarize([2.5])
    assert s["sd"] == 0.0
    assert s["mean"] == s["median"] == 2.5


def test_summarize_empty():
    s = ss.summarize([])
    assert s["n"] == 0
    assert math.isnan(s["mean"])


# - wilson

def test_wilson_reference_values():
    low, high = ss.wilson_ci(50, 100)
    assert low == approx(0.4038315, tol=1e-6)
    assert high == approx(0.5961685, tol=1e-6)


def test_wilson_small_sample():
    low, high = ss.wilson_ci(1, 10)
    assert low == approx(0.0178762, tol=1e-6)
    assert high == approx(0.4041500, tol=1e-6)


def test_wilson_zero_hits():
    low, high = ss.wilson_ci(0, 100)
    assert low == 0.0
    assert high > 0.0


def test_wilson_n_zero():
    low, high = ss.wilson_ci(0, 0)
    assert math.isnan(low) and math.isnan(high)


# - theory / rates

def test_theoretical_p():
    assert ss.theoretical_p(2.0) == approx(0.485)
    assert ss.theoretical_p(1.0) == 1.0
    assert ss.theoretical_p(1.5) == approx(0.6466667, tol=1e-6)
    assert ss.theoretical_p(100.0) == approx(0.0097)


def test_theoretical_rate_kinds():
    assert ss.theoretical_rate("ge", 2.0) == approx(0.485)
    assert ss.theoretical_rate("ge", 1.0) == 1.0
    assert ss.theoretical_rate("lt", 1.1) == approx(1.0 - 0.97 / 1.1)


def test_expected_value_negative():
    assert ss.expected_value(2.0, 0.485) == approx(0.485 * 2.0 - 1.0)
    assert ss.expected_value(2.0) == approx(-0.03)
    assert ss.expected_value(3.0) == approx(ss.theoretical_p(3.0) * 3.0 - 1.0)


# - count/rate

def test_count_and_rate_where():
    values = [1.0, 1.5, 2.0, 3.0]
    assert ss.count_where(values, lambda v: v < 2.0) == 2
    assert ss.rate_where(values, lambda v: v < 2.0) == approx(0.5)


# - streaks

def test_streak_info():
    values = [3.0, 1.5, 1.2, 4.0, 1.1, 1.9]
    info = ss.streak_info(values, lambda v: v < 2.0)
    assert info["longest"] == 2
    assert info["current"] == 2
    info2 = ss.streak_info(values, lambda v: v < 1.0)
    assert info2["current"] == 0
    assert info2["longest"] == 0


def test_streak_info_break_in_middle():
    values = [1.0, 1.0, 5.0, 1.0, 1.0, 1.0]
    info = ss.streak_info(values, lambda v: v < 2.0)
    assert info["longest"] == 3
    assert info["current"] == 3


def test_streak_empty():
    info = ss.streak_info([], lambda v: v < 2.0)
    assert info["current"] == 0 and info["longest"] == 0


# - gaps

def test_gaps_info():
    values = [1.0, 12.0, 1.0, 1.0, 1.0, 15.0, 1.0]
    info = ss.gaps_info(values, lambda v: v >= 10.0)
    assert info["gaps"] == [3]
    assert info["current"] == 1


def test_gaps_info_no_hit():
    info = ss.gaps_info([1.0, 1.5, 2.0], lambda v: v >= 10.0)
    assert info["gaps"] == []
    assert info["current"] is None


# - rolling

def test_rolling_median_partial_windows():
    assert ss.rolling_median([1, 2, 3, 4], 2) == [1, 1.5, 2.5, 3.5]


def test_rolling_rate():
    rates = ss.rolling_rate([1.0, 3.0, 1.0, 1.0], 2, lambda v: v < 2.0)
    assert rates == [1.0, 0.5, 0.5, 1.0]


def test_rolling_empty():
    assert ss.rolling_median([], 3) == []


# - visibility

def test_p_value_visible():
    assert not ss.p_value_visible(20)
    assert ss.p_value_visible(30)
    assert ss.p_value_visible(300)


# - verdicts

def test_verdict_insufficient_window():
    assert ss.verdict(299, 5000, 0.001) == "insufficient"
    assert ss.verdict(0, 0, None) == "insufficient"


def test_verdict_hidden_small_group():
    assert ss.verdict(400, 29, 0.001) == "hidden"
    assert ss.verdict(300, 29, 0.001) == "hidden"


def test_verdict_insufficient_takes_precedence():
    assert ss.verdict(29, 29, 0.001) == "insufficient"


def test_verdict_significant():
    assert ss.verdict(400, 400, 0.049) == "significant"


def test_verdict_not_significant():
    assert ss.verdict(400, 400, 0.05) == "not_significant"
    assert ss.verdict(400, 400, 0.9) == "not_significant"
    assert ss.verdict(400, 400, None) == "not_significant"


# - mann-whitney

def test_mw_matches_scipy_asymptotic():
    import scipy.stats as sps

    a = [1, 2, 3, 4, 5, 6, 7, 8]
    b = [11, 12, 13, 14, 15, 16, 17, 18]
    result = ss.mann_whitney_u(a, b)
    expected = sps.mannwhitneyu(a, b, alternative="two-sided",
                                method="asymptotic", use_continuity=True)
    assert result["u1"] == 0
    assert result["p"] == approx(expected.pvalue, tol=1e-12)
    assert result["r"] == approx(-1.0)


def test_mw_mid_ranks_direction_and_magnitude():
    result = ss.mann_whitney_u([1, 2, 2, 3], [2, 3, 3, 4])
    assert result["u1"] == 3
    assert result["u2"] == 13
    assert result["p"] < 0.2
    assert result["r"] == approx(-0.625)
    assert result["z"] < 0


def test_mw_p_value_is_two_sided_reasonable():
    result = ss.mann_whitney_u([1, 2, 3], [10, 11, 12])
    assert 0.0 < result["p"] < 0.1
    assert result["r"] == -1.0


def test_mw_all_tied():
    result = ss.mann_whitney_u([2, 2, 2], [2, 2, 2])
    assert result["p"] == 1.0
    assert result["r"] == 0.0


def test_mw_empty_groups():
    result = ss.mann_whitney_u([], [1, 2])
    assert math.isnan(result["p"])


# - fisher

def test_fisher_reference_values():
    assert ss.fisher_exact([[3, 1], [1, 3]])["p"] == approx(0.4857142857, tol=1e-9)
    assert ss.fisher_exact([[10, 0], [0, 10]])["p"] == approx(
        1.082508822446903e-05, tol=1e-12
    )
    assert ss.fisher_exact([[1, 1], [1, 1]])["p"] == 1.0


def test_fisher_all_zero():
    result = ss.fisher_exact([[0, 0], [0, 0]])
    assert math.isnan(result["p"])
    assert math.isnan(result["oddsratio"])


# - chi-square GOF

def test_chi_square_gof_reference():
    result = ss.chi_square_gof([16, 18, 16, 14, 12, 16], [15, 15, 15, 15, 15, 15])
    assert result["chi2"] == approx(1.4666667, tol=1e-6)
    assert result["p"] == approx(0.9168841, tol=1e-6)
    assert result["dof"] == 5


def test_chi_square_gof_totals_need_not_match():
    result = ss.chi_square_gof([10, 20], [12, 15])
    assert result["p"] == result["p"]


def test_chi_square_gof_too_few_buckets():
    result = ss.chi_square_gof([5], [5])
    assert math.isnan(result["p"])


# - chi-square ind.

def test_chi_square_independence_reference():
    result = ss.chi_square_independence([[10, 20], [20, 10]])
    assert result["chi2"] == approx(6.6666667, tol=1e-6)
    assert result["p"] == approx(0.0098232745, tol=1e-9)
    assert result["cramers_v"] > 0


def test_chi_square_independence_zero_expected():
    result = ss.chi_square_independence([[1, 0], [0, 1]])
    assert math.isnan(result["chi2"]) or result["p"] == result["p"]


# - adjust

def test_bh_adjust_reference():
    assert ss.bh_adjust([0.01, 0.04, 0.03, 0.005]) == [
        approx(0.02), approx(0.04), approx(0.04), approx(0.02)
    ]


def test_bh_adjust_monotone():
    adjusted = ss.bh_adjust([0.001, 0.02, 0.5, 0.9])
    assert adjusted[0] <= adjusted[1] <= adjusted[2] <= adjusted[3]
    assert adjusted[3] <= 1.0


def test_bonferroni_adjust_reference():
    assert ss.bonferroni_adjust([0.01, 0.5, 0.9]) == [
        approx(0.03), 1.0, 1.0
    ]


def test_adjust_empty():
    assert ss.bh_adjust([]) == []
    assert ss.bonferroni_adjust([]) == []


# - buckets

def test_bucket_index_and_counts():
    counts = ss.hist_counts([1.0, 1.05, 1.2, 1.6, 2.5, 3.5, 6.0, 25.0])
    assert len(counts) == len(ss.HIST_BUCKETS)
    assert counts[0] == 2
    assert counts[4] == 1
    assert counts[6] == 1


def test_hist_bucket_index_edge():
    assert ss.hist_bucket_index(1.0) == 0
    assert ss.hist_bucket_index(1.1) == 1
    assert ss.hist_bucket_index(1.5) == 2
    assert ss.hist_bucket_index(2.0) == 3
    assert ss.hist_bucket_index(3.0) == 4
    assert ss.hist_bucket_index(5.0) == 5
    assert ss.hist_bucket_index(10.0) == 6
    assert ss.hist_bucket_index(1000.0) == 6


def test_theoretical_shares_sum_to_one():
    total = sum(bucket["share"] for bucket in ss.BUCKETS)
    assert total == approx(1.0)


def test_gap_counts():
    counts = ss.gap_counts([1, 4, 7, 15, 60])
    assert len(counts) == len(ss.GAP_BUCKETS)
    assert counts[0] == 1
    assert counts[1] == 1
    assert counts[2] == 1
    assert counts[3] == 1
    assert counts[5] == 1


# - clustering

def test_clustering_stats_merges_sparse():
    values = [1.02] * 40 + [2.5] * 5 + [6.0] * 5 + [15.0] * 10
    result = ss.clustering_stats(values)
    assert result["n"] == 60
    assert len(result["labels"]) == len(result["counts"]) == len(result["expected"])
    assert all(e >= 5 for e in result["expected"]) or result["sparse"]
    assert sum(result["expected"]) == approx(result["n"])
    assert result["p"] == result["p"]


def test_clustering_labels_read_left_to_right():
    labels = [bucket["label"] for bucket in ss.BUCKETS]
    assert labels[0] == "1.00x"
    assert labels[-1] == "10x+"
    assert labels[1] == "1.01-1.99x"
    assert labels[2] == "2-2.99x"


def test_low_round_clustering_baseline():
    values = [1.0] * 30 + [3.0] * 70
    result = ss.low_round_clustering(values, low_threshold=1.1)
    # transitions are pairs, so 30 low rounds leave 30 following rounds
    assert result["n"] == 30
    assert result["baseline_low"] == approx(29 / 99)
    assert result["next_low"] == approx(29 / 30)
    assert result["low_threshold"] == 1.1
    assert sum(result["counts"]) == 30
    assert result["p"] == result["p"]


def test_low_round_clustering_small_sample():
    result = ss.low_round_clustering([1.0, 1.0, 3.0])
    assert result["n"] == 2


# - simulation

def test_simulate_flat_zero_edge_deterministic():
    result = ss.simulate_strategy(
        cashout=2.0, hit_rate=0.97 / 2.0, bankroll=1000.0, stake=10.0,
        rounds=50, strategy="flat", runs=500, seed=7,
    )
    assert 0.0 <= result["p_ruin"] <= 1.0
    assert len(result["finals"]) == 500
    assert result["finals"][0] == result["finals"][0]


def test_simulate_reproducible_with_seed():
    kwargs = dict(cashout=1.5, hit_rate=0.6, bankroll=200.0, stake=5.0,
                  rounds=40, strategy="mart", runs=50, seed=0)
    first = ss.simulate_strategy(**kwargs)
    second = ss.simulate_strategy(**kwargs)
    assert list(first["finals"]) == list(second["finals"])
    assert first["p_ruin"] == second["p_ruin"]
    assert len(first["finals"]) == 50


def test_simulate_certain_loss_ruins():
    result = ss.simulate_strategy(
        cashout=1.5, hit_rate=0.0, bankroll=50.0, stake=10.0,
        rounds=20, strategy="flat", runs=20, seed=0,
    )
    assert result["p_ruin"] == 1.0
    assert result["worst_drawdown"] >= 0.0


def test_simulate_martingale_ruin_vs_flat():
    flat = ss.simulate_strategy(
        cashout=1.5, hit_rate=0.6, bankroll=100.0, stake=5.0,
        rounds=100, strategy="flat", runs=300, seed=3,
    )
    mart = ss.simulate_strategy(
        cashout=1.5, hit_rate=0.6, bankroll=100.0, stake=5.0,
        rounds=100, strategy="mart", runs=300, seed=3,
    )
    assert mart["p_ruin"] >= flat["p_ruin"]


def test_simulate_bankroll_wrapper():
    result = ss.simulate_bankroll(
        cashout=2.0, hit_rate=0.485, start=500.0, base_stake=10.0,
        rounds=100, runs=50, seed=1,
    )
    assert "p_ruin" in result
    assert result["hit_rate"] == approx(0.485)
    assert result["strategy"] == "flat"
    assert len(result["finals"]) == 50


def test_simulate_bet_capped_at_balance():
    result = ss.simulate_strategy(
        cashout=1.1, hit_rate=0.9, bankroll=30.0, stake=10.0,
        rounds=10, strategy="mart", runs=100, seed=5,
    )
    assert all(final >= 0.0 for final in result["finals"])


def test_simulate_rounds_below_stake_ruin():
    result = ss.simulate_strategy(
        cashout=1.5, hit_rate=0.9, bankroll=8.0, stake=10.0,
        rounds=10, strategy="flat", runs=50, seed=1,
    )
    assert result["p_ruin"] == 1.0


# - constants

def test_threshold_rows():
    labels = [row["label"] for row in ss.THRESHOLD_ROWS]
    assert "At/above 2.00" in labels and "At/above 10.00" in labels
    assert any(label.startswith("Under") for label in labels)
    kinds = {row["kind"] for row in ss.THRESHOLD_ROWS}
    assert kinds == {"lt", "ge"}


def test_default_windows_in_range():
    for start, end in ss.DEFAULT_WINDOWS:
        assert 0 <= start <= 59
        assert 0 <= end <= 59


def test_verdict_threshold_is_300():
    assert ss.MIN_N_FOR_VERDICT == 300
    assert ss.MIN_N_FOR_P == 30


# - prediction

def _sample(n: int, unsafe: int, band: int) -> list[float]:
    return [1.05] * unsafe + [1.60] * band + [3.00] * (n - unsafe - band)


def test_theory_band_and_unsafe():
    assert ss.theory_band() == approx(0.97 / 1.10 - 0.97 / 2.50, tol=1e-9)
    assert ss.theory_unsafe() == approx(1.0 - 0.97 / 1.10, tol=1e-9)
    assert ss.theory_band(2.0, 3.0) == approx(0.97 / 2.0 - 0.97 / 3.0)
    assert math.isnan(ss.theory_band(3.0, 2.0))
    assert ss.theory_band() + ss.theory_unsafe() + ss.theoretical_p(2.50) == approx(1.0)


def test_predict_go_data_backed():
    pred = ss.predict_next(
        minute=46,
        windows=[(45, 47)],
        window_values=_sample(400, 40, 200),
        other_values=_sample(600, 120, 240),
        minute_values=_sample(40, 4, 20),
    )
    assert pred['decision'] == 'go'
    assert pred['confidence'] == 'data-backed'
    assert pred['matched_window'] == (45, 47)
    assert pred['headline_band'] == approx(200 / 400)
    assert pred['comparison'] is not None
    assert 0.0 <= pred['comparison']['p'] <= 1.0
    assert 0.0 <= pred['headline_band'] <= 1.0


def test_predict_provisional_sample_falls_back_to_theory():
    pred = ss.predict_next(
        minute=30,
        windows=[(29, 31)],
        window_values=_sample(100, 15, 45),
        other_values=_sample(600, 120, 240),
    )
    assert pred['decision'] == 'go'
    assert pred['confidence'] == 'provisional'
    assert pred['headline_band'] == approx(ss.theory_band())
    assert 'theory' in pred['headline_source']


def test_predict_theoretical_when_no_window_data():
    pred = ss.predict_next(
        minute=4,
        windows=[(3, 4)],
        window_values=_sample(10, 2, 4),
        other_values=[],
    )
    assert pred['decision'] == 'go'
    assert pred['confidence'] == 'theoretical'
    assert pred['headline_band'] == approx(ss.theory_band())
    assert pred['comparison'] is None


def test_predict_skip_outside_windows():
    pred = ss.predict_next(
        minute=10,
        windows=[(45, 47)],
        window_values=_sample(400, 40, 200),
        other_values=_sample(600, 120, 240),
        minute_values=_sample(40, 20, 10),
    )
    assert pred['decision'] == 'skip'
    assert pred['matched_window'] is None
    assert pred['headline_band'] == approx(ss.theory_band())


def test_predict_data_favored_minute():
    pred = ss.predict_next(
        minute=10,
        windows=[(45, 47)],
        window_values=_sample(400, 40, 200),
        other_values=_sample(600, 120, 240),
        minute_values=_sample(40, 1, 25),
        favored=[10],
    )
    assert pred['decision'] == 'go_data'
    assert pred['favored'] is True
    assert pred['matched_window'] is None


def test_predict_window_beats_favored():
    pred = ss.predict_next(
        minute=46,
        windows=[(45, 47)],
        window_values=_sample(400, 40, 200),
        other_values=_sample(600, 120, 240),
        favored=[46],
    )
    assert pred['decision'] == 'go'


def test_predict_normalises_minute():
    pred = ss.predict_next(
        minute=64, windows=[(4, 5)],
        window_values=[], other_values=[],
    )
    assert pred['minute'] == 4
    assert pred['decision'] == 'go'


def test_favored_minutes_picks_clear_winners():
    minute_values = {7: [1.05] + [3.0] * 39}
    for m in range(8, 14):
        minute_values[m] = [1.05] * 24 + [3.0] * 36
    rows = ss.favored_minutes(minute_values)
    minutes = [row['minute'] for row in rows]
    assert 7 in minutes
    assert all(row['unsafe_ci'][1] < 0.4 for row in rows)
    assert rows == sorted(rows, key=lambda r: (r['unsafe_rate'], r['minute']))


def test_favored_minutes_excludes_small_samples():
    rows = ss.favored_minutes({3: [1.05] * 20 + [3.0] * 10})
    assert rows == []


def test_favored_minutes_empty_data():
    assert ss.favored_minutes({}) == []
    assert ss.favored_minutes({1: []}) == []


def test_favored_minutes_skips_baseline_ties():
    values = [1.05, 3.0] * 40
    rows = ss.favored_minutes({1: values, 2: values})
    assert rows == []


# - odd suggestion

def test_suggest_odd_default_without_data():
    odd, promoted = ss.suggest_odd()
    assert odd == approx(1.10)
    assert not promoted
    odd, promoted = ss.suggest_odd([], default=1.25)
    assert odd == approx(1.25)
    assert not promoted


def test_suggest_odd_quantizes_to_005_grid():
    odd, _ = ss.suggest_odd([1.0, 1.31, 1.6])
    assert odd * 20 == pytest.approx(round(odd * 20))
    assert 1.0 <= odd <= ss.ODD_SOFT_CAP


def test_suggest_odd_small_sample_never_promotes():
    odd, promoted = ss.suggest_odd([9.0] * 20)
    assert not promoted
    assert odd <= ss.ODD_SOFT_CAP


def test_suggest_odd_mixed_data_stays_caution_band():
    values = [1.05, 1.1, 1.2, 1.3, 1.4, 1.6, 1.9, 2.4, 3.5, 8.0]
    odd, promoted = ss.suggest_odd(values)
    assert not promoted
    assert 1.0 <= odd <= ss.ODD_SOFT_CAP


def test_suggest_odd_promotes_on_very_certain_data():
    odd, promoted = ss.suggest_odd([2.6] * 40)
    assert promoted
    assert odd == approx(ss.ODD_MAX)


def test_suggest_odd_promotion_picks_highest_supported():
    # every round reached 1.55 but not 1.60 -> highest supported is 1.55
    odd, promoted = ss.suggest_odd([1.55] * 40)
    assert promoted
    assert odd == approx(1.55)


def test_suggest_odd_high_band_requires_confident_reach():
    # 70% of 30 rounds reached 2.00 - rate below 75%, so no promotion
    values = [2.5] * 21 + [1.2] * 9
    odd, promoted = ss.suggest_odd(values)
    assert not promoted
    assert odd <= ss.ODD_SOFT_CAP


def test_high_odd_grid_only_above_soft_cap():
    assert min(ss.HIGH_ODD_GRID) > ss.ODD_SOFT_CAP
    assert max(ss.HIGH_ODD_GRID) == approx(ss.ODD_MAX)
    assert len(ss.HIGH_ODD_GRID) == 11
