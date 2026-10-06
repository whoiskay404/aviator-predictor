"""Statistical helpers for Aviator Analyzer.

All functions are pure (no Streamlit, no I/O) and operate on plain
sequences of crash multipliers so they can be unit tested directly.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

HOUSE_EDGE = 0.03
HOT_THRESHOLD = 2.0
VERY_HOT_THRESHOLD = 10.0
THRESHOLDS = {"hot": HOT_THRESHOLD, "very_hot": VERY_HOT_THRESHOLD}

MIN_N_FOR_P = 30
MIN_N_FOR_VERDICT = 300
ALPHA = 0.05


def _theoretical_shares() -> list[float]:
    """Probability mass of each bucket under P(multiplier >= x) = 0.97/x."""
    shares = [HOUSE_EDGE]
    edges = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    for lo, hi in zip(edges, edges[1:]):
        shares.append((1 - HOUSE_EDGE) / lo - (1 - HOUSE_EDGE) / hi)
    shares.append((1 - HOUSE_EDGE) / 10.0)
    return shares


BUCKET_LABELS = (
    ["1.00x", "1.01-1.99x"]
    + [f"{i}-{i}.99x" for i in range(2, 10)]
    + ["10x+"]
)
BUCKETS = tuple(
    {"label": label, "share": share}
    for label, share in zip(BUCKET_LABELS, _theoretical_shares())
)

# Display buckets used for histograms and the clustering check.
HIST_BUCKETS = (
    ("1.00-1.09", 1.0, 1.1),
    ("1.10-1.49", 1.1, 1.5),
    ("1.50-1.99", 1.5, 2.0),
    ("2.00-2.99", 2.0, 3.0),
    ("3.00-4.99", 3.0, 5.0),
    ("5.00-9.99", 5.0, 10.0),
    ("10+", 10.0, math.inf),
)

GAP_BUCKETS = (
    ("1-2", 1, 2),
    ("3-5", 3, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51+", 51, math.inf),
)

THRESHOLD_ROWS = (
    {"label": "Under 1.10", "kind": "lt", "x": 1.10},
    {"label": "Under 1.50", "kind": "lt", "x": 1.50},
    {"label": "Under 2.00", "kind": "lt", "x": 2.00},
    {"label": "At/above 2.00", "kind": "ge", "x": 2.00},
    {"label": "At/above 3.00", "kind": "ge", "x": 3.00},
    {"label": "At/above 5.00", "kind": "ge", "x": 5.00},
    {"label": "At/above 10.00", "kind": "ge", "x": 10.00},
    {"label": "At/above 50.00", "kind": "ge", "x": 50.00},
)

DEFAULT_WINDOWS = (
    (3, 4), (20, 22), (29, 31), (40, 42), (45, 47), (50, 52), (57, 59),
)


def as_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def percentile(values: Sequence[float], q: float) -> float:
    """R-7 percentile (the default linear interpolation used by numpy/R)."""
    arr = as_array(values)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q, method="linear"))


def summarize(values: Sequence[float]) -> dict:
    arr = as_array(values)
    n = int(arr.size)
    if n == 0:
        nan = float("nan")
        return {"n": 0, "min": nan, "max": nan, "mean": nan,
                "median": nan, "sd": nan}
    return {
        "n": n,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "sd": 0.0 if n == 1 else float(np.std(arr, ddof=1)),
    }


def wilson_ci(successes: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    successes = min(max(int(successes), 0), int(n))
    low, high = proportion_confint(successes, n, alpha=alpha, method="wilson")
    return float(low), float(high)


def theoretical_p(cashout: float) -> float:
    """P(multiplier >= cashout) = (1 - house edge) / cashout, clamped to 1."""
    if not math.isfinite(cashout) or cashout <= 0:
        return float("nan")
    if cashout <= 1.0:
        return 1.0
    return float(min(1.0, (1 - HOUSE_EDGE) / cashout))


def expected_value(cashout: float, hit_rate: Optional[float] = None) -> float:
    """Expected profit per unit stake when cashing out at `cashout`."""
    if hit_rate is None:
        hit_rate = theoretical_p(cashout)
    if not math.isfinite(hit_rate):
        return float("nan")
    return float(hit_rate) * float(cashout) - 1.0


def count_where(values: Sequence[float], predicate: Callable[[float], bool]) -> int:
    return int(sum(1 for v in as_array(values) if predicate(v)))


def rate_where(values: Sequence[float], predicate: Callable[[float], bool]) -> float:
    arr = as_array(values)
    if arr.size == 0:
        return float("nan")
    return count_where(arr, predicate) / int(arr.size)


def streak_info(values: Sequence[float], predicate: Callable[[float], bool]) -> dict:
    """Longest and currently-running streak of values satisfying `predicate`."""
    arr = as_array(values)
    flags = [bool(predicate(v)) for v in arr]
    longest = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    current = 0
    for flag in reversed(flags):
        if not flag:
            break
        current += 1
    return {"current": current, "longest": longest}


def gaps_info(values: Sequence[float], predicate: Callable[[float], bool]) -> dict:
    """Completed gaps between matches plus the current (possibly open) gap.

    ``gaps`` holds finished distances between consecutive matches and
    ``current`` is the number of rounds since the last match (0 when the
    last round matched, ``None`` when the predicate never matched).
    """
    arr = as_array(values)
    matches = [i for i, v in enumerate(arr) if predicate(v)]
    gaps = [b - a - 1 for a, b in zip(matches, matches[1:])]
    current = None if not matches else int(arr.size) - 1 - matches[-1]
    return {"gaps": gaps, "current": current}


def rolling_median(values: Sequence[float], window: int) -> list[float]:
    """Rolling median; partial windows are used until `window` points exist."""
    arr = as_array(values)
    window = max(int(window), 1)
    return [float(np.median(arr[max(0, i + 1 - window):i + 1]))
            for i in range(int(arr.size))]


def rolling_rate(values: Sequence[float], window: int,
                 predicate: Callable[[float], bool]) -> list[float]:
    arr = as_array(values)
    window = max(int(window), 1)
    flags = np.array([1.0 if predicate(v) else 0.0 for v in arr])
    return [float(flags[max(0, i + 1 - window):i + 1].mean())
            for i in range(int(arr.size))]


def p_value_visible(n: int) -> bool:
    return int(n) >= MIN_N_FOR_P


def verdict(n_window: int, n_other: int, p: Optional[float] = None) -> str:
    """Classify a window comparison.

    Returns one of: ``insufficient`` (windows hold fewer than 300 rounds),
    ``hidden`` (a group is too small for an honest p-value),
    ``significant``, ``not_significant``.
    """
    if int(n_window) < MIN_N_FOR_VERDICT:
        return "insufficient"
    if int(n_window) < MIN_N_FOR_P or int(n_other) < MIN_N_FOR_P:
        return "hidden"
    if p is None or not math.isfinite(p):
        return "not_significant"
    return "significant" if p < ALPHA else "not_significant"


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> dict:
    """Two-sided Mann-Whitney U test (asymptotic, tie corrected, continuity)."""
    arr_a = as_array(a)
    arr_b = as_array(b)
    n1, n2 = int(arr_a.size), int(arr_b.size)
    empty = {"u1": float("nan"), "u2": float("nan"), "p": float("nan"),
             "r": float("nan"), "n1": n1, "n2": n2}
    if n1 == 0 or n2 == 0:
        return empty
    combined = np.concatenate([arr_a, arr_b])
    if combined.min() == combined.max():
        u = n1 * n2 / 2.0
        return {"u1": u, "u2": u, "p": 1.0, "r": 0.0, "n1": n1, "n2": n2}
    res = sps.mannwhitneyu(arr_a, arr_b, alternative="two-sided",
                           method="asymptotic", use_continuity=True)
    u1 = float(res.statistic)
    p = float(res.pvalue)
    r = 2.0 * u1 / (n1 * n2) - 1.0
    z = float(sps.norm.isf(p / 2.0)) * (1.0 if r >= 0 else -1.0) if p > 0 else float("nan")
    return {"u1": u1, "u2": n1 * n2 - u1, "p": p, "r": r, "z": z,
            "n1": n1, "n2": n2}


def fisher_exact(table: Sequence[Sequence[float]]) -> dict:
    arr = np.asarray(table, dtype=float)
    if arr.shape != (2, 2) or arr.sum() <= 0:
        return {"oddsratio": float("nan"), "p": float("nan")}
    res = sps.fisher_exact(arr)
    if hasattr(res, "pvalue"):
        return {"oddsratio": float(res.statistic), "p": float(res.pvalue)}
    oddsratio, p = res
    return {"oddsratio": float(oddsratio), "p": float(p)}


def chi_square_independence(table: Sequence[Sequence[float]]) -> dict:
    arr = np.asarray(table, dtype=float)
    nan = float("nan")
    out = {"chi2": nan, "p": nan, "dof": 1, "min_expected": nan,
           "expected_ok": False, "cramers_v": nan,
           "expected": [[nan, nan], [nan, nan]]}
    if arr.shape != (2, 2) or arr.sum() <= 0 or (arr.sum(axis=1) == 0).any() \
            or (arr.sum(axis=0) == 0).any():
        return out
    chi2, p, dof, expected = sps.chi2_contingency(arr, correction=False)
    min_expected = float(expected.min())
    n = float(arr.sum())
    return {
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "min_expected": min_expected,
        "expected_ok": min_expected >= 5.0,
        "cramers_v": float(math.sqrt(chi2 / n)),
        "expected": expected.tolist(),
    }


def chi_square_gof(observed: Sequence[float],
                   expected: Sequence[float]) -> dict:
    """Chi-square goodness of fit (computed directly so the observed and
    expected totals are allowed to differ slightly, as in the classic
    die-throw example)."""
    obs = np.asarray(observed, dtype=float)
    exp = np.asarray(expected, dtype=float)
    nan = float("nan")
    out = {"chi2": nan, "p": nan, "dof": 0}
    if obs.size == 0 or obs.size != exp.size:
        return out
    mask = exp > 0
    if not mask.any():
        return out
    obs, exp = obs[mask], exp[mask]
    dof = int(obs.size - 1)
    if dof < 1:
        return out
    chi2 = float(np.sum((obs - exp) ** 2 / exp))
    return {"chi2": chi2, "p": float(sps.chi2.sf(chi2, dof)), "dof": dof}


def _multipletest(pvals: Sequence[float], method: str) -> list[float]:
    vals = [float(v) for v in pvals]
    if not vals:
        return []
    if any(not math.isfinite(v) for v in vals):
        return [float("nan")] * len(vals)
    adjusted = multipletests(vals, alpha=ALPHA, method=method)[1]
    return [float(v) for v in adjusted]


def bh_adjust(pvals: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg false discovery rate adjusted p-values."""
    return _multipletest(pvals, "fdr_bh")


def bonferroni_adjust(pvals: Sequence[float]) -> list[float]:
    return _multipletest(pvals, "bonferroni")


def bucket_index(multiplier: float) -> int:
    if not math.isfinite(multiplier) or multiplier <= 1.0:
        return 0
    if multiplier < 10.0:
        return int(math.floor(multiplier))
    return 10


def theoretical_rate(kind: str, x: float) -> float:
    """Theory for a fair 97% RTP game: P(m >= x) = 0.97/x (or its complement)."""
    if not math.isfinite(x) or x <= 0:
        return float("nan")
    if kind == "ge":
        return 1.0 if x <= 1.0 else float(min(1.0, (1 - HOUSE_EDGE) / x))
    return float(min(1.0, max(0.0, 1.0 - (1 - HOUSE_EDGE) / x)))


def hist_counts(values: Sequence[float]) -> list[int]:
    arr = as_array(values)
    counts = [0] * len(HIST_BUCKETS)
    for v in arr:
        counts[hist_bucket_index(float(v))] += 1
    return counts


def hist_bucket_index(multiplier: float) -> int:
    if not math.isfinite(multiplier) or multiplier < HIST_BUCKETS[0][1]:
        return 0
    for i, (_, lo, hi) in enumerate(HIST_BUCKETS):
        if lo <= multiplier < hi:
            return i
    return len(HIST_BUCKETS) - 1


def gap_counts(gaps: Sequence[int]) -> list[int]:
    counts = [0] * len(GAP_BUCKETS)
    for g in gaps:
        for i, (_, lo, hi) in enumerate(GAP_BUCKETS):
            if lo <= g <= hi:
                counts[i] += 1
                break
    return counts


def clustering_stats(values: Sequence[float]) -> dict:
    """Observed bucket counts vs theoretical shares with chi-square GOF.

    Buckets whose expected count is below 5 are merged with the previous
    bucket before the test; merged label groups are reported.
    """
    arr = as_array(values)
    n = int(arr.size)
    counts = [0] * len(BUCKETS)
    for v in arr:
        counts[bucket_index(float(v))] += 1

    groups = _merge_small(counts, [b["share"] for b in BUCKETS],
                           [b["label"] for b in BUCKETS], n)
    return _gof_result(groups, n)


def low_round_clustering(values: Sequence[float],
                         low_threshold: float = 1.1) -> dict:
    """Does a sub-threshold round predict another one?

    Compares the bucket distribution of rounds *following* a low round
    against the baseline distribution of all later rounds (chi-square GOF).
    """
    arr = as_array(values)
    if arr.size >= 2:
        base = arr[1:]
        following = arr[1:][arr[:-1] < low_threshold]
    else:
        base = arr[:0]
        following = arr[:0]

    base_counts = hist_counts(base)
    base_total = int(sum(base_counts))
    props = [c / base_total for c in base_counts] if base_total else [0.0] * len(base_counts)
    obs = hist_counts(following)
    n = int(following.size)
    groups = _merge_small(obs, props, [b[0] for b in HIST_BUCKETS], n)
    out = _gof_result(groups, n)
    base_low = rate_where(base, lambda x: x < low_threshold) if base.size else float("nan")
    out.update({
        "n": n,
        "baseline_low": base_low,
        "next_low": rate_where(following, lambda x: x < low_threshold) if n else float("nan"),
        "low_threshold": low_threshold,
    })
    return out


def _merge_small(counts: Sequence[int], shares: Sequence[float],
                 labels: Sequence[str], n: int) -> list[dict]:
    """Accumulate buckets left to right until the expected count reaches 5."""
    groups: list[dict] = []
    pending_labels: list[str] = []
    pending_count = 0
    pending_share = 0.0
    for label, count, share in zip(labels, counts, shares):
        pending_labels.append(label)
        pending_count += count
        pending_share += share
        if pending_share * n >= 5.0 or len(pending_labels) + len(groups) >= len(labels):
            text = pending_labels[0] if len(pending_labels) == 1 else \
                f"{pending_labels[0]} … {pending_labels[-1]}"
            groups.append({"label": text, "count": pending_count,
                           "share": pending_share})
            pending_labels, pending_count, pending_share = [], 0, 0.0
    return groups


def _gof_result(groups: list[dict], n: int) -> dict:
    expected = [g["share"] * n for g in groups]
    chi2 = p = dof = float("nan")
    testable = n > 0 and len(groups) >= 2 and all(e > 0 for e in expected)
    if testable:
        result = chi_square_gof([g["count"] for g in groups], expected)
        chi2, p, dof = result["chi2"], result["p"], result["dof"]
    return {
        "n": n,
        "labels": [g["label"] for g in groups],
        "counts": [g["count"] for g in groups],
        "expected": expected,
        "shares": [g["share"] for g in groups],
        "chi2": chi2,
        "p": p,
        "dof": dof,
        "min_expected": min(expected) if expected else float("nan"),
        "sparse": bool(expected) and min(expected) < 5.0,
    }


def simulate_strategy(cashout: float = 1.5,
                      hit_rate: Optional[float] = None,
                      bankroll: float = 500.0,
                      stake: float = 10.0,
                      rounds: int = 300,
                      strategy: str = "flat",
                      runs: int = 10000,
                      seed: int = 0) -> dict:
    """Monte Carlo bankroll simulation (flat stake or Martingale).

    Mirrors the honest reference behaviour: a run stops once the balance
    drops below one base stake (ruin), bets are capped at the balance and
    Martingale doubles after a loss and resets after a win. `hit_rate`
    defaults to P(multiplier >= cashout) for a fair 97% RTP game.
    """
    if hit_rate is None:
        hit_rate = theoretical_p(cashout)
    hit_rate = float(min(max(float(hit_rate), 0.0), 1.0))
    cashout = max(float(cashout), 1.01)
    bankroll = float(bankroll)
    stake = float(stake)
    runs = max(int(runs), 1)
    rounds = max(int(rounds), 1)
    mart = str(strategy).lower().startswith("mart")

    instant_ruin = stake > bankroll
    rng = np.random.default_rng(int(seed))
    bal = np.full(runs, bankroll, dtype=float)
    bet = np.full(runs, stake, dtype=float)
    peak = bal.copy()
    dd_max = np.zeros(runs, dtype=float)
    active = np.ones(runs, dtype=bool)
    ruined = np.zeros(runs, dtype=bool)

    for _ in range(rounds):
        if instant_ruin:
            ruined[:] = True
            break
        broke = active & (bal < stake)
        ruined |= broke
        active &= ~broke
        if not active.any():
            break
        amount = np.minimum(bet, bal)
        win = (rng.random(runs) < hit_rate) & active
        bal = np.where(active,
                       np.where(win, bal + amount * (cashout - 1.0), bal - amount),
                       bal)
        if mart:
            bet = np.where(win, stake, np.where(active, amount * 2.0, bet))
        peak = np.maximum(peak, bal)
        dd_max = np.maximum(dd_max, peak - bal)

    ruined |= bal < stake
    finals = bal
    return {
        "runs": runs,
        "rounds": rounds,
        "cashout": cashout,
        "hit_rate": hit_rate,
        "strategy": "mart" if mart else "flat",
        "bankroll": bankroll,
        "stake": stake,
        "p_ruin": float(ruined.mean()),
        "mean_final": float(finals.mean()),
        "median_final": float(np.median(finals)),
        "p10_final": float(np.percentile(finals, 10)),
        "worst_drawdown": float(dd_max.max()),
        "median_drawdown": float(np.median(dd_max)),
        "p_profit": float((finals > bankroll).mean()),
        "expected_final": bankroll + rounds * stake * (hit_rate * cashout - 1.0),
        "finals": finals,
        "dd_max": dd_max,
    }


def simulate_bankroll(cashout: float = 2.0,
                      hit_rate: Optional[float] = None,
                      runs: int = 10000,
                      rounds: int = 200,
                      base_stake: float = 1.0,
                      start: float = 100.0,
                      seed: int = 0) -> dict:
    """Flat-stake convenience wrapper around :func:`simulate_strategy`."""
    return simulate_strategy(cashout=cashout, hit_rate=hit_rate,
                             bankroll=start, stake=base_stake, rounds=rounds,
                             strategy="flat", runs=runs, seed=seed)


# - prediction

SAFE_LO = 1.10
SAFE_HI = 2.50


def theory_band(lo: float = SAFE_LO, hi: float = SAFE_HI) -> float:
    """P(lo <= multiplier <= hi) for a fair 97% RTP crash game."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0 or hi <= lo:
        return float("nan")
    return float(max(0.0, min(1.0, theoretical_p(lo) - theoretical_p(hi))))


def theory_unsafe(threshold: float = SAFE_LO) -> float:
    """P(multiplier < threshold) for a fair 97% RTP crash game."""
    if not math.isfinite(threshold) or threshold <= 0:
        return float("nan")
    return float(max(0.0, min(1.0, 1.0 - theoretical_p(threshold))))


def _sample_band(values: Sequence[float],
                 lo: float = SAFE_LO, hi: float = SAFE_HI) -> dict:
    arr = as_array(values)
    n = int(arr.size)
    if n == 0:
        return {"n": 0}
    band_hits = int(((arr >= lo) & (arr <= hi)).sum())
    unsafe_hits = int((arr < lo).sum())
    band_ci = wilson_ci(band_hits, n)
    unsafe_ci = wilson_ci(unsafe_hits, n)
    return {
        "n": n,
        "band_hits": band_hits,
        "unsafe_hits": unsafe_hits,
        "band_rate": band_hits / n,
        "band_ci": band_ci,
        "unsafe_rate": unsafe_hits / n,
        "unsafe_ci": unsafe_ci,
    }


def favored_minutes(minute_values: dict,
                    min_n: int = MIN_N_FOR_P,
                    threshold: float = SAFE_LO) -> list[dict]:
    """Minutes whose unsafe-crash rate is provably below the overall rate.

    A minute qualifies only when its Wilson interval for P(m < threshold)
    sits entirely below the baseline rate of the whole sample - that keeps
    lucky-looking slices out.
    """
    all_values: list[float] = []
    for values in minute_values.values():
        all_values.extend(float(v) for v in values)
    arr = as_array(all_values)
    if arr.size == 0:
        return []
    baseline = float((arr < threshold).mean())
    out: list[dict] = []
    for minute, values in minute_values.items():
        sample = as_array(values)
        n = int(sample.size)
        if n < min_n:
            continue
        hits = int((sample < threshold).sum())
        low, high = wilson_ci(hits, n)
        if high < baseline:
            out.append({
                "minute": int(minute) % 60,
                "n": n,
                "unsafe_rate": hits / n,
                "unsafe_ci": (low, high),
            })
    out.sort(key=lambda row: (row["unsafe_rate"], row["minute"]))
    return out


def predict_next(minute: int,
                 windows: Sequence[tuple[int, int]],
                 window_values: Sequence[float],
                 other_values: Sequence[float],
                 minute_values: Sequence[float] = (),
                 favored: Sequence[int] = ()) -> dict:
    """Best-effort prediction for the round falling at `minute` (0-59).

    Decisions:
    - ``go``          minute lies inside one of your configured safe windows
    - ``go_data``     minute is outside the windows but its unsafe-crash rate
                      is significantly below baseline in your own data
    - ``skip``        minute is outside every window with no data support

    The headline probability uses your window data once it holds at least
    ``MIN_N_FOR_VERDICT`` rounds; below that it honestly falls back to the
    verified 97% RTP theory and says so.
    """
    minute = int(minute) % 60
    matched: Optional[tuple[int, int]] = None
    for start, end in windows:
        lo, hi = min(int(start), int(end)), max(int(start), int(end))
        if lo <= minute <= hi:
            matched = (lo, hi)
            break

    band_t, unsafe_t = theory_band(), theory_unsafe()
    emp_w = _sample_band(window_values)
    emp_o = _sample_band(other_values)
    emp_m = _sample_band(minute_values)
    favored_set = {int(m) % 60 for m in favored}
    is_favored = minute in favored_set

    notes: list[str] = []
    comparison: Optional[dict] = None
    if emp_w["n"] >= MIN_N_FOR_P and emp_o["n"] >= MIN_N_FOR_P:
        low_w = emp_w["unsafe_hits"]
        low_o = emp_o["unsafe_hits"]
        comparison = {
            "p": fisher_exact([[low_w, emp_w["n"] - low_w],
                               [low_o, emp_o["n"] - low_o]])["p"],
            "unsafe_window": emp_w["unsafe_rate"],
            "unsafe_other": emp_o["unsafe_rate"],
        }
    elif emp_w["n"] or emp_o["n"]:
        notes.append(
            f"Window sample n={emp_w['n']}, other minutes n={emp_o['n']} - "
            "need n ≥ 30 in both groups before the comparison p-value is shown."
        )

    if matched is not None:
        decision = "go"
        n_w = emp_w["n"]
        if n_w >= MIN_N_FOR_VERDICT:
            confidence = "data-backed"
            headline_band = emp_w["band_rate"]
            headline_source = f"your window data (n={n_w})"
        elif n_w >= MIN_N_FOR_P:
            confidence = "provisional"
            headline_band = band_t
            headline_source = (
                f"97% RTP theory (your windows only hold {n_w} rounds - "
                f"under {MIN_N_FOR_VERDICT}, so theory stays the headline)"
            )
            notes.append(
                f"Your windows look {'safer' if emp_w['unsafe_rate'] < unsafe_t else 'similar'} "
                f"than theory so far: unsafe rate {pct_text(emp_w['unsafe_rate'])} "
                f"over {n_w} rounds (theory {pct_text(unsafe_t)})."
            )
        else:
            confidence = "theoretical"
            headline_band = band_t
            headline_source = (
                f"97% RTP theory (only {n_w} timestamped rounds inside your "
                "windows - not enough data yet)"
            )
        notes.append(
            f"Window {matched[0]:02d}-{matched[1]:02d} matches your safe "
            "minute plan; stake only inside it."
        )
    elif is_favored:
        decision = "go_data"
        confidence = "data-favored"
        headline_band = band_t if emp_m.get("n", 0) < MIN_N_FOR_P else emp_m["band_rate"]
        source_n = emp_m.get("n", 0)
        headline_source = (
            "your own data for this minute" if source_n >= MIN_N_FOR_P
            else f"97% RTP theory, flagged by your data (minute sample n={source_n})"
        )
        notes.append(
            "Minute is outside your saved windows but its unsafe-crash rate is "
            "significantly below baseline in your data (Wilson interval clear "
            "of the overall rate)."
        )
    else:
        decision = "skip"
        confidence = "theoretical"
        headline_band = band_t
        headline_source = "97% RTP theory"
        notes.append(
            f"Minute {minute:02d} is outside every safe window and your data "
            "does not validate it - sit this one out."
        )

    notes.append(
        "Every round is independent (provably fair SHA-512): this is a "
        "best-effort estimate, never a guarantee."
    )

    return {
        "minute": minute,
        "matched_window": matched,
        "favored": is_favored,
        "decision": decision,
        "confidence": confidence,
        "headline_band": float(headline_band),
        "headline_source": headline_source,
        "theory": {"band": band_t, "unsafe": unsafe_t,
                   "reach_1_10": theoretical_p(SAFE_LO),
                   "reach_1_50": theoretical_p(1.5),
                   "reach_2_50": theoretical_p(SAFE_HI)},
        "window": emp_w,
        "other": emp_o,
        "minute_stats": emp_m,
        "comparison": comparison,
        "notes": notes,
    }


def pct_text(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "-"
    return f"{float(value) * 100:.1f}%"


# - odd suggestion

ODD_MIN = 1.00
ODD_SOFT_CAP = 1.45
ODD_MAX = 2.00
ODD_PROMOTE_MIN_N = 30
ODD_PROMOTE_RATE = 0.75
ODD_PROMOTE_LB = 0.60

# 0.05 grid from ODD_MAX down to just above the soft cap (2.00 … 1.50)
HIGH_ODD_GRID: tuple[float, ...] = tuple(
    step / 20.0
    for step in range(int(round(ODD_MAX * 20)), int(round(ODD_SOFT_CAP * 20)), -1)
)


def odd_quantize(value: float) -> float:
    """Snap to the 0.05 grid and keep inside [ODD_MIN, ODD_MAX]."""
    snapped = round(round(float(value) * 20) / 20, 2)
    return float(min(ODD_MAX, max(ODD_MIN, snapped)))


def suggest_odd(values: Sequence[float] = (),
                default: float = 1.10) -> tuple[float, bool]:
    """Suggest a cash-out odd: ``(odd, promoted)``.

    Most suggestions stay in the cautious **1.00-1.45** band: the base odd is
    the 25th percentile of `values`, capped at ``ODD_SOFT_CAP``.

    An odd above 1.45 (up to ``ODD_MAX``) is returned only when the data is
    *very certain the plane will go far*: at least ``ODD_PROMOTE_MIN_N``
    rounds, at least ``ODD_PROMOTE_RATE`` of them reached the candidate odd,
    and the Wilson lower bound stays above ``ODD_PROMOTE_LB``. The highest
    candidate that qualifies wins. With no data, `default` is returned.
    """
    vals = [float(v) for v in values]
    if not vals:
        return odd_quantize(default), False
    base = odd_quantize(min(ODD_SOFT_CAP, percentile(vals, 25)))
    n = len(vals)
    if n >= ODD_PROMOTE_MIN_N:
        for candidate in HIGH_ODD_GRID:
            hits = sum(1 for v in vals if v >= candidate)
            if hits / n >= ODD_PROMOTE_RATE:
                if wilson_ci(hits, n)[0] >= ODD_PROMOTE_LB:
                    return float(candidate), True
    return base, False
