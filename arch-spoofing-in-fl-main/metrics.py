"""Numeric primitives shared across the defence, the attacks and the analysis.

These live apart from any of their callers on purpose. `_js_divergence` used to
sit in `defense.py`, on the reasoning that the distribution verifier was the
first thing to need it. It then acquired callers in `fingerprint_lab`, `attacks`
and both notebooks, and once the probe helpers also needed it the import graph
had a cycle (`defense` -> probe helpers -> `defense`). A module with no project
imports of its own is the thing that cannot be part of a cycle.

Nothing here touches TensorFlow, so it stays cheap to import from a notebook
that only wants to score some vectors.
"""

from __future__ import annotations

import numpy as np


def js_divergence(p, q, eps: float = 1e-9) -> float:
    """Jensen-Shannon divergence between two discrete distributions.

    Symmetric, bounded, and 0 when the two are identical. `p` and `q` need not
    already be normalised.

    Used rather than cosine similarity because cosine did not separate honest
    from spoofed at this data scale: a spread softmax vector against a peaked
    declared histogram scores high on cosine whether or not the two describe
    the same distribution. JS is built for the comparison actually being made.
    """
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def expected_calibration_error(confidence, correct, n_bins: int = 10) -> float:
    """Mean |confidence - accuracy| over `n_bins` equal-width confidence bins.

    A well-calibrated model that reports 0.8 confidence is right 80% of the
    time. Overconfidence is one of the few behavioural traits that survives the
    move to a fixed-dimension probe space, which is why it is a fingerprint
    feature rather than just a reporting statistic.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total, err = len(confidence), 0.0
    if total == 0:
        return 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        err += mask.sum() / total * abs(confidence[mask].mean() - correct[mask].mean())
    return float(err)


def normalise(v) -> np.ndarray:
    """Clip negatives and scale to sum 1, falling back to uniform on all-zero."""
    v = np.clip(np.asarray(v, dtype=np.float64), 0, None)
    s = v.sum()
    return v / s if s > 0 else np.ones_like(v) / len(v)


# Legacy alias. Both notebooks and several modules import `_js_divergence`, and
# renaming it everywhere is churn with no benefit; the underscore is simply a
# historical accident from when it was private to `defense.py`.
_js_divergence = js_divergence

# JS divergence here is in NATS, so it is bounded by ln 2, not by 1. The honest
# baseline of 0.251 measured in the distribution study is therefore about a
# THIRD of the maximum possible disagreement, not a quarter. Anything that
# renders a JS value as a fraction of its range must divide by this.
JS_MAX = float(np.log(2.0))


# --------------------------------------------------------------------------- #
# Cluster separation
# --------------------------------------------------------------------------- #
def mean_pairwise_js(group_a_hists, group_b_hists) -> float:
    """Mean JS divergence over every CROSS-group pair of label histograms.

    THE cluster-separation metric for this project.

    Silhouette is not used and should not be: forcing a handful of points into
    two groups always separates something, so silhouette reports a healthy score
    for a clusterer that has learned nothing about the data. Mean pairwise JS
    asks the question actually being asked, which is whether the two recovered
    groups hold different label distributions.
    """
    A = np.atleast_2d(np.asarray(group_a_hists, dtype=np.float64))
    B = np.atleast_2d(np.asarray(group_b_hists, dtype=np.float64))
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    return float(np.mean([js_divergence(a, b) for a in A for b in B]))


def within_group_js(group_hists) -> float:
    """Mean JS divergence over every WITHIN-group pair.

    The contrast that makes `mean_pairwise_js` interpretable. A separation only
    means something if the cross-group divergence exceeds the within-group
    divergence; a large cross-group number on its own can simply mean the
    clients are all far apart, cluster or no cluster.
    """
    H = np.atleast_2d(np.asarray(group_hists, dtype=np.float64))
    n = len(H)
    if n < 2:
        return 0.0
    return float(np.mean([js_divergence(H[i], H[j])
                          for i in range(n) for j in range(i + 1, n)]))


def separation_ratio(groups) -> dict:
    """Cross-group against within-group JS, for a list of per-group histogram sets.

    Returns the two means and their ratio. A ratio at or below 1 means the
    clusterer has not separated anything, whatever its silhouette says.
    """
    groups = [np.atleast_2d(np.asarray(g, dtype=np.float64)) for g in groups
              if len(g)]
    if len(groups) < 2:
        return {"cross": float("nan"), "within": float("nan"),
                "ratio": float("nan"), "n_groups": len(groups)}

    cross = [mean_pairwise_js(groups[i], groups[j])
             for i in range(len(groups)) for j in range(i + 1, len(groups))]
    within = [within_group_js(g) for g in groups if len(g) >= 2]

    c = float(np.mean(cross)) if cross else float("nan")
    w = float(np.mean(within)) if within else 0.0
    return {"cross": c, "within": w,
            "ratio": (c / w) if w > 0 else float("inf"),
            "n_groups": len(groups)}


# --------------------------------------------------------------------------- #
# Update geometry
# --------------------------------------------------------------------------- #
def summarise(values) -> dict:
    """mean / std / min / max / median of a set of values, ignoring non-finite.

    Used for both cosine similarities and update norms, which is why it is one
    function rather than a `cosine_stats` and a `norm_stats` that would drift.
    """
    v = np.asarray([x for x in np.ravel(values) if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {k: float("nan") for k in ("mean", "std", "min", "max", "median")}
    return {"mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
            "median": float(np.median(v))}


def z_score(value, reference) -> float:
    """Where `value` sits in the distribution of `reference`, in standard deviations.

    The stealth measure. An attacker inside the honest band scores near 0, one
    outside it scores large. Sign is kept because direction matters: an update
    that is unusually SIMILAR to the target cluster is as suspicious as one that
    is unusually dissimilar, and collapsing to an absolute value hides which.
    """
    r = np.asarray([x for x in np.ravel(reference) if np.isfinite(x)], dtype=np.float64)
    if r.size < 2:
        return float("nan")
    sd = float(r.std())
    return float((float(value) - float(r.mean())) / sd) if sd > 0 else 0.0


# --------------------------------------------------------------------------- #
# Comparisons
# --------------------------------------------------------------------------- #
def intervals_disjoint(alo: float, ahi: float, blo: float, bhi: float) -> bool:
    """Do two confidence intervals fail to overlap?

    DIRECTION AGNOSTIC, and that is the entire point.

    The first version of this check in the old notebook was written as
    `ahi < blo`, on the assumption that `a` was whichever one scored lower.
    That is only correct when lower is better, and it reported
    oracle 1.000 [1.000, 1.000] against blind 0.500 [0.250, 0.750] as
    "no difference". Do not write one branch per direction.
    """
    return (ahi < blo) or (bhi < alo)


def compare(a_ci, b_ci, a_name: str = "A", b_name: str = "B",
            higher_is_better: bool = True, fmt: str = "{:.3f}") -> str:
    """A sentence about two bootstrap CIs, DERIVED FROM THE NUMBERS.

    Both notebooks previously looked their numbers up correctly and then printed
    a hand-written sentence beside them, so a reversed result was reported as
    though it had confirmed the hypothesis. Every claim goes through a function
    like this one, and overlapping intervals read as "no measurable difference"
    rather than as a win for whichever mean happened to be larger.

    Each argument is the `(mean, lo, hi)` tuple `run_context.boot_ci` returns.
    """
    am, alo, ahi = a_ci
    bm, blo, bhi = b_ci
    a_txt = f"{a_name} {fmt.format(am)} [{fmt.format(alo)}, {fmt.format(ahi)}]"
    b_txt = f"{b_name} {fmt.format(bm)} [{fmt.format(blo)}, {fmt.format(bhi)}]"

    if not (np.isfinite(am) and np.isfinite(bm)):
        return f"{a_txt} against {b_txt}: not comparable, a mean is undefined"
    if not intervals_disjoint(alo, ahi, blo, bhi):
        return f"{a_txt} against {b_txt}: intervals overlap, no measurable difference"

    a_wins = (am > bm) if higher_is_better else (am < bm)
    winner, loser = (a_name, b_name) if a_wins else (b_name, a_name)
    return f"{a_txt} against {b_txt}: {winner} beats {loser}, intervals disjoint"
