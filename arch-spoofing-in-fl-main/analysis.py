"""Post-hoc analysis and reporting over FederatedServer.fit() history.

A `history` here is the list FederatedServer.fit() returns: one dict per round
with keys round, clusters, metrics (accuracy per cluster), losses, fairness,
detection, client_log. These functions turn that into the tables/plots the
report's evaluation section (aim #4) asks for: convergence, accuracy, attack
success, and cross-cluster fairness.
"""

import csv
from pathlib import Path

import sys

import matplotlib

# Force the headless backend only OUTSIDE a notebook kernel. Scripts and Docker
# runs need Agg because there is no display; inside ipykernel the inline backend
# is what embeds figures into the executed notebook, and overriding it with Agg
# makes `plt.show()` a silent no-op so every chart is missing from the output.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# The report palette, shared with run_context so figures here and figures made
# in the notebook are the same system rather than two that merely coexist.
from run_context import SERIES, INK, SECOND, MUTED, GRID


def final_accuracy(history, family):
    return history[-1]["metrics"].get(family)


def attack_success_rate(baseline_history, attack_history, family, margin=0.05):
    """Fraction of rounds where `family`'s accuracy under attack is at least
    `margin` below the same round's baseline accuracy ("corrupted" rounds)."""
    rounds = min(len(baseline_history), len(attack_history))
    corrupted = 0
    for r in range(rounds):
        base_acc = baseline_history[r]["metrics"].get(family)
        atk_acc = attack_history[r]["metrics"].get(family)
        if base_acc is None or atk_acc is None:
            continue
        if atk_acc < base_acc - margin:
            corrupted += 1
    return corrupted / rounds if rounds else 0.0


def victim_degradation(baseline_history, attack_history, family):
    """Final-round accuracy drop (baseline - attack) for `family`."""
    return final_accuracy(baseline_history, family) - final_accuracy(attack_history, family)


def client_cluster_accuracy(history, client_id):
    """Per round, the accuracy of whichever cluster `client_id` actually landed
    in that round. Architecture-mode cluster ids are fixed arch names, but
    distribution-mode cluster ids (e.g. "cluster_0") are reassigned by clustering
    each round and are not guaranteed to name the same group of clients across
    rounds -- use this instead of a hardcoded family name to track a specific
    client's cluster (e.g. a spoofer's victim) reliably in that mode."""
    out = []
    for entry in history:
        cluster_id = next((c["cluster"] for c in entry["client_log"]
                           if c["client_id"] == client_id), None)
        acc = entry["metrics"].get(cluster_id) if cluster_id is not None else None
        out.append({"round": entry["round"], "cluster": cluster_id, "accuracy": acc})
    return out


def summarize(history, family=None):
    """Headline numbers for a single run, formatted as printable text."""
    families = [family] if family else sorted(history[-1]["metrics"].keys())
    lines = [f"Rounds: {len(history)}"]
    for fam in families:
        accs = [h["metrics"].get(fam) for h in history]
        losses = [h["losses"].get(fam) for h in history]
        lines.append(f"  {fam}: final accuracy={accs[-1]:.4f} "
                     f"(started {accs[0]:.4f}), final loss={losses[-1]:.4f}")
    last_fair = history[-1].get("fairness")
    if last_fair:
        lines.append(f"  fairness: variance={last_fair['variance']:.5f} "
                     f"worst-group={last_fair['worst_group']:.4f}")
    last_det = history[-1].get("detection")
    if last_det:
        lines.append(f"  final-round detection: precision={last_det['precision']:.3f} "
                     f"recall={last_det['recall']:.3f} f1={last_det['f1']:.3f}")
    return "\n".join(lines)


def history_to_rows(history, label=""):
    """Flatten a history into one row per (round, cluster) for CSV export."""
    rows = []
    for entry in history:
        fairness = entry.get("fairness") or {}
        detection = entry.get("detection") or {}
        for family, acc in entry["metrics"].items():
            rows.append({
                "run": label,
                "round": entry["round"],
                "cluster": family,
                "accuracy": acc,
                "loss": entry["losses"].get(family),
                "fairness_variance": fairness.get("variance"),
                "fairness_worst_group": fairness.get("worst_group"),
                "detection_precision": detection.get("precision"),
                "detection_recall": detection.get("recall"),
                "detection_f1": detection.get("f1"),
            })
    return rows


def save_round_csv(histories, path):
    """histories: {label: history}. Writes one row per (run, round, cluster)."""
    rows = []
    for label, history in histories.items():
        rows.extend(history_to_rows(history, label=label))
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[analysis] wrote {len(rows)} rows -> {path}")


def save_client_csv(histories, path):
    """histories: {label: history}. Writes one row per (run, round, client),
    including declared/true architecture and defence status - the raw material
    for a confusion-matrix or per-client case study in the report."""
    rows = []
    for label, history in histories.items():
        for entry in history:
            for client in entry["client_log"]:
                rows.append({"run": label, **client})
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[analysis] wrote {len(rows)} rows -> {path}")


def alpha_summary(grid_results, family, margin=0.05):
    """grid_results: {alpha: {False: baseline_history, True: attack_history}}.
    One row per alpha isolating the causal effect of spoofing from data
    heterogeneity: baseline/attack final accuracy, the degradation attributable
    to spoofing at that alpha, and fairness either side of the attack."""
    rows = []
    for alpha in sorted(grid_results.keys()):
        base = grid_results[alpha][False]
        atk = grid_results[alpha][True]
        base_acc = final_accuracy(base, family)
        atk_acc = final_accuracy(atk, family)
        base_fair = (base[-1].get("fairness") or {}).get("variance")
        atk_fair = (atk[-1].get("fairness") or {}).get("variance")
        rows.append({
            "alpha": alpha,
            "baseline_accuracy": base_acc,
            "attack_accuracy": atk_acc,
            "degradation": base_acc - atk_acc,
            "attack_success_rate": attack_success_rate(base, atk, family, margin=margin),
            "baseline_fairness_variance": base_fair,
            "attack_fairness_variance": atk_fair,
        })
    return rows


def print_alpha_summary(grid_results, family, margin=0.05):
    rows = alpha_summary(grid_results, family, margin=margin)
    print(f"{'alpha':<8}{'baseline':<12}{'attack':<12}{'degradation':<14}"
          f"{'success%':<10}{'fair_base':<12}{'fair_atk':<12}")
    for r in rows:
        print(f"{r['alpha']:<8}{r['baseline_accuracy']:<12.4f}{r['attack_accuracy']:<12.4f}"
              f"{r['degradation']:<+14.4f}{r['attack_success_rate']:<10.0%}"
              f"{r['baseline_fairness_variance']:<12.5f}{r['attack_fairness_variance']:<12.5f}")
    return rows


def save_alpha_summary_csv(grid_results, family, path, margin=0.05):
    rows = alpha_summary(grid_results, family, margin=margin)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[analysis] wrote {len(rows)} rows -> {path}")


def save_grid_round_csv(grid_results, path):
    """grid_results: {alpha: {False: history, True: history}}. Per-round detail
    (not just the final-round summary above), one row per (alpha, spoofing,
    round, cluster)."""
    histories = {
        f"alpha={alpha}|spoof={spoofing}": history
        for alpha, by_spoof in grid_results.items()
        for spoofing, history in by_spoof.items()
    }
    save_round_csv(histories, path)


def plot_alpha_grid(grid_results, family, path, title=None,
                    xlabel="Dirichlet alpha (lower = more non-IID)"):
    """Final-round victim-cluster accuracy vs alpha, baseline and attack as two
    lines -- the clean causal comparison the alpha grid exists to produce."""
    alphas = sorted(grid_results.keys())
    base_accs = [final_accuracy(grid_results[a][False], family) for a in alphas]
    atk_accs = [final_accuracy(grid_results[a][True], family) for a in alphas]

    plt.figure(figsize=(7, 4.5))
    plt.plot(alphas, base_accs, "o-", label="baseline (no attack)")
    plt.plot(alphas, atk_accs, "x--", label="attack (spoof + poison)")
    plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(f"'{family}' cluster final accuracy")
    plt.title(title or f"Effect of data heterogeneity on the spoofing attack: '{family}'")
    plt.legend()
    plt.tight_layout()
    _save(path)


def plot_degradation_vs_alpha(grid_results, family, path, title=None,
                              xlabel="Dirichlet alpha (lower = more non-IID)"):
    """Accuracy degradation (baseline - attack) vs alpha: does the attack get
    more or less damaging as data heterogeneity changes?"""
    alphas = sorted(grid_results.keys())
    degradations = [
        final_accuracy(grid_results[a][False], family) - final_accuracy(grid_results[a][True], family)
        for a in alphas
    ]
    plt.figure(figsize=(7, 4.5))
    plt.plot(alphas, degradations, "o-", color="crimson")
    plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy degradation from spoofing (baseline - attack)")
    plt.title(title or f"Spoofing impact vs. data heterogeneity: '{family}'")
    plt.tight_layout()
    _save(path)


def plot_accuracy(histories, family, path, title=None):
    """histories: {label: history}. One accuracy-vs-round line per label."""
    plt.figure(figsize=(7, 4.5))
    for label, history in histories.items():
        rounds = [h["round"] for h in history]
        accs = [h["metrics"].get(family) for h in history]
        plt.plot(rounds, accs, marker="o", label=label)
    plt.xlabel("Communication round")
    plt.ylabel(f"'{family}' cluster accuracy")
    plt.title(title or f"Convergence: '{family}' cluster")
    plt.legend()
    plt.tight_layout()
    _save(path)


def plot_fairness(histories, path, title="Cross-cluster fairness"):
    """histories: {label: history}. Lower variance = fairer across clusters."""
    plt.figure(figsize=(7, 4.5))
    for label, history in histories.items():
        rounds = [h["round"] for h in history]
        variances = [(h.get("fairness") or {}).get("variance") for h in history]
        plt.plot(rounds, variances, marker="o", label=label)
    plt.xlabel("Communication round")
    plt.ylabel("Accuracy variance across clusters")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    _save(path)


def plot_detection(history, path, title="Spoofer detection quality"):
    rounds, precision, recall, f1 = [], [], [], []
    for h in history:
        d = h.get("detection")
        if d is None:
            continue
        rounds.append(h["round"])
        precision.append(d["precision"])
        recall.append(d["recall"])
        f1.append(d["f1"])
    if not rounds:
        return
    plt.figure(figsize=(7, 4.5))
    plt.plot(rounds, precision, marker="o", label="precision")
    plt.plot(rounds, recall, marker="s", label="recall")
    plt.plot(rounds, f1, marker="^", label="f1")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Communication round")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    _save(path)


def _save(path):
    """Write the figure to disk AND display it inline, then close it.

    Both destinations are needed. The file feeds the report; the inline copy is
    what a reader of the executed notebook actually sees, and someone sent only
    the `.ipynb` has no access to `docs/figures/`. Closing without showing first
    discards the figure, leaving a bare "saved plot" line where the chart should
    be, so `show()` has to come before `close()`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130)
    plt.show()
    plt.close()
    print(f"[analysis] saved plot -> {path}")


# =========================================================================== #
# Inferred cluster assignment: does the server see the structure at all?
# =========================================================================== #
# Everything above analyses a run in which the server was TOLD which cluster
# each client belongs to. These analyse a run in which the server INFERS it from
# the submitted updates, which is what Clustered Federated Learning (CFL)
# actually does in the literature.
#
# The first question is not "how well does the attack work" but "can the server
# recover the true grouping when nobody is attacking at all". If it cannot, then
# an infiltration rate measures nothing: an attacker cannot be said to have
# broken into a cluster that the server was never able to form.
#
# `clustering_signal` is the function that answers it, and it answers it against
# GROUND TRUTH rather than against the clusterer's own confidence. A clusterer
# that produces confident, well-separated, entirely arbitrary groups is the
# failure mode here, and it is invisible to any metric computed from the
# clusterer's output alone. That is why Jensen-Shannon (JS) separation between
# recovered groups is reported alongside, never instead.

def clustering_signal(deltas, groups, index=None) -> dict:
    """Can the server's similarity matrix tell group-mates from strangers?

    `deltas` is the (n_clients, n_params) matrix of updates measured against the
    shared global model. `groups` is the PLANTED group of each client, which the
    server never sees and the analysis is allowed to. `index` restricts the
    similarity to a subset of parameters, normally `models.head_indices`, and
    must match whatever scope the clusterer itself is using.

    GROUND TRUTH IS PLANTED MEMBERSHIP, NOT LABEL HISTOGRAMS. Scoring the
    similarity matrix against Jensen-Shannon (JS) divergence between label
    histograms only works for label-DISTRIBUTION shift. The Clustered Federated
    Learning (CFL) literature plants clusters by CONCEPT shift (a different
    label permutation or rotation per group), under which every group has an
    identical label marginal by construction, so a histogram-based target matrix
    is ~0 everywhere and carries no information. Planted membership is the
    ground truth that holds whichever way the clusters were created.

    Returns the area under the ROC curve (AUC) for separating same-group pairs
    from different-group pairs by their similarity:

        0.5   chance. The similarity matrix carries no information about who
              belongs together, so any grouping built from it is arbitrary
              however confident or well separated it looks.
        1.0   perfect. Every within-group pair is more similar than every
              cross-group pair.

    AUC rather than a raw cosine mean because it needs no baseline to interpret,
    and rather than a rank correlation because the ground truth here is binary.
    `within` and `cross` are returned alongside so this agrees with
    `steering_geometry`, which reports the same two quantities.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from aggregation import cosine_matrix

    D = np.asarray(deltas, dtype=np.float64)
    if index is not None:
        D = D[:, np.asarray(index)]
    g = np.asarray(groups)
    n = len(g)
    if n < 3:
        return {"auc": float("nan"), "n": n,
                "note": "need at least 3 clients"}

    S = cosine_matrix(D)
    iu = np.triu_indices(n, 1)
    same = (g[iu[0]] == g[iu[1]]).astype(int)
    sims = S[iu]

    # Degenerate when every client is in one planted group, or all in different
    # ones: AUC is undefined with a single class present.
    auc = (float(roc_auc_score(same, sims)) if 0 < same.sum() < len(same)
           else float("nan"))
    within = sims[same == 1]
    cross = sims[same == 0]

    return {"auc": auc,
            "within": float(within.mean()) if len(within) else float("nan"),
            "cross": float(cross.mean()) if len(cross) else float("nan"),
            "margin": (float(within.mean() - cross.mean())
                       if len(within) and len(cross) else float("nan")),
            "n": n, "scope": "head" if index is not None else "full"}


def signal_over_rounds(history, groups, index=None) -> "pd.DataFrame":
    """`clustering_signal` per round.

    The round axis matters and the direction is not something to assume: on
    label-distribution shift the signal ROSE with training, on planted concept
    shift it DECAYED. Measure it rather than quoting a single round, and say
    which round any headline number came from.

    Pass `index=clusterer.head_index` to measure what the server actually
    decides on. Passing nothing measures the full parameter vector, which under
    concept shift is near chance because the disagreement is confined to the
    classifier head.
    """
    import pandas as pd

    rows = []
    for entry in history:
        D = entry.get("deltas")
        if D is None:
            continue
        s = clustering_signal(D, groups, index=index)
        rows.append({"round": entry["round"],
                     **{k: s[k] for k in ("auc", "within", "cross", "margin", "scope")}})
    return pd.DataFrame(rows)


def recovery_over_rounds(history, groups) -> "pd.DataFrame":
    """How well the recovered clustering matches the planted one, per round.

    Scored by the Adjusted Rand Index (ARI), the standard chance-corrected
    measure for comparing a clustering against a known ground truth:

        0.0   no better than a random partition of the same shape
        1.0   identical to the planted grouping, up to relabelling

    ARI compares two partitions directly, which is what makes it indifferent to
    how the heterogeneity was created. A cross-over-within JS ratio computed
    from label histograms cannot do that job under concept shift, because those
    histograms are identical across groups and the ratio sits at ~1 however good
    or bad the clustering is.

    ARI rather than silhouette because silhouette needs no ground truth: forcing
    a handful of points into two groups always separates something, so it scores
    an arbitrary partition as healthy. ARI is measured against the planted
    grouping and therefore cannot flatter one.

    `n_clusters` is reported beside it because ARI alone hides fragmentation: a
    clusterer that shatters into singletons and one that merges everything can
    both score 0.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score

    g = np.asarray(groups)
    rows = []
    for entry in history:
        members = [m for m in entry["membership"].values() if len(m)]
        labels = np.full(len(g), -1)
        for cid, ms in enumerate(members):
            for m in ms:
                labels[m] = cid
        seen = labels >= 0
        rows.append({"round": entry["round"], "n_clusters": len(members),
                     "sizes": sorted(len(m) for m in members),
                     "ari": float(adjusted_rand_score(g[seen], labels[seen]))
                     if seen.any() else float("nan")})
    return pd.DataFrame(rows)


# Retained under its old name so existing callers fail loudly rather than
# silently receiving a meaningless number. See F_ATTEMPTS.md C6.
def separation_over_rounds(history, true_hists):
    raise NotImplementedError(
        "separation_over_rounds scored recovered groups by Jensen-Shannon (JS) "
        "divergence between their label histograms, which is only meaningful "
        "for label-DISTRIBUTION shift. Under the concept shift the Clustered "
        "Federated Learning literature uses, every group has an identical label "
        "marginal and the ratio sits at ~1 regardless of clustering quality. "
        "Use recovery_over_rounds(history, bundle.groups), which scores the "
        "partition directly with the Adjusted Rand Index (ARI).")


def infiltration_over_rounds(history, target_members, attacker: int) -> "pd.DataFrame":
    """Did the attacker land in the same cluster as the majority of the target?

    Defined by MEMBERSHIP, never by cluster name. Cluster identifiers are
    reassigned every round, and under recursive bipartition the number of
    clusters changes too, so `cluster_0` in round 2 is not the group that was
    `cluster_0` in round 1. Tracking the name instead of the members produces an
    infiltration series that looks like the attack switching on and off.
    """
    import pandas as pd

    rows = []
    for entry in history:
        cluster_of = {}
        for cid, members in entry["membership"].items():
            for m in members:
                cluster_of[m] = cid

        counts = {}
        for m in target_members:
            cid = cluster_of.get(m)
            if cid is not None:
                counts[cid] = counts.get(cid, 0) + 1
        target_cluster = max(counts, key=counts.get) if counts else None

        landed = cluster_of.get(attacker)
        with_attacker = [m for m in target_members if cluster_of.get(m) == landed]
        rows.append({
            "round": entry["round"],
            "attacker_cluster": landed,
            "target_cluster": target_cluster,
            "infiltrated": bool(landed is not None and landed == target_cluster),
            "target_members_with_attacker": len(with_attacker),
            "target_size": len(target_members),
            "share_of_target": len(with_attacker) / max(len(target_members), 1),
        })
    return pd.DataFrame(rows)


def target_similarity(deltas, client_ids, groups, attacker: int,
                      target_group: int, index=None, home_group=None) -> dict:
    """Does the attacker's update actually RESEMBLE the group it is placed in?

    The mechanism check, and it is separate from whether the attack succeeded.
    Infiltration is an OUTCOME: it says the server put the attacker in the
    target cluster. It does not say the attacker's update looked like a member
    of that cluster, and a right answer from a wrong cause is this project's
    signature failure. If placement is being driven by something other than the
    similarity we claim, every conclusion built on it is unsafe.

    Returns the attacker's mean cosine to the TARGET group and to its own HOME
    group, alongside the honest within-group and cross-group bands measured over
    the other clients only. A successful spoof by the claimed mechanism should
    show `to_target` sitting up in the within-group band and `to_home` down in
    the cross-group band, i.e. the attacker now resembles strangers less than it
    resembles the group it does not belong to.

    `index` must match the clusterer's scope, normally `models.head_indices`.

    `home_group` matters as soon as K > 2. Without it, `to_home` is computed as
    "everything that is not the target", which at K=2 IS the home group but at
    K>2 pools the attacker's real home with every other bystander group. Those
    are different quantities: a spoof should pull the attacker away from its
    OWN group specifically, not away from all groups at once. Pass it whenever
    the group count exceeds two. Omitted, the old behaviour is kept so existing
    callers are unaffected.
    """
    import numpy as np

    from aggregation import cosine_matrix

    D = np.asarray(deltas, dtype=np.float64)
    if index is not None:
        D = D[:, np.asarray(index)]
    g = np.asarray(groups)
    pos = {c: i for i, c in enumerate(client_ids)}
    S = cosine_matrix(D)

    others = [c for c in client_ids if c != attacker]
    to_target = [S[pos[attacker], pos[c]] for c in others if g[c] == target_group]
    if home_group is None:
        to_home = [S[pos[attacker], pos[c]] for c in others if g[c] != target_group]
    else:
        to_home = [S[pos[attacker], pos[c]] for c in others if g[c] == home_group]

    # Honest bands exclude the attacker entirely, so they describe what the
    # federation looks like without it rather than being shifted by it.
    within, cross = [], []
    for a in others:
        for b in others:
            if a >= b:
                continue
            (within if g[a] == g[b] else cross).append(S[pos[a], pos[b]])

    out = {"to_target": float(np.mean(to_target)) if to_target else float("nan"),
           "to_home": float(np.mean(to_home)) if to_home else float("nan"),
           "honest_within": float(np.mean(within)) if within else float("nan"),
           "honest_cross": float(np.mean(cross)) if cross else float("nan"),
           "scope": "head" if index is not None else "full"}
    # Where the attacker sits between the two honest bands. 1.0 means it looks
    # exactly like a target group-mate, 0.0 like a stranger to them.
    span = out["honest_within"] - out["honest_cross"]
    out["target_resemblance"] = (float((out["to_target"] - out["honest_cross"]) / span)
                                 if span > 1e-12 else float("nan"))
    return out


def steering_geometry(deltas, client_ids, groups, attacker_self_cosine=None,
                      index=None) -> dict:
    """Where an attacker's steering sits relative to HONEST client variation.

    A cosine similarity is uninterpretable on its own. Reporting that a steered
    update is 0.991 similar to the honest one it replaced sounds like an
    imperceptible nudge, but every client delta in this federation sits above
    0.97 similarity to every other, so 0.991 could equally be a large move. The
    number only becomes a claim once it is placed against the honest baseline,
    exactly as a detection rate only becomes a claim beside a false-alarm rate.

    Returns the two honest reference distributions and, if given the attacker's
    honest-to-steered cosine, where it falls between them:

      within    mean cosine between honest clients in the SAME planted group.
                How alike two legitimate group-mates look.
      cross     mean cosine between honest clients in DIFFERENT planted groups.
                How unalike two legitimate strangers look.
      traversal the attacker's displacement as a fraction of the within-to-cross
                gap. 0 means it did not move; 1 means it moved as far as a
                typical stranger sits.

    `within` and `cross` also measure whether the clusterer has any signal at
    all: if their confidence intervals overlap, the server cannot separate
    group-mates from strangers and nothing built on top of it means anything.

    `index` restricts the similarity to a subset of parameters and MUST match
    the scope the clusterer decides on, normally `models.head_indices`. Reporting
    full-vector geometry while the server clusters on the head is not a small
    discrepancy: measured on label-permuted data the within-minus-cross margin
    is 0.0001 on the full vector and 0.875 on the head, so the full-vector view
    shows no structure at all where the server sees a clean separation.
    """
    import numpy as np

    from aggregation import cosine_matrix

    D = np.asarray(deltas, dtype=np.float64)
    if index is not None:
        D = D[:, np.asarray(index)]
    groups = np.asarray(groups)
    pos = {c: i for i, c in enumerate(client_ids)}
    S = cosine_matrix(D)

    within, cross = [], []
    for a in client_ids:
        for bb in client_ids:
            if a >= bb:
                continue
            (within if groups[a] == groups[bb] else cross).append(S[pos[a], pos[bb]])

    out = {"within": float(np.mean(within)) if within else float("nan"),
           "cross": float(np.mean(cross)) if cross else float("nan"),
           "within_min": float(np.min(within)) if within else float("nan"),
           "n_within": len(within), "n_cross": len(cross),
           "scope": "head" if index is not None else "full"}

    if attacker_self_cosine is not None and np.isfinite(out["within"]):
        gap = out["within"] - out["cross"]
        out["attacker_self_cosine"] = float(attacker_self_cosine)
        out["traversal"] = (float((out["within"] - attacker_self_cosine) / gap)
                            if gap > 0 else float("nan"))
    return out


# =========================================================================== #
# Declared-channel inspection
# =========================================================================== #
# Adapted from the CNN branch's `cluster_viz.py`, which inspects the declared
# label-histogram channel WITHOUT training anything. Three reasons it earns a
# place here rather than staying in that branch:
#
#   1. It is the cheapest possible check on tier 1 of the spoofing ladder: it
#      answers "could a server cluster on the declared histograms at all?" in
#      milliseconds, with no federation to run.
#   2. Under concept shift the answer is visually striking. Every group has a
#      near-identical label marginal by construction, so the heatmap shows flat
#      uniform rows where a Dirichlet partition shows sharp blocks. That picture
#      IS the argument for why the server has to infer rather than read.
#   3. It gives the write-up a figure of the data itself, not only of results.

# =========================================================================== #
# Baselines: what each statistic reads when nothing is happening
# =========================================================================== #
# A measured value is uninterpretable on its own. 0.665 is either good or
# useless depending on what a null result looks like, and the null differs per
# statistic: some are chance-corrected so their floor is 0, some have a floor of
# 1/K, and some have no analytic floor at all and must be measured.
#
# This table is the reference. `compare_to_baseline` formats any statistic
# against it so no number in a write-up appears without its norm beside it.

BASELINES = {
    "ari": dict(
        null=0.0, name="random partition of the same shape",
        note="chance-corrected by construction, so a random grouping scores 0, "
             "not 0.5. Everyone-in-one-cluster also scores 0.",
        higher_is_better=True),
    "auc": dict(
        null=0.5, name="chance",
        note="0.5 means the similarity ordering carries no information about "
             "who belongs together.",
        higher_is_better=True),
    "separation_ratio": dict(
        null=1.0, name="groups no more different than their own members",
        note="the ratio of cross-group to within-group divergence. At 1.0 the "
             "two groups are indistinguishable; the MEASURED median over all "
             "random splits is the honest null and sits near 0.99.",
        higher_is_better=True),
    "infiltration": dict(
        null=None, name="the measured honest control",
        note="NOT 1/K. An honest client is not placed at random: a working "
             "clusterer puts it in its own group, so its infiltration is 0. "
             "Where the control is above 0 that is the environment failing to "
             "recover the grouping, and every attacked number must be read "
             "against that rather than against zero.",
        higher_is_better=True),
    "estimation_error": dict(
        null=None, name="blind, i.e. assume uniform and never probe",
        note="an estimator that cannot beat 'assume uniform' is extracting "
             "nothing from the model. Lower is better here.",
        higher_is_better=False),
    "accuracy": dict(
        null=None, name="the no-attack control on the same setup",
        note="the chance floor is 1/num_classes, but the meaningful reference "
             "is what this federation reaches with no attacker present.",
        higher_is_better=True),
}


def compare_to_baseline(stat: str, value: float, baseline: float = None,
                        interval=None, baseline_interval=None) -> dict:
    """One statistic beside its norm, with a verdict that survives either direction.

    `baseline` overrides the analytic null for statistics that have none, such
    as infiltration and accuracy, where the reference has to be measured.

    The verdict is direction-agnostic: two intervals are called different only
    when one lies entirely outside the other. Writing a test that assumes higher
    is better silently reverses on error-style statistics, which has produced a
    wrong "no difference" verdict in this project before.
    """
    spec = BASELINES.get(stat, dict(null=None, name="unspecified",
                                    note="", higher_is_better=True))
    null = spec["null"] if baseline is None else baseline
    out = {"statistic": stat, "measured": value, "baseline": null,
           "baseline_is": spec["name"]}

    if null is None:
        out["verdict"] = "no baseline supplied"
        return out

    gap = value - null
    better = gap > 0 if spec["higher_is_better"] else gap < 0
    out["gap"] = gap

    if interval is not None and baseline_interval is not None:
        lo, hi = interval
        blo, bhi = baseline_interval
        separated = (hi < blo) or (bhi < lo)
        out["verdict"] = ("above baseline" if separated and better else
                          "below baseline" if separated else
                          "not distinguishable from baseline")
    else:
        out["verdict"] = "above baseline" if better else "at or below baseline"
    return out


def cluster_by_label_hist(hists, n_clusters: int = 2):
    """Cluster clients on their declared label histograms, with no training.

    Returns `(labels, assignment)` where `labels[i]` is client i's cluster index
    and `assignment` maps cluster id to member list. Uses the same
    `DistributionClusterer` the federation would use, so this is what a tier-1
    server would actually decide, not an approximation of it.
    """
    import numpy as np

    from client import ClientUpdate
    from clustering import DistributionClusterer

    H = np.asarray(hists, dtype=np.float64)
    updates = [ClientUpdate(client_id=i, weights=[], metadata={"label_hist": H[i]})
               for i in range(len(H))]
    groups = DistributionClusterer(n_clusters=n_clusters).cluster(updates)

    assignment = {cid: [u.client_id for u in ups] for cid, ups in groups.items()}
    labels = np.zeros(len(H), dtype=int)
    for k, (_, ids) in enumerate(sorted(assignment.items())):
        for i in ids:
            labels[i] = k
    return labels, assignment


def plot_label_histograms(hists, path, groups=None, title=None):
    """Heatmap of each client's label distribution, with a planted-group strip.

    Rows are clients, columns are classes, and the narrow strip on the left
    marks which group each client truly belongs to. Under concept shift the
    rows look alike and the strip looks arbitrary beside them, which is the
    point: the declared channel does not separate the groups the way the
    underlying task does.
    """
    import numpy as np

    H = np.asarray(hists, dtype=np.float64)
    has_strip = groups is not None
    fig, axes = plt.subplots(
        1, 2 if has_strip else 1, figsize=(7.5, 4.2),
        gridspec_kw={"width_ratios": [1, 14]} if has_strip else None,
        squeeze=False)

    if has_strip:
        g = np.asarray(groups).reshape(-1, 1)
        axes[0][0].imshow(g, aspect="auto", cmap="tab10", interpolation="nearest")
        axes[0][0].set_xticks([])
        axes[0][0].set_ylabel("client")
        axes[0][0].set_title("group", fontsize=9)

    ax = axes[0][-1]
    im = ax.imshow(H, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("class")
    if not has_strip:
        ax.set_ylabel("client")
    else:
        ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="share of client's data")
    fig.suptitle(title or "Declared label histogram per client")
    fig.tight_layout()
    _save(path)


def plot_label_mass(hists, path, labels=None, title=None):
    """Stacked label mass per client, coloured by class.

    The companion to the heatmap: the heatmap shows the pattern, this shows the
    composition. `labels` (a cluster assignment) is used only to order the bars,
    so clients the server grouped together stand side by side and any block
    structure is visible without reading the axis.
    """
    import numpy as np

    H = np.asarray(hists, dtype=np.float64)
    order = np.arange(len(H)) if labels is None else np.argsort(np.asarray(labels),
                                                                kind="stable")
    H = H[order]

    plt.figure(figsize=(7.5, 4))
    bottom = np.zeros(len(H))
    for c in range(H.shape[1]):
        plt.bar(range(len(H)), H[:, c], bottom=bottom, width=0.85,
                label=f"class {c}" if H.shape[1] <= 10 else None)
        bottom += H[:, c]
    plt.xticks(range(len(H)), [str(i) for i in order], fontsize=8)
    plt.xlabel("client" + ("" if labels is None else ", ordered by assigned cluster"))
    plt.ylabel("share of client's data")
    plt.title(title or "Label mass per client")
    if H.shape[1] <= 10:
        plt.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout()
    _save(path)


def mean_cluster_accuracy(history) -> "pd.DataFrame":
    """Per round, the mean and spread of accuracy across clusters.

    The CNN branch's benchmark summary reports `mean_acc` across clusters, which
    is the right single number for a NO-ATTACK baseline: it asks whether the
    federation as a whole is learning. It is the wrong number once an attacker is
    present, because averaging a healthy cluster with a destroyed one hides the
    destruction, so the spread and the minimum are reported beside it.
    """
    import numpy as np
    import pandas as pd

    rows = []
    for e in history:
        vals = [v for v in (e.get("metrics") or {}).values() if np.isfinite(v)]
        rows.append({"round": e["round"], "n_clusters": len(vals),
                     "mean_acc": float(np.mean(vals)) if vals else float("nan"),
                     "min_acc": float(np.min(vals)) if vals else float("nan"),
                     "spread": float(np.max(vals) - np.min(vals)) if vals else float("nan")})
    return pd.DataFrame(rows)


# =========================================================================== #
# Headline figures
# =========================================================================== #
# Each of these compares two measures that mean different things (did the
# attacker get in; what happened to the victim). They go in SEPARATE PANELS
# rather than on twin y-axes: a dual-axis chart lets the reader infer a
# relationship from where two lines happen to cross, which is an artefact of the
# two scales chosen rather than anything in the data.
#
# The control is drawn in grey in every figure. It is not a result, it is the
# thing every result is read against, and colouring it like the treatments
# invites the eye to compare attack strengths to each other instead of to it.

_CONTROL_GREY = "#898781"


def _bar_labels(ax, bars, values, fmt="{:.3f}", pad=0.012):
    """Direct-label every bar, in ink rather than in the series colour.

    The palette's green sits below 3:1 against white, so the colour guidance
    requires relief wherever it appears: visible labels or a table. Every figure
    here gets both.
    """
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + pad, bar.get_y() + bar.get_height() / 2,
                fmt.format(v), va="center", ha="left", fontsize=9, color=INK)


def plot_headline_grid(df, path, title=None):
    """Placement and damage per condition, five seeds.

    `df` is the grid5 table. Two panels because infiltration and accuracy are
    different quantities that happen to share a 0 to 1 range; putting them on one
    axis would imply they are comparable.
    """
    import numpy as np

    agg = (df.groupby("condition")
             .agg(infiltration=("infiltration", "mean"),
                  victim=("victim_final", "mean")).reset_index())
    # Control first, then the rest by how much damage they did: the reader
    # should meet the baseline before the treatments.
    agg["_order"] = np.where(agg.condition == "control", -1, agg["victim"])
    agg = agg.sort_values("_order", ascending=False)
    colours = [_CONTROL_GREY if c == "control" else SERIES[0]
               for c in agg.condition]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    ypos = np.arange(len(agg))

    b1 = axes[0].barh(ypos, agg["infiltration"], color=colours, height=0.68)
    axes[0].set_title("Did the attacker get in?")
    axes[0].set_xlabel("infiltration rate")
    axes[0].set_xlim(0, 1.18)
    axes[0].set_yticks(ypos, agg["condition"])
    _bar_labels(axes[0], b1, agg["infiltration"].values)

    b2 = axes[1].barh(ypos, agg["victim"], color=colours, height=0.68)
    axes[1].set_title("What happened to the victim cluster?")
    axes[1].set_xlabel("final accuracy")
    axes[1].set_xlim(0, 1.18)
    _bar_labels(axes[1], b2, agg["victim"].values)

    fig.suptitle(title or "Placement and damage, five seeds (grey = control)")
    fig.tight_layout()
    _save(path)


def plot_aggregator_outcome(df, path, title=None,
                            attacked_label="relabel + boost2",
                            acc_col="victim_final"):
    """Placement is invariant across aggregators; damage is not.

    The two panels carry the whole finding: the left one is deliberately flat,
    and that flatness is the result.

    `attacked_label` and `acc_col` are parameters because the same figure is
    drawn for two different studies. Under concept shift the attacked condition
    is `relabel + boost2` and damage is read on the victim cluster; under
    distribution shift it is `resample + boost x2` read on overall accuracy,
    since the planted groups there differ in label mass rather than in concept
    and there is no separate victim task to score.
    """
    import numpy as np

    atk = df[df.condition == attacked_label]
    ctrl = df[df.condition == "control"].groupby("aggregator")[acc_col].mean()
    agg = (atk.groupby("aggregator")
              .agg(infiltration=("infiltration", "mean"),
                   victim=(acc_col, "mean")).reset_index())
    # Accuracy LOST against each rule's own control, rather than accuracy
    # remaining. The quantity of interest is how much the attack destroyed, so
    # the rule that fails is the long bar and the rules that hold are short
    # ones. Each rule is its own baseline because they do not all reach the same
    # accuracy when nobody is attacking.
    agg["lost"] = [max(ctrl.get(a, np.nan) - v, 0.0)
                   for a, v in zip(agg["aggregator"], agg["victim"])]
    agg = agg.sort_values("lost")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    ypos = np.arange(len(agg))

    b1 = axes[0].barh(ypos, agg["infiltration"], color=SERIES[0], height=0.68)
    axes[0].set_title("Placement: identical under every rule")
    axes[0].set_xlabel("infiltration rate")
    axes[0].set_xlim(0, 1.18)
    axes[0].set_yticks(ypos, agg["aggregator"])
    _bar_labels(axes[0], b1, agg["infiltration"].values)

    b2 = axes[1].barh(ypos, agg["lost"], color=SERIES[1], height=0.68)
    axes[1].set_title("Damage: only one rule lets it through")
    axes[1].set_xlabel("victim accuracy lost, against that rule's own control")
    axes[1].set_xlim(0, 1.18)
    _bar_labels(axes[1], b2, agg["lost"].values)

    fig.suptitle(title or "Robust aggregation contains damage but not intrusion")
    fig.tight_layout()
    _save(path)


def plot_knowledge_ladder(df, path, title=None):
    """How placement and resemblance decay as the attacker's knowledge worsens.

    Plotted against FIDELITY, not swap count: transpositions can cancel, so five
    swaps can leave the same fidelity as three and the swap axis is not monotone.

    Each cluster count is read against ITS OWN measured control, drawn as a
    dashed rule in the same colour. At K=3 that control is 0.333 rather than 0,
    because the server does not recover three planted groups cleanly, and
    comparing the K=3 curve to zero would overstate the attack.
    """
    import numpy as np

    attacked = df[df.condition != "control"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    # Swap count is the experimental level; fidelity is what it achieved. Each
    # seed draws its own permutation, so one swap count lands at a slightly
    # different fidelity per seed. Averaging within a swap level and plotting at
    # the mean fidelity keeps one point per level per K, which is what makes the
    # curve a seed-average rather than a scatter of individual runs.
    for i, k in enumerate(sorted(df["n_groups"].unique())):
        sub = (attacked[attacked.n_groups == k]
               .groupby("swaps")
               .agg(fidelity=("fidelity", "mean"),
                    infiltration=("infiltration", "mean")).reset_index()
               .sort_values("fidelity", ascending=False))
        axes[0].plot(sub["fidelity"], sub["infiltration"], "o-",
                     color=SERIES[i], markersize=7, label=f"K = {k}")
        ctrl = df[(df.n_groups == k) & (df.condition == "control")]["infiltration"].mean()
        axes[0].axhline(ctrl, color=SERIES[i], ls=":", lw=1.4)
        axes[0].text(0.135, ctrl + 0.03, f"K={k} control {ctrl:.2f}",
                     fontsize=8, color=SECOND)

    axes[0].set_xlabel("fidelity of the attacker's copy of the target concept")
    axes[0].set_ylabel("infiltration rate")
    axes[0].set_title("Placement decays gradually")
    axes[0].invert_xaxis()          # knowledge worsens left to right
    axes[0].set_ylim(-0.05, 1.1)
    axes[0].legend(fontsize=9)

    res = (attacked.groupby("fidelity")["target_resemblance"].mean()
           .reset_index().sort_values("fidelity", ascending=False))
    axes[1].plot(res["fidelity"], res["target_resemblance"], "o-",
                 color=SERIES[2], markersize=7)
    axes[1].axhline(0, color=GRID, lw=1)
    axes[1].set_xlabel("fidelity of the attacker's copy of the target concept")
    axes[1].set_ylabel("resemblance to the target group")
    axes[1].set_title("And so does resemblance, smoothly")
    axes[1].invert_xaxis()

    fig.suptitle(title or "How much must the spoofer know?")
    fig.tight_layout()
    _save(path)


def plot_confound_panel(df, path, title=None):
    """Placement AND resemblance against group count: spoof, or elimination?

    The figure exists to settle one question. At K=2 an attacker evicted from
    its home group has exactly one other place to go, so high infiltration is
    consistent with two different stories, and the left panel alone cannot
    separate them.

    The right panel is what separates them. Elimination moves a client AWAY from
    its own group without making it look like anyone in particular, so
    resemblance would stay flat or negative. Imitation raises it. Reading the
    two panels together is the whole point, which is why they are one figure.

    Each K is drawn against ITS OWN measured control, as a dashed rule in the
    same colour, because the honest control is not zero everywhere: at higher K
    the server misplaces honest clients too, and comparing to zero would credit
    the attack for the environment's own failures.
    """
    import numpy as np

    ks = sorted(df["n_groups"].unique())
    atk = df[~df["is_control"].astype(bool)]
    ctl = df[df["is_control"].astype(bool)]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

    a_inf = atk.groupby("n_groups")["infiltration"].mean()
    c_inf = ctl.groupby("n_groups")["infiltration"].mean()
    axes[0].plot(ks, [a_inf[k] for k in ks], "o-", color=SERIES[0],
                 markersize=8, label="attacked", zorder=3)
    axes[0].plot(ks, [c_inf[k] for k in ks], "s--", color=_CONTROL_GREY,
                 markersize=7, label="honest control", zorder=2)
    # 1/K is what pure chance would give, and is NOT the baseline -- the
    # measured control is. It is drawn only to show the attacked curve is not
    # tracking it, which is the elimination reading's prediction.
    axes[0].plot(ks, [1.0 / k for k in ks], ":", color=SECOND, lw=1.3,
                 label="1/K (chance, not the baseline)")
    axes[0].set_xlabel("number of planted groups, K")
    axes[0].set_ylabel("infiltration rate")
    axes[0].set_title("Placement holds as destinations multiply")
    axes[0].set_ylim(-0.05, 1.12)
    axes[0].set_xticks(ks)
    axes[0].legend(fontsize=8)

    a_res = atk.groupby("n_groups")["target_resemblance"].mean()
    c_res = ctl.groupby("n_groups")["target_resemblance"].mean()
    axes[1].plot(ks, [a_res[k] for k in ks], "o-", color=SERIES[1],
                 markersize=8, label="attacked", zorder=3)
    axes[1].plot(ks, [c_res[k] for k in ks], "s--", color=_CONTROL_GREY,
                 markersize=7, label="honest control", zorder=2)
    axes[1].axhline(0, color=GRID, lw=1.2)
    axes[1].set_xlabel("number of planted groups, K")
    axes[1].set_ylabel("resemblance to the target group")
    axes[1].set_title("And the attacker genuinely looks like a member")
    axes[1].set_xticks(ks)
    axes[1].legend(fontsize=8)

    fig.suptitle(title or "Imitation, not elimination")
    fig.tight_layout()
    _save(path)


def plot_structure_envelope(df, path, title=None):
    """Where planted groups exist at all, against the Dirichlet ceiling.

    `df` is the structure sweep: one row per (partition, K, concentration,
    overlap) carrying `best_split`, the separation ratio.

    The Dirichlet rows are drawn as a shaded BAND rather than as another line,
    because they are not a competing setting -- they are the ceiling that the
    standard non-IID generator cannot exceed, and any planted curve sitting
    above it is the argument for planting groups in the first place. The band's
    top is the best two-way split found anywhere in Dirichlet data by exhaustive
    search, so nothing can be dismissed as a clusterer failing to look hard
    enough.

    A ratio of 1.0 is the no-structure null and gets its own rule.
    """
    import numpy as np

    planted = df[df.partition == "clustered"]
    dirich = df[df.partition == "dirichlet"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

    # Left: overlap is the difficulty knob most directly under our control.
    base_conc = 20.0
    sub = planted[planted.concentration == base_conc]
    for i, k in enumerate(sorted(sub["n_groups"].unique())):
        s = sub[sub.n_groups == k].sort_values("overlap")
        axes[0].plot(s["overlap"], s["best_split"], "o-", color=SERIES[i % len(SERIES)],
                     markersize=6, label=f"K = {int(k)}")
    axes[0].set_xlabel("overlap (mass each group shares with all classes)")
    axes[0].set_ylabel("separation ratio")
    axes[0].set_title(f"Planted structure, concentration {base_conc:g}")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8, ncol=2)

    # Right: concentration, at the operating overlap.
    sub2 = planted[planted.overlap == 0.3]
    for i, k in enumerate(sorted(sub2["n_groups"].unique())):
        s = sub2[sub2.n_groups == k].sort_values("concentration")
        axes[1].plot(s["concentration"], s["best_split"], "o-",
                     color=SERIES[i % len(SERIES)], markersize=6, label=f"K = {int(k)}")
    axes[1].set_xlabel("concentration (how tightly clients hug their profile)")
    axes[1].set_title("Planted structure, overlap 0.3")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")

    if len(dirich):
        top = float(dirich["best_split"].max())
        for ax in axes:
            ax.axhspan(0.9, top, color=_CONTROL_GREY, alpha=0.22, zorder=0)
            ax.axhline(1.0, color=GRID, lw=1.2, ls="--")
        axes[0].text(0.02, top * 1.04,
                     f"Dirichlet ceiling {top:.2f} — no grouping exists above this",
                     fontsize=8, color=SECOND)

    fig.suptitle(title or "Which settings contain groups at all?")
    fig.tight_layout()
    _save(path)


def plot_estimation_error(df, path, title=None):
    """How well the attacker can work out the target's label distribution.

    `df` carries one row per seed per knowledge level, with `error_vs_truth` as
    Jensen-Shannon divergence between the estimate and the truth. LOWER IS
    BETTER, which inverts the usual reading, so the axis says so and `oracle`
    sits at the short end.

    `blind` -- assume uniform, probe nothing -- is drawn as a dashed rule across
    the whole axis rather than as just another bar. It is not a competing
    method, it is the line an estimator has to beat to be doing anything at all,
    and a bar reaching that line is a negative result.

    Whiskers are bootstrap intervals over seeds. Each seed plants its own
    profiles, so a gap smaller than the whiskers is a draw, not a win.
    """
    import numpy as np
    from run_context import boot_ci

    order = [m for m in ("oracle", "inferred", "blind")
             if m in set(df["knowledge"])]
    order += [m for m in sorted(set(df["knowledge"])) if m not in order]

    means, los, his = [], [], []
    for mode in order:
        m, lo, hi = boot_ci(df.loc[df.knowledge == mode, "error_vs_truth"])
        means.append(m); los.append(m - lo); his.append(hi - m)

    blind = means[order.index("blind")] if "blind" in order else None
    colours = [_CONTROL_GREY if m == "blind" else SERIES[0] for m in order]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ypos = np.arange(len(order))
    bars = ax.barh(ypos, means, xerr=[los, his], color=colours, height=0.6,
                   error_kw={"ecolor": INK, "elinewidth": 1.2, "capsize": 4})
    ax.set_yticks(ypos, order)
    ax.invert_yaxis()
    ax.set_xlabel("estimation error against the truth "
                  "(Jensen-Shannon divergence, lower is better)")
    ax.set_xlim(0, max(np.array(means) + np.array(his)) * 1.35)
    _bar_labels(ax, bars, means, fmt="{:.4f}")

    if blind is not None:
        ax.axvline(blind, color=_CONTROL_GREY, ls="--", lw=1.4)
        ax.text(blind, -0.72, " blind baseline", fontsize=8, color=SECOND,
                va="center", ha="left")

    fig.suptitle(title or "What can the attacker learn from the broadcast model?")
    fig.tight_layout()
    _save(path)


def plot_clustering_signal(table, path, title=None):
    """Separability of group-mates from strangers, against round.

    The headline diagnostic for whether inferred clustering can work at all in
    this setting. Pass a table from `signal_over_rounds`; if it holds more than
    one `scope`, both are drawn, which is the figure that shows the head carries
    the signal and the full vector does not.

    The chance line at 0.5 is the point of the figure. An AUC hovering there
    means the server is guessing, whatever the recovered clusters look like.
    """
    plt.figure(figsize=(7, 4.5))
    if "scope" in table.columns and table["scope"].nunique() > 1:
        for label, sub in table.groupby("scope"):
            sub = sub.sort_values("round")
            plt.plot(sub["round"], sub["auc"], "o-" if label == "head" else "s--",
                     label=f"{label} parameter vector")
        plt.legend()
    else:
        sub = table.sort_values("round")
        plt.plot(sub["round"], sub["auc"], "o-")
    plt.axhline(0.5, color="#898781", lw=1, ls=":")
    plt.text(0.01, 0.505, "chance", color="#898781", fontsize=9,
             transform=plt.gca().get_yaxis_transform())
    plt.ylim(0.0, 1.05)
    plt.xlabel("Communication round")
    plt.ylabel("AUC separating same-group from cross-group pairs")
    plt.title(title or "Can the server see the cluster structure?")
    plt.tight_layout()
    _save(path)
