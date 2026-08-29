"""Parameter sweeps for the distribution-shift study, into one fixed schema.

Every result in `RESULTS.md` Part A sits at a single configuration: 12 clients,
2 groups, 6 rounds, 2 epochs, MAX_TRAIN=12000, concentration 20, overlap 0.3,
one attacker. Five seeds deep and one point wide. This module widens it.

WHY THIS FILE EXISTS RATHER THAN MORE SCRATCH SCRIPTS
-----------------------------------------------------
The six earlier sweeps (grid5, agg_sweep, arch_sweep, ladder, beta_sweep,
payload_sweep) were one-off scripts outside the repository, and each wrote its
own column set. That is survivable for one sweep and not for a campaign: seven
incompatible files cannot be pooled, compared, or guarded. Everything here
writes `run_context.SWEEP_COLUMNS`, so any two sweeps concatenate and any row
says what produced it.

THE ORDER IS DELIBERATE
-----------------------
`groups` runs FIRST, before the expensive extremes, because it can falsify the
headline. At K=2 an attacker that stops resembling its home group has exactly
one other place to go, so 1.000 infiltration may be ELIMINATION rather than
imitation. `scratchpad/ladder.py` found this for the concept-shift study and
broke it by sweeping K. The distribution study has never swept K and has never
measured resemblance, so it cannot currently tell the two apart. Spending two
hours on extremes that assume the headline holds, before checking whether it
holds, would be the wrong order.

RESUMABILITY IS NOT OPTIONAL HERE
---------------------------------
This machine runs at roughly 0.4 GB free RAM, where a process death is an
out-of-memory kill rather than a bug, and it is silent. Every sweep therefore
appends row by row and skips completed cells on restart, so a death costs the
current cell rather than the run.

WHERE IT LANDS
--------------
One folder per experiment, and THE FOLDER NAME STATES THE CHANGE, so the values
tried are visible from a directory listing rather than only from inside a file:

    experiments/
      README.md                        index
      01_structure_envelope/
        README.md                      what it asked, what it found
        data/results.csv
        graphs/*.png
        manifest.json
        log.txt
      02_n_groups_2-3-4-6/
      03_attackers_1-2-3/
      04_rounds_6-12-20-30/
      05_epochs_1-2-5-10/
      06_max_train_6k-12k-30k-60k/
      07_concentration_5-10-20-50-100/
      08_overlap_0-15-30-50pct/
      09_aggregators_6-rules/
      campaign.csv                     every experiment concatenated

    python sweeps.py structure    # free, no training
    python sweeps.py groups       # the confound gate
    python sweeps.py attackers
    python sweeps.py extremes     # optionally: extremes rounds
    python sweeps.py agg
    python sweeps.py campaign     # rebuild experiments/campaign.csv
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import analysis
from aggregation import stack
from attacks import ClusterSteering
from clustering import SattlerCosineClusterer
from data import label_histogram
from fl_loop import TorchFederatedServer
from lab_data import load_bundle_torch, partition_label_hists
from metrics import separation_ratio
from run_context import (SWEEP_COLUMNS, blank_row, build_campaign, code_digest,
                         experiment_csv, experiment_dir, git_state,
                         save_experiment)
from seeding import describe_device, set_global_seeds
from sklearn.metrics import adjusted_rand_score

# The operating point every sweep varies ONE factor away from. These are the
# values behind RESULTS.md Part A, so any sweep's baseline cell reproduces a
# number already in the write-up -- which is what makes the harness checkable
# against the notebook rather than merely self-consistent.
BASE = dict(
    dataset="mnist", arch="mlp_flat", partition="clustered",
    num_clients=12, n_groups=2, concentration=20.0, overlap=0.3,
    max_train=12000, warmup=4, rounds=6, epochs=2,
    n_attackers=1, mechanism="data", payload="none", payload_scale=2.0,
    aggregator="fedavg", scope="head", recursive=False,
)

SEEDS = (0, 1, 2)          # 3 for sweeps, matching ladder/agg_sweep precedent
HEADLINE_SEEDS = (0, 1, 2, 3, 4)

_RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
_GIT = git_state(REPO)
_CODE = code_digest()
_LOG_LINES: list[str] = []

# Resolved once at import, not inside __main__, so importing this module for a
# single sweep function gets the same provenance as running it from the CLI.
try:
    import torch as _torch

    _TORCH = _torch.__version__
except Exception:                                    # pragma: no cover
    _TORCH = "n/a"
_DEVICE = describe_device()


def log(msg: str) -> None:
    """Print and retain, so the per-cell trace survives into the run folder.

    The `[sattler]` threshold lines are the reason this exists. Threshold tuning
    has failed silently twice in this project, and today those lines live only
    in notebook scrollback. A sweep that ran for an hour should leave a record
    of how each fit behaved, not just its numbers.
    """
    print(msg, flush=True)
    _LOG_LINES.append(msg)


# --------------------------------------------------------------------------- #
# Shared experiment machinery
# --------------------------------------------------------------------------- #
def bundle(cfg: dict, seed: int):
    """One experimental world for a seed, at this configuration."""
    return load_bundle_torch(
        cfg["dataset"], num_clients=cfg["num_clients"], partition=cfg["partition"],
        n_groups=cfg["n_groups"], seed=seed, max_train=cfg["max_train"],
        concentration=cfg["concentration"], overlap=cfg["overlap"])


def sides(b, attacker: int):
    """(home group, target group, target member ids), for ANY number of groups.

    The notebook's own helper computes `1 - home`, which is correct only at
    K=2. This uses the pattern already proven in scratchpad/ladder.py: take the
    first other group in sorted order, deterministically, so the choice does not
    vary between the control and the attacked run of the same seed.
    """
    home = int(b.groups[attacker])
    groups = sorted({int(g) for g in b.groups})
    target_g = next(g for g in groups if g != home)
    members = [i for i in range(len(b.groups)) if int(b.groups[i]) == target_g]
    return home, target_g, members


def group_profile(b, g: int):
    H = partition_label_hists(b)
    return H[np.asarray(b.groups) == g].mean(axis=0)


def infiltration_of(history, attackers, target_members):
    """Fraction of POST-SPLIT rounds where an attacker sat with the target group.

    `attackers` may be one client id or several; the result is the mean over
    them, so a two-attacker run reports the fraction of (attacker, round) pairs
    that landed rather than collapsing to whether any one of them did.

    Three properties worth keeping:
      * rounds before the first split are skipped, not scored. Before a split
        every client shares one cluster, so the attacker is trivially "with" the
        target and so is everybody else. Scoring those rounds either way biases
        the control and the attack together.
      * membership is followed by WHO, never by cluster name, because names are
        reassigned each round.
      * the target cluster is decided by majority vote of its true members,
        since the group can itself be split across clusters.
    """
    ids = [attackers] if isinstance(attackers, (int, np.integer)) else list(attackers)
    hits, split_round = [], None
    for r, e in enumerate(history):
        if len(e["membership"]) < 2:
            continue
        if split_round is None:
            split_round = r
        where = {c: cid for cid, ms in e["membership"].items() for c in ms}
        votes = {}
        for m in target_members:
            votes[where.get(m)] = votes.get(where.get(m), 0) + 1
        tc = max(votes, key=votes.get)
        for a in ids:
            hits.append(where.get(a) == tc)
    return hits, split_round


def _row(cfg: dict, seed: int, condition: str, is_control: bool,
         sweep: str, factor: str, secs: float, **measured) -> dict:
    """Assemble a schema-complete row: provenance + config + condition + result."""
    return blank_row(
        run_id=_RUN_ID, sweep=sweep, factor=factor,
        git_commit=_GIT["git_commit"], git_dirty=_GIT["git_dirty"],
        code_digest=_CODE, timestamp=datetime.now().isoformat(timespec="seconds"),
        secs=round(secs, 1), device=_DEVICE, torch_version=_TORCH,
        seed=seed, condition=condition, is_control=is_control,
        **{k: cfg.get(k) for k in
           ("dataset", "arch", "partition", "num_clients", "n_groups",
            "concentration", "overlap", "max_train", "warmup", "rounds", "epochs",
            "n_attackers", "attacker_ids", "mechanism", "beta", "payload",
            "payload_scale", "aggregator", "scope", "recursive", "max_clusters",
            "eps1", "eps2")},
        **measured)


def run_cell(cfg: dict, seed: int, attacked: bool, sweep: str, factor: str,
             condition: str) -> dict:
    """One (config, seed, condition) cell: warm up, tune, measure, fit, score.

    The control and the attacked run go through THIS function, differing only in
    `beta`. Routing both through one path is what makes any difference between
    them attributable to the attack rather than to the pipeline.
    """
    t0 = time.time()
    cfg = dict(cfg)
    cfg["beta"] = 1.0 if attacked else 0.0
    cfg["max_clusters"] = cfg["n_groups"]
    if not attacked:
        cfg["payload"] = "none"

    b = bundle(cfg, seed)
    home, target_g, members = sides(b, 0)

    # EVERY attacker must start OUTSIDE the target group, or the experiment
    # measures the wrong thing. Taking clients 0..n-1 by index does not
    # guarantee that: group assignment is shuffled per seed, and on all three
    # seeds here one of clients {0,1,2} is already a member of the target. Such
    # a client has nothing to infiltrate, so counting it inflates the attacked
    # rate and the control alike, and the multi-attacker result then answers
    # "n clients, some already inside" rather than the question asked.
    attackers = [i for i in range(cfg["num_clients"])
                 if int(b.groups[i]) == home][:cfg["n_attackers"]]
    if len(attackers) < cfg["n_attackers"]:
        raise ValueError(
            f"only {len(attackers)} clients in the home group, cannot place "
            f"{cfg['n_attackers']} attackers outside the target at K="
            f"{cfg['n_groups']}")
    cfg["attacker_ids"] = ",".join(str(a) for a in attackers)
    profile = group_profile(b, target_g)

    # ---- attacker-free warmup, and thresholds tuned on it ------------------
    # Tuning on a run containing the attacker would fit the split criterion to
    # the attack it is meant to be blind to, and every number after it would
    # describe a server that had already seen what we claim it cannot see.
    set_global_seeds(seed, backend="torch")
    probe = SattlerCosineClusterer(eps1=-1.0, scope=cfg["scope"])
    warm = TorchFederatedServer(b.input_shape, b.num_classes, probe,
                                arch=cfg["arch"], n_clusters=cfg["n_groups"], seed=seed)
    for r in range(cfg["warmup"]):
        _, wu, _ = warm.run_round(b.client_data, round_num=r, epochs=cfg["epochs"])
    Dw = stack([u.metadata["delta_vec"] for u in wu])[:, probe.head_index]
    e1, e2 = SattlerCosineClusterer(scope=cfg["scope"]).tune_thresholds(Dw, margin=1.5)
    cfg["eps1"], cfg["eps2"] = round(float(e1), 4), round(float(e2), 4)

    def make_attack():
        a = ClusterSteering(
            malicious_clients=attackers, target_members=members,
            mechanism=cfg["mechanism"], beta=cfg["beta"],
            num_classes=b.num_classes, arch=cfg["arch"],
            input_shape=b.input_shape, seed=seed,
            payload=cfg["payload"], payload_scale=cfg["payload_scale"])
        a.target_hist = profile
        a.own_data = {c: b.client_data[c] for c in attackers}
        return a

    # ---- (a) MECHANISM, measured PRE-SPLIT --------------------------------
    # Held open for the same number of rounds as the warmup, so every client's
    # delta is still referenced against one shared model. Measuring after a
    # split compares deltas taken from different starting points, which is not
    # like-for-like and produced a wrong conclusion in this project before.
    set_global_seeds(seed, backend="torch")
    probe_a = SattlerCosineClusterer(eps1=-1.0, scope=cfg["scope"])
    srv_a = TorchFederatedServer(b.input_shape, b.num_classes, probe_a,
                                 arch=cfg["arch"], attack=make_attack(),
                                 n_clusters=cfg["n_groups"], seed=seed)
    for r in range(cfg["warmup"]):
        _, ua, _ = srv_a.run_round(b.client_data, round_num=r, epochs=cfg["epochs"])
    Da = stack([u.metadata["delta_vec"] for u in ua])
    sim = analysis.target_similarity(
        Da, [u.client_id for u in ua], b.groups, attacker=attackers[0],
        target_group=target_g, index=probe_a.head_index, home_group=home)

    # ---- (b) PLACEMENT, with the real clusterer ---------------------------
    clus = SattlerCosineClusterer(eps1=e1, eps2=e2, scope=cfg["scope"],
                                  recursive=cfg["recursive"],
                                  max_clusters=cfg["n_groups"])
    set_global_seeds(seed, backend="torch")
    srv = TorchFederatedServer(b.input_shape, b.num_classes, clus.reset(),
                               arch=cfg["arch"], attack=make_attack(),
                               aggregator=cfg["aggregator"],
                               n_clusters=cfg["n_groups"], seed=seed)
    hist = srv.fit(b.client_data, b.X_test, b.y_test, rounds=cfg["rounds"],
                   epochs=cfg["epochs"], client_groups=b.groups)

    hits, split_round = infiltration_of(hist, attackers, members)
    accs = [np.mean(list(e["metrics"].values())) for e in hist]

    # How well the server recovered the planted grouping AT ALL, in the final
    # round. Without this a K sweep cannot be read: infiltration measured
    # against a clusterer that has stopped recovering the groups is a number
    # about noise, and at higher K that failure is the thing to rule out first.
    ari = np.nan
    for e in reversed(hist):
        if len(e["membership"]) >= 2:
            labels = np.zeros(cfg["num_clients"], dtype=int)
            for k, (_, ms) in enumerate(sorted(e["membership"].items())):
                for c in ms:
                    labels[c] = k
            ari = float(adjusted_rand_score(b.groups, labels))
            break

    return _row(
        cfg, seed, condition, not attacked, sweep, factor, time.time() - t0,
        infiltration=float(np.mean(hits)) if hits else np.nan,
        post_split_rounds=len(hits) // max(1, len(attackers)),
        split_round=split_round, ari=ari,
        target_resemblance=sim.get("target_resemblance"),
        to_target=sim.get("to_target"), to_home=sim.get("to_home"),
        honest_within=sim.get("honest_within"), honest_cross=sim.get("honest_cross"),
        final_accuracy=float(accs[-1]) if accs else np.nan)


# --------------------------------------------------------------------------- #
# Resumable append
# --------------------------------------------------------------------------- #
# A sweep is a list of cells. Each is written the moment it completes, and on
# restart the cells already present are skipped. Without this an out-of-memory
# kill 39 minutes into a 40-minute sweep costs the whole sweep.

RESUME_KEY = ["sweep", "factor", "condition", "seed", "n_groups", "rounds",
              "epochs", "max_train", "concentration", "overlap", "n_attackers",
              "aggregator"]


def _done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        d = pd.read_csv(path)
    except Exception:
        return set()
    if not set(RESUME_KEY).issubset(d.columns):
        return set()
    return {tuple(str(v) for v in t)
            for t in d[RESUME_KEY].itertuples(index=False, name=None)}


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])[SWEEP_COLUMNS]
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def drive(name: str, cells: list, sweep: str, factor: str, note: str):
    """Run a list of (cfg, seed, attacked, condition) cells, resumably."""
    experiment_dir(name)
    path = experiment_csv(name)
    done = _done_keys(path)
    log(f"[{sweep}] {len(cells)} cells planned, {len(done)} already done")
    t0 = time.time()

    for i, (cfg, seed, attacked, condition) in enumerate(cells, 1):
        key = tuple(str(v) for v in (
            sweep, factor, condition, seed, cfg["n_groups"], cfg["rounds"],
            cfg["epochs"], cfg["max_train"], cfg["concentration"], cfg["overlap"],
            cfg["n_attackers"], cfg["aggregator"]))
        if key in done:
            log(f"  [{i}/{len(cells)}] skip (done): {condition} seed={seed}")
            continue
        row = run_cell(cfg, seed, attacked, sweep, factor, condition)
        _append(path, row)
        # Release the allocator's cached blocks between cells. Each cell builds
        # three servers and discards them, and a long sweep on an 8 GB card
        # accumulated enough fragmentation to fail a 128x784 matmul with
        # CUBLAS_STATUS_EXECUTION_FAILED partway through the rounds sweep.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        log(f"  [{i}/{len(cells)}] {factor}={cfg.get(factor)} {condition:>22} "
            f"seed={seed}  inf={row['infiltration']}  "
            f"res={row['target_resemblance']:+.3f}  "
            f"acc={row['final_accuracy']:.3f}  {row['secs']}s")

    df = pd.read_csv(path)
    d = save_experiment(name, df, log_text="\n".join(_LOG_LINES),
                        source="sweeps.py", note=note, sweep=sweep, factor=factor,
                        cells_planned=len(cells), cells_present=len(df),
                        total_secs=round(time.time() - t0, 1),
                        device=_DEVICE, torch=_TORCH, base=BASE,
                        run_id=_RUN_ID, **_GIT, code_digest=_CODE)
    log(f"[{sweep}] done in {time.time()-t0:.0f}s -> {d}")
    return df


# --------------------------------------------------------------------------- #
# Phase 0: structure. No training, so this is free.
# --------------------------------------------------------------------------- #
def sweep_structure():
    """Which (K, concentration, overlap) settings contain groups at all?

    Pure histogram arithmetic -- no model is trained -- so the whole envelope
    can be mapped exhaustively in seconds, before any GPU time is spent. Every
    later sweep then runs only where infiltration is a meaningful statement.

    Scored by the ratio of cross-group to within-group Jensen-Shannon
    divergence. 1.0 means the groups are no more different from each other than
    their own members are, i.e. no structure. For the Dirichlet reference rows
    the score is the MAXIMUM over every possible 2-way split, which is a ceiling
    no clusterer can beat -- so a low value says the structure is absent rather
    than that the algorithm failed to find it.
    """
    rows = []
    t0 = time.time()

    for alpha in (0.05, 0.2, 1.0):
        b = load_bundle_torch(BASE["dataset"], num_clients=BASE["num_clients"],
                              partition="dirichlet", alpha=alpha, seed=0,
                              max_train=BASE["max_train"])
        H = partition_label_hists(b)
        n = len(H)
        ratios = []
        for mask in range(1, 2 ** (n - 1)):
            left = [i for i in range(n) if (mask >> i) & 1]
            right = [i for i in range(n) if not (mask >> i) & 1]
            if min(len(left), len(right)) < 2:
                continue
            ratios.append(separation_ratio([H[left], H[right]])["ratio"])
        rows.append(dict(partition="dirichlet", alpha=alpha, n_groups=np.nan,
                         concentration=np.nan, overlap=np.nan,
                         best_split=max(ratios), median_split=float(np.median(ratios)),
                         min_group=np.nan, classes_per_group=np.nan))
        log(f"  dirichlet a={alpha}: best={max(ratios):.2f} "
            f"median={np.median(ratios):.2f}")

    for k in (2, 3, 4, 6):
        for conc in (5.0, 10.0, 20.0, 50.0, 100.0):
            for ov in (0.0, 0.15, 0.3, 0.5):
                cfg = {**BASE, "n_groups": k, "concentration": conc, "overlap": ov}
                b = bundle(cfg, 0)
                H = partition_label_hists(b)
                g = np.asarray(b.groups)
                sizes = [int((g == x).sum()) for x in sorted(set(g.tolist()))]
                # separation_ratio needs >= 2 members per group for a
                # within-group term. At 12 clients that caps K at 6.
                ratio = separation_ratio([H[g == x] for x in sorted(set(g.tolist()))])["ratio"]
                rows.append(dict(partition="clustered", alpha=np.nan, n_groups=k,
                                 concentration=conc, overlap=ov,
                                 best_split=ratio, median_split=np.nan,
                                 min_group=min(sizes),
                                 classes_per_group=round(10 / k, 2)))
        log(f"  planted K={k}: {len(rows)} rows so far")

    df = pd.DataFrame(rows)
    df["run_id"], df["git_commit"], df["code_digest"] = _RUN_ID, _GIT["git_commit"], _CODE
    df["timestamp"] = datetime.now().isoformat(timespec="seconds")
    save_experiment("01_structure_envelope", df, log_text="\n".join(_LOG_LINES),
                    source="sweeps.py", sweep="structure",
                    note="separation ratio over K x concentration x overlap, "
                         "plus Dirichlet reference ceilings. No training.",
                    total_secs=round(time.time() - t0, 1),
                    device=_DEVICE, torch=_TORCH, run_id=_RUN_ID,
                    **_GIT, code_digest=_CODE)

    base = df[(df.n_groups == 2) & (df.concentration == 20.0) & (df.overlap == 0.3)]
    log(f"\nSANITY: K=2, conc=20, overlap=0.3 -> {float(base['best_split'].iloc[0]):.2f} "
        f"(RESULTS.md Part A says 4.51)")
    log(f"[structure] {len(df)} rows in {time.time()-t0:.0f}s")
    return df


# --------------------------------------------------------------------------- #
# Phase 1: the confound gate.
# --------------------------------------------------------------------------- #
def sweep_groups():
    """Does the attack survive more than two groups, and by what mechanism?

    THE QUESTION THIS ANSWERS. At K=2 a client that stops resembling its home
    group has exactly one other place to go, so placement in the target may be
    ELIMINATION rather than imitation. Sweeping K breaks that: with four or six
    groups, being evicted from home does not tell you where you land, so
    infiltration that survives is placement by resemblance.

    Read alongside `target_resemblance`, which is the mechanism check the
    distribution study has never had. Infiltration is an OUTCOME; resemblance
    says whether the attacker's update actually looks like a member.

    DECISION RULE, fixed before the run:
      high infiltration AND positive resemblance -> genuine spoof
      infiltration decaying to the per-K control AND negative resemblance
                                                 -> elimination artefact
      high infiltration BUT negative resemblance -> a third mechanism; stop.
    """
    cells = []
    for k in (2, 3, 4, 6):
        cfg = {**BASE, "n_groups": k}
        for seed in SEEDS:
            cells.append((cfg, seed, False, "control"))
            cells.append((cfg, seed, True, "resample to target"))
    return drive("02_n_groups_2-3-4-6", cells, "groups", "n_groups",
                 "K sweep with resemblance; each K against its own control")


# --------------------------------------------------------------------------- #
# Phase 2: attacker fraction.
# --------------------------------------------------------------------------- #
def sweep_attackers():
    """1, 2 or 3 attackers of 12 -- 8.3%, 16.7%, 25%.

    Closes a named limitation: RESULTS.md concedes one attacker in twelve where
    the literature runs 1 to 30%. Not automatically easier with more attackers,
    because the LIE bound in attacks.py tightens as the malicious count grows.
    """
    cells = []
    for n_att in (1, 2, 3):
        cfg = {**BASE, "n_attackers": n_att}
        for seed in SEEDS:
            cells.append((cfg, seed, False, "control"))
            cells.append((cfg, seed, True, f"{n_att} attacker(s)"))
    return drive("03_attackers_1-2-3", cells, "attackers", "n_attackers",
                 "attacker fraction 8.3 / 16.7 / 25 percent")


# --------------------------------------------------------------------------- #
# Phase 3: the extremes, one factor at a time.
# --------------------------------------------------------------------------- #
# One experiment per factor, and the folder name states the change. Bundling
# five independent factors into one file made the values impossible to see from
# a directory listing, which is the thing a person actually scans.
EXTREMES = {
    "rounds":        ((6, 12, 20, 30),            "04_rounds_6-12-20-30"),
    "epochs":        ((1, 2, 5, 10),              "05_epochs_1-2-5-10"),
    "max_train":     ((6000, 12000, 30000, 60000), "06_max_train_6k-12k-30k-60k"),
    "concentration": ((5.0, 10.0, 20.0, 50.0, 100.0), "07_concentration_5-10-20-50-100"),
    "overlap":       ((0.0, 0.15, 0.3, 0.5),      "08_overlap_0-15-30-50pct"),
}


def sweep_extremes(only: str | None = None):
    """One factor at a time from the operating point, one folder per factor.

    Full factorial over five factors is hundreds of hours. One-factor-at-a-time
    is linear in the number of values and answers the question actually being
    asked, which is where each knob breaks the attack rather than how the knobs
    interact.
    """
    out = {}
    for factor, (values, name) in EXTREMES.items():
        if only and factor != only:
            continue
        cells = []
        for v in values:
            cfg = {**BASE, factor: v}
            for seed in SEEDS:
                cells.append((cfg, seed, False, f"control ({factor}={v})"))
                cells.append((cfg, seed, True, f"attack ({factor}={v})"))
        out[factor] = drive(name, cells, "extremes", factor,
                            f"{factor} swept over {list(values)}, "
                            f"everything else at the operating point")
    return out


# --------------------------------------------------------------------------- #
# Phase 4: aggregators, at one setting only.
# --------------------------------------------------------------------------- #
AGGREGATORS = ("fedavg", "krum", "multikrum", "median", "trimmed", "bulyan")


def sweep_agg(**overrides):
    """The six rules, at whichever setting Phase 3 shows most stressed.

    36 fits, the most expensive thing here, so it runs at ONE point rather than
    across the sweep. Placement and damage are scored separately because they
    have different answers: robust aggregation governs whose update counts, not
    who gets grouped with whom.
    """
    cells = []
    for agg in AGGREGATORS:
        cfg = {**BASE, **overrides, "aggregator": agg, "payload": "boost"}
        for seed in SEEDS:
            cells.append((cfg, seed, False, "control"))
            cells.append((cfg, seed, True, "resample + boost x2"))
    return drive("09_aggregators_6-rules", cells, "agg", "aggregator",
                 f"six aggregation rules at {overrides or 'the operating point'}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    log(f"[sweeps] {_DEVICE}")
    log(f"[sweeps] run_id={_RUN_ID} git={_GIT['git_commit']} "
        f"dirty={_GIT['git_dirty']} code={_CODE}")

    what = sys.argv[1] if len(sys.argv) > 1 else "structure"
    arg = sys.argv[2] if len(sys.argv) > 2 else None

    if what == "structure":
        sweep_structure()
    elif what == "groups":
        sweep_groups()
    elif what == "attackers":
        sweep_attackers()
    elif what == "extremes":
        sweep_extremes(only=arg)
    elif what == "agg":
        sweep_agg()
    elif what == "campaign":
        build_campaign()
    else:
        print(__doc__)
        sys.exit(1)
