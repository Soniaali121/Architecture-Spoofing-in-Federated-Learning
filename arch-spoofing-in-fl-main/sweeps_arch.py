"""Architecture-channel experiments, on the same machinery as the Part A campaign.

WHAT THIS IS FOR. The architecture channel is Amin's slice: clients run different
models, the server groups them by the architecture they DECLARE, and an attacker
lies about it. Until now that channel could only run on the TensorFlow
`FederatedServer`, which is CPU-only on this machine and predates the 51-column
schema and the loader guard. `TorchFederatedServer` now takes `client_archs`, so
these sweeps write rows that pool into `experiments/campaign.csv` and are refused
by `run_context.load_experiment` if their configuration is not what the caller
claims.

THE TWO TIERS, WHICH ARE THE WHOLE STUDY

    naive     declares architecture X, trains its own. The submitted vector does
              not fit cluster X's model, so the SHAPE CHECK rejects it. Free.
    adaptive  declares X and trains X. Shape-valid by construction, so the shape
              check has nothing to find and the update is aggregated into a
              cluster the client does not belong to.

The gap between those two is the contribution, and M1 measures it directly.

WHAT IS DELIBERATELY NOT HERE. No behavioural fingerprint verifier. That defence
already has a result, reproduced by `verify_arch_channel.py` and written up in
`ARCHITECTURE_CHANNEL.md`: it flags an adaptive spoofer 25.0% of the time against
a 34.3% false-alarm rate on honest clients, so it is below noise. Re-running it
here would need a probe-model corpus these sweeps do not build. Placement and
damage are what this driver measures.

    python sweeps_arch.py tiers      M1: naive vs adaptive, does the shape check hold
    python sweeps_arch.py pool       M2: how many architectures are in the federation
    python sweeps_arch.py agg        M3: the six aggregation rules
    python sweeps_arch.py collect    runs nothing, rebuilds campaign.csv
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from aggregation import stack
from attacks import ArchSpoof, CompositeAttack, WeightBoost
from clustering import ArchitectureClusterer
from fl_loop import TorchFederatedServer
from lab_data import load_bundle_torch
from run_context import (CONFIG_COLUMNS, SWEEP_COLUMNS, blank_row, code_digest,
                         experiment_csv, experiment_dir, git_state,
                         save_experiment)
from seeding import describe_device, set_global_seeds

import lab_models  # noqa: F401  registers the seven lab architectures by import

REPO = Path(__file__).resolve().parent

# The operating point. One malicious client of twelve, declaring `dnn_flat` while
# really running `mlp_flat`. Kept deliberately close to the Part A operating point
# so the two channels are comparable: same dataset, same client count, same rounds.
BASE = dict(
    dataset="mnist", partition="clustered", num_clients=12, n_groups=2,
    concentration=20.0, overlap=0.3, max_train=12000, warmup=0,
    rounds=6, epochs=2, n_attackers=1,
    mechanism="arch_spoof", beta=1.0, payload="none", payload_scale=2.0,
    aggregator="fedavg", scope="full", recursive=False, max_clusters=2,
    arch="mlp_flat", target_rank=0,
)

# Two dense architectures with DIFFERENT parameter counts, which is what makes the
# shape check meaningful. mlp_flat is 784-128-64-10; dnn_flat is 512-256-128-64.
HOME_ARCH, TARGET_ARCH = "mlp_flat", "dnn_flat"

SEEDS = (0, 1, 2)

_RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
_GIT = git_state(REPO)
_CODE = code_digest()
_LOG: list = []

try:
    import torch as _torch
    _TORCH, _DEVICE = _torch.__version__, describe_device()
except Exception:
    _TORCH, _DEVICE = None, "cpu"


def log(msg: str) -> None:
    print(msg)
    _LOG.append(msg)


def assign_archs(num_clients: int, pool) -> list:
    """Round-robin architectures across clients, deterministically.

    Round-robin rather than random so every architecture gets the same number of
    clients and no cluster is short-handed by luck of the draw. A cluster of one
    would make its aggregation meaningless.
    """
    pool = list(pool)
    return [pool[i % len(pool)] for i in range(num_clients)]


def infiltration_of(history, attacker: int, target_arch: str):
    """Fraction of rounds the attacker sat in the target ARCHITECTURE's cluster.

    Simpler than the distribution study's version and deliberately so: cluster
    ids here ARE architecture names, assigned from a declared field, so there is
    no majority vote to take and no renaming between rounds. Membership is still
    read per round rather than assumed.
    """
    hits = []
    for e in history:
        where = {c: cid for cid, members in e["membership"].items() for c in members}
        hits.append(where.get(attacker) == target_arch)
    return hits


def run_cell(cfg: dict, seed: int, attacked: bool, adaptive: bool,
             sweep: str, factor: str, condition: str) -> dict:
    """One (config, seed, condition) cell. Control and attack share this path.

    The control is `attacked=False`, which is the same client doing nothing. Every
    number here is read against it, and a cell without one is refused by the guard
    downstream.
    """
    t0 = time.time()
    cfg = dict(cfg)
    pool = cfg.get("arch_pool") or [HOME_ARCH, TARGET_ARCH]
    archs = assign_archs(cfg["num_clients"], pool)

    # The attacker is the first client whose true architecture is the home one, so
    # it always has somewhere to infiltrate FROM. Picking client 0 blindly would
    # sometimes pick a client already running the target architecture, which has
    # nothing to spoof; the distribution study hit exactly that bug.
    attacker = next(i for i, a in enumerate(archs) if a == HOME_ARCH)
    cfg["attacker_ids"] = str(attacker)

    b = load_bundle_torch(cfg["dataset"], num_clients=cfg["num_clients"],
                          partition=cfg["partition"], n_groups=cfg["n_groups"],
                          seed=seed, max_train=cfg["max_train"],
                          concentration=cfg["concentration"], overlap=cfg["overlap"])

    attack = None
    if attacked:
        attack = ArchSpoof(malicious_clients=[attacker], spoof_as=TARGET_ARCH,
                           adaptive=adaptive)
        # ArchSpoof is a PLACEMENT mechanism only: it decides which cluster the
        # attacker lands in and leaves the weights honest. Asking whether an
        # aggregation rule blunts the attack is only a question once there is a
        # payload to blunt, so compose one on.
        if cfg["payload"] == "boost":
            attack = CompositeAttack(
                attack,
                WeightBoost(malicious_clients=[attacker],
                            boost=cfg["payload_scale"]))
        elif cfg["payload"] != "none":
            raise ValueError(f"unsupported payload {cfg['payload']!r}")

    set_global_seeds(seed, backend="torch")
    srv = TorchFederatedServer(
        b.input_shape, b.num_classes, ArchitectureClusterer(),
        attack=attack, aggregator=cfg["aggregator"], seed=seed,
        client_archs=archs)
    hist = srv.fit(b.client_data, b.X_test, b.y_test, rounds=cfg["rounds"],
                   epochs=cfg["epochs"])

    hits = infiltration_of(hist, attacker, TARGET_ARCH)
    # Victim accuracy is the TARGET cluster's own accuracy, which is the thing the
    # attack damages. Global accuracy would average the victim's loss away against
    # clusters the attacker never touched.
    victim = [e["metrics"].get(TARGET_ARCH, np.nan) for e in hist]
    accs = [np.mean([v for v in e["metrics"].values() if v == v]) for e in hist]
    rejected = sum(1 for r in srv.shape_rejections if r[1] == attacker)

    return blank_row(
        run_id=_RUN_ID, sweep=sweep, factor=factor,
        git_commit=_GIT["git_commit"], git_dirty=_GIT["git_dirty"],
        code_digest=_CODE, timestamp=datetime.now().isoformat(timespec="seconds"),
        secs=round(time.time() - t0, 1), device=_DEVICE, torch_version=_TORCH,
        seed=seed, condition=condition, is_control=not attacked,
        **{k: cfg.get(k) for k in CONFIG_COLUMNS if k != "seed"},
        infiltration=float(np.mean(hits)) if hits else np.nan,
        post_split_rounds=len(hits),
        final_accuracy=float(accs[-1]) if accs else np.nan,
        victim_accuracy=float(victim[-1]) if victim else np.nan,
        declared_arch=TARGET_ARCH if attacked else HOME_ARCH,
        true_arch=HOME_ARCH,
        shape_rejected_rounds=int(rejected),
    )


def drive(name: str, cells: list, sweep: str, factor: str, note: str):
    """Run a list of cells, resumably, into experiments/<name>/."""
    experiment_dir(name)
    path = experiment_csv(name)
    done = set()
    if path.exists():
        d = pd.read_csv(path)
        done = {(str(r.condition), int(r.seed), str(r.get("aggregator")),
                 str(r.get("arch_pool_size"))) for r in d.itertuples()}
    log(f"[{sweep}] {len(cells)} cells planned")
    t0 = time.time()

    for i, (cfg, seed, attacked, adaptive, condition) in enumerate(cells, 1):
        key = (condition, seed, str(cfg.get("aggregator")),
               str(len(cfg.get("arch_pool") or [HOME_ARCH, TARGET_ARCH])))
        if key in done:
            log(f"  [{i}/{len(cells)}] skip (done): {condition} seed={seed}")
            continue
        row = run_cell(cfg, seed, attacked, adaptive, sweep, factor, condition)
        pd.DataFrame([row])[SWEEP_COLUMNS].to_csv(
            path, mode="a", header=not path.exists(), index=False)
        log(f"  [{i}/{len(cells)}] {condition:>26} seed={seed}  "
            f"inf={row['infiltration']:.3f}  victim={row['victim_accuracy']:.3f}  "
            f"shape_rejected={row['shape_rejected_rounds']}  {row['secs']}s")

    df = pd.read_csv(path)
    d = save_experiment(name, df, log_text="\n".join(_LOG), source="sweeps_arch.py",
                       note=note, sweep=sweep, factor=factor,
                       cells_planned=len(cells), cells_present=len(df),
                       total_secs=round(time.time() - t0, 1), device=_DEVICE,
                       torch=_TORCH, base=BASE, run_id=_RUN_ID, **_GIT,
                       code_digest=_CODE)
    log(f"[{sweep}] done in {time.time()-t0:.0f}s -> {d}")
    return df


def sweep_tiers():
    """M1. Naive against adaptive: does the free shape check actually hold?

    THE QUESTION. A metadata-only lie should be stopped by shape validation
    alone, without any defence. An attacker willing to train the model it claims
    should walk straight past it. If both are stopped, the shape check is doing
    more than it should and no fingerprinting result means anything. If neither
    is, it is not wired in.

    DECISION RULE, fixed before the run:
      naive rejected every round AND adaptive never rejected  -> as designed
      neither rejected                                        -> shape check absent, stop
      both rejected                                           -> the adaptive path
                                                                 is not training the
                                                                 declared architecture
    """
    cells = []
    for seed in SEEDS:
        cells.append((BASE, seed, False, False, "control, no spoof"))
        cells.append((BASE, seed, True, False, "naive spoof"))
        cells.append((BASE, seed, True, True, "adaptive spoof"))
    return drive("M1_spoof_tiers", cells, "arch_tiers", "condition",
                 "naive vs adaptive spoof against the weight-shape check")


def sweep_pool():
    """M2. Does the attack survive more architectures in the federation?

    The analogue of the distribution study's K sweep. More architectures means
    more clusters to be wrong about, so if placement decays the attack was
    landing somewhere by default rather than by declaration.
    """
    pools = [[HOME_ARCH, TARGET_ARCH],
             [HOME_ARCH, TARGET_ARCH, "mlp_plain"],
             [HOME_ARCH, TARGET_ARCH, "mlp_plain", "dnn_flat2"]]
    pools = [[a for a in p if a in _available()] for p in pools]
    cells = []
    for pool in pools:
        cfg = {**BASE, "arch_pool": pool, "max_clusters": len(pool)}
        for seed in SEEDS:
            cells.append((cfg, seed, False, False, f"control, {len(pool)} archs"))
            cells.append((cfg, seed, True, True, f"adaptive, {len(pool)} archs"))
    return drive("M2_arch_pool", cells, "arch_pool", "max_clusters",
                 "how many architectures the federation runs")


def sweep_agg():
    """M3. The six aggregation rules, so this channel and Part A can be compared.

    Part A found robust aggregation contains the damage but not the intrusion.
    The prediction here is the same, and for the same reason: aggregation decides
    whose update counts, clustering decides who you sit with.

    A PAYLOAD IS REQUIRED for this question to mean anything. M1 shows placement
    alone does no damage at all -- the adaptive spoofer even helps slightly -- so
    without a payload there is nothing for an aggregation rule to contain and all
    six rules would tie trivially.

    `payload_scale=10.0` is `WeightBoost`'s own default and was chosen by probing
    2x / 5x / 10x on seed 0: victim accuracy 0.9080 / 0.8800 / 0.7150 against a
    0.9330 control, with finite logits throughout. 10x is the smallest of the
    three that damages FedAvg unambiguously without the update diverging.
    """
    cells = []
    for agg in ("fedavg", "krum", "multikrum", "median", "trimmed", "bulyan"):
        cfg = {**BASE, "aggregator": agg, "payload": "boost", "payload_scale": 10.0}
        for seed in SEEDS:
            cells.append((cfg, seed, False, False, f"control, {agg}"))
            cells.append((cfg, seed, True, True, f"adaptive, {agg}"))
    return drive("M3_aggregators", cells, "arch_agg", "aggregator",
                 "six aggregation rules against an adaptive architecture spoof")


def _available():
    from models import ARCH_REGISTRY
    return set(ARCH_REGISTRY)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "tiers"
    log(f"[sweeps_arch] {_DEVICE}, torch {_TORCH}")
    log(f"[sweeps_arch] run_id={_RUN_ID} git={_GIT['git_commit']} code={_CODE}")
    if what == "tiers":
        sweep_tiers()
    elif what == "pool":
        sweep_pool()
    elif what == "agg":
        sweep_agg()
    elif what in ("collect", "campaign"):
        from run_context import build_campaign
        build_campaign()
    else:
        print(__doc__)
        sys.exit(1)
