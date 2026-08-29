# Architecture Spoofing Attacks in Federated Learning

MSc AI group project. In model-heterogeneous / clustered Federated Learning (CFL),
the server groups clients into per-architecture clusters (e.g. "mlp", "dnn") based
on metadata the client itself declares, and trusts that declaration without
verification. This project demonstrates that a malicious client can **spoof its
declared architecture** to infiltrate a cluster it doesn't belong to, poison that
cluster's global model from within, and defeats existing Byzantine-robust defences
because they run *after* clustering, not at cluster assignment. It also implements
a **behavioural-fingerprinting defence**: the server verifies a client's declared
architecture against the actual behaviour of the update it submits (probed on
server-held reference data) and quarantines mismatches before aggregation.

Research question: *How can Clustered Federated Learning systems identify and
defend against architecture spoofing attacks where malicious actors misrepresent
their device properties to gain unauthorized entry into localized clusters?*

## Setup

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

Python 3.11 recommended (TensorFlow 2.15). If you have multiple Python installs,
make sure the one you run has TensorFlow — check with:
```bash
python -c "import tensorflow; print(tensorflow.__version__)"
```

## Quickstart

```bash
python make_example_data.py   # once: generates data/example_ids_iot.csv
python CFL.py                 # baseline vs attack vs attack+defence, on the IDS pipeline
```

`CFL.py` prints a full comparison and writes `results/` (git-ignored):
- `round_metrics.csv` — per-round accuracy/loss/fairness/detection, one row per (run, round, cluster)
- `client_log.csv` — per-round, per-client audit trail (declared vs. true architecture, spoof ground truth, accept/reject status, local train loss/accuracy)
- `accuracy_dnn.png`, `fairness.png`, `detection.png` — comparison plots across the three runs

To point at the real dataset instead of the synthetic example, pass
`csv_path="Process_1 IDS-IoT-2024.csv"` to `main()` in `run_arch_cfl.py`/`CFL.py`
(the file is not included in this repo).

## Project layout

### Core FL framework (shared by both pipelines below)
| File | Purpose |
|---|---|
| `models.py` | Architecture registry: `mlp`, `dnn` factories |
| `client.py` | Local training; returns a `ClientUpdate` (weights + declared metadata) |
| `clustering.py` | `ArchitectureClusterer` (groups by declared arch), `DistributionClusterer` (groups by label-histogram similarity) |
| `aggregation.py` | FedAvg (plain and sample-weighted) |
| `attacks.py` | `ArchSpoof`, `WeightSignFlip`, `LabelHistSpoof`, `CompositeAttack` |
| `defense.py` | `FingerprintVerifier` — the behavioural-fingerprinting defence |
| `fl_loop.py` | `FederatedServer` — orchestrates rounds, clustering, defence, and observability |
| `analysis.py` | Post-hoc metrics (attack success rate, victim degradation), CSV export, comparison plots |
| `data.py` | IDS-IoT CSV loading, IID/Dirichlet client partitioning |

### Architecture-clustered IDS pipeline (the report's main experiment)
| File | Purpose |
|---|---|
| `make_example_data.py` | Generates a small synthetic IDS-style CSV so the pipeline runs without the private dataset |
| `run_arch_cfl.py` | `main()` entry point: architecture-clustered FL with optional spoof/poison/defence flags |
| `run_data_cfl.py` | Distribution-clustered FL variant (clusters on data, not declared architecture) |
| `CFL.py` | Runnable demo: baseline vs. attack vs. attack+defence, with full analysis export |

### Standalone MNIST attack/defence scripts (independent of the above; own mini-framework)
| File | Purpose |
|---|---|
| `fl_common.py` | Shared MNIST data/model/simulation code |
| `attack_spoofing.py` | `python attack_spoofing.py` — spoofing attack demo with impact metrics |
| `defense_fingerprint.py` | `python defense_fingerprint.py` — high-precision fingerprinting defence demo |

### Legacy
`aimens-code.py` — original single-file reference implementation (flat FL, no
clustering or spoofing). Superseded by the modules above; kept for reference.

## How the attack works

A spoofing client declares an architecture that isn't its own, trains (and
warm-starts from) *that* architecture so its update is shape-valid for the target
cluster, and only then applies its payload (`WeightSignFlip` and/or
`LabelHistSpoof`). This is what makes the attack land: a client that merely lies
about its label without training the matching shape gets discarded by the shape
check and does no damage — see `attacks.py`'s `training_arch()` hook.

## How the defence works

`defense.py`'s `FingerprintVerifier` trains a few honest "shadow" models per
architecture on server-held reference data to learn a normal behaviour band
(probe-set accuracy floor + loss ceiling). Every round, before aggregation, each
client's update is loaded into a model of its *declared* architecture and probed
on held-out data; if its behaviour falls outside the honest band (or produces
non-finite output, e.g. a sign-flipped BatchNorm model), it's quarantined. This
verifies the declared metadata against actual behaviour instead of trusting it.

## Notes

- `run_arch_cfl.py` defaults to non-IID (Dirichlet) client partitioning to match
  the report's setting; pass `partition="iid"` for the simpler case.
- `results/`, `data/*.csv`, `__pycache__/`, and `*.png` are git-ignored (generated
  artifacts). Regenerate the example data with `make_example_data.py`.
