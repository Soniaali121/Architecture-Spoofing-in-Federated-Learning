# Architecture Spoofing Attacks in Federated Learning

MSc AI group project. In model-heterogeneous / clustered Federated Learning (CFL),
the server groups clients into per-architecture clusters (e.g. "mlp", "dnn") based
on metadata the client itself declares, and trusts that declaration without
verification. This project shows that a malicious client can **spoof its declared
architecture** to infiltrate a cluster it doesn't belong to and poison that
cluster's global model from within. It also implements a
**behavioural-fingerprinting defence**, in which the server verifies a client's
declared architecture against the actual behaviour of the update it submits,
probed on server-held reference data.

Research question: *How can Clustered Federated Learning systems identify and
defend against architecture spoofing attacks where malicious actors misrepresent
their device properties to gain unauthorized entry into localized clusters?*

## Headline result

> **Read "Project status" below first if you are new to this.** The supervisor has
> since redirected the work toward data distribution and statistical heterogeneity,
> and the current results live in `cfl-distribution.ipynb`. What follows in this
> section is the architecture-spoofing study that came before it.

Measured, not asserted (see [Results](#results) for the numbers and
`old_code/arch-spoofing-analysis.ipynb` for the figures):

- The attack **works**, costing the victim cluster **65 to 79% of its accuracy** at
  every data-skew level tested, across 3 seeds with confidence intervals.
- Architecture fingerprinting **cannot detect the attack that matters**. An
  adaptive spoofer trains the architecture it declares, so its update genuinely is
  a well-formed member of the victim cluster. It is flagged 25% of the time
  against a **34% false-alarm rate on honest clients**, i.e. below noise.
- The detection that does occur tracks the **payload**, not the spoof.

That last point changes what the report can claim. "Behavioural fingerprinting
verifies a declared architecture" holds only for a *naive* spoofer, and a naive
spoofer is already stopped by a weight-shape check that needs no research at all.

**The report's claim that spoofing defeats Byzantine-robust aggregation is now
measured, and it is wrong as written.** Krum, Multi-Krum, coordinate-wise median,
trimmed mean and Bulyan are implemented in `aggregation.py` and swept in both live
notebooks. The measured result:

| | plain FedAvg | every robust rule |
|---|---|---|
| attacker placed in the target cluster | yes | **yes, all five** |
| damage lands once inside | yes | **no** |

Placement is 1.000 under all six rules against a 0.000 honest control. Damage
lands only under plain FedAvg, which loses 0.41 accuracy against its own control
while every robust rule loses nothing. **Clustering decides who you are grouped
with; aggregation decides whose update counts, and a robust aggregator governs
only the second.** The supportable sentence is that spoofing defeats robust
*clustering* completely and robust *aggregation* not at all.

**It is measured in both studies**, with different heterogeneity, different
partitioners and different attacks, which makes it a replication rather than one
experiment. See [RESULTS.md](RESULTS.md) §A7, §B5, and "The replication".

One claim remains **unsupported** and should not appear in the report: that
fingerprinting detects spoofing. It needs a *stealthy* attacker, which does not
exist here (`DEFENSE_NOTES.md` section 4.1).

## Project status

The supervisor redirected this work toward **data distribution and statistical
heterogeneity**, and that is the current direction. **Three** notebooks now carry live
results, across the project's two attack channels:

| notebook | axis | setting | what it establishes |
|---|---|---|---|
| [cfl-distribution.ipynb](cfl-distribution.ipynb) | **data** | distribution shift, planted label profiles | the current direction: both clustering channels are spoofable, imitation is free, and the binding constraint is the attacker's *knowledge* of the target distribution |
| [cfl-spoofing.ipynb](cfl-spoofing.ipynb) | **data** | concept shift, permuted labels | the prior study that motivated it, on the literature-standard Clustered Federated Learning setup |
| [cfl-architecture.ipynb](cfl-architecture.ipynb) | **model** | model heterogeneity, declared architecture | added 30 Aug 2026: placement is total under both spoof tiers, the shape check is all-or-nothing, fingerprinting fails, and the binding constraint is *capability* rather than knowledge |

**The project attacks one trust boundary through two channels**, and they are different
axes rather than a sequence. The server groups on `metadata['label_hist']` (data) or
`metadata['arch']` (model); neither field is verified. The two **fail for opposite
reasons**, which is the sharpest finding in the project: distribution spoofing is bounded
by knowledge and free in capability, architecture spoofing is bounded by capability and
free in knowledge. See [ARCHITECTURE_CHANNEL.md](ARCHITECTURE_CHANNEL.md).

[RESULTS.md](RESULTS.md) is the write-up for the data channel;
[ARCHITECTURE_CHANNEL.md](ARCHITECTURE_CHANNEL.md) for the model channel. **All three
settings agree** on the aggregation result above, which makes it a replication rather
than a single experiment.

The **attack and its analysis** are the focus. The **defence** is implemented and
working but deliberately parked (`enable_defense=False` by default), see
[DEFENSE_NOTES.md](DEFENSE_NOTES.md) for what it does, what's weak about it, and
what the analysis phase found out about it. Don't extend the defence before
reading that file, in particular sections 9 and 10.

`aimens-code.py` is **read-only**. It's the reference implementation we started
from, kept verbatim so our own contribution stays visible against it. Never edit it.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11 recommended. Two backends are in play and which one you need depends
on what you are running:

- **PyTorch** for the three live notebooks. Verified: torch 2.5.1+cu121 on an
  RTX 4060 Laptop, using the GPU directly from native Windows, no container.
- **TensorFlow** for the older pipelines (`CFL.py`, `run_arch_cfl.py`,
  `run_data_cfl.py`). Verified: TensorFlow 2.15.0, scikit-learn 1.4.2, pandas
  2.2.2, NumPy 1.26.4.

If you have multiple Python installs, check the one you are about to run:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import tensorflow; print(tensorflow.__version__)"
```

### GPU: use the container for TensorFlow only

**TensorFlow cannot see an NVIDIA GPU from native Windows**, and no CUDA install
fixes it: TF's Windows wheels have no GPU code compiled in (`is_built_with_cuda()`
is `False`), because native-Windows GPU support was dropped after TF 2.10. PyTorch
still ships Windows CUDA builds, which is why it works and TF does not.

The [Dockerfile](Dockerfile) solves this with a Linux TF build reaching the GPU
through Docker Desktop's WSL2 backend, using the host driver. Verified on an
RTX 4060 Laptop: `built_with_cuda: True`, compute capability 8.9, ~5.5 GB usable.

```bash
docker build -t archspoof:gpu .
docker run --gpus all --rm archspoof:gpu \
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

See the Dockerfile header for the notebook and long-job recipes. Two gotchas are
documented there: use **named detached containers** for long runs (so `docker stop`
really stops them), and Git Bash needs `MSYS_NO_PATHCONV=1`.

Everything runs on CPU too, only corpus size and runtime change.

## Reproducibility

Every entry point takes `seed=` and calls `seeding.set_global_seeds(seed)` before
the first model is built, pinning Python, NumPy, and whichever framework RNG the
backend selects (`backend="torch"` for the live notebooks, TensorFlow otherwise).
Same seed → identical results; different seed → different results.

The live notebooks go further: **every headline number is a mean over five seeds
with a bootstrap 95% confidence interval, reported beside a control measured on
the same setup.** A gap smaller than the intervals is reported as *not
distinguishable* rather than as a result.

Seed-to-seed spread is **large** at small round/epoch counts (10+ accuracy points
has been observed between seeds on a 1-round run), so any headline number must be
averaged over several seeds with a reported spread. Single-run numbers are for
smoke-testing only, not for the report.

```python
from run_arch_cfl import main
runs = [main(rounds=3, epochs=5, seed=s) for s in range(5)]
```

## Quickstart

**For the current results**, open [cfl-distribution.ipynb](cfl-distribution.ipynb)
and Run All, then [cfl-spoofing.ipynb](cfl-spoofing.ipynb). Both are self-contained:
they download MNIST on first use, need no pre-built corpus, and write named runs
under `results/runs/`. Restart the kernel first if you have run one before. On an
RTX 4060 the distribution notebook takes roughly 25 minutes end to end, the
aggregator sweep in Part 7 being most of it; both fall back to CPU automatically
and take proportionally longer.

**The standalone demos**, which are self-contained and need no corpus:

```bash
python make_example_data.py   # once: generates data/example_ids_iot.csv
python CFL.py                 # architecture pipeline: baseline vs attack vs attack+defence
python run_data_cfl.py demo   # distribution pipeline: same 3-way comparison
```

> The architecture-spoofing corpus generators (`fingerprint_lab.py`, `dist_lab.py`,
> `corpus.py`, `experiment_grid.py`) and the two notebooks they fed now live in
> [old_code/](old_code/), along with their output in `old_code/results/`. They were
> retired because they partition clients with `partition_dirichlet`, which contains
> no group structure for a clustering study to recover. See
> [old_code/README.md](old_code/README.md) for the measurement behind that, and do
> not compare current numbers against anything in there.

`CFL.py` and `run_data_cfl.py demo` print a full comparison and write `results/`
(git-ignored):
- `round_metrics.csv` / `dist_round_metrics.csv`, per-round accuracy/loss/fairness/detection, one row per (run, round, cluster)
- `client_log.csv` / `dist_client_log.csv`, per-round, per-client audit trail (declared vs. true architecture, spoof ground truth, accept/reject status, local train loss/accuracy)
- comparison plots across the three runs (convergence, fairness, detection quality)

The copies of those four CSVs currently in `old_code/results/` are from the retired
Dirichlet line, not from a fresh run of these demos.

To point at the real dataset instead of the synthetic example, pass
`csv_path="Process_1 IDS-IoT-2024.csv"` to `main()` in `run_arch_cfl.py` / `CFL.py`
(the file is not included in this repo).

## Project layout

### The live study
| File | Purpose |
|---|---|
| `LECTURE.md` | **Start here if you are new to the project.** The whole thing from scratch, no code required |
| `HANDOFF.md` | **Start here if you are picking the work up.** Section 0 is the current state; the trap list in section 6 is the most useful part of the file |
| `cfl-distribution.ipynb` | **Current direction.** Distribution shift and statistical heterogeneity: whether Dirichlet has group structure at all, both clustering channels, resampling against prior-editing, the knowledge ladder, and the aggregator sweep. **Generated**, see `notebooks/` |
| `cfl-spoofing.ipynb` | The prior concept-shift study, on the literature-standard Clustered Federated Learning setup. **Generated**, see `notebooks/`. Its saved outputs are a single-seed `QUICK` run; §B4's five-seed numbers come from `part_b/grid5.py` |
| `cfl-architecture.ipynb` | **The model-heterogeneity channel.** Clients run different architectures, the server groups them by the one they *declare*, and an attacker lies about it. **Generated**, see `notebooks/`. Parts 3 and 8 train 45 federations live, so a full run takes about 16 minutes |
| `notebooks/` | **The notebook generators.** All three notebooks above are build outputs. Edit the generator, never the `.ipynb` |
| `sweeps_arch.py` | The architecture channel's campaign driver, mirroring `sweeps.py`. `python sweeps_arch.py tiers\|pool\|agg`. Shares `run_cell` with the notebook so the two cannot drift |
| `verify_arch_channel.py` | Re-derives every architecture-channel figure from its source and asserts it: the archived corpus, then the live M1/M3 experiments. Run before citing anything from that channel |
| `ARCHITECTURE_CHANNEL.md` | The model channel's write-up, to the same standard as `RESULTS.md` |
| `ARCHITECTURE_WALKTHROUGH.md` | `cfl-architecture.ipynb` explained from scratch: every metric defined, every number given a verdict, supervisor Q&A |
| `PRESENTATION_ARCH.md` | The model channel as a presentation, figures in place. Counterpart to `PRESENTATION.md` |
| `part_b/` | The seven drivers behind every Part B table, with a README mapping each to its run folder |
| `RESULTS.md` | The write-up for the two **data**-channel studies, every number traced to a named run |
| `experiments/` | **The parameter campaign.** One folder per experiment, named for the change it makes, each with its own `data/`, `graphs/` and write-up. Driven by `sweeps.py` |
| `sweeps.py` | The campaign driver. `python sweeps.py structure\|groups\|attackers\|extremes\|agg` runs experiments; `collect` runs nothing and rebuilds `campaign.csv` from the folders already on disk. Resumable: appends row by row and skips completed cells on restart |
| `DISTRIBUTION_WALKTHROUGH.md` | `cfl-distribution.ipynb` transcribed in full, every cell annotated with what it does, how, and why. Read this instead of the notebook if you are not going to run it |
| `SUPERVISOR_SCRIPT.md` | Speaking notes for presenting the above |
| `DEFENSE_NOTES.md` | What a defence would have to do, and what the results rule out |
| `REPORT_NOTES.md` | What to change in the written report |
| `F_ATTEMPTS.md` | Approaches that failed, and why, so they are not retried |

### Core FL framework (shared by every pipeline below)
| File | Purpose |
|---|---|
| `models.py` | Architecture registry: `mlp`, `dnn` factories. `ARCH_REGISTRY` is an extension point, `lab_models.py` adds seven more at import time. `head_indices` scopes a flat weight vector to the classifier head |
| `client.py` | Local training; returns a `ClientUpdate` (weights + declared metadata) |
| `clustering.py` | `SattlerCosineClusterer` (cosine similarity of update deltas, the published Clustered Federated Learning method), `IFCAClusterer` (client picks its own best-fitting cluster), `ArchitectureClusterer` (declared arch), `DistributionClusterer` (declared label histogram) |
| `aggregation.py` | FedAvg (plain and sample-weighted), plus Krum, Multi-Krum, coordinate-wise median, trimmed mean, Bulyan and FoolsGold, reimplemented from the published equations and cross-checked against the ByzFL library. `aggregate_flat(name, ...)` dispatches by name |
| `attacks.py` | Every attack: `ClusterSteering` (the live one: mechanisms `declare`/`weight`/`relabel`/`data`/`prior`, constraints `lie`/`minmax`, payloads `boost`/`flip`), `ArchSpoof`, `WeightSignFlip`, `WeightBoost`, `LabelHistSpoof`, `AdaptiveLabelHistSpoof`, `TargetEstimator`, `CompositeAttack` |
| `defense.py` | `FingerprintVerifier` (architecture) / `DistributionFingerprintVerifier` (label histogram), behavioural-fingerprinting defences |
| `fl_loop.py` | `FederatedServer` (TensorFlow) and `TorchFederatedServer` (PyTorch, what the live notebooks use), orchestrating rounds, clustering, defence, and observability |
| `analysis.py` | Post-hoc metrics (clustering signal, recovery over rounds, target similarity), the `BASELINES` registry and `compare_to_baseline`, CSV export, comparison plots |
| `data.py` | IDS-IoT CSV loading, scaling, IID/Dirichlet client partitioning, the planted-group partitioners (`partition_clustered`, `partition_concept_permuted`, `partition_concept_rotated`, `partition_natural_permuted`), server reference split |
| `metrics.py` | Jensen-Shannon divergence, calibration error, and `separation_ratio` (cross-group over within-group divergence, which is how "is there any group structure here at all?" gets answered). Imports nothing from the project, which is what keeps it out of the import cycle its callers would otherwise form |
| `seeding.py` | `set_global_seeds(seed, backend=...)`, pinning Python/NumPy and either TensorFlow or PyTorch RNGs for reproducible runs |

### Architecture-clustered IDS pipeline (the report's main experiment)
| File | Purpose |
|---|---|
| `make_example_data.py` | Generates a small synthetic IDS-style CSV so the pipeline runs without the private dataset |
| `run_arch_cfl.py` | `main()` entry point: architecture-clustered FL with optional spoof/poison/defence flags |
| `run_data_cfl.py` | Distribution-clustered FL variant (clusters on data, not declared architecture) |
| `CFL.py` | Runnable demo: baseline vs. attack vs. attack+defence (architecture pipeline), with analysis export |
| `run_data_cfl.py demo` | Same 3-way demo for the distribution pipeline. This absorbed `DCFL.py`, which was a thin runner around it and has been deleted |

### The lab layer (Mohamed's slice)
| File | Purpose |
|---|---|
| `lab_models.py` | GPU configuration, the seven lab architectures (3 dense + 4 CNN), the per-dataset architecture pools, and the PyTorch equivalents the live notebooks train |
| `lab_data.py` | Dataset loaders (MNIST/CIFAR-10/FEMNIST/IDS) and the `Bundle` client/test/probe split. FEMNIST preserves its writer split, which is what makes `partition="natural"` a genuine non-IID partition rather than a reshuffle |
| `lab_probe.py` | The probe-model cache, output priors, and fingerprint feature extraction. The one place that loads submitted weights and measures behaviour |
| `run_context.py` | Per-run output folders and manifests (`save_run`), plus the shared notebook helpers (palette, bootstrap CI, `show`, `per_seed_breakdown`) |
| `Dockerfile` | GPU runtime for the TensorFlow pipelines (TF cannot use the GPU from native Windows, see the file for why). The live notebooks are PyTorch and need no container |

The retired corpus generators (`fingerprint_lab.py`, `dist_lab.py`, `corpus.py`,
`experiment_grid.py`) and the two notebooks they fed are in [old_code/](old_code/).
The sections further down describing their results are the prior study, kept
because the report cites it.

### Results layout

Every notebook execution writes into its own timestamped folder, so reruns never
overwrite each other and any figure traces back to the run that made it:

```
results/
  runs/
    20260901-025647_dist_knowledge/   <- illustrative; the timestamp changes per run
      dist_knowledge.csv   the table itself
      figures/             plots for this run
      tables/              the CSV behind each figure
      manifest.json        row count, columns, content digest, dataset, source notebook
  cache/                   expensive intermediates reused across runs
  dist_knowledge.csv       the LATEST copy of each named run, overwritten every time
```

Each named run is written twice on purpose. The timestamped folder is the
permanent record, so a number quoted in the write-up traces to the exact run that
produced it; the flat copy at the top of `results/` is a **pointer** to whichever
run is current, so a notebook has a stable path to load. Cite the timestamped
folder, never the flat file.

`cache/` stays **shared rather than per-run** because `cache/attack_impact.csv`
costs 18 federated runs to rebuild. Each run's manifest pins it by content hash,
so a run stays reproducible without copying it.

`run_context.latest_run()` returns the newest **completed** run, the manifest is
written last, so a run that died partway is skipped rather than mistaken for
current. `run_context.list_runs()` shows all of them with a completion flag.

Output from the retired Dirichlet line is in `old_code/results/` and is **not**
comparable to anything above. Keeping the two apart is deliberate: `HANDOFF.md`
records four separate incidents of stale numbers being pooled with fresh ones.

### Superseded standalone MNIST scripts
`fl_common.py`, `attack_spoofing.py`, `defense_fingerprint.py`. Not part of the
notebook results, and nothing in the working tree imports them. They are the first
working version of the experiment, on MNIST, with their own mini-framework. Kept
because they still run and the report cites them.

| File | Purpose |
|---|---|
| `fl_common.py` | MNIST data/model/simulation in one module, **the origin of the 6-D fingerprint feature vector** (`FINGERPRINT_FEATURES`) that `lab_probe.SHAPE_AGNOSTIC_FEATURES` grew out of |
| `attack_spoofing.py` | `python attack_spoofing.py`, spoofing attack demo with impact metrics |
| `defense_fingerprint.py` | `python defense_fingerprint.py`, fingerprinting defence demo, including the RandomForest architecture meta-classifier (the Ateniese/Ganju paradigm the report cites) |

Every concept in there exists again in the live stack and the two have already
drifted (the `dnn` factories no longer share an optimiser). Prefer the live
modules; treat these as historical.

### Reference: do not modify
`aimens-code.py`, the original single-file implementation we were given as our
starting example (flat FL, no clustering or spoofing). Kept **verbatim and
read-only** so we can always show exactly what we started from and what is our
own work on top. Superseded in practice by the modules above; never edit,
refactor, or import from it.

## How the attacks work

There are two independent spoofing surfaces in this codebase, matching the two
clustering strategies in `clustering.py`. Both lie about declared metadata the
server trusts; they differ in *which* metadata.

### Architecture spoof (`run_arch_cfl.py`, `CFL.py`)

All clients share the same data distribution assumption; clusters are formed by
**declared architecture** (`ArchitectureClusterer`). Every client carries two
distinct fields: `declared_arch` (what the server sees, drives cluster
assignment, `ClientUpdate.metadata['arch']`) and `true_arch` (the client's real,
honest architecture, `ClientUpdate.true_arch`). An honest client always has
them equal; `ArchSpoof` sets them apart for its malicious clients. What training
actually uses depends on `ArchSpoof(..., adaptive=...)`:

- `adaptive=True` (default), the client trains (and warm-starts from) the
  *declared* architecture, so its update is shape-valid for the target cluster
  and survives the server's shape check. Paired with `WeightSignFlip` (via
  `CompositeAttack`), this is the attack that actually lands: verified to crash
  victim-cluster accuracy (see the alpha grid below).
- `adaptive=False`: the client trains its *true* architecture and only lies in
  the declared metadata (`client_log`'s `status` column reads `shape_rejected`).
  In this shape-constrained setting, a naive metadata-only lie is discarded by
  the server's shape check before it can do any damage, a useful baseline
  showing why the interesting/dangerous case is the adaptive one.

### Distribution spoof (`run_data_cfl.py`)

Here every client trains the *same* architecture, and clusters are formed instead
by **declared label distribution** (`DistributionClusterer`, agglomerative
clustering on each client's reported label histogram). This pathway is a purer
version of the same declared/true split: `LabelHistSpoof` only rewrites
`update.metadata['label_hist']` (the declared histogram), it never touches the
weights, so the submitted update still reflects the client's true data. This is
exactly what `DistributionFingerprintVerifier` exploits (see below). Run it with
the attack enabled (the default `python run_data_cfl.py` run is baseline-only):

```python
from run_data_cfl import main
main(rounds=3, malicious_clients=(0,), enable_label_hist_spoof=True,
     enable_sign_flip=True, spoof_target_class=0, dirichlet_alpha=0.5,
     enable_defense=True)
```

## Data partitioning: IID vs. Dirichlet non-IID

`data.py` offers two client partitioners, and which one you use changes what each
attack above is actually testing:

- `partition_iid`: every client gets an independent random subsample; all
  clients' data looks statistically alike.
- `partition_dirichlet(..., alpha)`: labels are split across clients by
  Dirichlet(alpha) proportions per class. Small `alpha` (e.g. 0.1-0.5) means
  strong label skew (each client sees mostly a few classes); large `alpha`
  (e.g. 10+) approaches IID. This is the standard non-IID benchmark setting in
  FL literature and is what the report's evaluation assumes.

**Dirichlet plays a different role for each attack:**

- **Architecture spoof**: Dirichlet is *context*, not the attack mechanism. The
  lie is about model type, not data. Under IID data (the `run_arch_cfl.py`
  default is actually `"dirichlet"`, but pass `partition="iid"` to compare) a
  spoofer's update stands out more starkly since every honest client looks the
  same; under Dirichlet non-IID, honest clients are already diverse, so the
  spoofer's update, and its poison, has natural variance to hide inside,
  making the attack stealthier and detection harder.
- **Distribution spoof**: Dirichlet *is* the attack surface. `DistributionClusterer`
  only produces meaningful, separated clusters when clients' label histograms
  actually differ, which requires non-IID (skewed) data. Under IID partitioning
  every client's histogram is close to uniform, there are no real data-clusters
  to infiltrate, and `LabelHistSpoof` has nothing to exploit. `dirichlet_alpha`
  directly controls how distinct (and thus how attractive/vulnerable) the target
  clusters are.

## Isolating spoofing from data heterogeneity: the alpha grid

A single baseline-vs-attack comparison confounds two things: how much damage the
attack does, and how heterogeneous the data happened to be when it was measured.
`old_code/experiment_grid.py` separates them by running the full attack (adaptive
architecture spoof + sign-flip) at three Dirichlet settings, severe non-IID
(`alpha=0.1`), mild (`0.5`), and near-IID (`10.0`), with and without spoofing at
each, so the degradation at a given alpha is attributable to spoofing alone:

Current numbers, **3 seeds per cell** with bootstrap 95% confidence intervals,
produced by Section 4 of `old_code/arch-spoofing-analysis.ipynb` (3 rounds, 3 epochs,
6 clients, victim cluster `dnn`):

| alpha | baseline | attack | degradation (95% CI) | relative loss |
|---|---|---|---|---|
| 0.1 | 0.492 | 0.169 | 0.323 [0.248, 0.470] | 65% |
| 0.5 | 0.834 | 0.167 | 0.668 [0.485, 0.762] | 79% |
| 10.0 | 0.995 | 0.258 | 0.738 [0.735, 0.740] | 74% |

Two findings:

- **Absolute damage is largest where the baseline was strongest.** At near-IID
  (`alpha=10`) the honest baseline is nearly perfect (0.995) and the attack takes
  0.738 of it; at severe non-IID (`alpha=0.1`) natural heterogeneity has already
  degraded the baseline to 0.492, so there is less left to destroy.
- **In relative terms the attack is severe at every alpha**, removing roughly two
  thirds to four fifths of whatever accuracy existed. So "non-IID hides the
  attack" is a claim about *detectability*, not about how much damage lands.

A **variance decomposition** confirms these are treatment effects rather than
noise: the attack condition explains **75.9%** of the variation in victim-cluster
accuracy, alpha **13.5%**, and the random seed only **0.3%**.

> That seed figure is configuration-dependent and worth understanding before you
> trust any single run. At 1 round / 1 epoch, seed accounted for roughly 20% of
> variance and alpha for almost none, i.e. the opposite conclusion. Short runs
> have not converged and are dominated by initialisation. Use at least the 3
> rounds / 3 epochs above before drawing conclusions, and always report seeds.

`old_code/experiment_grid.py` remains the standalone version of this sweep and validates
its own baseline before trusting the attack numbers: for each alpha it runs and
sanity-checks the honest run (cluster sizes, accuracy) before introducing the
spoofer, so a partitioning bug shows up as an obviously-wrong baseline rather
than getting misread as an attack effect. Note it does not yet do multiple seeds;
the notebook is the source for report-quality numbers.

## Results

All from `old_code/arch-spoofing-analysis.ipynb`, corpus of 1512 fingerprinted client
updates across 126 configurations (IDS + MNIST, 3 seeds, 3 alphas, 5
architectures). Figures land in `results/runs/<timestamp>_analysis/figures/`.

### Can behaviour identify an architecture at all?

Yes, on honest clients only, with whole seeds held out of validation:

| dataset | architectures | accuracy | majority baseline | lift |
|---|---|---|---|---|
| ids | 3 | 0.81 | 0.33 | +0.47 |
| mnist | 4 | 0.51 | 0.33 | +0.18 |

The confusion matrix qualifies this: `mlp_plain` is identified 89% of the time,
but `dnn_flat` only 72%, with 22% of it mistaken for `mlp_flat`. **Most of the
separability is one architecture being structurally unusual** (no BatchNorm or
Dropout), not fine discrimination between similar networks.

### The shape leak

Allow features derived from model size and architecture inference becomes
perfect, and meaningless:

| feature set | ids | mnist |
|---|---|---|
| shape-agnostic (honest) | 0.81 | 0.51 |
| + shape-dependent | 1.00 | 1.00 |
| `n_params` alone | 1.00 | 1.00 |

The server already knows the declared shape, so a classifier reading it has
learned nothing, and an adaptive spoofer's shape matches by construction. This is
why `old_code/fingerprint_lab.py` tags every feature as `SHAPE_AGNOSTIC_FEATURES` or
`SHAPE_DEPENDENT_FEATURES`, and only the former go into the headline classifier.

### Which mechanism actually catches the spoofer

Three mechanisms, reported separately and never pooled. `false_alarm` is the same
fingerprint rule applied to honest clients, which is what makes the column
interpretable at all:

| condition | shape check | nonfinite | fingerprint | vs 34% false alarm |
|---|---|---|---|---|
| naive, no payload | **1.00** | 0.00 | 0.97 | signal |
| naive, sign flip | **1.00** | 1.00 | n/a | unscorable |
| adaptive, no payload | 0.00 | 0.00 | 0.25 | **below noise** |
| adaptive, boost | 0.00 | 0.00 | 0.47 | indistinguishable |
| adaptive, sign flip | 0.00 | 0.00 | 0.92 | signal (the *payload*) |

Naive spoofing is caught outright by the free weight-shape check. Adaptive
spoofing passes every structural check, and the fingerprint adds nothing unless
the payload has wrecked the model.

### Distribution clustering only means anything under skew

Mean pairwise Jensen-Shannon divergence between client label histograms:

| alpha | 0.05 | 0.1 | 0.25 | 0.5 | 1.0 | 2.0 | 5.0 | 10.0 |
|---|---|---|---|---|---|---|---|---|
| pairwise JS | 0.526 | 0.498 | 0.409 | 0.298 | 0.192 | 0.116 | 0.048 | **0.024** |

A twenty-two-fold collapse: as data approaches IID, clients become
indistinguishable and there is no real data-cluster left to infiltrate.

> Silhouette score was tried first and **does not** answer this question. With 6
> clients forced into k=2 the clusterer always separates something, so it wanders
> between 0.10 and 0.31 with no trend. The notebook plots it alongside as an
> explicitly-labelled secondary, and the earlier version of this analysis that
> relied on it asserted a trend the data never showed.

## How the defence works

> **Parked, see [DEFENSE_NOTES.md](DEFENSE_NOTES.md) before extending this.**
> The defence is off by default while we focus on attack analysis. Two things
> about it are now **measured**, not suspected, and both limit what it can claim:
>
> 1. **Its perfect precision/recall is NaN detection.** Sign-flipping a network
>    with batch normalisation makes the running variance negative, so the forward
>    pass returns non-finite values, which any check catches. Re-run against a
>    BatchNorm-free architecture pool and recall drops to **0.00**. This is why
>    `old_code/fingerprint_lab.py`'s `ARCH_POOLS` includes `mlp_plain` as a control.
> 2. **It is blind to an adaptive spoofer by construction.** That client trains
>    the architecture it declares, so a verifier asking "does this behave like
>    what it claims?" correctly answers yes. Flagged 25% against a 34%
>    false-alarm rate on honest clients, i.e. below noise.
>
> Neither is a bug in the implementation. They are properties of the threat model,
> and a future defence has to detect the payload or the data distribution instead.

Both verifiers in `defense.py` share the same idea: calibrate a "normal"
behaviour band from honest shadow clients, then flag any update whose behaviour
falls outside it, rather than trusting what the client declared.

- **`FingerprintVerifier`** (architecture mode) trains a few honest shadow models
  per architecture on server-held reference data to learn a probe-set accuracy
  floor and loss ceiling. Every round, each client's update is loaded into a
  model of its *declared* architecture and probed; behaviour outside the honest
  band, or non-finite output (e.g. a sign-flipped BatchNorm model), is
  quarantined.
- **`DistributionFingerprintVerifier`** (distribution mode) trains honest shadow
  clients under Dirichlet-skewed label splits to learn how far a model's mean
  predicted-class distribution (soft, averaged softmax over a probe set, less
  noisy than a hard argmax-vote histogram at small data scale) normally diverges
  from its own declared label histogram, measured via Jensen-Shannon divergence.
  A client that declares a histogram it didn't actually train on shows much
  higher divergence than any honest client; non-finite output is treated as
  maximally suspicious, same as the architecture verifier. On the example data
  this reaches precision/recall/F1 = 1.0 for the demonstrated attack, but the
  underlying signal (output-prior shift from label skew) is inherently weaker
  and noisier than the architecture verifier's accuracy/loss signal, cosine
  similarity between predicted and declared distributions was tried first and
  did *not* separate honest from spoofed reliably at this data scale, which is
  why JS divergence is used instead. Treat this verifier's robustness as less
  battle-tested than the architecture one if you extend the attack (e.g. a
  smaller `spoof_target_class` skew, more classes, larger real dataset).

## Data pipeline

`prepare_fl_data` splits in this order:

```
full data ──> (training pool, test set)          stratified, seeded
training pool ──> (server reference, client pool) server_ref_frac, default 0.10
client pool ──> per-client partitions             IID or Dirichlet
```

- **Features** are every column except the target, selected by numeric dtype and
  with the target dropped **by name** (not by position), so a dataset whose label
  column isn't last can't leak the label into `X`. Non-finite values are zeroed
  with a printed count.
- **Scaling**: `StandardScaler` fitted on the training split only, then applied
  to test and reference. Near-no-op on the synthetic CSV (already roughly
  standardised); it matters for the real dataset, whose raw features span very
  different magnitudes. Disable with `scale=False`.
- **Server reference set** (`FLData.X_ref`/`y_ref`) is carved out of the training
  pool, never given to any client, and disjoint from the test set. It is what
  server-side machinery is allowed to see. It is carved out **unconditionally**,
  whether or not a defence is enabled, so client partitions are identical across
  the baseline/attack/defence conditions, otherwise those runs would train on
  different data and the comparison between them would be confounded.

## Aggregation

`FederatedServer(weighted_aggregation=True)` is the default and applies standard
sample-weighted FedAvg. Set `False` for the plain unweighted mean.

This is not cosmetic. Dirichlet partitioning makes client sizes very uneven, so
an unweighted mean gives a small malicious client the same influence as a large
honest one, inflating apparent attack strength and confounding "how damaging is
spoofing" with "which aggregation rule we happened to use". The flag exists so
that confound can be measured as an ablation rather than left implicit.

## Notes

- `run_arch_cfl.py` defaults to non-IID (Dirichlet, `alpha=0.5`) client
  partitioning to match the report's setting; pass `partition="iid"` for the
  simpler case.
- `run_data_cfl.py` always uses Dirichlet partitioning (`dirichlet_alpha=0.5` by
  default) since distribution-based clustering is meaningless under IID data -
  see above.
- `results/`: `data/*.csv`, `__pycache__/`, and `*.png` are git-ignored (generated
  artifacts). Regenerate the example data with `make_example_data.py`.
