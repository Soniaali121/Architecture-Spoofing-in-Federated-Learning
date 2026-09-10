# Membership spoofing in Clustered Federated Learning

**What this is.** An attack study. A client holding data from one group makes the server
place it in another, and from inside corrupts a model the other members depend on.

No defence is built or proposed here. Where visibility is measured, that describes the
attack's signature rather than a countermeasure.

**Two studies, two kinds of heterogeneity.** Clustered Federated Learning (CFL) groups
clients because they differ, and there are two ways they can differ. This document reports
both, because **they agree**, and a structural result holding in two independent settings is
a replication rather than a single experiment.

| | Part A, **distribution shift** | Part B, **concept shift** |
|---|---|---|
| groups differ in | *how often* each class appears | what an input *means* |
| planted by | per-group label profiles | per-group label permutation |
| notebook | `cfl-distribution.ipynb` | `cfl-spoofing.ipynb` |
| status | **the current direction**, after the supervisor's redirection toward data distribution and statistical heterogeneity | the prior study that motivated it, and the literature's standard CFL setup |

**How to read it.** Every number traces to a saved run: the notebook results to
`results/runs/`, the parameter campaign to [`experiments/`](experiments/), both listed under
[Provenance](#provenance). Every headline figure is a mean over seeds with a bootstrap 95%
confidence interval, reported beside a control measured on the same setup. **A gap smaller
than the intervals is reported as *not distinguishable*, never as a result.**

**§A6 is the parameter campaign**, which widens every other Part A number from the single
configuration they were measured at. Read it before quoting any of them as general.

For how any Part A number was actually produced, [DISTRIBUTION_WALKTHROUGH.md](DISTRIBUTION_WALKTHROUGH.md)
transcribes `cfl-distribution.ipynb` in full and annotates every cell.

**Three claims in here contradict things we have previously written**, one of them in the
report. All are flagged in place: see §A7, §B5 and §B6.

---
---

> **Writing this up for the report?** The drafted Experimental Results section is
> [`docs/report_experimental_results.md`](docs/report_experimental_results.md), rewritten
> 31 Aug 2026 to cover both parts with distribution leading. Every figure in it is taken
> from this document. `REPORT_NOTES.md` carries the outstanding decisions, including the
> word budget, which the draft currently exceeds.

# Part A. Distribution shift, and statistical heterogeneity

MNIST, `mlp_flat`, 12 clients, 2 planted groups, 6 rounds of 2 epochs, one attacker (8.3%),
five seeds. Groups are planted by sampling each group's clients from its own label profile,
at concentration 20.0 and overlap 0.3.

## A1. The setting is not a free choice: Dirichlet has no groups in it

Statistical heterogeneity only supports a *clustering* study when the heterogeneity is
**grouped**. Scoring every possible two-way split of 12 clients by `separation_ratio`, the
ratio of cross-group to within-group Jensen-Shannon divergence:

| partition | best split available anywhere | median random split | x above baseline |
|---|---|---|---|
| Dirichlet(0.05) | 1.65 | 0.99 | 1.65 |
| Dirichlet(0.2) | 1.43 | 0.99 | 1.43 |
| Dirichlet(1.0) | 1.46 | 0.99 | 1.46 |
| **planted, overlap 0.5** | 2.38 | n/a | 2.38 |
| **planted, overlap 0.3** | **4.51** | n/a | **4.51** |
| **planted, overlap 0.0** | 17.31 | n/a | 17.31 |

A ratio of 1.0 means the two halves are no more different from each other than their own
members are, and the measured median over random splits sits at 0.99, which is the honest
null rather than an assumed one.

**Dirichlet partitioning, the standard non-Independent-and-Identically-Distributed generator,
produces individually skewed clients and no group structure.** The best grouping that exists
*anywhere* in it is barely above a random split, at every concentration tested. Planted
profiles are an order of magnitude stronger.

> This is why the earlier Dirichlet-based line in `old_code/` was retired. An infiltration
> rate into an arbitrary grouping describes nothing, and it is the most likely explanation
> for the unresolved puzzle in `HANDOFF.md` where infiltration sat flat at 0.67 across every
> mechanism including the do-nothing control: placement was never driven by data similarity,
> so imitating a data distribution could not move it.

## A2. Both clustering channels carry real signal

| channel | what the server uses | recovery of the planted grouping | baseline |
|---|---|---|---|
| **declared** | the client's reported label histogram | ARI **1.000**, all five seeds | 0.000, random partition of the same shape |
| **inferred**, full vector | cosine similarity of whole update deltas | AUC 0.8852, margin **0.0002** | 0.5, chance |
| **inferred**, classifier head | cosine similarity of head deltas only | AUC **1.0000**, margin **0.7545** | 0.5, chance |

Adjusted Rand Index (ARI) is chance-corrected, so 0.0 is both a random partition and the
degenerate one-cluster answer. AUC 1.0000 means **every** within-group pair is more similar
than **every** cross-group pair.

> **The declared channel scores 1.000 here and 0.665 in Part B, and both are correct.** The
> declared label histogram *is* the distribution, so under distribution shift it is a faithful
> signal and recovers the grouping exactly. Under concept shift the groups share a label
> marginal by design and the channel only leaks through residual class imbalance, which is the
> weaker 0.665. The difference between the two numbers is a finding about what the channel
> measures, not an inconsistency between the notebooks.

Head scoping matters as much here as under concept shift: the full-vector margin is 0.0002,
which is a hair above chance, against 0.7545 on the head. See §B6 for what destroys the
full-vector signal.

## A3. Imitating a distribution: two routes, and they differ in kind

**Resampling.** The attacker redraws its own training set, with replacement, to match the
target's profile. On seed 0, where the attacker is client 0 born into group 1 and targeting
group 0:

| | class 0 | class 1 | **class 2** | class 3 | class 4 | class 5 | class 6 | class 7 | class 8 | class 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| attacker's own data | 0.004 | 0.032 | **0.000** | 0.067 | 0.020 | 0.207 | 0.123 | 0.071 | 0.398 | 0.079 |
| the target's profile | 0.147 | 0.237 | **0.146** | 0.176 | 0.150 | 0.038 | 0.033 | 0.035 | 0.021 | 0.018 |
| achieved by resampling | 0.172 | 0.277 | **0.000** | 0.206 | 0.175 | 0.044 | 0.038 | 0.042 | 0.025 | 0.021 |
| shortfall | 0.000 | 0.000 | **0.146** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Resampling is bounded by holdings.** Sampling is with replacement, so a class the attacker
holds *any* of is reproduced closely and only classes it holds **none** of stay at zero. Here
that is one class carrying 0.146 of the target's mass.

**Prior-editing.** The attacker leaves its data alone and shifts the classifier's output
bias, following the logit-adjustment identity (Menon et al., ICLR 2021): adding `log q - log p`
to the logits moves the model's reported prior from `p` to `q`. The same class 2 the attacker
holds none of, whose model outputs 0.0000: a bias shift of **+18.8** makes it claim 0.146.

Run as an actual attack across five seeds, on every class the attacker holds none of:

| | value | baseline |
|---|---|---|
| attacker's holdings | 0.000 | n/a |
| **resampling achieves** | **0.000** | this *is* the baseline: a class you do not hold cannot be drawn |
| the target wants | 0.146 | n/a |
| **prior-editing claims** | **0.155** | against the 0.000 resampling reaches |

**The two routes differ in kind, not degree. Resampling yields a model that has *learned* the
distribution; prior-editing yields one that merely *reports* it.** Prior-editing is bounded by
nothing, which is its strength as an attack and its weakness against any server that checks
behaviour rather than claims.

## A4. The binding constraint is knowledge, not capability

Every result above hands the attacker the target's profile. Can it work the profile out for
itself, by probing the model the server broadcasts? Scored by Jensen-Shannon divergence
between the estimate and the truth, **lower is better**:

| knowledge level | what it assumes | error [95% CI] |
|---|---|---|
| `oracle` | told the target's profile exactly | **0.0000** [0.0000, 0.0000] |
| `inferred` | probes the broadcast model on a neutral probe set | 0.0844 [0.0737, 0.0945] |
| `blind` | assumes uniform, probes nothing | 0.0887 [0.0792, 0.0970] |

**Probing the broadcast model is not measurably better than assuming a uniform distribution
and never probing at all.** The improvement is 5% and the intervals overlap. The model does
leak its cluster's label distribution, but only by a couple of percentage points per class,
while the truth it is trying to recover is strongly skewed.

> This is the binding constraint on the whole distribution channel, and it is the sharpest
> limitation in this study. **An attacker *told* the target distribution imitates it perfectly
> and for free. An attacker that must *work it out* from what the server broadcasts cannot.**
> Every infiltration number in §A5 is therefore an upper bound conditioned on oracle
> knowledge, and closing that gap is what would turn this from a demonstrated capability into
> a threat model.

`inferred` probes on a **neutral probe set**, not on the attacker's own data. Averaging
outputs over the attacker's own inputs mixes in whatever that attacker happens to hold, which
on a skewed client dominates and yields a picture of the attacker rather than of the target.

## A5. Both channels move the server's decision, and imitation is free

**Declared channel**, five seeds, no retraining and no weight modification:

| condition | joined the target cluster |
|---|---|
| declares the target's profile | **1.000**, all five seeds |
| honest (control) | **0.000**, all five seeds |

The baseline is the **measured** honest control at 0.000, not 1/K. An honest client is not
placed at random: a working clusterer puts it in its own group.

**Inferred channel**, resample-to-target against Sattler clustering on head deltas, five
seeds:

| condition | infiltration [95% CI] | final accuracy [95% CI] |
|---|---|---|
| control (`beta=0`) | **0.000** [0.000, 0.000] | 0.833 [0.806, 0.855] |
| resample to target | **1.000** [1.000, 1.000] | 0.843 [0.817, 0.868] |

| statistic | measured | baseline | baseline is | gap | verdict |
|---|---|---|---|---|---|
| infiltration | 1.0000 | 0.0000 | the measured honest control | 1.0000 | **above baseline** |
| accuracy | 0.8429 | 0.8328 | the no-attack control, same setup | 0.0101 | **not distinguishable** |

**Imitation is free.** Accuracy under attack is 0.843 against a control of 0.833, which is
not distinguishable from it. Resampling changes *which* distribution the model learns, not
*how well* it learns one, so the attacker pays no visible price for the disguise. **A defence
looking for a degraded attacker will not find one.**

## A6. It is imitation, not elimination, and the parameter campaign

Everything in §A1 to §A5 was measured at **one configuration**: 12 clients, 2 groups, 6
rounds, 2 epochs, `MAX_TRAIN=12000`, concentration 20, overlap 0.3, one attacker. Five seeds
deep and one point wide. A campaign of seven experiments widened it. Full detail, per
experiment, in [`experiments/`](experiments/); the master table is `experiments/campaign.csv`.

### The confound that had to be cleared first

At K=2 the 1.000 infiltration in §A5 is **ambiguous**. An attacker that merely stops
resembling its *home* group has exactly one other place to go, so it would land in the target
**by elimination**, having imitated nothing. That produces 1.000 just as a spoof does, and
nothing in this study previously distinguished them: it never swept K and never measured
resemblance.

| K | infiltration | its own control | 1/K | **resemblance** | control |
|---|---|---|---|---|---|
| 2 | **1.000** | 0.000 | 0.500 | **+0.725** | −0.101 |
| 3 | **1.000** | 0.333 | 0.333 | **+1.109** | +0.066 |
| 4 | **1.000** | 0.000 | 0.250 | **+0.986** | +0.073 |
| 6 | **1.000** | 0.222 | 0.167 | **+1.219** | +0.065 |

**It is imitation.** Infiltration does not move as destinations multiply, at K=6 there are
five wrong clusters and one right one, where elimination predicts roughly 1/(K−1), and
resemblance is positive and *rising*, where elimination predicts flat or negative. Experiment
01 independently measured K=4 as having the **strongest** planted structure (separation 6.02
against 4.51 at K=2), so success there cannot be attributed to weaker groups either.

Verified that the K=6 run genuinely forms six clusters rather than two: splits persist across
rounds, so the tree accumulates 2 → 4 → **6** by round 2 and holds.

### Placement is invariant across four factors

| factor | range tested | infiltration | control |
|---|---|---|---|
| attackers | 1, 2, 3 of 12 (8.3 → **25%**) | **1.000** throughout | 0.000 |
| rounds | 6, 12, 20, **30** | **1.000** throughout | 0.000 |
| local epochs | 1, 2, 5, **10** | **1.000** throughout | 0.000 |
| training pool | 6k, 12k, 30k, **60k** (full MNIST) | **1.000** throughout | 0.000 |

This closes limitation 2 below (attacker fraction) and rules out the two most obvious
objections: it is neither a small-data nor a short-run artefact.

**Two predictions failed, informatively.** More local epochs was expected to sharpen the
server's signal and blunt the attack; placement never moves, but resemblance decays +0.758 →
+0.665, so the mechanism weakens without the outcome changing, the attack has *margin*, and
this measures it. More data was expected to help the server; it helps the **attacker**
(+0.785 → +0.908), because the attacker resamples from its own data too. The attack does not
exploit noise, so removing noise does not remove it.

### Two frontiers, failing for opposite reasons

| setting | attack | control | separation (§A1 metric) | what failed |
|---|---|---|---|---|
| concentration 5 | 0.000 | **0.333** | 2.14 | the **environment** |
| concentration 10 | 0.667 | 0.000 | 3.45 | transition |
| **overlap 0.0** | 0.000 | **0.000** | **17.31** | the **attack** |
| overlap 0.15 | 0.667 | 0.000 | 7.81 | transition |

**Only the controls separate these.** At concentration 5 the clusterer misplaces *honest*
clients a third of the time and separation is barely above the Dirichlet ceiling of 1.65,
there is no coherent group to join, so this is not a statement about the attack at all.

At **overlap 0** the environment is at its strongest anywhere in the campaign (separation
17.31, control a clean 0.000) and the attack still achieves nothing, with resemblance
collapsing to +0.084. The reason is the bound in §A3: disjoint groups mean the attacker holds
**none** of the target's classes, so resampling can manufacture nothing. **That bound,
previously shown arithmetically on a single class, is confirmed at the extreme where it binds
on every class at once.**

> The configuration that defeats this attack is disjoint groups, which is also the
> configuration where clustered federated learning has least reason to exist, since disjoint
> groups can simply be trained as separate models. Worth stating as a tension rather than
> presenting zero overlap as a mitigation.

**Resemblance tracks overlap almost linearly** (+0.084 → +0.491 → +0.725 → +1.299), which is
the mechanism stated directly: the attack's strength is the size of the intersection between
what the attacker holds and what the target looks like.

### A limitation the campaign found in itself

**Three seeds resolve a frontier's endpoints but not its middle.** Intermediate points read
0.667 with a verdict of *not distinguishable*, because a 2-of-3 outcome cannot be separated
from a 0-of-3 control at that sample size. Where each frontier lies exactly is **unresolved**
and would need 5+ seeds at concentration ∈ {8,10,12,15} and overlap ∈ {0.05,0.1,0.15,0.2}.
The endpoints (0.000 against 1.000) are unambiguous; the transitions are not, and quoting
0.667 as a point estimate would overstate what three seeds support.

---

## A7. Robust aggregation does not prevent the intrusion

Six aggregation rules, three seeds, attacker resamples to the target and then boosts:

| aggregator | placement, attacked | placement, control | accuracy lost vs its own control |
|---|---|---|---|
| **fedavg** | **1.000** | 0.000 | **+0.4119** |
| multikrum | **1.000** | 0.000 | +0.0098 |
| bulyan | **1.000** | 0.000 | -0.0025 |
| median | **1.000** | 0.000 | -0.0173 |
| trimmed | **1.000** | 0.000 | -0.0184 |
| krum | **1.000** | 0.000 | -0.0254 |

Each rule is read against **its own** control, not against FedAvg. The rules do not all reach
the same accuracy with no attacker present (krum 0.756, median 0.853), so a shared reference
would credit or penalise them for a difference the attack did not cause. Negative values are
noise around zero, not the attack helping.

**Placement is flat at 1.000 across every rule, range 1.000 to 1.000.** Damage lands only
under plain Federated Averaging. **Clustering decides who you are grouped with; aggregation
decides whose update counts, and a robust aggregator governs only the second.**

> **This replicates §B5 exactly, under a completely different kind of heterogeneity, and it
> corrects a claim in our report.** The report states that spoofing defeats Byzantine-robust
> defences. It does not. See [The replication](#the-replication) below.

---
---

# Part B. Concept shift, the prior study

Label-permuted MNIST, faithful Sattler clustering, 12 clients, 2 planted groups, one
attacker, five seeds.

## B1. The setting, and why it is not a free choice

The Clustered Federated Learning (CFL) literature plants clusters by **concept shift**.
Sattler gives each group a different random label permutation; the Iterative Federated
Clustering Algorithm (IFCA) rotates each group's images. Groups disagree about what an input
*means*, not about how often each class appears.

We use **label-permuted MNIST**. The Intrusion Detection System (IDS) data was ruled out by
measurement rather than preference: its classes are imbalanced 1.57:1, so permuting labels
also shifts the label marginal by 0.0712 against a within-group sampling scatter of 0.0448.
Concept shift and distribution shift are confounded there, so no effect measured on it could
be credited to either.

The server is **faithful Sattler**: the published split criterion on absolute norms
(`||sum d|| < eps1` and `max ||d|| > eps2`), complete-linkage bipartition on cosine
similarity, and thresholds tuned per experiment as the paper prescribes. The tuning happens
on an **attacker-free warmup**, so the criterion has never seen the attack it is judged
against.

**One documented departure.** Similarity is computed on the classifier head, following FedRep
and FedPer. Justified in §B6 and flagged wherever it appears.

## B2. The spoofing ladder: three tiers, three costs

| tier | server decides membership by | attacker does | cost | result |
|---|---|---|---|---|
| 1 | reading a **declared field** | edits the field | none | channel carries signal, ARI **0.665** |
| 2 | **IFCA**: client reports which model fits | reports the target index | none | **0.000 to 1.000** infiltration |
| 3 | **infers** from the submitted update | genuinely trains the target's task | retraining | **1.000** infiltration |

**The cost rises as the server trusts less, and never becomes prohibitive.**

Tier 1 was expected to be dead under concept shift, on the reasoning that permutation leaves
label marginals identical. It is not. That holds only for perfectly balanced classes, and
MNIST is 1.13:1. The declared channel still recovers the planted grouping at ARI 0.665, so
the earlier declared-metadata work in `old_code/` attacks a live channel. Compare §A2, where
the same channel reaches 1.000 because under distribution shift the histogram *is* the thing
the groups differ in.

## B3. The tier-3 spoof works by the mechanism claimed

Infiltration is an outcome. It says the server placed the attacker in the target cluster; it
does not say the attacker's update *resembled* that cluster. Measured directly, on the head,
before the split:

| | to target | to home | honest within | honest cross | resemblance |
|---|---|---|---|---|---|
| control (`beta=0`) | -0.192 | **+0.708** | +0.704 | -0.167 | **-0.03** |
| `relabel` (`beta=1`) | **+0.648** | -0.177 | +0.707 | -0.170 | **+0.93** |

The attacker's update **swaps allegiance**. It moves from sitting where an honest member of
its own group belongs, to sitting inside the target's within-group band. Resemblance goes
from 0.00 to 0.96 across three seeds.

**This is why consistency checking cannot catch it.** The attacker is not lying about
anything. It trained the target's task and submitted the honest result.

## B4. The full attack, five seeds

> **Provenance, corrected 31 Aug 2026.** These numbers are now produced by
> `cfl-spoofing.ipynb` Part 7, which runs the grid itself via
> `part_b.grid5.run_grid()` and cross-checks the saved `results/grid5.csv`
> against its own run. Previously the notebook loaded that CSV, so the figures
> below came from a script rather than from the notebook.

| condition | infiltration [95% CI] | victim final [CI] | rounds collapsed | signature |
|---|---|---|---|---|
| **control** | **0.000** [0.000, 0.000] | 0.950 [0.944, 0.956] | 0.0 / 6 | none |
| `relabel` | **1.000** [1.000, 1.000] | 0.950 [0.948, 0.954] | 0.0 / 6 | **none exists** |
| **`relabel` + `boost x2`** | **1.000** [1.000, 1.000] | **0.000** [0.000, 0.000] | **5.8 / 6** | **none exists** |
| `weight` + LIE | 1.000 | 0.943 | 0.0 / 6 | +0.101 |
| `weight` + Min-Max | 1.000 | 0.942 | 0.0 / 6 | +0.782 |
| `weight` beta=1.0 | 1.000 | 0.949 | 0.0 / 6 | -0.069 |

**The complete attack reaches 1.000 placement and drives the victim cluster to 0.000
accuracy, with zero-width confidence intervals on five seeds, while submitting an update with
no perturbation to detect.**

**Damage is reported as collapse count and final round, never as a mean.** The victim series
oscillates between collapsed and recovered, and averaging it produced a non-monotonic
artefact in which `boost x2` appeared to do more damage than `x10`. What `boost` actually
does is collapse the model to a **constant predictor**: all weights stay finite, and it emits
one class for every input.

**The attack strength is derived, not chosen.** `beta` was our own dial. LIE (Baruch et al.,
2019) and Min-Max (Shejwalkar and Houmansadr, 2021) derive the bound from the honest clients
instead. Min-Max permits only a **25% blend** toward the target and that is already enough,
keeping cosine **+0.782** to the attacker's own honest update against -0.069 at `beta=1.0`.
The unconstrained attack looked conspicuous only because it was pushed four times harder than
it needed to go.

## B5. Byzantine-robust aggregation does not prevent the intrusion

| aggregator | placement | victim under full attack | attacker's update selected |
|---|---|---|---|
| **fedavg** | **1.000** | **0.000** | 1.00 |
| krum | **1.000** | 0.942 | 0.00 |
| multikrum | **1.000** | 0.946 | 0.00 |
| bulyan | **1.000** | 0.950 | 0.00 |
| median | **1.000** | 0.951 | 1.00 |
| trimmed | **1.000** | 0.951 | 1.00 |

Control is 0.000 under all six.

**Clustering and aggregation are separate decisions.** The spoof defeats robust *clustering*
completely, since placement is identical under every aggregator, while robust *aggregation*
contains the damage entirely. Two distinct mechanisms do the containing. Krum, Multi-Krum and
Bulyan **reject** the update outright. Median and trimmed mean **accept** it, because
coordinate-wise rules use every client, and neutralise the boosted coordinates during
combination.

**FoolsGold never flags the attacker.** Weight 1.00 for attacker and honest clients alike, in
every condition. It detects clients whose update histories are unusually similar; ours is
similar to the target group by design, but not anomalously so. Computed as an observation,
not deployed.

## B6. Where the cluster signal lives, and what destroys it

| architecture | head share | BatchNorm coords | margin full | margin head | infiltration |
|---|---|---|---|---|---|
| dnn | 0.112% | 1,924 | **0.0001** | 0.648 | 1.000 |
| mlp_flat | 0.590% | 386 | **0.0001** | 0.961 | 1.000 |
| cifar_cnn | 0.253% | 0 | 1.002 | 1.505 | 1.000 |
| mnist_cnn | 0.308% | 0 | 1.134 | 1.550 | 1.000 |
| mlp_plain | 0.549% | 0 | 0.720 | 1.203 | 1.000 |

**The attack generalises.** Infiltration is 1.000 on all five architectures across both
seeds, with attacked Adjusted Rand Index (ARI) 0.6648 throughout.

**BatchNorm destroys the cluster signal in the full parameter vector.** On `mlp_flat` the
BatchNorm running statistics are **0.35% of coordinates and 100% of the update's magnitude**,
with pairwise cosine **+1.0000** across all twelve clients. They track *inputs*, and concept
shift leaves inputs unchanged, so they are identical regardless of group and swamp everything
else. The learned parameters sit at +0.045 by comparison.

> A server clustering on full weight vectors of a BatchNorm network is clustering on input
> statistics. It groups at random while its similarity matrix reads a healthy +0.9997.

> **This corrects an earlier explanation of ours.** We attributed full-vector blindness to the
> classifier head being small. Head share does not explain it: `dnn` has the *smallest* head
> and behaves like `mlp_flat`, while `cifar_cnn` is smaller still and behaves like the other
> convolutional networks. Head scoping helps on every architecture, but on BatchNorm networks
> much of that benefit comes from **excluding the buffers** rather than from isolating the
> classifier.

## B7. How much must the spoofer know?

Every result above hands the attacker the target's concept exactly. Degrading that copy by
*k* random label transpositions, and reading each panel against its own **measured control**:

| fidelity | K=2 infiltration | K=3 infiltration | resemblance |
|---|---|---|---|
| control | **0.000** | **0.333** | -0.013 |
| 1.000 | 1.000 | 1.000 | +0.958 |
| 0.800 | 1.000 | 0.722 | +0.705 |
| 0.633 | 1.000 | 0.722 | +0.519 |
| 0.467 | 1.000 | 0.722 | +0.402 |
| 0.367 | 1.000 | 0.389 | +0.343 |
| 0.133 | 0.667 | 0.333 | +0.186 |

**The spoof degrades gracefully.** At half-correct knowledge it still reaches 0.722 at K=3
against a control of 0.333, and resemblance falls smoothly rather than collapsing, so the
attacker keeps genuinely resembling the target on a badly degraded copy. It reaches control
level only at fidelity 0.13. **The oracle assumption is not doing the work.**

The K=3 control is **0.333, not 0.000**. With three clusters the server does not recover the
planted grouping cleanly, so even an honest client is sometimes misplaced. That is a property
of the environment, and every K=3 number has to be read against it rather than against zero.

> **This is where the two studies diverge, and it is worth stating.** Here, degraded knowledge
> of the target *concept* still infiltrates. In §A4, degraded knowledge of the target
> *distribution* is not measurably better than no knowledge at all. Concept shift is learnable
> from a corrupted copy; a label profile is not readable from the broadcast model. So the
> distribution channel is the more knowledge-limited of the two.

## B8. A natural non-IID partition

Everything above imposes concept shift on an Independent and Identically Distributed (IID)
split, so the heterogeneity is ours. FEMNIST's writer split is the canonical *natural*
partition: each client is one real person's handwriting.

| condition | infiltration [95% CI] | ARI | victim final |
|---|---|---|---|
| control | **0.000** [0.000, 0.000] | **1.000** | 0.109 |
| `relabel` | **1.000** [1.000, 1.000] | 0.665 | 0.107 |
| `relabel` + `boost x2` | **1.000** [1.000, 1.000] | 0.665 | 0.117 |

**Placement transfers.** The server recovers the planted grouping exactly with no attacker
(ARI 1.000), and the spoof reaches 1.000 infiltration on both seeds against a 0.000 control.
The attack is not an artefact of synthetic partitioning.

**Head scoping is stronger here than on MNIST.** Full-vector margin is **0.0015**, near
chance, against head AUC 1.000. On real writer data the full parameter vector carries almost
no cluster information at all.

> **Payload was NOT MEASURED on FEMNIST.** Victim accuracy is about 0.11 in every condition
> *including the control*, and the control itself "collapses" in 2.5 of 8 rounds. FEMNIST
> gives each writer roughly 280 samples across 62 classes, so eight rounds of two epochs is
> badly undertrained and there is no headroom for damage to register. A damage number sitting
> at the control level is not evidence of anything, so it is reported as absent rather than as
> a result.

**A loader defect found on the way.** `_load_femnist` selected writers, pooled their rows and
shuffled, discarding the writer split that its own docstring called the reason for the
dependency. Every client came out with exactly 2143 samples. `partition="natural"` now uses
the real shards: 239 to 320 samples per client, 46 to 55 distinct classes each. Had this been
run before the fix, "natural non-IID" would have been false in a way no output revealed.

## B9. Fidelity to the published algorithms

**Sattler's published constants (0.4 and 1.6) never fire on our data.** They assume a
converged federation in which client updates cancel and the summed update approaches zero.
Over eight rounds ours does not approach it. Sattler treats `eps1` and `eps2` as
per-experiment quantities, so tuning them follows the paper rather than departing from it.
The tuning must happen on an attacker-free run, or the criterion has already seen the attack
it is meant to be blind to.

**The tier-2 spoof transfers to IFCA's own heterogeneity.** Rotated MNIST rather than the
label permutation borrowed from Sattler, and the declared-index spoof still succeeds. So that
result is a property of trusting a client-reported value, not of how the clusters were
planted.

---
---

# The replication

The two studies were built separately, on different heterogeneity, with different partitioners
and different attacks. They agree on the structural result:

| | Part A, distribution shift | Part B, concept shift |
|---|---|---|
| placement under **plain FedAvg** | 1.000 | 1.000 |
| placement under **all five robust rules** | **1.000** | **1.000** |
| honest control, every rule | 0.000 | 0.000 |
| damage under **plain FedAvg** | 0.41 accuracy lost | victim to 0.000, 5.8/6 rounds collapsed |
| damage under **every robust rule** | none | none |

**Clustering and aggregation are separate decisions, and a Byzantine-robust aggregator
governs only the second.** It protects the model without preventing the intrusion, so it is
not a defence against membership spoofing. An attacker seeking *access* to a cluster's model
rather than its destruction pays no price under any of the six rules, in either setting.

> **This corrects a claim in our report.** The report states that spoofing defeats
> Byzantine-robust defences. It does not, and the correction is not a hedge: the measurement
> is stronger than the original claim, because it separates two things the claim conflated.
> The supportable sentence is that **spoofing defeats robust *clustering* completely and
> robust *aggregation* not at all.** See `REPORT_NOTES.md` §2 for the wording.

One number crosses the two studies and looks like a contradiction until you read what it
measures: the declared channel recovers the grouping at **1.000** in Part A and **0.665** in
Part B. Explained at §A2. It is a finding about the channel, not a discrepancy.

---

# Limitations

Applying to both studies unless marked.

1. **One dataset, one architecture per study.** MNIST throughout. Part B extends to five
   architectures (§B6) and to FEMNIST (§B8); Part A does neither.
2. ~~**Twelve clients, one attacker (8.3%).**~~ **CLOSED for Part A** by §A6: swept to 3
   attackers of 12 (25%), placement 1.000 throughout against a 0.000 control. Still open for
   Part B, which was measured at one attacker only.
3. **Knowledge is the binding constraint in Part A** (§A4). Every distribution-channel
   infiltration number is an upper bound conditioned on oracle knowledge of the target
   profile, and probing the broadcast model does not recover it.
4. **`weight` and tier 2 still assume oracle knowledge in Part B.** `weight` reads the target
   members' actual deltas, and tier 2 reads their reported indices. Only `relabel` has been
   tested under degraded knowledge (§B7), where it survives.
5. **Stealth is reduced, not established.** Min-Max shrinks the signature from -0.069 to
   +0.782, but that compares the attacker to *its own* honest update, while the honest bands
   compare *different* clients. A client-versus-its-own-history baseline is needed before any
   indistinguishability claim.
6. **Payload is unmeasured outside MNIST.** FEMNIST confirms placement (§B8) but its models
   are too undertrained for damage to register.
7. **Prior-editing is measured for what it CLAIMS** (§A3), through the model's output
   distribution, not for whether that claim survives a server that checks behaviour.
8. **The Part A aggregator sweep uses three seeds**, not five, being the most expensive cell
   in that notebook. Part B's uses five.
11. **The parameter campaign (§A6) uses three seeds**, which resolves a frontier's endpoints
    but not its middle: a 2-of-3 outcome cannot be separated from a 0-of-3 control, so
    intermediate points read *not distinguishable*. The flat results are unambiguous; the two
    frontier locations are not, and would need 5+ seeds at concentration ∈ {8,10,12,15} and
    overlap ∈ {0.05,0.1,0.15,0.2}.
12. **The campaign is one-factor-at-a-time**, so interactions between parameters are not
    measured. Full factorial over five factors was estimated at hundreds of GPU-hours.
9. **Method details rest on a secondary reference.** Sattler (TNNLS 2020) and Ghosh et al.
   (NeurIPS 2020) are not in the repo, so the eps values and experimental setups need
   checking against the primary sources before submission.
10. **No defence measured**, by choice. There is therefore no false positive rate anywhere in
    this document.

---

# Provenance

Every table traces to a named run under `results/runs/<timestamp>_<label>/`, each with a
`manifest.json` recording what produced it. The flat `results/<label>.csv` is a pointer to the
latest run and is overwritten by design, so cite the timestamped folder.

**Part A**, all written by `cfl-distribution.ipynb`, executed 25 Aug 2026:

| § | run label | rows | what it holds |
|---|---|---|---|
| A4 | `dist_knowledge` | 15 | oracle / inferred / blind estimation error, 5 seeds x 3 levels |
| A5 | `dist_attack` | 10 | resample-to-target against the inferred channel, 5 seeds x 2 conditions |
| A7 | `dist_aggregators` | 36 | six aggregation rules, 3 seeds x 6 rules x 2 conditions |

The three tables in §A1, §A2 and §A3 are computed inline from the notebook's own partition and
are reproduced by re-running Parts 1 to 3; they are cheap and hold no run of their own.

**The §A6 campaign**, in [`experiments/`](experiments/). One folder per experiment, named for
the change it makes, each holding its own `data/results.csv`, `graphs/`, `README.md`,
`manifest.json` and `log.txt`:

| experiment | rows | what it varies |
|---|---|---|
| `01_structure_envelope` | 83 | K × concentration × overlap, no training |
| `02_n_groups_2-3-4-6` | 24 | the confound gate |
| `03_attackers_1-2-3` | 18 | 8.3 → 25% malicious |
| `04_rounds_6-12-20-30` | 24 | federation length |
| `05_epochs_1-2-5-10` | 24 | local training |
| `06_max_train_6k-12k-30k-60k` | 24 | data per client, to full MNIST |
| `07_concentration_5-10-20-50-100` | 30 | how tight the groups are |
| `08_overlap_0-15-30-50pct` | 24 | how distinct the groups are |
| `10_target_choice` | 80 | **which** of the other groups is attacked, at K=4 and K=6 |

Reproduce any of them with `python sweeps.py <name>`. `experiments/campaign.csv` concatenates
all eight training experiments into one 248-row table.

**Every campaign row carries its complete configuration**: 47 columns covering provenance
(run id, git commit, code digest, device, timing), the full experimental state including the
per-seed tuned `eps1`/`eps2`, the condition, and the measurements. That is deliberate: the
older files recorded only the varied parameter, so a row could not say what produced it, two
experiments could not be pooled, and a guard could not check the configuration it was
presented as. `run_context.load_experiment(name, expect=...)` now refuses a file whose pinned
configuration does not match, whose pinned column *varies*, or where any treatment cell lacks
a control. `HANDOFF.md` records four incidents of stale results being pooled with fresh ones;
this is the machinery that makes a fifth detectable rather than silent.

> **Reproducibility, checked rather than assumed.** The notebook was executed twice from a
> clean kernel, and every number in Part A came out **bit-for-bit identical** across the two
> runs, down to the last reported decimal. Seeding is pinned per condition via
> `set_global_seeds(seed, backend="torch")` before each server is constructed, so this is the
> expected behaviour rather than a lucky draw, but it had not been verified before.

**Part B**, produced by the drivers in [`part_b/`](part_b/), which also holds a
README mapping each script to its run folder and grid. Those scripts were rescued
into the repo on 25 Aug 2026; until then they existed only in a session temp
directory, which is why older citations name them as `scratchpad/<name>.py`:

| § | run label | rows | produced by |
|---|---|---|---|
| B4 | `grid5` | 35 | `part_b/grid5.py` |
| B5 | `agg_sweep` | 54 | `part_b/agg_sweep.py` |
| B6 | `arch_sweep` | 10 | `part_b/arch_sweep.py` |
| B7 | `ladder` | 42 | `part_b/ladder.py` |
| B8 | `femnist` | 6 | `part_b/femnist.py` |

Output from the retired Dirichlet line is quarantined in `old_code/results/` and is **not**
comparable to anything here. See `old_code/README.md`.

Aggregators are reimplemented from the published equations rather than adapted from the
authors' repositories, several of which carry no licence. The self-test in `aggregation.py`
cross-checks them against **ByzFL** (MIT).

---

# Method notes worth carrying forward

Five results in this study began as a confident number that agreed with the hypothesis while
the mechanism underneath had changed. Each was caught by a breakdown, never by the summary:

| the number | what it hid | what caught it |
|---|---|---|
| full-vector cosine +0.9997 | BatchNorm buffers, not weights | per-scope split |
| `boost x2` worse than `x10` | an oscillating series | per-round trace |
| infiltration 1.000 under corruption | measurement taken after the split | resemblance column |
| "attack is loud", cosine -0.035 | a dial pushed past what was needed | derived constraints |
| JS separation of recovered groups | identical marginals under concept shift | ground-truth swap |
| infiltration flat at 0.67 including control | Dirichlet had no groups to infiltrate | exhaustive split search (§A1) |

**A summary statistic that confirms the hypothesis is the one to distrust.**

The full account of what failed and why is in `F_ATTEMPTS.md`.
