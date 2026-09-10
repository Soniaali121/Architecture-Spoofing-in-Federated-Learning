# Membership spoofing under distribution shift

**Part A of the Clustered Federated Learning attack study, presented with the figures.**

> **Which file do I want?**
> This one is **what you show**, read top to bottom, with every figure in place.
> [`SLIDES.md`](SLIDES.md) is the same content as a Marp deck, one idea per slide.
> [`SUPERVISOR_SCRIPT.md`](SUPERVISOR_SCRIPT.md) is **what you say**, with timings and
> anticipated questions. [`RESULTS.md`](RESULTS.md) is the full write-up including Part B.

**Scope.** Part A only, the distribution-shift study. MNIST, 12 clients, 2 planted
groups, 6 rounds of 2 epochs, one attacker, five seeds, unless a section says otherwise.

## The one-sentence version

> Following the steer toward data distribution, the attacker can imitate a target
> group's distribution **for free**, but it cannot **work out** what that distribution
> is. The attack is limited by **knowledge**, not by capability.

Two results are worth your time above the others. **The correction**: robust aggregation
does not stop the intrusion, only the damage, which contradicts a paragraph in our own
literature review. **The limitation**: the knowledge channel that was meant to be the
contribution did not survive its own measurement.

## 0. Method, and the five questions I expect first

These are the questions that decide whether anything after them is worth hearing, so
they are answered up front rather than in an appendix.

### 0.1 How is the clustering actually performed?

**Sattler's cosine method, unmodified.** The server computes pairwise cosine similarity
between the *update deltas* clients submit, and bipartitions a group when it looks like
it contains more than one data distribution. The split criterion is:

```
split when   || mean(d_i) || < eps1    and    max_i || d_i || > eps2
```

The reasoning is that at a stationary point of a **single** distribution the client
deltas cancel, so their mean goes quiet while the individual deltas do not. A group whose
mean update has gone quiet while its members are still moving hard is being pulled in two
directions at once.

Four things worth stating because they are where this could go wrong:

- **Nothing is declared.** The clusterer sees submitted weights only. The planted group
  labels exist for scoring and are never passed to it.
- **Similarity is scoped to the classifier head**, following FedRep and FedPer. On the
  full vector the cross-group cosine is +0.9990 and the criterion can **never** fire; on
  the head it is -0.1925, which is the opposed-gradient condition the criterion detects.
- **`eps1` and `eps2` are tuned on an attacker-free warmup** of 4 rounds, per seed, and
  the chosen values are recorded in every row. A criterion tuned with the attacker present
  would be a defence that has already seen the attack.
- **It is the published algorithm and we did not improve it.** We are analysing an attack,
  so the clusterer is the *environment*, not our contribution. A clusterer we invented
  would make this a result about a straw man.

### 0.2 How does the imitation work, and how do you know it worked?

The attacker resamples **its own data**, with replacement, toward a blended target:

```
blended = (1 - beta) * its_true_histogram  +  beta * target_group_profile
```

`beta` is the commitment dial, and **`beta = 0` is the control**: the same client, same
code path, no imitation.

We do not assume the resampling worked. It is measured three ways:

| check | what it shows | result |
|---|---|---|
| achieved histogram vs requested | did the data actually shift? | matches on every class the attacker holds; falls short **only** on classes it holds none of |
| `target_resemblance` | does the *update* look like a member's? | **+0.725** at the operating point, against an honest control near **0.000** |
| `shortfall` recorded per row | how much mass could not be manufactured | **0.146** on the zero-held class, exactly the target's mass there |

That middle row is the one that matters. Infiltration says the server **placed** the
attacker; resemblance says it genuinely **looked like** a member. Both are needed, because
placement alone cannot distinguish imitation from being evicted into the only other group.

### 0.3 Are the malicious clients really training on their own data?

**Yes, through the identical code path as every honest client.** This is the question I
would ask too, so here is the evidence rather than an assurance.

In `fl_loop.py` the loop is the same for everyone. The only attacker-specific step is
which rows it trains on:

```python
X_train, y_train = self.attack.training_data(client_id, X, y)   # honest: returns X, y unchanged
update = train_client_torch(client_id, X_train, y_train, ...)   # SAME function for all 12 clients
```

- The attacker gets **no extra epochs, no different architecture, no privileged data**.
  It trains on a resampled draw from **its own** examples.
- The resampling cannot invent data. A class it holds none of stays at zero, which is
  measured and is itself a reported result.
- **The server never branches on who the attacker is.** `is_attacker` is written into the
  record twice, both times for scoring, and appears nowhere in `clustering.py` or
  `aggregation.py`. The attack code says so explicitly: this flag "must never route
  control flow through it: a loop that treats the attacker differently because it knows
  who the attacker is has assumed away the problem."

**And it does contribute into the wrong group.** After clustering, its update is
aggregated into whichever cluster it landed in. That is the harm: at the operating point
it lands with the target group in **100%** of post-split rounds while an honest client
lands there in **0%**, and under plain FedAvg its boosted contribution costs that group
**41 percentage points** of accuracy.

### 0.4 What exactly was run?

| | operating point | swept over |
|---|---|---|
| dataset | MNIST | (one dataset, stated as a limitation) |
| architecture | `mlp_flat`, 784-128-64-10 with BatchNorm | 5 architectures in the concept-shift study |
| **clients** | **12** | fixed |
| **malicious** | **1, so 8.3%** | **1, 2, 3, up to 25%** |
| **distinct groups** | **2** | **K = 2, 3, 4, 6** |
| rounds | 6, after a 4-round attacker-free warmup | 6, 12, 20, 30 |
| local epochs | 2 | 1, 2, 5, 10 |
| training images | 12,000 | 6k, 12k, 30k, 60k |
| group tightness | concentration 20 | 5, 10, 20, 50, 100 |
| group distinctness | overlap 0.3 | 0.0, 0.15, 0.3, 0.5 |
| aggregation | FedAvg | 6 rules incl. Krum, Bulyan, median |
| **seeds** | **5** in the notebook | **3** in the campaign, **5** in experiment 10 |

The notebook results are five seeds. The campaign sweeps are three, which is stated as a
limitation: three seeds resolve a frontier's endpoints but not its middle.

### 0.5 How was the attack evaluated?

Four measurements, and **every one is quoted beside a control measured on the same setup**,
never against a theoretical baseline.

| measure | what it answers | its baseline |
|---|---|---|
| **infiltration** | did it get in? | the **measured honest control**, not 1/K |
| **target_resemblance** | did it get in *by imitating*? | the honest cross-group band |
| **final accuracy** | did the disguise cost it anything? | the no-attack control on the same setup |
| **accuracy lost** | did the payload do damage? | **each aggregation rule's own** control |

Three rules govern how those numbers are read:

1. **Infiltration counts only post-split rounds.** Before the first split every client
   shares one cluster, so the attacker is trivially "with" the target and so is everyone.
2. **Membership is tracked by client id, never by cluster name**, because cluster labels
   are reassigned every round. Each round the target's cluster is re-derived by majority
   vote of its known members.
3. **Every headline is a mean over seeds with a bootstrap 95% interval, and where two
   intervals overlap the result is reported as *not distinguishable*.** Section 5 is
   exactly that case and it costs us one of our own hypotheses.

## 1. The measurement that forced the setup to change

*Code: `cfl-distribution.ipynb` **Cell 7**.*

Before asking whether an attacker can join group B, "group B" has to be a real thing in
the data. Scoring **every possible** two-way split of 12 clients by separation ratio,
cross-group divergence over within-group divergence, gives a **ceiling**: the best
grouping that exists anywhere, which no algorithm can beat.

![Separation ratio](docs/figures/lecture_2_separation.png)

| partition | best split anywhere | median random split |
|---|---|---|
| Dirichlet(0.05) | **1.65** | 0.99 |
| Dirichlet(0.2) | 1.43 | 0.99 |
| Dirichlet(1.0) | 1.46 | 0.99 |
| planted, overlap 0.5 | 2.38 | n/a |
| **planted, overlap 0.3** | **4.51** | n/a |
| planted, overlap 0.0 | 17.31 | n/a |

**Dirichlet partitioning, the field's standard non-Independent-and-Identically-Distributed
(non-IID) generator, contains no group structure.** It skews every client
*independently*, so you get twelve individually unusual clients and no groups. A ratio of
1.0 means no structure, and the measured median over random splits is 0.99, so the
instrument reads correctly where the answer is already known.

**Why this mattered enough to restart.** An earlier line of work measured infiltration
into Dirichlet data and found it flat at 0.67 across every mechanism including the
do-nothing control. The reason is now plain: there was nothing to infiltrate. That work
is retired in `old_code/` and is not cited.

## 2. What the setting looks like instead

*Code: `cfl-distribution.ipynb` **Cell 8**.*

![The setup](docs/figures/lecture_1_setup.png)

Each row is a client, each column a digit class, darker means more of it. On the right,
sorted by planted group, the block structure is what the server is trying to recover. On
the left, the standard recipe, there is nothing to recover.

## 3. Both membership channels carry real signal, and both are spoofable

*Code: `cfl-distribution.ipynb` **Cells 10 and 11**. Head scoping: `models.head_indices`.*

| channel | what the server uses | recovery |
|---|---|---|
| **declared** | the client's reported label histogram | ARI **1.000**, all five seeds, against 0.000 for a random partition |
| **inferred**, full vector | cosine similarity of whole update deltas | AUC 0.8852, margin **0.0002** |
| **inferred**, classifier head | cosine similarity of head deltas only | AUC **1.0000**, margin **0.7545** |

![The two channels](docs/figures/lecture_3_channels.png)

**The full vector is a trap.** An Area Under the Curve (AUC) of 0.8852 looks respectable
until you read the margin: within-group similarity 0.9992 against cross-group 0.9990.
Every client's full update is 99.9% similar to every other's, so any real threshold has
nowhere to sit. Restrict to the classifier head and the margin becomes 0.7545.

**This is not the groups holding similar data.** The declared channel recovers the same
grouping at ARI 1.000, and separation is 4.51. It is the same clients measured two ways,
and one way looks at the wrong part of the model. `mlp_flat` is 784 to 128 to 64 to 10:

| part of the model | values | share |
|---|---|---|
| first dense layer, 784 to 128 | 100,480 | 91.22% |
| second dense layer, 128 to 64 | 8,256 | 7.49% |
| BatchNorm scale and shift | 384 | 0.35% |
| BatchNorm running mean and variance | 384 | 0.35% |
| **classifier head, 64 to 10 plus bias** | **650** | **0.59%** |
| total | 110,154 | |

**The head is the only place a class preference can live.** If a client's data is 40%
eights, gradient descent pushes the class-8 row and bias in a specific direction. The
trunk learns reusable shape features that every client needs identically, because ink is
ink no matter which digits you hold.

**Cosine is driven by magnitude, not by parameter count**, and that is what makes the
full vector useless. The BatchNorm running statistics are only **0.35% of coordinates but
carry 100% of the update's magnitude**, with pairwise cosine **+1.0000** across all twelve
clients. They track *inputs*, which are near-identical for everyone reading handwritten
digits. So cosine on the full vector measures the one component the whole federation
shares.

**The clinching number.** Both halves from Cell 11's single run, so this is one
measurement rather than two stitched together. On the head, cross-group cosine is
**-0.1925**. Negative means **opposed** gradients, which is exactly the condition Sattler's
split criterion is built to detect. On the full vector it is **+0.9990**, positive, so the
criterion **can never fire at all**. Head scoping is not a tuning preference, it is what
makes the published method work on this architecture.

Both channels work, so neither is a straw man. The declared channel scoring 1.000 is
exactly what makes it worth attacking: the server relies on a field the client writes and
never checks.

## 4. Two imitation routes, and they differ in kind

*Code: `cfl-distribution.ipynb` **Cells 13, 14 and 21**.*

![Two routes](docs/figures/lecture_4_routes.png)

Measured on the classes the attacker holds **none** of, averaged over five seeds:

| | value |
|---|---|
| attacker's holdings | 0.000 |
| the target's profile wants | 0.146 |
| **resampling achieves** | **0.000** |
| **prior-editing claims** | **0.155** |

**Resampling** redraws the attacker's own training set toward the target profile. It is
bounded by holdings: sampling with replacement cannot invent an example you do not have.

**Prior-editing** leaves the data alone and shifts the final layer's bias, following the
logit-adjustment identity (Menon et al., ICLR 2021). For a class the model genuinely
outputs 0.0000 on, a bias shift of **+18.8** makes it claim 0.146.

**The two differ in kind, not degree.** Resampling produces a model that has *learned*
the distribution. Prior-editing produces one that merely *reports* it. That makes
prior-editing strictly stronger against a server reading what a model claims, and
strictly weaker against one that tests what it can do. **Nobody has built that test here,
so it is a lead rather than a result.**

## 5. The limitation, volunteered

*Code: `cfl-distribution.ipynb` **Cell 16**.*

Every result above assumes the attacker knows the target distribution. Where would it get
that? Three knowledge levels, scored by estimation error against the truth, lower better:

![The knowledge ladder](docs/figures/lecture_5_knowledge.png)

| knowledge | error | 95% interval |
|---|---|---|
| blind, assume uniform | 0.0887 | [0.0792, 0.0970] |
| **inferred, probe the model** | **0.0844** | [0.0737, 0.0945] |
| oracle, told the answer | 0.0000 | [0.0000, 0.0000] |

`inferred` beats `blind` by about 5%, **and the intervals overlap.** Per seed it is
actually *worse* than blind on seed 2. So the honest reading is **not distinguishable**.

**A model trained on a skewed distribution does leak that skew, but only by a couple of
percentage points per class, while the truth it is trying to recover is strongly skewed.**
The leak is real and far too weak to attack through. This was the intended contribution
and it did not survive its own measurement.

**What that means practically:** the attack needs an informed insider or a leaked case-mix
summary. It is not something a random participant mounts from what the federation
broadcasts.

## 6. Imitation is free

*Code: `cfl-distribution.ipynb` **Cells 18 and 19**.*

Against the inferred channel the attacker must genuinely change its update, so it
resamples to the target and trains.

| statistic | measured | baseline | verdict |
|---|---|---|---|
| infiltration | **1.0000** | 0.0000, the honest control | above baseline |
| final accuracy | 0.8429 | 0.8328, the no-attack control | **not distinguishable** |

The attacker is very slightly *better* than the control, and the gap of 0.0101 sits well
inside the intervals. Resampling changes *which* distribution the model learns, not *how
well* it learns one.

**A defence that goes looking for a degraded or unusual-looking attacker will not find
one.** Had the disguise cost accuracy, that cost would itself have been a detection
signal. It does not.

## 7. The correction: robust aggregation contains damage, not intrusion

*Code: `cfl-distribution.ipynb` **Cell 23**.*

Our own literature review claims spoofing defeats Byzantine-robust defences, with no
experiment behind it. This settles it, and the answer is neither yes nor no.

![Aggregation](docs/figures/lecture_6_aggregation.png)

| aggregator | placement, attacked | placement, control | accuracy lost vs its own control |
|---|---|---|---|
| **fedavg** | **1.000** | 0.000 | **+0.4119** |
| multikrum | **1.000** | 0.000 | +0.0098 |
| bulyan | **1.000** | 0.000 | -0.0025 |
| median | **1.000** | 0.000 | -0.0173 |
| trimmed | **1.000** | 0.000 | -0.0184 |
| krum | **1.000** | 0.000 | -0.0254 |

Read the columns separately, because they say opposite things. **Placement is 1.000 under
every rule**: robust aggregation does not stop the attacker getting in. **Damage is
stopped completely**: plain averaging loses 41 percentage points, every robust rule loses
essentially nothing, and four of six end up marginally better than their own control.

**Why.** These are two different jobs. **Clustering decides who you are grouped with.
Aggregation decides whose update counts.** A robust aggregator only ever touches the
second, so it can discard the poisoned contribution while the attacker still sits inside
the target group receiving its model.

So the correct claim is: **robust aggregation contains the damage but not the intrusion.**
The concept-shift study reaches the same result independently, which makes the correction
a replication rather than a single experiment.

Each rule is scored against **its own** control, because the rules do not all reach the
same accuracy with no attacker present.

## 8. The campaign

*Code: driven by [`sweeps.py`](sweeps.py), one folder each under [`experiments/`](experiments/). The grid figure is `cfl-distribution.ipynb` **Cell 30**.*

Everything above was measured at **one configuration**. A campaign of nine experiments
widened it. Eight of them train and pool into one table; the first does no training.

| experiment | the question | rows |
|---|---|---|
| `01_structure_envelope` | is there group structure to find at all? | 83 |
| `02_n_groups_2-3-4-6` | imitation, or elimination? | 24 |
| `03_attackers_1-2-3` | does it need many malicious clients? | 18 |
| `04_rounds_6-12-20-30` | can a defender wait it out? | 24 |
| `05_epochs_1-2-5-10` | does more local training change it? | 24 |
| `06_max_train_6k-12k-30k-60k` | does it survive on full MNIST? | 24 |
| `07_concentration_5-10-20-50-100` | how tight must groups be? | 30 |
| `08_overlap_0-15-30-50pct` | how distinct must groups be? | 24 |
| `10_target_choice` | does it matter **which** group is attacked? | 80 |

**`experiments/campaign.csv` holds 248 rows across 100 experimental cells, 0 of which
lack a control.** Every row carries its **complete** 51-column configuration, not just the
varied factor. That is what lets any two experiments be pooled, and lets a loader refuse a
file whose configuration is not what the caller claims. This project has four recorded
incidents of stale results being silently mixed with fresh ones; the schema is what makes
a fifth loud instead of silent.

There is no experiment 09. It was specced as aggregators-at-an-extreme and deliberately
not run, since aggregation governs whose update counts rather than who is grouped with
whom, and section 7 already answered it at lower cost.

![The campaign grid](docs/figures/campaign_grid.png)

> **Caption, stated honestly.** This grid was produced by the notebook from experiments
> 02 to 08 and **does not include experiment 10**, which postdates it. Refreshing it
> requires re-executing the notebook. The flat panels are as much of the result as the
> two that fall.

## 9. It is imitation, not elimination

*Code: `cfl-distribution.ipynb` **Cell 25**, which loads [`experiments/02_n_groups_2-3-4-6/`](experiments/02_n_groups_2-3-4-6/) rather than recomputing.*

With two groups, an attacker that merely stops resembling its own group lands in the
other by default. That would be **elimination**, not imitation, and it would make the
headline far less interesting. The test, stated before the run: add groups. If it is
elimination, infiltration should collapse toward roughly 1/(K-1).

![Imitation not elimination](docs/figures/lecture_7_imitation.png)

| groups (K) | infiltration | resemblance | control | control resemblance |
|---|---|---|---|---|
| 2 | **1.000** | 0.725 | 0.000 | -0.101 |
| 3 | **1.000** | 1.109 | 0.333 | 0.066 |
| 4 | **1.000** | 0.986 | 0.000 | 0.073 |
| 6 | **1.000** | 1.219 | 0.222 | 0.065 |

Infiltration holds at 1.000 to six groups, where there are five wrong clusters and one
right one. Elimination is ruled out. And resemblance does not merely stay positive, it
**rises**, from +0.725 to +1.219, while honest controls sit near zero. The attacker is
actively coming to look like a member.

**Note the controls at K=3 and K=6 are 0.333 and 0.222, not zero.** The clusterer
misplaces honest clients at those settings too, which is exactly why every number here is
read against its own measured control rather than against zero.

## 10. Placement is invariant across four factors

*Code: `cfl-distribution.ipynb` **Cells 27 and 29**, loading experiments 03 to 06.*

| factor | swept over | infiltration | control |
|---|---|---|---|
| malicious clients | 1, 2, 3 (8.3% to 25%) | 1.000 throughout | 0.000 |
| federated rounds | 6, 12, 20, 30 | 1.000 throughout | 0.000 |
| local epochs | 1, 2, 5, 10 | 1.000 throughout | 0.000 |
| training images | 6k, 12k, 30k, 60k | 1.000 throughout | 0.000 |

The attack does not need numbers: one client that genuinely resembles the target is
already enough, so more attackers cannot improve on 1.000. And a defender cannot wait it
out, because once the cluster tree splits it never re-merges. **The flatness is the
result**, and it is what stops the headline being an artefact of one configuration.

## 11. Two frontiers, failing for opposite reasons

*Code: `cfl-distribution.ipynb` **Cell 29**, loading experiments 07 and 08.*

![Two frontiers](docs/figures/lecture_8_frontiers.png)

| concentration | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| infiltration | **0.000** | 0.667 | 1.000 | 1.000 | 1.000 |
| **control** | **0.333** | 0.000 | 0.000 | 0.000 | 0.000 |

| overlap | 0.00 | 0.15 | 0.30 | 0.50 |
|---|---|---|---|---|
| infiltration | **0.000** | 0.667 | 1.000 | 1.000 |
| **control** | **0.000** | 0.000 | 0.000 | 0.000 |
| resemblance | **0.084** | 0.491 | 0.725 | 1.299 |

**Both read 0.000 and they mean opposite things. Only the controls separate them.**

**Concentration 5, the environment fails.** The control is 0.333, so the clusterer is
misplacing honest clients too, and separation there is 2.14, barely above the Dirichlet
ceiling of 1.65. Nothing can be concluded about the attack here.

**Overlap 0, the attack fails.** A clean 0.000 control and separation 17.31, the strongest
setting measured. The environment is perfect and the attacker breaks: disjoint groups mean
it holds none of the target's classes, so resampling can manufacture nothing and
resemblance collapses to 0.084.

That second one is the only clean defensive win in the study, and it costs completely
disjoint classes between groups, which is also the configuration where Clustered Federated
Learning has least reason to exist. **A boundary on the attack, not a deployable
mitigation.**

## 12. Victim choice, and a prediction we got wrong

*Code: `cfl-distribution.ipynb` **Cell 32** (Part 11, added 27 Aug 2026). Produced by `sweeps.sweep_targets()` in [`sweeps.py`](sweeps.py) via `python sweeps.py targets`; data in [`experiments/10_target_choice/`](experiments/10_target_choice/).*

At K>2 the attacker has a choice of victim that no experiment had exercised: the code took
"the first other group in sorted order" and nothing varied it.

**The prediction, fixed before the run:** the **most diffuse** target should be easiest,
because a diffuse profile sits near uniform and uniform is what a blind attacker declares.

![Which target](docs/figures/lecture_9_targets.png)

| half of the data | mean spread | mean resemblance | 95% interval |
|---|---|---|---|
| diffuse targets | 0.1680 | **0.9941** | [0.8559, 1.1177] |
| peaked targets | 0.2380 | **1.1784** | [1.1300, 1.2303] |

**Falsified, and consistently. All 10 of the 10 (K, seed) blocks run the wrong way for
the prediction, sign test p = 0.0020.** The obvious confound is clean: distance from the
attacker's own group correlates at rho +0.037, p = 0.821.

**Why it reverses.** Resemblance is a *relative* measure. Imitating a diffuse target makes
the attacker look like **everybody**, which marks it as a member of nobody. A peaked
target is distinctive, so imitating it lands somewhere **specific**. The prediction
reasoned from estimation error, which turns out not to govern resemblance at all.

Two honest notes. The intervals clear each other by only 0.0123, so the half-split alone
is marginal; **the consistency across all ten blocks is the evidence, not the size of the
gap.** And the pooled correlation reads +0.260 with an interval crossing zero, which looks
like no effect: that pooled figure is the wrong test, because target rank is not a stable
identity across seeds.

**Because the prediction was written down before the run, we can say we were wrong
plainly, rather than quietly reinterpreting it.**

## 13. What this does not establish

- **There is no defence here, and therefore no false positive rate.** This is an attack
  study. Where visibility is measured, that describes the attack's signature, not a
  countermeasure.
- **The `inferred` knowledge channel does not work**, and that was the intended
  contribution.
- **One dataset, one architecture.** MNIST with a small dense network. CIFAR-10 untested.
- **Twelve clients is small.** Adjusted Rand Index (ARI) is coarse at that size: one
  misplaced client already costs a third of the scale.
- **The frontier midpoints are unresolved** at three seeds, and K=6 gives two-member
  groups where "majority of the target group" is a weak notion.
- **Two of our own stated predictions have been falsified by our own data**, the
  knowledge channel and the diffuse-target hypothesis. Both were pre-registered, which is
  the only reason that is a strength.

## 14. Three things I would like your view on

**First**, is the distribution study now the centre of the report, with the concept-shift
work as supporting replication, or the other way round? It is written up with distribution
leading, because that is the direction you pointed me in, but the concept-shift study is
the more complete one.

**Second**, are you comfortable with the correction to the Byzantine-robustness claim? It
affects a paragraph in the literature review that Amin and Sonia wrote, and it now holds
in two independent settings.

**Third**, should the research question be narrowed to match the evidence, or should I
build a detector and measure its false positive rate? I think narrowing is right, because
the negative result is the stronger contribution, but that is a change to what we said we
would do.
