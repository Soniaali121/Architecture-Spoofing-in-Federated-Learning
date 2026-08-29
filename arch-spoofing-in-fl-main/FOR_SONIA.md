# Sonia — start here

Written by Mohamed, 25 August 2026.

This is an introduction to the whole project: what it is, what Amin is building, what I have
built, how the two fit together, and what all of it means for the report. It assumes you have
not run any of the code and are not going to, so nothing here requires you to.

It is long because you asked for detail and because there is one finding in here that
**changes a paragraph in the Relevant Work section you and Amin wrote**. That part is in
§7, and I would rather you got the full picture before you got to it.

---

## Contents

1. [The project in one page](#1-the-project-in-one-page)
2. [The background you need](#2-the-background-you-need)
3. [How the three of us split the work](#3-how-the-three-of-us-split-the-work)
4. [What Amin is doing](#4-what-amin-is-doing)
5. [What I have done](#5-what-i-have-done)
6. [The findings, stated plainly](#6-the-findings-stated-plainly)
7. [What has to change in the report](#7-what-has-to-change-in-the-report)
8. [What is NOT established, and why saying so matters](#8-what-is-not-established-and-why-saying-so-matters)
9. [Which file to open for what](#9-which-file-to-open-for-what)
10. [Things you might reasonably ask me](#10-things-you-might-reasonably-ask-me)

---

## 1. The project in one page

**The system we are studying.** In **Federated Learning (FL)**, many devices train a shared
model without ever sending their data anywhere. Each device trains locally on its own private
data and sends only the resulting *model update* to a central server. The server averages all
the updates into one global model and sends it back. Repeat for a number of rounds. The data
never moves, which is the entire point — it is how you train on hospital records or phone
keyboards without collecting them.

**The problem that creates.** Averaging assumes everyone is trying to learn the same thing.
When devices hold very different data, the average is a compromise that suits nobody.

**The fix, and our subject.** **Clustered Federated Learning (CFL)** sorts clients into groups
and trains one model *per group*. Now each group gets a model suited to its own data. But this
forces the server to make a new decision: **which group does each client belong to?**

**Our attack.** That grouping decision is a trust boundary, and it is the one we attack. A
malicious client makes the server put it in a group it does not belong to. From inside that
group it receives the group's specialised model and its own updates are mixed into that
group's model. It can then either steal the model or poison it.

**Our research question, as currently written:**

> *How can Clustered Federated Learning systems identify and defend against architecture
> spoofing attacks where malicious actors misrepresent their device properties to gain
> unauthorized entry into localized clusters?*

**Keep that wording in mind. §7 explains why I think it has to narrow.**

---

## 2. The background you need

You do not need to code, but you do need six ideas to read anything else in the repository.

### 2.1 A client's "update" is a vector of numbers

When a client trains locally, its model's weights change. The **update** (we also call it the
**delta**) is simply `weights after training − weights before training`. It is a long list of
numbers, and it is the only thing the client sends. Everything the server knows about a
client, it knows from that vector plus whatever the client *declares* about itself.

### 2.2 There are two ways a server can decide who belongs where

This distinction runs through the entire project.

| | **declared** | **inferred** |
|---|---|---|
| what the server uses | metadata the client reports about itself | the update the client submits |
| example | "I am running a CNN", "here is my label distribution" | cosine similarity between client updates |
| verified? | **no** — the server takes the client's word | yes, in the sense that the update is real |
| attacked by | editing a field | having to actually change what you submit |

**The declared channel is where our project started.** It is easy to attack — you overwrite a
field — but a critic can fairly say no serious system would trust unverified metadata.

**The inferred channel is what the published CFL literature actually uses.** Sattler's method
computes the cosine similarity between the updates clients submit and splits a group when it
looks like it contains two different data distributions. The client is asked nothing, so
**there is no field to lie about**. Attacking that is the harder and more interesting result.

### 2.3 Clients can differ in two different ways

This one matters a lot, and it is the axis the supervisor pushed us along.

| | **concept shift** | **distribution shift** |
|---|---|---|
| groups disagree about | what an input *means* | how *often* each class appears |
| example | group A calls a picture of a 3 a "three"; group B calls the same picture a "seven" | group A holds mostly digits 0–4; group B mostly 5–9 |
| how often each label appears | **identical between groups**, by design | **different between groups** — this *is* the difference |
| who uses it | Sattler, IFCA — the CFL literature standard | what a practitioner means by "statistical heterogeneity" |

They are not two flavours of the same thing. They are attacked differently, they leak
differently, and as you will see in §6 the *same* server mechanism scores completely
differently on them.

### 2.4 "Non-IID" and "clustered" are different properties

**IID** means Independent and Identically Distributed — every client's data looks statistically
like every other's. **Non-IID** means it does not. Almost every FL paper generates non-IID
data with **Dirichlet partitioning**, which gives each client its own random skew.

But Dirichlet skews each client *independently*. Every client differs from every other, and
**nothing connects one client to another**. So the data is genuinely non-IID and contains **no
groups**. That distinction is the subject of §5.2 and it is the measurement that changed our
experimental setup.

### 2.5 A control is not optional

Every number in my work is reported beside the same experiment with the attack **switched
off**, on the same random seeds. Without that, "the attacker got into the target cluster 100%
of the time" is meaningless — you cannot tell it from a clusterer that puts everyone
everywhere.

Where the difference between a measurement and its control is smaller than the statistical
uncertainty, I report it as **"not distinguishable"** rather than as a small effect. There are
places in the results where that is the honest answer and I have not dressed them up.

### 2.6 The three metrics you will see

- **ARI** (Adjusted Rand Index) — how well the server's grouping matches the true grouping.
  **1.0 is perfect, 0.0 is random.** It is chance-corrected, so a random guess really does
  score 0. For our setup of 12 clients in 2 groups of 6, **0.665 means exactly one client is
  in the wrong group.**
- **AUC** (Area Under the Curve) — 0.5 is a coin flip, 1.0 is perfect separation.
- **Infiltration rate** — the fraction of rounds where the attacker was grouped with the
  target group. **The baseline is the measured honest control, which is 0.000**, not 0.5.

---

## 3. How the three of us split the work

| | slice | output |
|---|---|---|
| **Amin** | the **environment**. Convolutional Neural Networks (CNNs), three datasets, and a benchmark comparing the two clustering modes | `arch-spoofing-in-fl-adding-cnns/arch-cfl.ipynb` |
| **Mohamed (me)** | the **attack analysis**. Two studies of how the grouping decision is spoofed, and what does and does not stop it | `cfl-spoofing.ipynb`, `cfl-distribution.ipynb`, `RESULTS.md` |
| **You** | the **write-up**, including the Relevant Work section you and Amin wrote | the report |

The two code slices are complementary rather than overlapping. **Amin measures how well the
system works when nobody is attacking it. I measure what happens when somebody is.** You need
both: an attack result is meaningless without a demonstration that the thing being attacked
works properly in the first place.

---

## 4. What Amin is doing

His work is in `arch-spoofing-in-fl-adding-cnns/`, a branch forked from an earlier commit and
never merged into the main tree. His notebook is `arch-cfl.ipynb`, 17 cells.

### 4.1 What it contains

**Six literature-grounded CNNs.** The original project used only small dense networks
(multi-layer perceptrons). Amin added proper convolutional architectures, which is what a
real image-classification deployment would use. This matters for credibility: an attack
demonstrated only on toy dense networks invites the objection that it would not survive on a
real model.

**Three datasets, with loaders for each.**

| dataset | what it is | how clients are split |
|---|---|---|
| MNIST | handwritten digits, 10 classes | Dirichlet, α = 0.5 |
| CIFAR-10 | small colour photos, 10 classes | Dirichlet, α = 0.5 |
| **FEMNIST** | handwritten characters, 62 classes, **labelled by which person wrote them** | **natural writer shards** |

**FEMNIST is the valuable one.** Each client is one real person's handwriting. That means the
non-IID structure was **not designed by us** — it is a genuine property of real data. It is
much more defensible against an examiner than a synthetic Dirichlet split, and it is the
canonical natural non-IID benchmark in the FL literature.

**A benchmark comparing the two clustering modes.** He runs the full CFL pipeline in two
configurations — clustering by declared **architecture**, and clustering by declared
**distribution** — and compares final accuracy across all three datasets, six runs in total.
His results, from the saved outputs in his notebook:

| dataset | mode | final accuracy |
|---|---|---|
| MNIST | architecture | leaf_cnn 0.972, mnist_cnn 0.976 |
| MNIST | distribution | cluster_0 **0.967**, cluster_1 **0.664** |
| CIFAR-10 | architecture | cifar_cnn 0.485, cifar_cnn_deep 0.473 |

**Distribution-clustering visualisations.** Label-histogram heatmaps (rows are clients,
coloured strip shows the assigned cluster), stacked label mass per client, and a cluster
assignment table. These are genuinely good figures and they show *what the clustering is
actually doing* rather than just its score. There are five of them saved in the notebook.

**One practical note before we push.** His notebook loads its numbers from
`results/notebook_cnn_benchmark.json`, and **that file is not in the repository**. The
notebook still reads fine because its outputs are saved, but nobody can re-run the loading
cell. Worth asking him to commit it.

### 4.2 What I have taken from his branch

His CNN architectures and his dataset loaders were adapted into my `lab_models.py` and
`lab_data.py`. **The FEMNIST writer-based loader in particular is his contribution and it
carries one of my results** — the natural non-IID condition in `RESULTS.md` §B8 exists because
of it.

I also took the idea behind his label-histogram heatmap. There is a version of it in my
distribution notebook, and the notebook says where it came from.

**I have not merged his branch and I do not intend to.** The two trees have diverged and a
merge would be a large, risky operation shortly before submission. Taking the specific pieces
and crediting them is safer.

### 4.3 One thing about his branch that is worth knowing

His branch forked from a commit *before* the fingerprinting work, and in that older code the
architecture-spoofing attack does not actually function: the spoofing client trains its *true*
architecture rather than the one it declares, so the shapes do not match and the server
discards its update before it can do anything.

**This does not affect his benchmark**, because his benchmark has no attacker in it — it
measures the honest system. It only means his branch cannot be used to run attack experiments
as it stands. I mention it so nobody quotes an attack number from that tree by accident.

---

## 5. What I have done

My slice is the attack analysis. It is **two studies**, and the reason there are two is worth
understanding because it is the strongest thing we have.

### 5.1 The two studies

| | **Study B: concept shift** | **Study A: distribution shift** |
|---|---|---|
| notebook | `cfl-spoofing.ipynb` | `cfl-distribution.ipynb` |
| groups differ in | what an input means | how often each class appears |
| status | the prior study | **the current direction**, after the supervisor's redirection |
| scale | 38 cells, 5 architectures, 2 datasets | 25 cells, 9 parts |

They are labelled A and B in `RESULTS.md`, with **A first** because the supervisor asked us to
focus on data distribution and statistical heterogeneity, and A is that work.

**Why keep both.** Because they agree. The central negative finding (§6.5) holds in both, with
different data, different partitioners and different attacks. One experiment showing something
is a finding. **Two independent settings showing the same thing is a replication**, and that is
what makes it strong enough to correct a claim in our report.

### 5.2 The measurement that changed our experimental setup

This is the single most useful thing I found, and it invalidated the direction the project was
originally going in.

Before running any attack, I asked a question nobody had asked: **does the data we are
clustering actually contain groups?**

I checked exhaustively. With 12 clients there are about two thousand ways to split them into
two halves. I scored **every single one** by how much more different the two halves are from
each other than their own members are from each other. A score of **1.0 means no group
structure at all.**

| how clients were split | best split available *anywhere* | a random split |
|---|---|---|
| Dirichlet, α = 0.05 | 1.65 | 0.99 |
| Dirichlet, α = 0.2 | 1.43 | 0.99 |
| Dirichlet, α = 1.0 | 1.46 | 0.99 |
| **planted groups (what I use now)** | **4.51** | — |
| planted groups, fully separated | **17.31** | — |

Read the Dirichlet rows carefully. That is **not** "our clustering algorithm failed to find the
groups". It is **"the best grouping that could possibly exist in this data is barely
distinguishable from a coin flip"**. No algorithm can find structure that is not there. That is
why I searched exhaustively instead of just running a clusterer — it removes "maybe the
algorithm is bad" as an explanation.

**Why this mattered so much.** Our earlier work partitioned clients with Dirichlet and then
asked a server to recover two clusters from it. Those "clusters" were arbitrary sets of
clients, so "the attacker infiltrated cluster B" was a statement about nothing in particular.

**And it solved a mystery.** For weeks our old results showed infiltration stuck at 0.67 across
*every* attack strategy — including the do-nothing control, where the attacker did nothing at
all. It looked like a bug and nobody could explain it. It was not a bug. **Placement was never
being driven by data similarity, because there was no data similarity to drive it**, so
changing the attack could not move a number the attack had never controlled.

Everything from that line is now quarantined in `old_code/` with an explanation, so nobody
cites it by accident.

> **This is relevant to Amin's benchmark and I want to be precise about it.** His
> distribution-clustering mode uses Dirichlet α = 0.5 on MNIST and CIFAR-10. My measurement
> says that partition contains no group structure, so the clusters his distribution mode
> recovers on those two datasets are essentially arbitrary. **This is not a criticism of his
> code, which does exactly what it says.** It is a property of the standard generator that
> neither of us knew about until I measured it, and I could not find it documented anywhere
> in the literature. His **FEMNIST** results are unaffected, because the writer split is real
> structure.
>
> **His own output already shows the symptom, which is what makes this worth raising rather
> than glossing.** On MNIST his distribution clustering splits the six clients as
> `cluster_0: [0,1,3,4,5]` and `cluster_1: [2]` — **five clients against one**. On CIFAR-10 it
> is 4 against 2. A clusterer forced to produce two groups from data containing none will
> always separate *something*, and what it separates is whichever client happens to look most
> unusual. The consequence is visible in the accuracy: the lone client in `cluster_1` reaches
> **0.664** while the group of five reaches **0.967**, because it is training by itself with
> no one to average with.
>
> **So there is a real, publishable finding sitting in his numbers**, and it is a better one
> than "distribution clustering scored X": *applying distribution-based clustering to
> Dirichlet-partitioned data produces unbalanced, arbitrary clusters and actively harms the
> isolated clients.* That is worth stating, and it is his result, not mine — my contribution
> is only the measurement explaining **why** it happens. The honest framing for the report is
> that we found this together, because it was reading his notebook that made me go and check.

### 5.3 What the attacker can actually do

Once there are real groups, the question is how a client fakes membership. There are three
routes and I measured all of them.

**Route 1 — just lie (`declare`).** The client reports the target group's label histogram
instead of its own. That is it. One field overwritten. **No retraining, no modification to what
it submits, no cost of any kind.** Success rate: **100%, on every seed**, against a control of
0%.

**Route 2 — actually become like them (`resample`).** Against a server that ignores what
clients say, the attacker has to change what it *submits*. So it redraws its own training data
so its label mix matches the target group's, and trains on that. The model then genuinely
learns the distribution it is pretending to have.

This has a hard limit, and finding it was the most interesting part of the work. Here is my
attacker, imitating the target group:

| | class 0 | class 1 | **class 2** | class 3 | class 8 |
|---|---|---|---|---|---|
| what the attacker holds | 0.004 | 0.032 | **0.000** | 0.067 | 0.398 |
| what the target looks like | 0.147 | 0.237 | **0.146** | 0.176 | 0.021 |
| what resampling achieved | 0.172 | 0.277 | **0.000** | 0.206 | 0.025 |

Look at class 0: the attacker holds **0.4%** of it and produces **17.2%**. It can do that
because it draws the same few images over and over. Any class it holds *even a little* of, it
can imitate almost perfectly.

But class 2 it holds **none** of, and there it achieves **exactly zero**. **You cannot draw
what you do not have.**

> I had assumed this bound would be gradual — that holding 3% of a class would let you reach
> some small multiple of 3%. It is not gradual. It is **exactly zero-or-fine**. Only classes
> with literally zero examples fail, and everything else is imitated perfectly. I got that
> wrong when planning the experiment and the measurement corrected me.

**Route 3 — lie with the model itself (`prior-edit`).** The attacker leaves its data alone and
edits the final layer of its network so the model *reports* a different class distribution
than it learned. The technique is **logit adjustment** (Menon et al., ICLR 2021), published as
a *fairness* method for correcting imbalanced training data. Run backwards, it is a disguise.

It has **no** limit. On the exact classes where resampling achieved 0.000, prior-editing makes
the model claim **0.155** against a target of 0.146 — from a model that has never seen a single
example of that class.

**So the two routes differ in kind, not degree.** Resampling produces a model that has
genuinely **learned** the distribution. Prior-editing produces one that merely **reports** it.
That distinction is the one place a defence has anything to work with (§6.6).

### 5.4 How much does the attacker have to know?

Every result above assumes the attacker is **told** what the target group's data looks like.
That is not realistic, so I tested whether it could work it out for itself.

The plausible method: every round, the server broadcasts the current model to everyone. A
model trained mostly on one class should predict that class more often. So the attacker
inspects the broadcast model and reads off what it seems to have been trained on. This
requires nothing that an ordinary honest participant does not already have.

| what the attacker knows | how wrong its estimate is |
|---|---|
| told the answer exactly | 0.0000 |
| **works it out by inspecting the model** | 0.0844 |
| **doesn't try — just guesses "everything equally likely"** | 0.0887 |

**Inspecting the model is a 5% improvement over not bothering, and the statistical intervals
overlap.** On one of the five seeds it was actually *worse* than guessing. There is a real
leak, but it is a couple of percentage points per class, while the thing being estimated is
strongly skewed. The leak is smaller than the target.

**This is the most important limitation in my work and I have put it in front of the results
rather than in a footnote.** The attack is limited by **knowledge, not capability**. An
attacker who is *told* the target's distribution imitates it perfectly and for free. An
attacker who has to *work it out* cannot.

**Interestingly, the two studies disagree here, and the disagreement is informative.** Under
concept shift, I corrupted the attacker's knowledge deliberately and it *still* got in — at
half-correct knowledge it succeeded 72% of the time against a control of 33%. A *concept* is
learnable from a bad copy. A *label distribution* is not readable from the broadcast model at
all. So the distribution channel is the more knowledge-limited of the two, and we should say
so rather than implying both are equally dangerous.

### 5.5 Everything I got wrong along the way

I am listing these because two of them changed conclusions, and because an examiner asking
"how do we know your other numbers are right?" is answered better by a documented error record
than by a claim of care. The full account is in `F_ATTEMPTS.md`.

**1. I explained a result with the wrong mechanism.** I found that clustering on a client's
*whole* update fails while clustering on just the final layer works, and I explained it by the
final layer being small. **That was wrong.** Testing five architectures showed the split falls
exactly on whether the network uses **batch normalisation**, a standard component that tracks
statistics of the *input data*. Since every client sees the same kind of images, those
statistics are nearly identical for everyone — and they account for essentially **all** of the
update's magnitude. A server clustering on whole weight vectors of such a network is clustering
on input statistics and grouping at random, **while its own diagnostics read a healthy 0.9997.**

**2. I concluded the attack was "loud" when it was just badly tuned.** I had invented a
strength dial and set it to maximum, then observed that the attack looked conspicuous.
Replacing my dial with a published constraint from the literature showed a **25% commitment is
already enough**, and at that strength the attacker's submission stays 78% similar to what it
would honestly have sent. My "the attack is detectable" conclusion was an artefact of pushing
it four times harder than it needed to go.

**3. I reported a nonsensical result and had to find out why.** A double-strength attack
appeared to do *more* damage than a ten-times-strength one. Looking at the per-round trace
showed the victim model oscillating between broken and recovered, so averaging it was averaging
an on/off signal. I now report the fraction of rounds broken and the final state instead.

**4. Our FEMNIST loader was silently destroying the thing that makes FEMNIST worth using.** It
selected writers, then pooled all their data together and shuffled it — discarding the
writer-based split that is the entire reason to use the dataset. Every client came out with
exactly 2143 samples, which is what gave it away. **Had I not caught this, "we evaluated on a
natural non-IID partition" would have been false in the report, in a way no output revealed.**

**The pattern in all four is the same**, and it is worth a sentence in the report's methodology:
a summary number agreed with what I expected while the mechanism underneath had changed, and
nothing ever raised an error. **Every one was caught by breaking the number down — by round, by
component, by seed. None was caught by the headline figure.**

---

## 6. The findings, stated plainly

All from five random seeds with confidence intervals, each against its own control.

### 6.1 The declared channel is a perfect signal, which is exactly what makes it dangerous

Under distribution shift, the server recovers the true grouping from declared label histograms
with **ARI 1.000** — flawless, on every seed.

That sounds like good news and it is the opposite. The histogram *is* the group structure, so
of course reading it works. But it is a field **the client writes and the server cannot check**.
Its accuracy is what makes trusting it fatal: the server gets a signal so good it has no reason
to look any further, and a client that simply lies goes wherever it likes.

### 6.2 The same channel scores 1.000 in one study and 0.665 in the other, and both are right

This looks like a contradiction and it is a finding. Under **distribution** shift the declared
histogram is the thing the groups differ in, so it recovers the grouping exactly. Under
**concept** shift the groups are built to have *identical* label frequencies, so the histogram
should carry nothing at all — and it scores 0.665 rather than 0.000 only because MNIST's classes
are slightly imbalanced, which leaves a faint trace.

**Quote both numbers together or neither.** Both notebooks now carry a note explaining it,
because a reader meeting them separately would assume one is broken.

### 6.3 Getting in is free

The attacker reaches **100% infiltration** while its accuracy goes from 0.833 to 0.843 —
**statistically indistinguishable** from the honest control.

Resampling changes *which* distribution the model learns, not *how well* it learns one. The
attacker trains on a perfectly coherent dataset; just not its own. There is no degradation, no
instability, nothing anomalous.

**A defence looking for a damaged or misbehaving attacker will find nothing, because there is
nothing wrong with it.**

### 6.4 The strongest version of the attack has nothing to detect

This is the concept-shift result and it is the sharpest thing in the project. The attacker
takes its own images, relabels them the way the target group would, trains on that, and submits
the result **completely unmodified**.

It is not tampering with anything. It really did train, and those really are the weights that
came out. It just trained the wrong task on purpose.

**So there is no inconsistency for a verifier to find.** Any defence that asks "does this update
match what this client claims to be?" gets the answer *yes*, correctly. Measured directly: the
attacker's update moves from sitting where an honest member of its own group belongs, to sitting
inside the target group's own band. It is not being placed there by mistake — **it genuinely
looks like one of them.**

### 6.5 Robust aggregation does not stop the intrusion — and this is the one that affects the report

There is an established family of defences called **Byzantine-robust aggregation** — Krum,
Multi-Krum, coordinate-wise median, trimmed mean, Bulyan. They exist to stop a malicious client
from corrupting the shared model. I implemented all five from their published equations,
cross-checked them against a reference library, and ran them.

| | plain averaging | all five robust rules |
|---|---|---|
| **attacker gets into the target cluster** | yes | **yes — every single one** |
| **damage lands once inside** | yes, badly | **no — every one blocks it** |

Placement is **100% under all six rules**, against a **0%** honest control. Damage lands only
under plain averaging.

**Here is why.** Clustering and aggregation are **two separate decisions**. Clustering decides
*who you are grouped with*. Aggregation decides *whose update counts*. A robust aggregator only
governs the second one. It protects the model without preventing the break-in.

**And I measured this twice**, in both studies, with different data and different attacks. It is
a replication, not a single result.

### 6.6 The one thing a defence could catch

Of everything I measured, exactly one mechanism leaves a trace: **prior-editing**. It produces a
model whose *claimed* distribution and *actual* behaviour have come apart. A server willing to
test the model's behaviour on its own held-out data — rather than reading what the client
declares — could in principle catch that.

**Three honest cautions**, because this is exactly the shape of claim I have already overstated
once in this project:

1. **It does not touch resampling at all**, which is the quieter and stronger route. A defence
   that catches only the loud attacker is a defence against the attacker who did not need to be
   loud.
2. **It is untested against an adaptive attacker.** Someone who knows the check exists can edit
   the bias *less*, trading disguise quality for consistency.
3. **There is no false-alarm rate for it.** Honest clients' behaviour drifts from their declared
   histogram too, especially early on. Without measuring how often it wrongly accuses an honest
   client, "it detects the attack" means nothing.

---

## 7. What has to change in the report

This is the part that affects you directly.

### 7.1 The Relevant Work sentence that contradicts our own results

The Relevant Work section currently contains, in the paragraph introducing the research gap:

> "...to manipulate cluster assignment and gain unauthorized access to a target cluster, where
> they can disrupt the specialized model and **amplify the impact of subsequent poisoning
> attacks**."

**Our own Experimental Results now contradict that.** Spoofing gains entry, but it does **not**
amplify poisoning against any Byzantine-robust aggregator, in either study. As written, our
literature review and our results section disagree with each other, which is the first thing a
marker will notice.

**Suggested replacement** — and this is a change of *claim*, not a wording fix, so all three of
us should agree before it goes in:

> Entry into the cluster is the objective in itself, since it grants access to the specialised
> model regardless of whether a poisoning payload survives aggregation.

**Why I think the corrected version is the better claim**, not a retreat:

- It is **more precise**. It separates two things the original conflated — getting in, and doing
  damage once in.
- It is **evidenced**, in two independent settings, which is stronger than an unsupported
  stronger-sounding claim.
- It **opens a threat model the original closes off**: an attacker whose goal is *access* to a
  specialised model — to steal it, to study it, to extract what it learned about other people's
  data — pays **no price at all** under any of the six rules. That is arguably a more realistic
  adversary than one who just wants to break things.

When we present it, say **"in both settings"** explicitly. One experiment contradicting a cited
claim invites "your setup was unusual". Two independent settings do not.

### 7.2 The research question is wider than our evidence

The question promises to *identify and defend against* the attack. We built no detector, by
decision — the attack had to be understood first. So we have **no false positive rate**, and
false positive rate is named in the template as one of three required evaluation dimensions.

Two options:

1. **Build a detector and measure its false positive rate.** Real work, and it reopens the study
   this close to submission.
2. **Narrow the question to what we can actually support.** Suggested:

> How do architecture and membership spoofing attacks succeed against Clustered Federated
> Learning, and what do their mechanisms require of any defence beyond those already deployed?

**I recommend option 2.** It makes the negative results the *contribution* rather than a gap, and
the analysis already answers it: six defence families are ruled out on measurement, and two
directions are identified. But it is a change to what we said we would do, so it is a
three-person decision, not mine.

### 7.3 The title is narrower than the work

Currently: *"Architecture Spoofing Attacks in Federated Learning: A Vulnerability Analysis"*.

Architecture spoofing is now **one instance of three**, and the distribution study — the leading
one after the supervisor's redirection — is not architecture spoofing in any sense. Either widen
the title, or state early in the Experimental Results that **architecture is the instance and
membership is the class**. The second is cheaper and keeps continuity with the literature review
as it stands.

### 7.4 A methodological point worth a paragraph

The four errors in §5.5 share one shape, and I think it is worth reporting rather than hiding:
**a summary statistic that agrees with your hypothesis is the one to distrust.** Every error was
caught by breaking a number down — by round, by component, by seed — and none by the headline
figure. That is a genuine methodological contribution and it costs us nothing to say.

---

## 8. What is NOT established, and why saying so matters

An examiner will find these. Volunteering them is both more honest and more persuasive than
being led to them, and every one of them is in `RESULTS.md` already.

1. **The distribution attack assumes the attacker is told the target's distribution.** §5.4 shows
   it cannot work this out for itself. Every infiltration number in that study is an upper bound.
2. **No detector, so no false positive rate.** By decision, but it is the gap in §7.2.
3. **Twelve clients, one attacker — 8.3%.** The literature runs 1% to 30%, so we are in range but
   at the low end for client count.
4. **The distribution study is one dataset and one architecture.** The concept-shift study extends
   to five architectures and to FEMNIST; the newer one does not yet.
5. **The robust-aggregation sweep in the distribution study used three seeds**, not five, because
   it is the most expensive thing in the notebook.
6. **Prior-editing is measured for what it *claims***, not for whether the claim survives a server
   that checks behaviour.
7. **Our method descriptions rest on a secondary survey**, not on the Sattler and Ghosh papers
   themselves, which are not in the repository. **Their threshold values and experimental setups
   need checking against the primary sources before we submit.** This one is a genuine action
   item, not just a caveat.

---

## 9. Which file to open for what

**Start with these three:**

| file | what it is |
|---|---|
| `RESULTS.md` | **the write-up.** Part A is distribution, Part B is concept shift, then a section on why they agree. Every number traced to a saved run |
| `REPORT_NOTES.md` | **decisions for the three of us**, with a checklist at the end. §0 covers the redirection, §2 is the Relevant Work sentence |
| `LECTURE.md` | the whole project explained from scratch, no code. Longer than this file and goes deeper on the background |

**If you want more:**

| file | what it is |
|---|---|
| `DISTRIBUTION_WALKTHROUGH.md` | the distribution notebook transcribed in full, every cell explained. This is what to read instead of the notebook |
| `SUPERVISOR_SCRIPT.md` | what I plan to say in the next supervision, with anticipated questions |
| `DEFENSE_NOTES.md` | what a defence would have to do, and the six families the results rule out |
| `F_ATTEMPTS.md` | everything that failed, in detail |
| `HANDOFF.md` | project history. §0 is current, the rest is the state as of 15 August |
| `old_code/README.md` | what was retired and why. **Do not cite anything in `old_code/`** |

**The notebooks themselves** are `cfl-distribution.ipynb` and `cfl-spoofing.ipynb`. Both have
been run and contain all their outputs and figures, so you can scroll them without running
anything.

---

## 10. Things you might reasonably ask me

**"Do I need to run any of this?"**
No. Both notebooks are saved with their outputs and figures. `DISTRIBUTION_WALKTHROUGH.md`
explains one of them line by line.

**"Why did the project change direction?"**
The supervisor's feedback was to focus on data distribution and statistical heterogeneity. I
built the distribution study in response. The earlier concept-shift work is still in and still
cited — it is what the second study replicates.

**"Is Amin's work still needed if you have two studies?"**
Yes, and for two reasons. His benchmark demonstrates the system works properly when nobody is
attacking it, which is what makes an attack result meaningful. And his CNNs and FEMNIST loader
are *inside* my results — the natural non-IID condition exists because of his loader.

**"Are you saying Amin's results are wrong?"**
No, and I would push back on anyone who read it that way. His code does what it says. My
measurement says the standard Dirichlet generator contains no group structure, which changes
**how his MNIST and CIFAR distribution-mode numbers should be described** — and in a direction
that makes them *more* interesting, not less. His 5-against-1 cluster split and the 0.664
accuracy of the isolated client are evidence that distribution clustering on Dirichlet data
actively harms clients rather than helping them. That is a finding. His FEMNIST results are
unaffected either way. And it was reading his notebook that sent me to check in the first
place.

**"How confident are you in the numbers?"**
The headline ones: confident. Five seeds, confidence intervals, a control in every table, and
every table saved to a timestamped folder with a content hash. I re-ran the distribution
notebook end to end on 25 August and it reproduced the previous run **identically, to the last
decimal**. Where I am *not* confident, it says so — §8 is that list, and the knowledge
limitation in §5.4 is stated before the results rather than after them.

**"What is the single most important thing for the report?"**
The Relevant Work sentence in §7.1. It is the only place where our own sections contradict each
other, and it is a five-minute fix now versus a question we cannot answer in a viva.

**"What should I do first?"**
Read `REPORT_NOTES.md` — it is written as decisions for the three of us and has a checklist.
Then tell me whether you want the Byzantine correction written as a paragraph you can drop in,
and I will draft it.

---

*Ask me anything. Nothing in this project is complicated once you know which of the two
heterogeneity types you are looking at, and I would rather explain it twice than have us
present numbers we cannot defend.*
