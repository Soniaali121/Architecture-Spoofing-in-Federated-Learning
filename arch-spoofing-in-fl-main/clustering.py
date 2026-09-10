"""Clustering strategies: how the server decides which cluster a client is in.

Three families, and the difference between them is the whole point of the study.

DECLARED          the server reads a field the client wrote about itself.
                  `ArchitectureClusterer` and `DistributionClusterer` below.
                  The client can simply lie, and nothing in the update
                  contradicts the lie once the attack is adaptive.

CLIENT-SIDE       the server asks the client to compute something and report the
                  answer. `IFCAClusterer`. Still a declared channel, but the
                  claim is now about a quantity the server could in principle
                  check, which is a strictly stronger position than trusting an
                  arbitrary string.

SERVER-SIDE       the server infers membership from the submitted update and
                  asks the client nothing. `SattlerCosineClusterer`. Nothing is
                  declared, so there is nothing to lie about: an attacker has to
                  SHAPE ITS UPDATE to be placed where it wants.

Essentially all of the CFL literature (Sattler, IFCA, MUDGUARD, EBS-CFL,
ClusterGuard) is in the second and third families. The first is the one this
project originally attacked, and it is the one a reader will push back on.

CLUSTERING RUNS ON DELTAS, NOT ON ABSOLUTE WEIGHTS
--------------------------------------------------
`SattlerCosineClusterer` reads `metadata['delta_vec']`, which the server sets to
`w_client - w_global` before calling. This is load-bearing and was verified
empirically: cosine over ABSOLUTE weight vectors of 12 independently initialised
clients measured mean +0.768 (range 0.453 to 0.953) on the IDS data, which looks
like a healthy, well-spread similarity structure and is almost entirely shared
initialisation rather than shared data. The same measurement on a 1.66M
parameter CNN gave mean +0.001, because in high dimension independent vectors
are near-orthogonal. Neither number says anything about which clients hold
similar data. Only the delta from a COMMON starting point does.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from client import ClientUpdate


def _agglomerative(n_clusters: int, metric: str, linkage: str):
    """AgglomerativeClustering across the sklearn versions that renamed `affinity`."""
    try:
        return AgglomerativeClustering(n_clusters=n_clusters, metric=metric,
                                       linkage=linkage)
    except TypeError:
        return AgglomerativeClustering(n_clusters=n_clusters, affinity=metric,
                                       linkage=linkage)


class ClusteringStrategy(ABC):
    @abstractmethod
    def cluster(self, updates: List[ClientUpdate]) -> Dict[str, List[ClientUpdate]]:
        """Return mapping cluster_id -> updates assigned to that cluster."""

    # Cluster ids are NOT stable between rounds. `cluster_0` in round 2 is not
    # the group that was `cluster_0` in round 1: the labels come out of whatever
    # order the clustering algorithm happened to produce, and under recursive
    # bipartition the number of clusters changes too. Anything tracking a group
    # across rounds must track MEMBERSHIP. `fl_loop` and the analysis both do.
    needs_delta = False


class ArchitectureClusterer(ClusteringStrategy):
    """Group clients by claimed architecture metadata (trusted or poisoned)."""

    def cluster(self, updates: List[ClientUpdate]) -> Dict[str, List[ClientUpdate]]:
        groups: Dict[str, List[ClientUpdate]] = defaultdict(list)
        for update in updates:
            arch = update.metadata.get("arch")
            if arch is None:
                raise ValueError(f"Client {update.client_id} missing metadata['arch']")
            groups[str(arch)].append(update)
        return dict(groups)


class DistributionClusterer(ClusteringStrategy):
    """
    Group clients by similarity of distribution metadata (default: label histogram).

    Uses agglomerative clustering on L2-normalized label histograms.
    Falls back to a single cluster if there are fewer clients than n_clusters.
    """

    def __init__(self, n_clusters=2, feature_key="label_hist"):
        self.n_clusters = n_clusters
        self.feature_key = feature_key

    def cluster(self, updates: List[ClientUpdate]) -> Dict[str, List[ClientUpdate]]:
        if not updates:
            return {}

        features = []
        for update in updates:
            feat = update.metadata.get(self.feature_key)
            if feat is None:
                raise ValueError(
                    f"Client {update.client_id} missing metadata['{self.feature_key}']"
                )
            features.append(np.asarray(feat, dtype=np.float64).ravel())

        X = np.vstack(features)
        # Avoid zero-norm rows breaking cosine/L2 assumptions.
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms

        k = min(self.n_clusters, len(updates))
        if k <= 1:
            return {"cluster_0": list(updates)}

        labels = _agglomerative(k, "euclidean", "ward").fit_predict(X)

        groups: Dict[str, List[ClientUpdate]] = defaultdict(list)
        for update, label in zip(updates, labels):
            groups[f"cluster_{int(label)}"].append(update)
        return dict(groups)


# =========================================================================== #
# Inferred cluster assignment
# =========================================================================== #
class IFCAClusterer(ClusteringStrategy):
    """Cluster identity chosen BY THE CLIENT, from which cluster model fits best.

    IFCA (Ghosh et al.): the server broadcasts K cluster models, each client
    evaluates all K on its own data, picks the one with the lowest loss, trains
    from that one, and reports which it picked. The server averages within each
    reported index.

    The client computes the assignment, so the server is still trusting a
    declaration. The difference from `ArchitectureClusterer` is that the claim
    is now about something CHECKABLE IN PRINCIPLE: an honest client's update
    really should fit the model it named better than the alternatives, whereas
    a declared architecture string has no such internal consistency requirement
    until the update is probed.

    The attack is therefore trivial (report the target index) but it is also the
    reason this tier is worth measuring: it establishes what infiltration looks
    like when it costs the attacker nothing, which is the baseline the
    server-side tier has to be compared against.
    """

    KEY = "cluster_choice"

    def __init__(self, n_clusters: int = 2):
        self.n_clusters = n_clusters

    def cluster(self, updates: List[ClientUpdate]) -> Dict[str, List[ClientUpdate]]:
        groups: Dict[str, List[ClientUpdate]] = defaultdict(list)
        for u in updates:
            choice = u.metadata.get(self.KEY)
            if choice is None:
                raise ValueError(
                    f"Client {u.client_id} did not report metadata['{self.KEY}']. "
                    "The server must ask every client to evaluate the K cluster "
                    "models before clustering; see fl_loop.")
            groups[f"cluster_{int(choice)}"].append(u)
        return dict(groups)


class SattlerCosineClusterer(ClusteringStrategy):
    """Server-side bipartition on the cosine similarity of client UPDATE DELTAS.

    Sattler, Mueller and Samek (2019). Nothing is declared: the server computes
    pairwise cosine similarity between the deltas clients submitted, and splits
    a group in two when the group looks like it contains more than one data
    distribution.

    THE SPLIT CRITERION
        split when   ||mean(d_i)|| < eps1   and   max_i ||d_i|| > eps2

    The reasoning: at a stationary point of a SINGLE distribution the client
    deltas cancel, so their mean is near zero (hence eps1) while the individual
    deltas are not (hence eps2). A group whose mean update has gone quiet while
    its members are still moving hard is a group being pulled in two directions
    at once, which is the signature of two distributions sharing one model.

    Both thresholds are scale-dependent and MUST be tuned on honest clients
    before any attack is run. Sattler's published values (0.4, 1.6) are for his
    setup, not ours. `tune_thresholds` below does this, and the chosen values
    belong in the run manifest: a split criterion tuned in the presence of the
    attacker is a defence that has seen the attack, which is a different and
    much weaker claim than the one this experiment makes.

    `recursive=True` reproduces the published algorithm, which keeps splitting
    until no group meets the criterion. `recursive=False` performs at most one
    split, which is the controlled version used for the two-cluster experiments,
    because the number of clusters is then fixed and infiltration means one
    thing rather than varying with the round.

    THIS IS THE PUBLISHED ALGORITHM. DO NOT "IMPROVE" IT.
    -----------------------------------------------------
    We are analysing an ATTACK, so the clustering is the environment rather than
    our contribution. A clusterer we invented would make the attack a result
    about a strawman we built rather than about deployed Clustered Federated
    Learning (CFL). Four tempting changes are therefore deliberately absent:

      relative thresholds    the paper specifies absolute norms; a scale-free
                             ratio is a different criterion, not a tidier one.
      centring               subtracting the mean delta before computing
                             similarity widens the spread 25-fold while dropping
                             rank correlation with the true grouping from 0.61 to
                             0.18. A wider range of a worse measure.
      recursive=False        the published algorithm is recursive.
      min_size / max_clusters  neither appears in the paper. Available as
                             explicit guards, defaulted OFF, never applied
                             silently.

    `eps1` and `eps2` are genuinely per-experiment in the paper, so tuning them
    is faithful rather than a deviation, but they MUST be tuned by
    `tune_thresholds` on an attacker-free run. A criterion tuned in the presence
    of the attacker has already seen the attack, which is a different and much
    weaker claim than the one this experiment makes.

    SCOPE: WHICH PARAMETERS THE SIMILARITY IS COMPUTED OVER
    -------------------------------------------------------
    `scope="full"` is vanilla Sattler, every parameter. `scope="head"` restricts
    it to the final Linear layer's weight and bias.

    This is not a tweak, it is the difference between the method working and not
    working here. Under concept shift every client sees the same inputs, so the
    trunk gradients are near-identical federation-wide and the task disagreement
    is confined to the classifier. Measured on label-permuted data at round 4,
    within-group minus cross-group cosine margin:

        dataset   scope="full"   scope="head"   head cross-group cosine
        MNIST           0.0001          0.875                    -0.171
        IDS             0.0032          1.388                    -0.499

    On the full vector every pair sits at +0.9997, nothing cancels, `||sum d||`
    stays at exactly `n` times the individual norm, and the split criterion can
    never fire. On the head the cross-group cosine is NEGATIVE, which is the
    opposed-gradient condition the criterion was designed around.

    `scope="head"` follows FedRep and FedPer, which separate a shared
    representation from a personalised head, and the common practice of
    computing client similarity on classifier updates. It is a departure from
    Sattler as published, so any run using it must say so, and `scope="full"`
    remains the faithful baseline to report alongside.
    """

    needs_delta = True
    DELTA_KEY = "delta_vec"
    SCOPES = ("full", "head")

    def __init__(self, eps1: float = 0.4, eps2: float = 1.6,
                 recursive: bool = True, min_size: int = 1,
                 max_clusters: Optional[int] = None, scope: str = "head"):
        # Defaults are Sattler's published values and the published recursion.
        # They are very unlikely to be right for our data (his are tuned for his
        # setup), which is what `tune_thresholds` is for. They are the defaults
        # anyway, so that constructing this class without tuning reproduces the
        # paper rather than reproducing our guesses.
        #
        # min_size=1 and max_clusters=None mean "no guard", i.e. exactly the
        # published behaviour. Raising min_size stops the criterion isolating a
        # single outlier, and capping max_clusters stops permanent splits
        # fragmenting the federation, but both are OUR additions and any run
        # using them must say so.
        if scope not in self.SCOPES:
            raise ValueError(f"scope must be one of {self.SCOPES}, got {scope!r}")
        self.eps1 = float(eps1)
        self.eps2 = float(eps2)
        self.recursive = bool(recursive)
        self.min_size = int(min_size)
        self.max_clusters = max_clusters
        self.scope = scope
        # Flat indices of the classifier head, set by the server which knows the
        # architecture. Left None for scope="full", and a hard error rather than
        # a silent fallback for scope="head", because falling back to the full
        # vector would quietly restore the configuration that cannot work.
        self.head_index: Optional[np.ndarray] = None
        # The cluster tree, carried across rounds. None until the first call.
        # Splits are permanent; see `cluster`.
        self.partition: Optional[List[List[int]]] = None
        # Populated on every call, so the experiment can record WHY a split did
        # or did not happen rather than only recording that it did.
        self.last_trace: List[dict] = []

    @staticmethod
    def residuals(D: np.ndarray) -> np.ndarray:
        """d_i - mean(d): what is left after removing the shared task direction."""
        D = np.asarray(D, dtype=np.float64)
        return D - D.mean(axis=0, keepdims=True) if len(D) > 1 else D

    def _deltas(self, updates: Sequence[ClientUpdate]) -> np.ndarray:
        from aggregation import stack

        vecs = []
        for u in updates:
            d = u.metadata.get(self.DELTA_KEY)
            if d is None:
                raise ValueError(
                    f"Client {u.client_id} has no metadata['{self.DELTA_KEY}']. "
                    "Sattler clustering runs on the delta from the shared global "
                    "model, not on absolute weights; the server must set it "
                    "before clustering. Clustering absolute weights measures "
                    "shared initialisation, not shared data.")
            vecs.append(np.asarray(d, dtype=np.float64).ravel())
        return self._scoped(stack(vecs))

    def _scoped(self, D: np.ndarray) -> np.ndarray:
        """Restrict the delta matrix to the parameters this clusterer decides on."""
        if self.scope == "full":
            return D
        if self.head_index is None:
            raise ValueError(
                "scope='head' needs head_index, which the server sets from the "
                "architecture. Falling back to the full vector would silently "
                "restore the configuration measured NOT to work: every pair sits "
                "at +0.9997 cosine and the split criterion can never fire.")
        return D[:, self.head_index]

    def should_split(self, D: np.ndarray) -> dict:
        """Evaluate the criterion and return the numbers AND the reason.

        The reason matters as much as the decision. A group that did not split
        because its mean delta is still large is a group the algorithm thinks is
        still learning a shared task; one that did not split because the
        bipartition would isolate a single client is a completely different
        situation. Recording only a boolean makes those indistinguishable in the
        corpus, and they get read as the same thing.
        """
        # Sattler's criterion, on ABSOLUTE norms:
        #     ||sum_i d_i|| < eps1   AND   max_i ||d_i|| > eps2
        # The reasoning: at a stationary point of a SINGLE distribution the
        # client updates cancel, so their sum goes near zero (eps1) while the
        # individual updates do not (eps2). A group whose combined update has
        # gone quiet while its members are still moving hard is being pulled in
        # several directions at once.
        #
        # The sum, not the mean. The paper's condition is on the summed update,
        # and the two differ by a factor of n, so a threshold tuned against one
        # is wrong by that factor against the other.
        sum_norm = float(np.linalg.norm(D.sum(axis=0))) if len(D) else 0.0
        ind = np.linalg.norm(D, axis=1) if len(D) else np.zeros(1)
        max_norm = float(ind.max())

        if len(D) < max(2, 2 * self.min_size):
            reason = (f"group of {len(D)} cannot be halved"
                      + (f" at min_size={self.min_size}" if self.min_size > 1 else ""))
        elif sum_norm >= self.eps1:
            reason = (f"||sum d|| {sum_norm:.4f} >= eps1 {self.eps1:.4f}: the group "
                      f"is still moving together, so it looks like one task")
        elif max_norm <= self.eps2:
            reason = (f"max ||d|| {max_norm:.4f} <= eps2 {self.eps2:.4f}: nobody is "
                      f"moving much, so the group looks converged")
        else:
            reason = "criterion met"

        return {"n": len(D), "sum_norm": sum_norm, "max_norm": max_norm,
                "mean_norm": float(np.linalg.norm(D.mean(axis=0))) if len(D) else 0.0,
                "eps1": self.eps1, "eps2": self.eps2,
                "split": reason == "criterion met", "reason": reason}

    def _bipartition(self, D: np.ndarray) -> np.ndarray:
        """Split into two by complete-linkage on cosine DISTANCE.

        Distance is `1 - similarity`, which is non-negative and is what sklearn
        expects from a precomputed metric. Sattler's own code passes the negated
        similarity, which is equivalent for complete linkage (it is monotone in
        the same order) but feeds sklearn negative "distances".

        Complete linkage, not average or ward: the criterion Sattler derives is
        about the WORST cross-group similarity, so the linkage has to be the one
        that merges on worst-case distance.
        """
        from aggregation import cosine_matrix

        S = cosine_matrix(D)
        return _agglomerative(2, "precomputed", "complete").fit_predict(1.0 - S)

    def similarity(self, D: np.ndarray) -> np.ndarray:
        """The similarity matrix this clusterer actually decides on.

        Exposed so the analysis can plot what the server sees rather than a
        reconstruction of it. Reporting one matrix while clustering on another
        would be a quiet, unfalsifiable error.
        """
        from aggregation import cosine_matrix

        return cosine_matrix(D)

    def reset(self):
        """Forget the cluster tree. Call between independent runs."""
        self.partition = None
        self.last_trace = []
        return self

    def cluster(self, updates: List[ClientUpdate]) -> Dict[str, List[ClientUpdate]]:
        """Refine the existing partition. Clusters PERSIST across rounds.

        This is the part of the published algorithm that is easy to miss and
        expensive to get wrong. Clustered Federated Learning (CFL) maintains a
        cluster TREE: a group that has been bipartitioned is never re-merged,
        only split further. Re-clustering all clients from scratch each round
        instead produced a cluster count that oscillated 1, 2, 2, 2, 1, 1 across
        six rounds, which makes infiltration undefined in the single-cluster
        rounds. Averaging those rounds in as successes or failures are both
        wrong, and neither is visible in the resulting rate.

        Persistence also changes the attack, in the direction that makes it more
        interesting: the attacker must get into the target group AND STAY IN,
        because next round it warm-starts from whichever cluster model it landed
        under, and a group it has been placed in is never dissolved for it.
        """
        self.last_trace = []
        updates = list(updates)
        if len(updates) < 2:
            return {"cluster_0": updates}

        by_id = {u.client_id: u for u in updates}
        index = {u.client_id: i for i, u in enumerate(updates)}
        D_all = self._deltas(updates)

        if getattr(self, "partition", None) is None:
            current = [[u.client_id for u in updates]]
        else:
            # Carry the tree forward, dropping any client that did not report
            # this round and adopting any that is new. A client appearing for
            # the first time joins the largest group; it has no history to
            # place it, and inventing a group for it would inflate the count.
            known = set()
            current = []
            for group in self.partition:
                kept = [c for c in group if c in by_id]
                if kept:
                    current.append(kept)
                    known.update(kept)
            newcomers = [u.client_id for u in updates if u.client_id not in known]
            if newcomers:
                if not current:
                    current = [newcomers]
                else:
                    max(current, key=len).extend(newcomers)

        next_partition: List[List[int]] = []
        pending = list(current)

        while pending:
            group = pending.pop()
            rows = [index[c] for c in group]
            trace = self.should_split(D_all[rows])
            trace["members"] = list(group)

            if (self.max_clusters is not None
                    and len(next_partition) + len(pending) + 2 > self.max_clusters):
                trace["split"] = False
                trace["reason"] = f"at the max_clusters ceiling of {self.max_clusters}"
            if not trace["split"]:
                self.last_trace.append(trace)
                next_partition.append(group)
                continue

            labels = self._bipartition(D_all[rows])
            left = [c for c, l in zip(group, labels) if l == 0]
            right = [c for c, l in zip(group, labels) if l == 1]

            # A split that isolates a single client is usually the criterion
            # firing on one outlier rather than on two distributions. Refusing
            # it keeps the cluster count meaningful: without it, one boosted
            # update fragments the federation a client at a time, and because
            # splits are permanent that damage is never undone.
            if min(len(left), len(right)) < self.min_size:
                trace["split"] = False
                trace["reason"] = (f"bipartition would leave a group of "
                                   f"{min(len(left), len(right))}, below "
                                   f"min_size={self.min_size}")
                self.last_trace.append(trace)
                next_partition.append(group)
                continue

            self.last_trace.append(trace)
            if self.recursive:
                pending.extend([left, right])
            else:
                next_partition.extend([left, right])

        self.partition = next_partition
        return {f"cluster_{i}": [by_id[c] for c in g]
                for i, g in enumerate(next_partition)}

    def tune_thresholds(self, D_honest: np.ndarray, margin: float = 1.2):
        """Pick eps1 and eps2 from an ATTACKER-FREE run, and say what was chosen.

        Sattler treats these as per-experiment hyperparameters, so tuning them
        is faithful to the paper rather than a deviation from it. What would NOT
        be faithful is tuning them on a run containing the attacker: the
        criterion would then have been fitted to the attack it is supposed to be
        blind to, and every downstream number would describe a server that had
        already seen what we claim it cannot see.

        **Pass the deltas from the LAST round of an attacker-free warmup, not the
        first.** Both criterion quantities decay as the federation trains, and
        they decay together. Measured on label-permuted MNIST over 8 rounds,
        head scope:

            round      ||sum d||     max ||d||
              1           8.9088        0.9351
              4           2.4175        0.3970
              8           1.2625        0.3482

        Tuning eps2 from round 1 gives `0.5 x 0.91 = 0.455`, but `max ||d||`
        falls to 0.348, so the condition `max ||d|| > eps2` becomes permanently
        unsatisfiable and the clusterer silently never splits. That failure is
        invisible: no error, no warning, just a federation that stays whole.

        eps1 must sit ABOVE the summed-delta norm at the end of warmup, so the
        criterion fires once the mixture has settled. eps2 must sit BELOW the
        largest individual norm there, so it keeps firing afterwards.
        """
        D_honest = np.asarray(D_honest, dtype=np.float64)
        sum_norm = float(np.linalg.norm(D_honest.sum(axis=0)))
        ind = np.linalg.norm(D_honest, axis=1)
        max_norm = float(ind.max())

        self.eps1 = float(sum_norm * margin)
        self.eps2 = float(max_norm * 0.5)
        print(f"[sattler] tuned on {len(D_honest)} attacker-free deltas "
              f"(scope={self.scope}): eps1={self.eps1:.4f} "
              f"(||sum d||={sum_norm:.4f} x {margin}), "
              f"eps2={self.eps2:.4f} (max ||d||={max_norm:.4f} x 0.5)")
        if sum_norm >= self.eps1 or max_norm <= self.eps2:
            print("[sattler] WARNING: these thresholds do not fire on the very "
                  "data they were tuned on, so the clusterer will never split.")
        return self.eps1, self.eps2
