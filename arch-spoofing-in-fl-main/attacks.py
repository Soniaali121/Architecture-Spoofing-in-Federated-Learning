"""Attack hooks applied to ClientUpdate before server-side clustering/aggregation."""

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Set

import numpy as np

from client import ClientUpdate
from lab_probe import mean_output_prior
from metrics import normalise as _normalise


class Attack(ABC):
    @abstractmethod
    def apply(self, update: ClientUpdate) -> ClientUpdate:
        ...

    def training_arch(self, client_id: int, true_arch: str) -> Optional[str]:
        """Which architecture this client should actually TRAIN (and warm-start from)
        this round, or None to train its true architecture. Architecture spoofing
        overrides this so a spoofer emits a shape-valid update for the cluster it
        infiltrates (otherwise the server's shape check discards it and the attack
        does nothing)."""
        return None

    def training_data(self, client_id: int, X, y):
        """What this client should actually train on, defaulting to its real data.

        The distribution analogue of `training_arch`: an attacker imitating a
        label distribution can resample its own data so the model genuinely learns
        the prior it is about to declare. `fl_loop.TorchFederatedServer` calls
        this every round; the older Keras `FederatedServer` does not, so attacks
        relying on it are limited to the torch loop and the corpus generator.
        """
        return X, y

    def targets(self, client_id: int) -> bool:
        """Is this client the attacker? Ground truth, for the analysis only.

        The server records this so a row can be scored, and must never route
        control flow through it: a loop that treats the attacker differently
        because it knows who the attacker is has assumed away the problem.
        """
        return client_id in getattr(self, "malicious_clients", set())

    def shape_update(self, update, context: dict):
        """Rewrite the update after training, before the server clusters it.

        This is the hook the SERVER-SIDE tier needs. When the server infers
        placement from the submitted update there is no declared field to
        falsify, so the only way to influence placement is to change the update
        itself. `apply` exists for the declared-metadata tiers and rewrites
        metadata; this one rewrites the weights and the delta.

        `context` carries what the attacker can see. Anything using
        `context['updates']` is reading other clients' submissions, which a real
        attacker cannot do; such attacks are upper bounds, and must be labelled
        as such wherever they are reported.

        Default is a no-op, so honest clients and the declared-metadata attacks
        pass through untouched.
        """
        return update


class NoAttack(Attack):
    def apply(self, update: ClientUpdate) -> ClientUpdate:
        return update


class WeightSignFlip(Attack):
    """Classic model poison: negate all weight tensors."""

    def __init__(self, malicious_clients: Optional[Iterable[int]] = None):
        self.malicious_clients: Set[int] = set(malicious_clients or [])

    def apply(self, update: ClientUpdate) -> ClientUpdate:
        if update.client_id not in self.malicious_clients:
            return update
        poisoned = ClientUpdate(
            client_id=update.client_id,
            weights=[-np.asarray(w) for w in update.weights],
            metadata=dict(update.metadata),
            true_arch=update.true_arch,
        )
        return poisoned


class WeightBoost(Attack):
    """Model-replacement / weight-boosting poison (Bagdasaryan et al., 2020).

    Amplifies the client's own delta from the global model by `boost`, so a
    single update can dominate the FedAvg mean. Unlike a sign flip it keeps the
    update's DIRECTION honest, which is precisely why it is harder to spot from
    probe behaviour and why it belongs in a fingerprint study.

    `global_weights` has to be set by the caller each round, since the delta is
    measured against whatever model the client warm-started from.
    """

    def __init__(self, malicious_clients: Optional[Iterable[int]] = None,
                 boost: float = 10.0):
        self.malicious_clients: Set[int] = set(malicious_clients or [])
        self.boost = boost
        self.global_weights: Optional[List[np.ndarray]] = None

    def apply(self, update: ClientUpdate) -> ClientUpdate:
        if update.client_id not in self.malicious_clients or self.global_weights is None:
            return update
        boosted = [np.asarray(g) + self.boost * (np.asarray(w) - np.asarray(g))
                   for w, g in zip(update.weights, self.global_weights)]
        return ClientUpdate(client_id=update.client_id, weights=boosted,
                            metadata=dict(update.metadata), true_arch=update.true_arch)


class ArchSpoof(Attack):
    """Architecture spoof: a malicious client declares an architecture other than
    its true one, so the server's declared-metadata clustering assigns it to a
    cluster it doesn't belong to.

    `adaptive` controls what the client actually trains, and is the difference
    between a spoof that does damage and one that doesn't:
      - adaptive=True  (default): the client trains (and warm-starts from) the
        DECLARED architecture, so its update is shape-valid for the target
        cluster and survives the server's shape check. This is the attack that
        actually infiltrates and, paired with WeightSignFlip, poisons the victim
        cluster -- verified to crash victim-cluster accuracy in this codebase.
      - adaptive=False: the client trains its TRUE architecture and only lies in
        the declared metadata. In this shape-constrained setting that produces a
        weight-shape mismatch against the victim cluster's model, so the update
        is discarded by the server's shape check before it can do any harm --
        i.e. a naive metadata-only lie is already defeated by basic shape
        validation. Useful as a baseline to show that shape-checking alone stops
        unsophisticated spoofing, motivating why the interesting/dangerous case
        is the adaptive one.
    """

    def __init__(self, malicious_clients: Optional[Iterable[int]] = None,
                 spoof_as: str = "dnn", adaptive: bool = True):
        self.malicious_clients: Set[int] = set(malicious_clients or [])
        self.spoof_as = spoof_as
        self.adaptive = adaptive

    def training_arch(self, client_id: int, true_arch: str) -> Optional[str]:
        if client_id in self.malicious_clients and self.adaptive:
            return self.spoof_as
        return None

    def apply(self, update: ClientUpdate) -> ClientUpdate:
        if update.client_id not in self.malicious_clients:
            return update
        metadata = dict(update.metadata)
        metadata["arch"] = self.spoof_as        # already set at train time; kept explicit
        metadata["spoofed"] = True
        return ClientUpdate(
            client_id=update.client_id,
            weights=update.weights,
            metadata=metadata,
            true_arch=update.true_arch,
        )


class LabelHistSpoof(Attack):
    """
    Metadata poison for distribution-based CFL: replace reported label histogram
    with a fake vector so the client joins another cluster.
    """

    def __init__(
        self,
        malicious_clients: Optional[Iterable[int]] = None,
        fake_hist: Optional[np.ndarray] = None,
        target_class: Optional[int] = None,
        num_classes: Optional[int] = None,
    ):
        self.malicious_clients: Set[int] = set(malicious_clients or [])
        if fake_hist is not None:
            self.fake_hist = np.asarray(fake_hist, dtype=np.float64)
        elif target_class is not None and num_classes is not None:
            self.fake_hist = np.zeros(num_classes, dtype=np.float64)
            self.fake_hist[target_class] = 1.0
        else:
            raise ValueError("Provide fake_hist or (target_class and num_classes)")

    def apply(self, update: ClientUpdate) -> ClientUpdate:
        if update.client_id not in self.malicious_clients:
            return update
        metadata = dict(update.metadata)
        metadata["label_hist"] = self.fake_hist.copy()
        metadata["spoofed"] = True
        return ClientUpdate(
            client_id=update.client_id,
            weights=update.weights,
            metadata=metadata,
            true_arch=update.true_arch,
        )


class ClusterSteering(Attack):
    """Get placed in a cluster you do not belong to, when nothing is declared.

    THE THREAT MODEL THIS ANSWERS
    -----------------------------
    Every attack above falsifies a field the client declares about itself: an
    architecture name, a label histogram, a cluster index. That works because
    the server trusts the declaration. Clustered Federated Learning (CFL) as
    actually published does not: Sattler infers cluster membership from the
    cosine similarity of the submitted update deltas, and asks the client
    nothing. There is no field to lie about.

    So the attacker has to shape the UPDATE. It holds data from a source group
    and wants to be aggregated into a target group, and its only lever is the
    delta it submits.

    WHAT THE MEASUREMENTS SAY THE ATTACKER IS UP AGAINST
    ----------------------------------------------------
    On 12 clients in 2 planted groups over 5 seeds, at concentration 20 and
    overlap 0.3, the server takes 3 to 5 rounds to split, and from the moment it
    splits it recovers the planted grouping EXACTLY and holds it. It is late,
    not wrong. Two consequences shape this attack:

      1. There is a window of 3 to 5 rounds in which one cluster exists and
         placement has not happened yet. Steering during that window is what
         decides the split.
      2. Splits are PERMANENT. The cluster tree is only ever refined, never
         re-merged, so a client placed on the wrong side is not given a second
         chance. The attacker gets one shot at the split, and after it lands the
         question changes from "can I get in" to "was I already in".

    MECHANISMS, selected by `mechanism`:

      none      train honestly, submit honestly. The control. `beta=0` reduces
                every other mechanism to this, and a run without it cannot
                support any claim that the attack CAUSED an effect.
      weight    train honestly, then blend the delta toward an estimate of the
                target's delta and rescale to the honest norm band. Unbounded by
                the attacker's data, but it is a lie the weights do not back up.
      relabel   apply the TARGET GROUP'S CONCEPT to the attacker's own data and
                train on that. Under concept shift this is the honest-looking
                attack and the strongest one available: the resulting update is
                a genuine member of the target cluster, because the attacker
                really did train the target's task. Nothing about it is
                internally inconsistent, so there is no discrepancy for a
                verifier to find. This is the same shape as the adaptive
                architecture spoof, which trains the architecture it declares.

    MECHANISMS THAT ONLY APPLY TO LABEL-DISTRIBUTION SHIFT, and which RAISE
    under concept shift rather than silently doing nothing:

      data      resample own data toward the target's estimated label
                distribution. Under concept shift every group has an IDENTICAL
                label marginal by construction, so there is nothing to resample
                toward and this is a no-op that would report as a failed attack.
      prior     logit adjustment so the model's output prior matches the
                target's. Same defect, from the same cause: the priors are
                identical, so the adjustment is the zero vector.

    Both are kept for the dormant `partition_clustered` condition and are
    guarded, because an attack that quietly does nothing is indistinguishable in
    the results table from an attack that was tried and failed.

    `beta` in [0, 1] is the placement-versus-payload dial: 0 is honest, 1 is
    full commitment to the disguise. It is the axis of the frontier figure and
    the reason this class takes a dial rather than a boolean.
    """

    MECHANISMS = ("none", "declare", "weight", "relabel", "data", "prior")
    # How far the attacker is ALLOWED to move. `beta` is a dial we chose;
    # these two derive the limit from the honest population itself.
    CONSTRAINTS = ("none", "lie", "minmax")
    # Meaningful only when groups differ in their label marginals.
    DISTRIBUTION_ONLY = ("data", "prior")
    PAYLOADS = ("none", "boost", "flip")

    def __init__(self, malicious_clients: Optional[Iterable[int]] = None,
                 target_members: Optional[Iterable[int]] = None,
                 mechanism: str = "weight", beta: float = 1.0,
                 estimator: Optional["TargetEstimator"] = None,
                 num_classes: int = 0, arch: str = "mlp_flat",
                 input_shape=None, match_norm: bool = True, seed: int = 0,
                 target_concept=None, target_group: Optional[int] = None,
                 home_group: Optional[int] = None,
                 payload: str = "none", payload_scale: float = 1.0,
                 constraint: str = "none"):
        if mechanism not in self.MECHANISMS:
            raise ValueError(f"mechanism must be one of {self.MECHANISMS}, "
                             f"got {mechanism!r}")
        if mechanism == "relabel" and target_concept is None:
            raise ValueError(
                "mechanism='relabel' needs target_concept, the callable "
                "(X, y, group) -> (X, y) that applies a group's concept. "
                "load_bundle_torch builds it for the concept-shift partitions; "
                "pass bundle's concept function and target_group.")
        self.malicious_clients: Set[int] = set(malicious_clients or [])
        self.target_members: Set[int] = set(target_members or [])
        self.mechanism = mechanism
        self.beta = float(beta)
        self.estimator = estimator
        self.num_classes = num_classes
        self.arch = arch
        self.input_shape = input_shape
        self.match_norm = match_norm
        self.target_concept = target_concept
        self.target_group = target_group
        # The attacker's OWN group. Its labels are already expressed in this
        # group's concept, so relabelling must invert it before applying the
        # target's, or the two compose into a third concept belonging to
        # neither group. See F_ATTEMPTS.md D4.
        self.home_group = home_group

        # PAYLOAD: what the attacker does once it is inside.
        #
        # Placement and damage are separate questions and this project has
        # conflated them before. Infiltration with payload="none" measures
        # whether the attacker can get in; a payload measures what it can do
        # from there, and how visible doing it makes them. Sweeping
        # `payload_scale` traces the frontier between the two.
        #
        #   none   infiltrate and behave. The placement-only control.
        #   boost  scale the delta, so the attacker's contribution dominates the
        #          cluster mean (Bagdasaryan et al., 2020, model replacement).
        #          Keeps the DIRECTION honest, which is what makes it harder to
        #          spot than a sign flip.
        #   flip   negate the delta, dragging the cluster model backwards.
        #          Known to be crude: at large scale it drives the model to NaN,
        #          which is a wrecked model rather than a hidden one.
        if payload not in self.PAYLOADS:
            raise ValueError(f"payload must be one of {self.PAYLOADS}, got {payload!r}")
        self.payload = payload
        self.payload_scale = float(payload_scale)

        # CONSTRAINT: where the limit on movement comes from.
        #
        # `beta` is a dial this project invented, and a reviewer reading Baruch
        # et al. will ask why the stealth bound was chosen rather than derived.
        # These two derive it from the honest clients themselves:
        #
        #   lie     A Little Is Enough (Baruch et al., NeurIPS 2019). Compute
        #           the per-coordinate mean and standard deviation of the honest
        #           updates, then the largest z that still leaves the attacker
        #           inside a majority, and clip the submission to mu +/- z*sigma.
        #           The bound is a property of the honest population's spread.
        #
        #   minmax  Manipulating the Byzantine (Shejwalkar & Houmansadr, NDSS
        #           2021). Binary-search the largest blend toward the target
        #           whose maximum distance to any honest update still does not
        #           exceed the largest honest-to-honest distance. Aggregator
        #           agnostic, which is why it is the standard strong baseline.
        #
        # Both are reimplemented from the equations rather than adapted from the
        # authors' repositories, several of which carry no licence.
        if constraint not in self.CONSTRAINTS:
            raise ValueError(f"constraint must be one of {self.CONSTRAINTS}, "
                             f"got {constraint!r}")
        self.constraint = constraint
        self.rng = np.random.default_rng(seed)
        # Flat indices of the classifier head, set by the server. The stealth
        # measurement MUST be taken here rather than on the full vector: the
        # head is 0.6% of the parameters on the MNIST Multi-Layer Perceptron
        # (MLP), so a blend that rewrites it entirely still leaves full-vector
        # cosine at 0.9998. Reporting that as "the attacker barely moved" is
        # false, and it is exactly what an earlier sweep did.
        self.head_index = None
        # Set by the caller when the attacker is given oracle knowledge of the
        # target's label distribution. `None` means it must estimate.
        self.target_hist = None
        self.own_data = {}
        # Recorded per round so the analysis can report what the attacker
        # actually achieved rather than what it attempted. The two differ, and
        # the difference is a result.
        self.trace: List[dict] = []

    def apply(self, update: ClientUpdate) -> ClientUpdate:
        """No-op, deliberately.

        `apply` is the declared-metadata hook: it rewrites the fields a client
        states about itself. This attack has nothing to state. Against a server
        that infers placement from the update there is no declaration to
        falsify, which is the whole point of the tier, so all of the work
        happens in `training_data` (before training) and `shape_update` (after
        it). Leaving this as a no-op rather than removing it keeps
        `ClusterSteering` usable inside a `CompositeAttack` alongside the
        declared-metadata attacks.
        """
        return update

    # -- data-side --------------------------------------------------------- #
    def training_data(self, client_id: int, X, y):
        """Resample toward the target distribution, for the `data` mechanism only.

        The honest mechanism, in the sense that the model genuinely learns the
        prior it will present. Also strictly bounded, and the bound is itself a
        result: a client holding no examples of a class cannot manufacture them,
        so the achieved histogram falls short by exactly the mass the target
        placed on classes it does not hold.
        """
        if client_id not in self.malicious_clients or self.beta <= 0:
            return X, y

        if self.mechanism == "relabel":
            # Train the target's task. `beta` is the fraction of the attacker's
            # own rows relabelled into the target concept, so beta=0 is honest
            # and beta=1 fully adopts the target's task. A partial mix is a
            # client training two contradictory labellings of the same inputs,
            # which is a genuinely different attack from either endpoint rather
            # than an interpolation between them, and it is worth measuring.
            n = len(np.asarray(y))
            k = int(round(self.beta * n))
            if k <= 0:
                return X, y
            idx = self.rng.permutation(n)[:k]
            Xt, yt = self.target_concept(np.asarray(X)[idx], np.asarray(y)[idx],
                                         self.target_group,
                                         from_group=self.home_group)
            X_new = np.concatenate([np.asarray(X)[np.setdiff1d(np.arange(n), idx)],
                                    np.asarray(Xt)])
            y_new = np.concatenate([np.asarray(y)[np.setdiff1d(np.arange(n), idx)],
                                    np.asarray(yt)])
            self.trace.append({"client_id": client_id, "mechanism": "relabel",
                               "beta": self.beta, "rows_relabelled": int(k),
                               "rows_total": int(n)})
            return X_new, y_new

        if self.mechanism != "data":
            return X, y

        target = self._target_for(client_id, y)
        if target is None:
            return X, y

        from data import label_histogram

        true_hist = label_histogram(np.asarray(y), self.num_classes)
        blended = _normalise((1.0 - self.beta) * true_hist + self.beta * target)
        Xr, yr, achieved = resample_for_target(X, y, blended, self.rng)
        self.trace.append({"client_id": client_id, "mechanism": "data",
                           "beta": self.beta, "requested": blended,
                           "achieved": achieved,
                           "shortfall": float(np.abs(blended - achieved).sum() / 2)})
        return Xr, yr

    def assert_applicable(self, groups, label_hists=None):
        """Fail loudly if this mechanism cannot work on this kind of shift.

        Call it once at setup. `data` and `prior` both steer toward a target
        LABEL DISTRIBUTION, so on concept-shift data (label permutation or
        rotation), where all groups share one label marginal, they are exact
        no-ops. Without this check they run, change nothing, and are recorded as
        an attack that was tried and failed, which is not the same claim at all.
        """
        import numpy as np

        if self.mechanism not in self.DISTRIBUTION_ONLY or label_hists is None:
            return True

        from sklearn.metrics import adjusted_rand_score

        from clustering import DistributionClusterer

        H = np.asarray(label_hists, dtype=np.float64)
        g = np.asarray(groups)

        # Run the actual clusterer and score it, rather than summarising the
        # histograms with a scalar. The question the mechanism depends on is
        # whether the declared channel carries RECOVERABLE structure, and a
        # clusterer answers that directly.
        #
        # A scalar comparison of between-group separation against within-group
        # scatter gives the wrong answer here: on label-permuted MNIST it reads
        # 0.0064 against 0.0162 and calls the channel noise, while clustering
        # those same histograms recovers the planted grouping at Adjusted Rand
        # Index (ARI) 0.665. The clusterer works on the whole class-dimensional
        # pattern, where a small deviation that is CONSISTENT within a group
        # separates cleanly even when its magnitude sits below per-client
        # scatter.
        from client import ClientUpdate

        ups = [ClientUpdate(client_id=i, weights=[], metadata={"label_hist": H[i]})
               for i in range(len(H))]
        n_groups = len(np.unique(g))
        recovered = DistributionClusterer(n_clusters=n_groups).cluster(ups)
        labels = np.full(len(H), -1)
        for ci, members in enumerate(recovered.values()):
            for u in members:
                labels[u.client_id] = ci
        ari = float(adjusted_rand_score(g, labels))

        if ari < 0.1:
            raise ValueError(
                f"mechanism={self.mechanism!r} steers toward a target label "
                f"distribution, but clustering the declared histograms recovers "
                f"the planted grouping at ARI {ari:.3f}, i.e. barely better than "
                f"chance. The declared channel carries no structure here, so the "
                f"attack would be an exact no-op and would be recorded as an "
                f"attack that was tried and failed. Use mechanism='relabel' "
                f"(train the target's concept) or 'weight' (steer the update).")
        return True

    def _target_for(self, client_id: int, y_own=None):
        if self.target_hist is not None:
            return _normalise(self.target_hist)
        if self.estimator is None:
            return None
        return self.estimator.estimate()

    # -- update-side ------------------------------------------------------- #
    def shape_update(self, update, context: dict):
        """Blend the delta toward the target, then restore an honest norm.

        Runs after training, because steering is defined RELATIVE to the honest
        direction: the attacker needs its own honest delta before it can decide
        how far to move away from it.
        """
        cid = update.client_id
        if cid not in self.malicious_clients:
            return update

        if self.mechanism == "declare" and self.beta > 0:
            # TIER 2. The Iterative Federated Clustering Algorithm (IFCA) has
            # each client evaluate the K cluster models and REPORT which fits
            # best. The server takes that report at face value, so spoofing it
            # costs the attacker nothing at all: no retraining, no weight edit,
            # no accuracy sacrifice. It overwrites one integer.
            #
            # This runs after the server has written the honest choice at
            # fl_loop.py, which is exactly the point: any value the client
            # states can be replaced between computing it and submitting it.
            declared = self._target_index(context)
            if declared is not None:
                self.trace.append({"client_id": cid, "mechanism": "declare",
                                   "honest_choice": update.metadata.get("cluster_choice"),
                                   "declared_choice": declared})
                update.metadata["cluster_choice"] = declared
            return update

        if self.beta <= 0 and self.payload == "none":
            return update

        from aggregation import cosine, flatten, unflatten

        delta = np.asarray(update.metadata.get("delta_vec"), dtype=np.float64)
        if delta is None or not delta.size:
            return update
        global_vec = np.asarray(update.vec, dtype=np.float64) - delta

        if self.mechanism in ("none", "data", "relabel") or self.beta <= 0:
            # Placement acts before this hook, in `training_data`, or not at all.
            # `relabel` submits weights that are honest for the task it trained,
            # so with payload="none" it passes through completely untouched --
            # that is the whole point of it. A payload still applies here, and
            # measuring it against this honest baseline isolates how visible the
            # DAMAGE is, separately from how visible the placement was.
            return self._apply_payload(update, delta, global_vec)

        if self.mechanism == "weight":
            target_delta = self._estimate_target_delta(context, cid)
            if target_delta is None:
                return update

            if self.constraint == "none":
                steered = (1.0 - self.beta) * delta + self.beta * target_delta
            else:
                # The published constraints are defined on the parameters the
                # server actually inspects, so they are applied on the head when
                # one is in use. Applying them to the full vector would bound a
                # quantity nobody is looking at: the head is 0.6% of parameters,
                # so a full-vector bound is satisfied almost automatically.
                idx = (np.asarray(self.head_index)
                       if self.head_index is not None else None)
                H = self._honest_deltas(context, index=idx)
                if H is None or len(H) < 2:
                    return update

                d_s = delta[idx] if idx is not None else delta
                t_s = target_delta[idx] if idx is not None else target_delta

                if self.constraint == "lie":
                    blended = (1.0 - self.beta) * d_s + self.beta * t_s
                    bounded, info = self._lie_bound(blended, H)
                else:                                   # minmax
                    bounded, info = self._minmax_bound(d_s, t_s, H)

                info.update(client_id=cid, constraint=self.constraint,
                            scope="head" if idx is not None else "full")
                self.trace.append(info)

                # Write the bounded values back into the full-length delta, so
                # only the constrained subspace is altered.
                steered = delta.copy()
                if idx is not None:
                    steered[idx] = bounded
                else:
                    steered = bounded
        else:                        # prior
            steered = self._prior_steer(update, delta, context)
            if steered is None:
                return update

        # Restore the honest norm. Without this the steered delta's magnitude
        # betrays it to any norm check, and more importantly a blend of two
        # vectors is systematically SHORTER than either, so an unrescaled
        # steered update would be quietly down-weighted by the aggregator and
        # the attack would look weaker than it is for a reason unrelated to
        # placement.
        if self.match_norm:
            honest_norms = [float(np.linalg.norm(u.metadata["delta_vec"]))
                            for u in context.get("updates", [])
                            if u.client_id not in self.malicious_clients
                            and u.metadata.get("delta_vec") is not None]
            wanted = float(np.median(honest_norms)) if honest_norms else float(
                np.linalg.norm(delta))
            got = float(np.linalg.norm(steered))
            if got > 0:
                steered = steered * (wanted / got)

        # Stealth is measured on the HEAD, because that is what the server
        # clusters on. The full-vector figure is kept beside it rather than
        # discarded: the gap between the two IS the finding, and hiding it is
        # F_ATTEMPTS.md C2 records what the full-vector figure measures instead.
        entry = {
            "client_id": cid, "mechanism": self.mechanism, "beta": self.beta,
            "cos_full": cosine(delta, steered),
            "norm_before": float(np.linalg.norm(delta)),
            "norm_after": float(np.linalg.norm(steered)),
        }
        if self.head_index is not None:
            idx = np.asarray(self.head_index)
            entry["cos_honest_to_steered"] = cosine(delta[idx], steered[idx])
            entry["scope"] = "head"
        else:
            entry["cos_honest_to_steered"] = entry["cos_full"]
            entry["scope"] = "full"
        self.trace.append(entry)

        update.vec = global_vec + steered
        update.metadata["delta_vec"] = steered
        update.metadata["steered"] = True
        return self._apply_payload(update, steered, global_vec, honest=delta)

    def _apply_payload(self, update, delta, global_vec, honest=None):
        """What the attacker does once placement is settled.

        `honest` is the update it would have submitted with no payload, which is
        the reference the visibility measurement is taken against. For `relabel`
        that reference is the relabelled-honest update, not the attacker's
        original one: the question a defence asks is whether THIS submission
        looks like a well-formed member of the cluster it was placed in, and
        after a successful relabel it genuinely is one.
        """
        if self.payload == "none":
            return update

        from aggregation import cosine

        if self.payload == "boost":
            poisoned = delta * self.payload_scale
        else:                                   # flip
            poisoned = -delta * self.payload_scale

        ref = delta if honest is None else honest
        entry = {"client_id": update.client_id, "payload": self.payload,
                 "payload_scale": self.payload_scale,
                 "payload_cos_full": cosine(ref, poisoned),
                 "payload_norm_ratio": (float(np.linalg.norm(poisoned)
                                              / max(np.linalg.norm(ref), 1e-12)))}
        if self.head_index is not None:
            idx = np.asarray(self.head_index)
            entry["payload_cos_head"] = cosine(ref[idx], poisoned[idx])
        self.trace.append(entry)

        update.vec = global_vec + poisoned
        update.metadata["delta_vec"] = poisoned
        update.metadata["payload"] = self.payload
        return update

    def _honest_deltas(self, context: dict, index=None):
        """The other clients' deltas, optionally restricted to a parameter scope.

        Reading these is an ORACLE capability: a real attacker cannot see what
        its peers submitted. Both published constraints are defined in terms of
        the honest population, so any attack using them is an upper bound on
        attacker power and must be reported as one.
        """
        import numpy as np

        out = []
        for u in context.get("updates") or []:
            if u.client_id in self.malicious_clients:
                continue
            d = u.metadata.get("delta_vec")
            if d is None:
                continue
            d = np.asarray(d, dtype=np.float64)
            out.append(d[np.asarray(index)] if index is not None else d)
        return np.vstack(out) if out else None

    def _lie_bound(self, steered, H):
        """A Little Is Enough: clip to mu +/- z_max * sigma, per coordinate.

            s     = floor(n/2 + 1) - m
            z_max = largest z with Phi(z) < (n - m - s) / (n - m)

        `n` counts all clients and `m` the malicious ones. The quantity being
        bounded is how far a submission can sit from the honest mean while a
        majority-based rule still accepts it, so the limit is set by the honest
        population's own spread rather than by a dial.
        """
        import numpy as np
        from scipy.stats import norm

        n = len(H) + len(self.malicious_clients)
        m = max(1, len(self.malicious_clients))
        s = int(np.floor(n / 2 + 1)) - m
        denom = max(n - m, 1)
        q = (n - m - s) / denom
        # Phi(z) < q, so z_max is the quantile just below q. Guard the
        # degenerate ends, where the inverse CDF is infinite.
        z_max = float(norm.ppf(min(max(q, 1e-6), 1 - 1e-6)))

        mu, sigma = H.mean(axis=0), H.std(axis=0)
        lo, hi = mu - abs(z_max) * sigma, mu + abs(z_max) * sigma
        return np.clip(steered, lo, hi), {"z_max": abs(z_max), "n": n, "m": m, "s": s}

    def _minmax_bound(self, honest_delta, target_delta, H):
        """Min-Max: the largest blend that stays inside the honest spread.

            maximise gamma  s.t.  max_i ||x - d_i|| <= max_{i,j} ||d_i - d_j||
            where          x = (1 - gamma) * honest + gamma * target

        Binary search on gamma, which is monotone in the constraint. Returns the
        admissible update and the gamma that produced it, so the experiment can
        report how much movement the honest spread actually permitted rather
        than how much we asked for.
        """
        import numpy as np

        pair_max = 0.0
        for i in range(len(H)):
            for j in range(i + 1, len(H)):
                pair_max = max(pair_max, float(np.linalg.norm(H[i] - H[j])))

        def feasible(gamma):
            x = (1.0 - gamma) * honest_delta + gamma * target_delta
            return max(float(np.linalg.norm(x - h)) for h in H) <= pair_max

        if not feasible(0.0):
            # Even the honest update violates the bound, which means the honest
            # spread is not what this measure assumes. Report rather than force.
            return honest_delta, {"gamma": 0.0, "pair_max": pair_max,
                                  "note": "honest update already outside the bound"}
        lo, hi = 0.0, 1.0
        if feasible(1.0):
            gamma = 1.0
        else:
            for _ in range(40):
                mid = (lo + hi) / 2
                if feasible(mid):
                    lo = mid
                else:
                    hi = mid
            gamma = lo
        x = (1.0 - gamma) * honest_delta + gamma * target_delta
        return x, {"gamma": float(gamma), "pair_max": pair_max}

    def _target_index(self, context: dict):
        """Which cluster index the target group's members reported.

        Reads the other clients' declarations, which a real attacker cannot do,
        so this is an ORACLE and an upper bound like `_estimate_target_delta`.
        For tier 2 that bound is close to tight: the realistic attacker only has
        to guess one integer out of K, and with K=2 guessing is a coin flip that
        it can simply repeat each round until it lands.
        """
        votes = {}
        for u in context.get("updates") or []:
            if u.client_id in self.target_members:
                c = u.metadata.get("cluster_choice")
                if c is not None:
                    votes[int(c)] = votes.get(int(c), 0) + 1
        return max(votes, key=votes.get) if votes else None

    def _estimate_target_delta(self, context: dict, client_id: int):
        """What direction would a member of the target group have submitted?

        With `target_members` set this reads their submitted deltas directly,
        which a real attacker CANNOT do. That makes it an upper bound on
        attacker power, and it must be labelled as one wherever it is reported.
        The realistic version estimates the target's label distribution and
        synthesises a delta by training on data resampled to it, which is the
        two-stage attack and is built on top of this once the bound is known.
        """
        updates = context.get("updates") or []
        deltas = [np.asarray(u.metadata["delta_vec"], dtype=np.float64)
                  for u in updates
                  if u.client_id in self.target_members
                  and u.metadata.get("delta_vec") is not None]
        if not deltas:
            return None
        return np.mean(deltas, axis=0)

    def _prior_steer(self, update, delta, context: dict):
        """Logit adjustment on the final-layer bias, expressed as a delta.

        Moves the model's output prior toward the target's label distribution.
        Unlike `weight` this changes what the model BEHAVES like, not only which
        direction it points, so it survives a behavioural check that a pure
        direction blend would fail.
        """
        from aggregation import flatten, unflatten
        from lab_probe import mean_output_prior_torch
        from models import build_torch_model, final_bias_key

        target = self._target_for(update.client_id)
        if target is None:
            return None

        model = build_torch_model(self.arch, self.input_shape, self.num_classes)
        template = model.state_dict()
        key = final_bias_key(model)

        source = mean_output_prior_torch(self.arch, update.vec, self.input_shape,
                                         self.num_classes,
                                         self.own_data.get(update.client_id, (None,))[0])
        blended = _normalise((1.0 - self.beta) * source + self.beta * np.asarray(target))

        state = unflatten(update.vec, template)
        shift = np.log(blended + 1e-9) - np.log(_normalise(source) + 1e-9)
        state[key] = state[key] + torch_as(shift, state[key])
        return flatten(state) - (np.asarray(update.vec, dtype=np.float64) - delta)


def corrupt_concept(concept_fn, num_classes: int, swaps: int, seed: int = 0):
    """Wrap a concept function so the attacker holds an IMPERFECT copy of it.

    Every attack in this study is handed the target group's concept exactly,
    which makes it an upper bound on attacker power rather than a threat model.
    A real attacker must infer the target's task, and will get it partly wrong.
    This degrades the knowledge by applying `swaps` random transpositions on top
    of the true target concept:

        swaps = 0                 oracle, the exact target concept
        swaps = 1 .. C/2          partial, progressively worse
        swaps >= C                effectively blind

    Sweeping `swaps` answers the question the oracle result cannot: does
    placement degrade gracefully as the attacker's knowledge worsens, or does it
    collapse the moment the knowledge is imperfect? Graceful degradation means
    the spoof is a realistic threat; a cliff at one swap means it depends on
    knowledge an attacker is unlikely to have, which is a genuine limitation and
    worth reporting as one.

    Composing on top of the true concept avoids needing the underlying
    permutation, so this works for label permutation and generalises to any
    other concept the bundle supplies.
    """
    rng = np.random.default_rng(seed)
    perm = np.arange(num_classes)
    for _ in range(max(0, int(swaps))):
        i, j = rng.choice(num_classes, size=2, replace=False)
        perm[i], perm[j] = perm[j], perm[i]

    def corrupted(X, y, group, from_group=None):
        Xc, yc = concept_fn(X, y, group, from_group=from_group)
        return Xc, perm[np.asarray(yc)]

    corrupted.swaps = int(swaps)
    corrupted.perm = perm
    # How much of the target concept survives the corruption, which is the
    # x-axis the ladder should actually be plotted against: `swaps` counts
    # transpositions, but two swaps can undo each other.
    corrupted.fidelity = float(np.mean(perm == np.arange(num_classes)))
    return corrupted


def torch_as(array, like):
    """Small helper: numpy array to a tensor matching `like`'s dtype and device."""
    import torch

    return torch.as_tensor(np.asarray(array), dtype=like.dtype, device=like.device)


class CompositeAttack(Attack):
    """Apply multiple attacks in sequence (e.g. arch spoof + sign flip)."""

    def __init__(self, *attacks: Attack):
        self.attacks = attacks

    def training_arch(self, client_id: int, true_arch: str) -> Optional[str]:
        for attack in self.attacks:
            arch = attack.training_arch(client_id, true_arch)
            if arch is not None:
                return arch
        return None

    def training_data(self, client_id: int, X, y):
        for attack in self.attacks:
            X, y = attack.training_data(client_id, X, y)
        return X, y

    def apply(self, update: ClientUpdate) -> ClientUpdate:
        for attack in self.attacks:
            update = attack.apply(update)
        return update


# --------------------------------------------------------------------------- #
# Adaptive distribution spoofing
# --------------------------------------------------------------------------- #
# `LabelHistSpoof` above lies about the declared label histogram but never
# touches the weights, so the submitted model still behaves like the attacker's
# real data and `defense.DistributionFingerprintVerifier` catches the mismatch.
# It gets into the cluster and is thrown straight back out.
#
# To stay in, an attacker has to satisfy two constraints at once:
#
#   PLACEMENT    the declared histogram alone decides the cluster, since
#                `clustering.DistributionClusterer` sees nothing else.
#   CONSISTENCY  the model's output prior on the server's probe set has to match
#                that declaration.
#
# The way to satisfy both is to MAKE THE LIE TRUE: change what the model behaves
# like so it matches what was declared. Same move as `ArchSpoof(adaptive=True)`,
# which trains the architecture it declares, applied to data instead.
#
# Verified before this was written: the bias edit below closes 95-100% of the JS
# gap to an arbitrary target, it survives the attacker measuring its own prior on
# its own data while the server measures on a different probe set, and it can
# imitate classes the attacker holds NONE of (a client with zero examples of a
# class was made to output it with probability 0.91). Resampling cannot do that
# last one at all, which makes calibration strictly the stronger mechanism.


def calibrate_output_prior(weights, target, source, eps: float = 1e-9):
    """Shift the final softmax bias so the model's output prior moves toward
    `target`. Returns new weights; the input is not mutated.

    This is logit adjustment (Menon et al., 2021, long-tail learning): adding
    log(target) - log(source) to the logits reweights the output distribution
    without retraining. Only the final Dense bias is touched, so learned features
    survive and the model still classifies sensibly; only its class priors move.

    It is an approximation, since the mean of a softmax is not the softmax of a
    mean, so callers record the prior actually achieved rather than the one asked
    for.
    """
    target = np.asarray(target, dtype=np.float64) + eps
    source = np.asarray(source, dtype=np.float64) + eps
    target = target / target.sum()
    source = source / source.sum()
    delta = np.log(target) - np.log(source)

    out = [np.array(w, copy=True) for w in weights]
    if out[-1].shape != delta.shape:
        raise ValueError(
            f"final bias has shape {out[-1].shape}, expected {delta.shape}. "
            "calibrate_output_prior assumes the model ends in a Dense softmax layer."
        )
    out[-1] = out[-1] + delta.astype(out[-1].dtype)
    return out


def resample_for_target(X, y, target_hist, rng, size: Optional[int] = None):
    """Resample the client's own data so its label mix approaches `target_hist`.

    Returns (X_new, y_new, achieved_hist).

    The honest mechanism: the model genuinely learns the prior it will declare.
    Also strictly bounded, and the bound is the interesting part. A client holding
    no examples of a class cannot manufacture them, so the achieved histogram
    falls short by exactly the mass the target placed on classes it does not hold,
    and that mass is redistributed over the classes it does. `achieved_hist` is
    returned so callers record what happened, not what was requested.
    """
    from data import label_histogram

    y = np.asarray(y)
    target = np.asarray(target_hist, dtype=np.float64)
    target = target / target.sum()
    num_classes = len(target)
    size = int(size or len(y))

    available = {c: np.where(y == c)[0] for c in range(num_classes)}
    available = {c: idx for c, idx in available.items() if len(idx)}

    reachable = np.array([target[c] if c in available else 0.0 for c in range(num_classes)])
    if reachable.sum() <= 0:
        return X, y, label_histogram(y, num_classes)
    reachable = reachable / reachable.sum()

    picks = []
    for c, share in enumerate(reachable):
        n = int(round(share * size))
        if n <= 0 or c not in available:
            continue
        pool = available[c]
        # With replacement, so a small class can be over-represented. Without it
        # the target is unreachable whenever it asks for more of a class than the
        # client holds, which is most of the interesting cases.
        picks.append(rng.choice(pool, size=n, replace=len(pool) < n))

    if not picks:
        return X, y, label_histogram(y, num_classes)

    idx = np.concatenate(picks)
    rng.shuffle(idx)
    return X[idx], y[idx], label_histogram(y[idx], num_classes)


class TargetEstimator:
    """Works out which distribution the attacker should imitate.

    Three levels of attacker knowledge, so results show how much knowledge the
    attack actually needs rather than assuming the best case:

      oracle    handed the target cluster's true mean histogram. Not realistic;
                an upper bound on attacker power.
      inferred  probes the global model the server hands it and reads the mean
                softmax. A cluster's global model leaks the aggregate label
                distribution of its members, because a model trained mostly on
                class 3 predicts class 3 more often. Needs no access beyond what
                every participant already receives.
      blind     uniform, or one-hot on a guessed class. Roughly what
                `LabelHistSpoof` does today, and the baseline the others must beat.
    """

    MODES = ("oracle", "inferred", "blind")

    def __init__(self, mode: str, num_classes: int, arch: str = "mlp_flat",
                 input_shape=None, oracle_hist=None, blind_class: Optional[int] = None,
                 blind_strength: float = 0.9, probe_X=None):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.num_classes = num_classes
        self.arch = arch
        self.input_shape = input_shape
        self.oracle_hist = None if oracle_hist is None else np.asarray(oracle_hist, float)
        self.blind_class = blind_class
        self.blind_strength = blind_strength
        # Neutral inputs to probe the broadcast model on. Without this the
        # estimate is taken over the attacker's own data and reflects the
        # attacker rather than the target; see `estimate`.
        self.probe_X = probe_X

    def estimate(self, global_weights=None, X_own=None) -> np.ndarray:
        if self.mode == "oracle":
            if self.oracle_hist is None:
                raise ValueError("oracle mode needs oracle_hist")
            return _normalise(self.oracle_hist)

        if self.mode == "blind":
            if self.blind_class is None:
                return np.ones(self.num_classes) / self.num_classes
            h = np.full(self.num_classes, (1 - self.blind_strength) / self.num_classes)
            h[self.blind_class] += self.blind_strength
            return _normalise(h)

        if global_weights is None or X_own is None:
            raise ValueError(
                "inferred mode needs the global model the server handed us plus a "
                "set of inputs to probe it on. Pass them to estimate(), as "
                "estimate(global_weights, X_probe), or use oracle/blind."
            )

        # PROBE ON NEUTRAL INPUTS, NOT THE ATTACKER'S OWN DATA.
        #
        # The quantity wanted is the TARGET's label distribution, read off the
        # model the server broadcast. Averaging the model's output over the
        # attacker's own inputs mixes in whatever the attacker's data happens to
        # be, and on a skewed client that term dominates: an attacker holding 90%
        # class 6 estimated 0.35 on class 6 when the target's true share was
        # 0.06, which is a blurred picture of itself rather than of the target.
        #
        # `probe_X`, when set, is used in preference to whatever is passed as
        # X_own, so a caller can hand over a held-out neutral set.
        X = self.probe_X if getattr(self, "probe_X", None) is not None else X_own

        # Dispatch on how the weights arrive. A flat float vector is the torch
        # representation; a list of arrays is Keras. Both backends register the
        # same architecture names, so `self.arch` is valid for either.
        vec = np.asarray(global_weights)
        if vec.ndim == 1 and vec.dtype.kind == "f":
            from lab_probe import mean_output_prior_torch

            prior = mean_output_prior_torch(self.arch, vec, self.input_shape,
                                            self.num_classes, X)
        else:
            prior = mean_output_prior(self.arch, global_weights, self.input_shape,
                                      self.num_classes, X)
        return _normalise(prior)


class AdaptiveLabelHistSpoof(Attack):
    """Declare a histogram that lands us in the target cluster, then make the
    model behave as though that declaration were true.

    `imitation` selects the mechanism:

      calibrate  edit the final-layer bias after training. Powerful: it imitates
                 classes the attacker holds none of, and being a pure weight edit
                 inside apply() it runs through the UNMODIFIED FederatedServer.
      resample   train on data resampled toward the declaration. More natural but
                 bounded by what the client holds, and it needs the
                 `training_data` hook, which only the corpus generator calls.
      none       declare the lie and change nothing, i.e. the naive baseline.
                 Kept here so every condition shares one code path and one trace.

    `lam` blends the attacker's real histogram with the target estimate:
        declared = (1 - lam) * true + lam * target
    lam=0 is honest, lam=1 is full imitation. It is the reach-versus-stealth dial.
    """

    IMITATIONS = ("calibrate", "resample", "none")

    def __init__(self, malicious_clients: Optional[Iterable[int]] = None,
                 estimator: Optional[TargetEstimator] = None,
                 lam: float = 1.0, imitation: str = "calibrate",
                 arch: str = "mlp_flat", input_shape=None, num_classes: int = 0,
                 seed: int = 0):
        if imitation not in self.IMITATIONS:
            raise ValueError(f"imitation must be one of {self.IMITATIONS}")
        self.malicious_clients: Set[int] = set(malicious_clients or [])
        self.estimator = estimator
        self.lam = float(lam)
        self.imitation = imitation
        self.arch = arch
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.rng = np.random.default_rng(seed)

        # Set by the caller each round. A real client obviously has the global
        # model, it was just sent it; the stock FederatedServer simply does not
        # route it to the Attack object, so the corpus generator supplies it.
        # A simulator limitation, not a concession in the threat model.
        self.global_weights = None
        self.own_data = {}          # client_id -> (X, y)

    def _declared(self, true_hist, target):
        d = (1.0 - self.lam) * np.asarray(true_hist, float) + self.lam * np.asarray(target, float)
        return _normalise(d)

    def target_for(self, client_id: int) -> np.ndarray:
        X = self.own_data.get(client_id, (None, None))[0]
        return self.estimator.estimate(global_weights=self.global_weights, X_own=X)

    def training_data(self, client_id: int, X, y):
        """Only the resample path changes anything. The stock FederatedServer
        never calls this, which is why resample needs the generator's own loop."""
        from data import label_histogram

        if client_id not in self.malicious_clients or self.imitation != "resample":
            return X, y
        declared = self._declared(label_histogram(np.asarray(y), self.num_classes),
                                  self.target_for(client_id))
        Xr, yr, _ = resample_for_target(X, y, declared, self.rng)
        return Xr, yr

    def apply(self, update: ClientUpdate) -> ClientUpdate:
        from data import label_histogram

        cid = update.client_id
        if cid not in self.malicious_clients:
            return update

        # What train_client recorded is the histogram of whatever we actually
        # trained on, which under `resample` is already the doctored mix.
        trained_hist = np.asarray(update.metadata.get("label_hist"), dtype=np.float64)

        X_own, y_own = self.own_data.get(cid, (None, None))
        true_hist = (label_histogram(np.asarray(y_own), self.num_classes)
                     if y_own is not None else trained_hist)

        target = self.target_for(cid)
        declared = self._declared(true_hist, target)

        weights = update.weights
        if self.imitation == "calibrate":
            # Measured on the attacker's OWN data, not the server's probe set,
            # because the attacker does not have the probe set. The resulting
            # mismatch is a real limitation and is left in rather than corrected.
            source = (mean_output_prior(self.arch, weights, self.input_shape,
                                        self.num_classes, X_own)
                      if X_own is not None
                      else np.ones(self.num_classes) / self.num_classes)
            weights = calibrate_output_prior(weights, declared, source)

        metadata = dict(update.metadata)
        metadata["label_hist"] = declared          # what the server clusters on
        metadata["spoofed"] = True
        metadata["dist_true_hist"] = true_hist
        metadata["dist_target_est"] = target
        metadata["dist_declared"] = declared
        metadata["dist_trained_hist"] = trained_hist
        metadata["dist_lambda"] = self.lam
        metadata["dist_knowledge"] = self.estimator.mode if self.estimator else "none"
        metadata["dist_imitation"] = self.imitation

        return ClientUpdate(client_id=cid, weights=weights, metadata=metadata,
                            true_arch=update.true_arch)
