"""Federated learning round orchestrator."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from aggregation import fedavg, fedavg_weighted
from attacks import Attack, NoAttack
from client import ClientUpdate, train_client
from clustering import ClusteringStrategy
from models import build_model


class FederatedServer:
    """
    Shared FL loop for architecture- and distribution-based clustering.

    - Architecture mode: one global model per architecture name (cluster ids == arch names).
    - Distribution mode: one global model per discovered cluster (shared base architecture).

    An optional `defense` verifies each client's declared metadata against the
    behaviour of its update and quarantines spoofers before aggregation:
    `defense.FingerprintVerifier` for architecture mode (declared architecture),
    `defense.DistributionFingerprintVerifier` for distribution mode (declared
    label histogram). Both expose the same `calibrate(...)`/`flag(update)`
    interface so `fit()` can drive either one generically.

    Observability: each round's history entry (see `fit`) carries per-cluster
    accuracy and loss, cross-cluster fairness, per-round detection quality, and a
    per-client log (which cluster it landed in, declared/true arch, spoof ground
    truth, local train loss/acc, and whether the client was rejected and why).
    See analysis.py to turn this into CSVs and plots.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        clusterer: ClusteringStrategy,
        client_archs: List[str],
        attack: Optional[Attack] = None,
        base_arch: str = "mlp",
        mode: str = "architecture",
        defense=None,
        weighted_aggregation: bool = True,
    ):
        if mode not in ("architecture", "distribution"):
            raise ValueError("mode must be 'architecture' or 'distribution'")

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.clusterer = clusterer
        self.client_archs = client_archs
        self.attack = attack or NoAttack()
        self.base_arch = base_arch
        self.mode = mode
        self.defense = defense
        # Standard FedAvg weights each client's contribution by its sample count.
        # Set False for the unweighted mean, which hands a client with very
        # little data the same influence as one with a lot -- relevant here
        # because Dirichlet partitioning makes client sizes very uneven, so an
        # unweighted mean would inflate a small malicious client's impact and
        # confound "attack strength" with "aggregation choice".
        self.weighted_aggregation = weighted_aggregation
        # (round, client_id, flagged, is_spoof, mechanism)
        self.detections: List[tuple] = []
        self.history: List[dict] = []       # populated by fit(); kept for post-hoc analysis

        self.global_models: Dict[str, object] = {}
        if mode == "architecture":
            for arch in sorted(set(client_archs)):
                self.global_models[arch] = build_model(arch, input_dim, num_classes)
        else:
            self.global_models["cluster_0"] = build_model(base_arch, input_dim, num_classes)

    def _arch_for_client(self, client_id: int) -> str:
        if self.mode == "architecture":
            return self.client_archs[client_id]
        return self.base_arch

    def _model_key_for_training(self, client_id: int) -> str:
        if self.mode == "architecture":
            return self.client_archs[client_id]
        return getattr(self, "_client_cluster", {}).get(client_id, "cluster_0")

    def _ensure_cluster_model(self, cluster_id: str):
        if cluster_id in self.global_models:
            return
        template_key = next(iter(self.global_models))
        template = self.global_models[template_key]
        new_model = build_model(self.base_arch, self.input_dim, self.num_classes)
        new_model.set_weights(template.get_weights())
        self.global_models[cluster_id] = new_model

    def _aggregate(self, usable: List[ClientUpdate]):
        """FedAvg over the accepted updates, sample-weighted unless disabled."""
        weight_lists = [u.weights for u in usable]
        if not self.weighted_aggregation:
            return fedavg(weight_lists)
        counts = [int(u.metadata.get("n_samples") or 0) for u in usable]
        if sum(counts) <= 0:
            return fedavg(weight_lists)
        return fedavg_weighted(weight_lists, counts)

    @staticmethod
    def _is_spoof(update: ClientUpdate) -> bool:
        """Ground truth: did an Attack falsify this client's declared metadata?
        Both ArchSpoof and LabelHistSpoof set metadata['spoofed'] = True when they
        rewrite a field, regardless of which metadata (architecture or label
        histogram) was lied about -- this is attack-agnostic on purpose."""
        return bool(update.metadata.get("spoofed", False))

    def run_round(self, client_data, round_num=0, epochs=10, batch_size=32, verbose=True):
        updates: List[ClientUpdate] = []

        for client_id, (X_client, y_client) in enumerate(client_data):
            home_arch = self._arch_for_client(client_id)
            train_arch = home_arch
            model_key = self._model_key_for_training(client_id)

            # Architecture spoofing: a malicious client trains (and warm-starts from)
            # the architecture it claims, so its update is shape-valid for the victim
            # cluster and survives the shape check below.
            spoof_arch = self.attack.training_arch(client_id, home_arch)
            if spoof_arch is not None:
                train_arch = spoof_arch
                if self.mode == "architecture":
                    model_key = spoof_arch

            if model_key not in self.global_models:
                model_key = next(iter(self.global_models))

            if verbose:
                note = f"  [spoofing {home_arch.upper()}->{train_arch.upper()}]" if spoof_arch else ""
                print(f"  Training client {client_id} using {train_arch.upper()} (from {model_key}){note}")

            update = train_client(
                client_id=client_id,
                X_client=X_client,
                y_client=y_client,
                arch_name=train_arch,
                global_weights=self.global_models[model_key].get_weights(),
                input_dim=self.input_dim,
                num_classes=self.num_classes,
                epochs=epochs,
                batch_size=batch_size,
                true_arch=home_arch,
            )
            update = self.attack.apply(update)
            if verbose and update.metadata.get("spoofed"):
                print(f"  [!] Client {client_id} infiltrated '{update.metadata['arch']}' "
                      f"cluster (true arch {update.true_arch})")
            updates.append(update)

        clusters = self.clusterer.cluster(updates)
        if verbose:
            summary = {cid: [u.client_id for u in ups] for cid, ups in clusters.items()}
            print(f"  Clusters: {summary}")

        # client_id -> "accepted" | "shape_rejected" | "quarantined"
        status: Dict[int, str] = {u.client_id: "accepted" for u in updates}

        self._client_cluster = {}
        for cluster_id, cluster_updates in clusters.items():
            for u in cluster_updates:
                self._client_cluster[u.client_id] = cluster_id

            if self.mode == "architecture":
                target_model = self.global_models.get(cluster_id)
                if target_model is None:
                    continue
                target_shapes = [w.shape for w in target_model.get_weights()]
                usable = []
                for u in cluster_updates:
                    shapes = [w.shape for w in u.weights]
                    if shapes != target_shapes:
                        status[u.client_id] = "shape_rejected"
                        # Recorded so the audit trail shows this client was
                        # blocked, but tagged "shape" so it is excluded from the
                        # fingerprint verifier's precision/recall -- the shape
                        # check stopped it, not the defence. Without the tag a
                        # naive (non-adaptive) spoofer would silently vanish
                        # from the detection stats entirely.
                        self.detections.append(
                            (round_num, u.client_id, True, self._is_spoof(u), "shape"))
                        if verbose:
                            print(f"  Skipping client {u.client_id} in cluster '{cluster_id}': "
                                  f"weight shape mismatch")
                        continue
                    # Behavioural-fingerprint verification of the declared metadata.
                    if self.defense is not None:
                        flagged = self.defense.flag(u)
                        self.detections.append(
                            (round_num, u.client_id, flagged, self._is_spoof(u), "fingerprint"))
                        if flagged:
                            status[u.client_id] = "quarantined"
                            if verbose:
                                print(f"  [defence] quarantined client {u.client_id} from "
                                      f"'{cluster_id}' (fingerprint inconsistent with declared arch)")
                            continue
                    usable.append(u)
                if not usable:
                    continue
                target_model.set_weights(self._aggregate(usable))
            else:
                self._ensure_cluster_model(cluster_id)
                usable = []
                for u in cluster_updates:
                    # Behavioural-fingerprint verification of the declared label
                    # histogram (distribution-mode analogue of the arch check above).
                    if self.defense is not None:
                        flagged = self.defense.flag(u)
                        self.detections.append(
                            (round_num, u.client_id, flagged, self._is_spoof(u), "fingerprint"))
                        if flagged:
                            status[u.client_id] = "quarantined"
                            if verbose:
                                print(f"  [defence] quarantined client {u.client_id} from "
                                      f"'{cluster_id}' (distribution fingerprint inconsistent "
                                      f"with declared label histogram)")
                            continue
                    usable.append(u)
                if not usable:
                    continue
                self.global_models[cluster_id].set_weights(self._aggregate(usable))

        client_log = [
            {
                "round": round_num + 1,
                "client_id": u.client_id,
                "cluster": self._client_cluster.get(u.client_id),
                "declared_arch": u.metadata.get("arch"),
                "true_arch": u.true_arch,
                "is_spoof": self._is_spoof(u),
                "status": status.get(u.client_id, "accepted"),
                "n_samples": u.metadata.get("n_samples"),
                "train_loss": u.metadata.get("train_loss"),
                "train_accuracy": u.metadata.get("train_accuracy"),
            }
            for u in updates
        ]
        return clusters, client_log

    def evaluate(self, X_test, y_test, verbose=True):
        """Return (accuracy_by_cluster, loss_by_cluster) on the held-out test set."""
        accuracy, loss = {}, {}
        for name, model in self.global_models.items():
            l, acc = model.evaluate(X_test, y_test, verbose=0)
            accuracy[name] = float(acc)
            loss[name] = float(l)
            if verbose:
                print(f"  {name}: accuracy={acc:.4f} loss={l:.4f}")
        if verbose and len(accuracy) > 1:
            accs = list(accuracy.values())
            print(f"  fairness: variance={np.var(accs):.5f}  worst-group={min(accs):.4f}")
        return accuracy, loss

    @staticmethod
    def _fairness(accuracy: Dict[str, float]):
        if len(accuracy) < 2:
            return None
        accs = list(accuracy.values())
        return {"variance": float(np.var(accs)), "worst_group": float(min(accs))}

    def _round_detection(self, round_num: int):
        """Precision/recall/F1 of the FINGERPRINT DEFENCE for this round only
        (None if no defence). Clients stopped upstream by the weight-shape check
        are excluded from these scores -- they never reached the defence, so
        crediting it with them would overstate recall. `blocked_by_shape` counts
        spoofers the shape check caught, reported alongside so the two
        mechanisms stay distinguishable."""
        if self.defense is None:
            return None
        rows = [(flagged, spoof) for (r, _, flagged, spoof, mech) in self.detections
                if r == round_num and mech == "fingerprint"]
        shape_blocked = sum(1 for (r, _, _, spoof, mech) in self.detections
                            if r == round_num and mech == "shape" and spoof)
        if not rows:
            return None
        y_pred = [int(f) for f, _ in rows]
        y_true = [int(s) for _, s in rows]
        return {
            "blocked_by_shape": shape_blocked,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "tp": sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 1),
            "fp": sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 0),
            "fn": sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 1),
            "tn": sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 0),
        }

    def fit(self, client_data, X_test, y_test, rounds=2, epochs=10, batch_size=32,
            X_ref=None, y_ref=None):
        """Run `rounds` federated rounds and return the per-round history.

        `X_ref`/`y_ref` is server-held reference data (see `data.FLData.X_ref`)
        used to calibrate the defence. It MUST NOT be the test set: the defence
        trains shadow models on it, so calibrating on X_test and then reporting
        accuracy on X_test trains the defence on the evaluation data. If no
        reference data is supplied the run falls back to X_test and says so
        loudly -- treat any result produced that way as not reportable.
        """
        if self.defense is not None and not self.defense.calibrated:
            if X_ref is None or y_ref is None:
                print("[warn] no server reference data supplied; calibrating the defence "
                      "on the TEST set. This leaks evaluation data into the defence -- "
                      "pass X_ref/y_ref from prepare_fl_data() before reporting results.")
                X_ref, y_ref = X_test, y_test
            self.defense.calibrate(
                np.asarray(X_ref), np.asarray(y_ref),
                sorted(set(self.client_archs)), self.input_dim, self.num_classes,
            )

        history = []
        for round_num in range(rounds):
            print(f"Round {round_num + 1}/{rounds}")
            clusters, client_log = self.run_round(
                client_data, round_num=round_num, epochs=epochs, batch_size=batch_size)
            accuracy, loss = self.evaluate(X_test, y_test)
            fairness = self._fairness(accuracy)
            detection = self._round_detection(round_num)
            if detection is not None:
                print(f"  [defence] round detection: precision={detection['precision']:.3f} "
                      f"recall={detection['recall']:.3f} f1={detection['f1']:.3f} "
                      f"(tp={detection['tp']} fp={detection['fp']} fn={detection['fn']})")

            history.append({
                "round": round_num + 1,
                "clusters": clusters,
                "metrics": accuracy,     # kept as plain {name: accuracy} for backward compatibility
                "losses": loss,
                "fairness": fairness,
                "detection": detection,
                "client_log": client_log,
            })

        if self.defense is not None:
            self._report_detection()
        self.history = history
        return history

    def _report_detection(self):
        """Fingerprint-defence scores over all client-rounds. Shape-check
        rejections are counted separately, not folded in -- see
        `_round_detection`."""
        rows = [(f, s) for (_, _, f, s, mech) in self.detections if mech == "fingerprint"]
        shape_blocked = sum(1 for (_, _, _, s, mech) in self.detections
                            if mech == "shape" and s)
        if not rows:
            return
        y_pred = [int(f) for f, _ in rows]
        y_true = [int(s) for _, s in rows]
        print("\n[defence] spoofer detection over all client-rounds: "
              f"precision={precision_score(y_true, y_pred, zero_division=0):.3f} "
              f"recall={recall_score(y_true, y_pred, zero_division=0):.3f} "
              f"f1={f1_score(y_true, y_pred, zero_division=0):.3f}")
        if shape_blocked:
            print(f"[defence] (a further {shape_blocked} spoofing client-round(s) were "
                  f"blocked upstream by the weight-shape check, not by the defence)")


# =========================================================================== #
# Torch server: inferred cluster assignment
# =========================================================================== #
# The server above places clients by reading declared metadata. This one infers
# placement from the updates themselves, which is what essentially all of the
# CFL literature actually does, and it is the setting the steering attacks are
# measured in.
#
# NO DEFENCE HERE, DELIBERATELY. This loop measures placement, payload and
# update geometry only. A verifier would add a detection rate, and a detection
# rate is meaningless without a matched false-alarm rate, which is a separate
# piece of work. `defense.py` still holds the verifiers for when that resumes.
#
# ROUND 1 STARTS EVERY CLIENT FROM ONE SHARED INITIALISATION. This is not a
# detail. Clustering compares deltas from a common point; if clients begin from
# independent random inits their weight vectors are near-orthogonal at high
# dimension (measured: mean pairwise cosine +0.001 on a 1.66M parameter CNN) or
# dominated by shared initialisation at low dimension (measured: +0.768 on a
# 14.7k parameter MLP). Neither number says anything about who holds similar
# data, and FedAvg over independent inits sits at chance.

class TorchFederatedServer:
    """Clustered FL where the server infers membership from submitted updates.

    One architecture across the whole federation: this study is about
    DISTRIBUTION clusters, so architecture is held constant and is not the
    channel under attack.

    `clusterer` is any `ClusteringStrategy`. If it declares `needs_delta`, this
    server sets `metadata['delta_vec']` on every update before clustering.
    `IFCAClusterer` instead has each client report which cluster model fits it
    best, which this server computes and records as `metadata['cluster_choice']`.
    """

    # The two clusterers need OPPOSITE initialisation regimes, and getting this
    # wrong produces a federation that never splits. See F_ATTEMPTS.md D1.
    #
    #   shared    every cluster model starts from ONE initialisation. Required
    #             by Sattler: cosine similarity between client deltas is only
    #             meaningful if the deltas are measured from a common point.
    #
    #   distinct  each of the K cluster models starts from its own random draw.
    #             Required by IFCA: the client picks the cluster whose model
    #             gives the lowest local loss, and if the K models are identical
    #             every loss is identical, every client picks index 0, and the
    #             federation never splits. Observed exactly that way before this
    #             was fixed, and "IFCA never splits" reads like a property of
    #             IFCA rather than the symmetry bug it actually is. The
    #             published algorithm initialises the K models differently for
    #             precisely this reason.
    INIT_MODES = ("shared", "distinct")

    def __init__(self, input_shape, num_classes, clusterer, arch: str = "mlp_flat",
                 attack=None, aggregator: str = "fedavg", n_clusters: int = 2,
                 weighted_aggregation: bool = True, byzantine_f: int = 1,
                 seed: int = 0, init_mode: Optional[str] = None):
        from attacks import NoAttack
        from models import build_torch_model
        from aggregation import flatten

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.clusterer = clusterer
        self.arch = arch
        self.attack = attack or NoAttack()
        self.aggregator = aggregator
        self.n_clusters = n_clusters
        self.weighted_aggregation = weighted_aggregation
        self.byzantine_f = byzantine_f
        self.seed = seed

        if init_mode is None:
            # Derived from what the clusterer needs, but overridable, because
            # "shared init" is also a legitimate thing to TEST IFCA under.
            init_mode = ("distinct"
                         if getattr(clusterer, "KEY", None) == "cluster_choice"
                         else "shared")
        if init_mode not in self.INIT_MODES:
            raise ValueError(f"init_mode must be one of {self.INIT_MODES}")
        self.init_mode = init_mode

        import torch

        k = max(1, n_clusters)
        self.cluster_models: Dict[str, np.ndarray] = {}
        if init_mode == "shared":
            torch.manual_seed(seed)
            init = flatten(build_torch_model(arch, input_shape, num_classes).state_dict())
            self.cluster_models = {f"cluster_{i}": init.copy() for i in range(k)}
        else:
            for i in range(k):
                torch.manual_seed(seed * 1000 + i)
                self.cluster_models[f"cluster_{i}"] = flatten(
                    build_torch_model(arch, input_shape, num_classes).state_dict())

        # Tell the clusterer where the classifier head lives, if it wants to
        # restrict similarity to it. The server knows the architecture; the
        # clusterer only sees flat vectors and cannot work this out itself.
        if getattr(clusterer, "scope", None) == "head":
            from models import head_indices

            idx = head_indices(build_torch_model(arch, input_shape, num_classes))
            clusterer.head_index = idx
            # The attack needs the same indices, so its stealth measurement is
            # taken on the parameters the server actually decides on. Measured
            # on the full vector instead, a blend that rewrites the head
            # entirely still reads 0.9998 and looks like no movement at all.
            if hasattr(self.attack, "head_index"):
                self.attack.head_index = idx

        self._membership: Dict[int, str] = {}
        # Populated by run_round, so cluster_test_sets works whether the caller
        # drives fit() or steps run_round() itself. Set only in fit() before,
        # which broke every caller that drove its own round loop.
        self._client_labels: List = []
        self.history: List[dict] = []

    # -- placement ---------------------------------------------------------- #
    def _home_model(self, client_id: int) -> Tuple[str, np.ndarray]:
        """Which cluster model this client warm-starts from this round."""
        cid = self._membership.get(client_id)
        if cid is None or cid not in self.cluster_models:
            cid = next(iter(self.cluster_models))
        return cid, self.cluster_models[cid]

    def _ifca_choice(self, client_id: int, X, y) -> int:
        """Which cluster model gives this client the lowest local loss.

        The client computes this, not the server: it needs the client's private
        data. That is what makes IFCA's assignment a DECLARED quantity even
        though it is derived rather than invented, and it is why an attacker can
        simply report a different index.
        """
        from client import evaluate_vec

        losses = []
        for cid in sorted(self.cluster_models):
            loss, _ = evaluate_vec(self.cluster_models[cid], self.arch,
                                   self.input_shape, self.num_classes, X, y)
            losses.append((loss, cid))
        best = min(losses)[1]
        return int(str(best).rsplit("_", 1)[-1])

    # -- one round ---------------------------------------------------------- #
    def run_round(self, client_data, round_num: int = 0, epochs: int = 2,
                  batch_size: int = 32, verbose: bool = False):
        from aggregation import aggregate_flat, learnable_mask, stack
        from client import train_client_torch
        from models import build_torch_model

        self._client_labels = [y for _, y in client_data]

        updates = []
        for client_id, (X, y) in enumerate(client_data):
            home_cid, global_vec = self._home_model(client_id)

            # The attacker may resample its data toward the distribution it is
            # imitating. Honest clients get their real data back unchanged.
            X_train, y_train = self.attack.training_data(client_id, X, y)

            update = train_client_torch(
                client_id, X_train, y_train, self.arch, global_vec,
                self.input_shape, self.num_classes, epochs=epochs,
                batch_size=batch_size, seed=self.seed * 1000 + round_num * 100 + client_id)

            # The delta is what the clusterer sees, and it is measured against
            # the model this client actually warm-started from.
            update.metadata["delta_vec"] = update.vec - global_vec
            update.metadata["home_cluster"] = home_cid
            update.metadata["is_attacker"] = self.attack.targets(client_id)

            if getattr(self.clusterer, "KEY", None) == "cluster_choice":
                update.metadata["cluster_choice"] = self._ifca_choice(client_id, X, y)

            updates.append(update)

        # The attack shapes the update AFTER training and after the honest delta
        # exists, because steering is defined relative to the honest direction.
        updates = [self.attack.shape_update(u, self._context(updates))
                   for u in updates]

        clusters = self.clusterer.cluster(updates)

        self._membership = {}
        for cid, members in clusters.items():
            for u in members:
                self._membership[u.client_id] = cid

        # Aggregate within each cluster.
        selected: Dict[int, bool] = {u.client_id: True for u in updates}
        new_models = {}
        for cid, members in clusters.items():
            M = stack([u.vec for u in members])
            agg, chosen = aggregate_flat(self.aggregator, M, f=self.byzantine_f)
            if chosen is not None:
                keep = {members[i].client_id for i in chosen}
                for u in members:
                    selected[u.client_id] = u.client_id in keep
            new_models[cid] = agg
        self.cluster_models = new_models

        return clusters, updates, selected

    def _context(self, updates) -> dict:
        """What an attacker is allowed to see: the other clients' submissions.

        A real attacker cannot see these. It is passed so that the ALIE and
        Min-Max baselines, which are defined in terms of the honest population's
        mean and variance, can be computed at all; those are omniscient by
        construction and are labelled as upper bounds rather than as realistic
        attacks. The steering attacks do not use it.
        """
        return {"updates": updates, "cluster_models": self.cluster_models,
                "arch": self.arch, "input_shape": self.input_shape,
                "num_classes": self.num_classes}

    # -- evaluation --------------------------------------------------------- #
    def evaluate(self, X_test, y_test, verbose: bool = False,
                 cluster_test=None):
        """Accuracy and loss per cluster.

        `cluster_test` maps a cluster identifier to its OWN `(X, y)` test set.
        Pass it whenever the clusters are genuinely specialised, which under
        `partition="clustered"` they are.

        WHY THIS MATTERS MORE THAN IT LOOKS. Scoring every cluster model on one
        global test set penalises correct specialisation: a cluster that has
        properly learned half the label space scores badly on a test set
        covering all of it. Measured on 12 clients in 2 planted groups, a run
        that split at round 2 plateaued at 0.627 global accuracy while a run
        that stayed unified until round 7 reached 0.805, and the difference is
        almost entirely this artefact rather than any real loss of capability.

        For the attack experiments this is not a cosmetic issue. The payload
        metric is the drop in the victim cluster's accuracy. Under global-test
        scoring, an attack that merely DELAYS a split registers as a large
        accuracy gain, and an attack that genuinely damages the victim is
        indistinguishable from healthy specialisation. Both directions of error
        are invisible in the number itself.
        """
        from client import evaluate_vec

        accuracy, loss = {}, {}
        for cid, vec in self.cluster_models.items():
            Xc, yc = (cluster_test or {}).get(cid, (X_test, y_test))
            l, a = evaluate_vec(vec, self.arch, self.input_shape,
                                self.num_classes, Xc, yc)
            accuracy[cid], loss[cid] = float(a), float(l)
            if verbose:
                scope = "own" if (cluster_test or {}).get(cid) else "global"
                print(f"  {cid}: accuracy={a:.4f} loss={l:.4f} ({scope} test set)")
        return accuracy, loss

    def concept_test_sets(self, group_test: dict, client_groups) -> Dict[str, tuple]:
        """Per-cluster test sets chosen by the MAJORITY CONCEPT of its members.

        For concept shift (label permutation, rotation), where every group shares
        one label marginal so there is nothing to subset by class.

        Majority rather than unanimity, because once an attacker infiltrates, a
        cluster contains a member from another concept. Grading the cluster by
        its majority is what makes the victim's accuracy the right payload
        metric: it asks how well the cluster serves the clients it is FOR, not
        how well it serves the intruder.
        """
        import numpy as np

        out = {}
        for cid, members in self._membership_groups().items():
            votes = {}
            for m in members:
                g = int(client_groups[m])
                votes[g] = votes.get(g, 0) + 1
            majority = max(votes, key=votes.get) if votes else 0
            out[cid] = group_test[majority]
        return out

    def cluster_test_sets(self, X_test, y_test, groups=None):
        """Build a per-cluster test set from the CURRENT membership.

        Each cluster's test set is the subset of the global test set whose
        labels the cluster's members actually hold, weighted to match their
        pooled label histogram. Rebuilt every round, because membership changes
        and a test set pinned to round 1's grouping would quietly measure the
        wrong clusters from round 2 onwards.

        FOR LABEL-DISTRIBUTION SHIFT ONLY. It selects a cluster's test rows by
        which classes its members hold, so under concept shift (label
        permutation, rotation), where every group holds every class, it returns
        the same global set for every cluster and grades a correctly-permuted
        cluster against another group's labels. Use `concept_test_sets` there;
        `fit` picks it automatically when given `group_test` and `client_groups`.
        """
        import numpy as np

        from data import label_histogram

        y_test = np.asarray(y_test)
        out = {}
        for cid, members in self._membership_groups().items():
            hist = np.mean([label_histogram(np.asarray(self._client_labels[m]),
                                            self.num_classes) for m in members], axis=0)
            # Keep test rows whose class the cluster actually holds. A class the
            # cluster never sees is not part of the task it was trained for, and
            # scoring it there measures the partition rather than the model.
            keep = np.isin(y_test, np.where(hist > 1e-9)[0])
            out[cid] = (X_test[keep], y_test[keep]) if keep.any() else (X_test, y_test)
        return out

    @staticmethod
    def _majority_group(members, client_groups) -> int:
        """The planted group most of `members` belong to.

        Majority rather than unanimity, because once an attacker infiltrates, a
        cluster contains a member from another group. The cluster is still
        serving the group that owns it, and that is what its accuracy should be
        attributed to.
        """
        votes: Dict[int, int] = {}
        for m in members:
            g = int(client_groups[m.client_id if hasattr(m, "client_id") else m])
            votes[g] = votes.get(g, 0) + 1
        return max(votes, key=votes.get) if votes else 0

    def _membership_groups(self) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = {}
        for client_id, cid in self._membership.items():
            groups.setdefault(cid, []).append(client_id)
        return groups

    def fit(self, client_data, X_test, y_test, rounds: int = 3, epochs: int = 2,
            batch_size: int = 32, verbose: bool = False,
            keep_deltas: Optional[bool] = None,
            group_test: Optional[dict] = None, client_groups=None):
        """Run `rounds` federated rounds and return the per-round history.

        Each history entry records the membership SETS, not just the cluster
        names: identifiers are reassigned every round, and under recursive
        bipartition the cluster count changes too, so anything tracking a group
        across rounds has to track who is in it.

        `keep_deltas` retains the full (n_clients, n_params) delta matrix per
        round, which `analysis.signal_over_rounds` needs to correlate the
        server's view against ground truth. It costs
        `n_clients * n_params * 8` bytes per round: about 1.4 MB per round for
        the 14.7k-parameter Multi-Layer Perceptron (MLP), but roughly 160 MB per
        round for the 1.66M-parameter Convolutional Neural Network (CNN). On a
        machine that routinely has 0.3 GB free that is the difference between a
        run and an Out Of Memory (OOM) kill, so it auto-disables above the
        threshold below and says so rather than dying silently.
        """
        from aggregation import cosine, cosine_matrix, norms, stack

        n_params = len(next(iter(self.cluster_models.values())))
        if keep_deltas is None:
            budget = len(client_data) * n_params * 8 * rounds
            keep_deltas = budget < 200_000_000
            if not keep_deltas:
                print(f"[server] not retaining deltas: {budget/1e6:.0f} MB over "
                      f"{rounds} rounds would risk an OOM kill on this machine. "
                      f"Pass keep_deltas=True to override, or analyse a smaller "
                      f"architecture.")

        # Cached so `cluster_test_sets` can rebuild per-cluster test sets from
        # membership without the caller threading the labels through.
        self._client_labels = [y for _, y in client_data]

        self.history = []
        for r in range(rounds):
            clusters, updates, selected = self.run_round(
                client_data, round_num=r, epochs=epochs, batch_size=batch_size,
                verbose=verbose)
            if group_test is not None and client_groups is not None:
                per_cluster = self.concept_test_sets(group_test, client_groups)
            else:
                per_cluster = self.cluster_test_sets(X_test, y_test)
            accuracy, loss = self.evaluate(X_test, y_test, verbose=verbose,
                                           cluster_test=per_cluster)
            global_accuracy, _ = self.evaluate(X_test, y_test)

            D = stack([u.metadata["delta_vec"] for u in updates])
            S = cosine_matrix(D)
            client_ids = [u.client_id for u in updates]

            entry = {
                "round": r + 1,
                "membership": {cid: [u.client_id for u in members]
                               for cid, members in clusters.items()},
                # `metrics` is scored on each cluster's OWN distribution, which
                # is the number the payload metric uses. `global_metrics` is the
                # same models on the full test set, kept alongside because the
                # two diverge sharply once clusters specialise and a reader
                # comparing against the pre-clustering literature will expect
                # the global one.
                "metrics": accuracy,
                "global_metrics": global_accuracy,
                # Accuracy indexed by PLANTED group rather than by cluster name,
                # so a payload has a referent that does not move when the
                # attacker changes cluster. The earlier "mean of non-target
                # clusters" column compared different things at different attack
                # strengths and produced an unexplainable 0.207.
                "group_metrics": (
                    {int(g): accuracy.get(cid, float("nan"))
                     for cid, members in clusters.items()
                     for g in [self._majority_group(members, client_groups)]}
                    if client_groups is not None else None),
                "losses": loss,
                "client_ids": client_ids,
                "cosine": S,
                "deltas": D if keep_deltas else None,
                "delta_norms": norms(D),
                "selected": dict(selected),
                "split_trace": list(getattr(self.clusterer, "last_trace", [])),
                "client_log": [
                    {"round": r + 1, "client_id": u.client_id,
                     "cluster": self._membership.get(u.client_id),
                     "home_cluster": u.metadata.get("home_cluster"),
                     "is_attacker": bool(u.metadata.get("is_attacker")),
                     "n_samples": u.metadata.get("n_samples"),
                     "train_loss": u.metadata.get("train_loss"),
                     "train_accuracy": u.metadata.get("train_accuracy"),
                     "delta_norm": float(np.linalg.norm(u.metadata["delta_vec"])),
                     "selected_by_aggregator": bool(selected.get(u.client_id, True)),
                     "cluster_choice": u.metadata.get("cluster_choice")}
                    for u in updates
                ],
            }
            self.history.append(entry)
            if verbose:
                print(f"Round {r + 1}: {entry['membership']}")
        return self.history
