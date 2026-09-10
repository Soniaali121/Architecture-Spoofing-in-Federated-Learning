"""Behavioural-fingerprinting defence for architecture-clustered FL.

The server does not trust a client's declared architecture metadata. Before
aggregation it verifies the DECLARED architecture against the behaviour of the
update the client actually submits: it loads the update into a model of the
declared architecture and probes it on server-held reference data. Honest members
of a cluster reach a consistent probe accuracy and loss; a spoofer that infiltrates
the cluster with a poisoned (or off-profile) update falls outside that band -- low
probe accuracy, high loss, or non-finite outputs (e.g. a sign-flipped BatchNorm
model) -- and is quarantined. This is the report's "fingerprinting as a
verification primitive" (Ateniese/Ganju/Suri meta-classifier paradigm),
operationalised as a robust, round-stable behaviour band.
"""

import numpy as np
from sklearn.metrics import accuracy_score

from data import label_histogram, partition_dirichlet
from lab_probe import probe
from metrics import js_divergence
from models import build_model

# Kept importable from here: this is where JS divergence lived before it moved to
# `metrics.py` to break an import cycle, and both notebooks still import it from
# this module.
_js_divergence = js_divergence


class FingerprintVerifier:
    def __init__(self, acc_margin=0.5, nll_sigma=4.0, n_shadow=4, shadow_epochs=1,
                 shadow_size=1000, probe_size=1000, seed=42):
        self.acc_margin = acc_margin          # floor = acc_margin * honest mean probe acc
        self.nll_sigma = nll_sigma            # ceiling = honest mean + nll_sigma * std
        self.n_shadow = n_shadow
        self.shadow_epochs = shadow_epochs
        self.shadow_size = shadow_size
        self.probe_size = probe_size
        self.rng = np.random.default_rng(seed)
        self.acc_floor = {}                    # declared arch -> probe-accuracy floor
        self.nll_ceiling = {}                  # declared arch -> probe-nll ceiling
        self.X_probe = None
        self.y_probe = None
        self.input_dim = None
        self.num_classes = None
        self.calibrated = False

    def calibrate(self, X_ref, y_ref, archs, input_dim, num_classes):
        """Split server-held reference data into a shadow-train pool and a probe set,
        train honest shadow models per architecture, and set a behaviour band
        (accuracy floor + loss ceiling) from their honest fingerprints."""
        self.input_dim = input_dim
        self.num_classes = num_classes

        n = len(X_ref)
        probe_n = max(1, min(self.probe_size, n // 2))
        perm = self.rng.permutation(n)
        probe_idx, shadow_idx = perm[:probe_n], perm[probe_n:]
        self.X_probe, self.y_probe = X_ref[probe_idx], y_ref[probe_idx]
        Xs, ys = X_ref[shadow_idx], y_ref[shadow_idx]

        for arch in archs:
            accs, nlls = [], []
            for _ in range(self.n_shadow):
                take = min(self.shadow_size, len(Xs))
                idx = self.rng.choice(len(Xs), size=take, replace=False)
                model = build_model(arch, input_dim, num_classes)
                model.fit(Xs[idx], ys[idx], epochs=self.shadow_epochs,
                          batch_size=32, verbose=0)
                acc, nll = self._fingerprint(arch, model.get_weights())
                accs.append(acc)
                nlls.append(nll)
            accs, nlls = np.array(accs), np.array(nlls)
            self.acc_floor[arch] = self.acc_margin * float(accs.mean()) if len(accs) else 0.0
            self.nll_ceiling[arch] = float(nlls.mean() + self.nll_sigma * (nlls.std() + 1e-6) + 2.0)

        self.calibrated = True
        print("[defence] calibrated behaviour band: "
              + ", ".join(f"{a} acc>={self.acc_floor[a]:.2f}/nll<={self.nll_ceiling[a]:.2f}"
                          for a in archs))

    def _fingerprint(self, arch, weights):
        """Return (probe_accuracy, probe_nll). Non-finite outputs (a corrupted model,
        e.g. sign-flipped BatchNorm) map to the worst possible fingerprint.

        Goes through `lab_probe.probe` for its model cache. This used to call
        `build_model` on every update, and `flag()` runs once per client per
        round, so the rebuild cost was being paid across the whole corpus run.
        """
        probs = probe(arch, weights, self.input_dim, self.num_classes, self.X_probe)
        if probs is None:
            return 0.0, 1e9
        eps = 1e-9
        probs = np.clip(probs, eps, 1.0)
        pred = np.argmax(probs, axis=1)
        acc = float(accuracy_score(self.y_probe, pred))
        nll = float(-np.log(probs[np.arange(len(self.y_probe)), self.y_probe]).mean())
        return acc, nll

    def flag(self, update):
        """True if the update's probe behaviour is inconsistent with the honest
        distribution of the architecture it declared (i.e. likely a spoofer)."""
        arch = update.metadata.get("arch")
        if arch not in self.acc_floor:
            return False
        acc, nll = self._fingerprint(arch, update.weights)
        return (acc < self.acc_floor[arch]) or (nll > self.nll_ceiling[arch])


class DistributionFingerprintVerifier:
    """Behavioural-fingerprinting defence for distribution-clustered FL.

    Here architecture is shared and not spoofable; instead a client declares a
    label histogram (`clustering.DistributionClusterer` groups on it) and
    `attacks.LabelHistSpoof` can falsify that declaration. Falsifying metadata
    doesn't change what the client actually trained on: `LabelHistSpoof` only
    rewrites `update.metadata['label_hist']`, the submitted weights still reflect
    the client's true data. Training on a label distribution skewed toward class
    k leaves a detectable signature -- the model's mean predicted probability
    over a probe set is pulled toward class k (an output-prior shift from label
    skew). This verifier compares that soft predicted-class distribution to the
    declared histogram via Jensen-Shannon divergence and flags large mismatches.

    JS divergence (not cosine similarity) is the metric that actually separates
    honest from spoofed here: on the example synthetic IDS data, honest shadow
    clients' divergence from their own true histogram maxes out around 0.26,
    while a spoofer declaring a one-hot histogram it didn't train on lands
    around 0.39 (~3.5 honest std above the honest mean) -- cosine similarity
    between a spread softmax vector and a peaked declared histogram turned out
    not to separate honest from spoofed well at this data scale (see dev notes /
    README), whereas JS divergence, built specifically to compare distributions,
    does.
    """

    def __init__(self, js_sigma=1.5, n_shadow=8, shadow_epochs=1,
                 shadow_size=1000, probe_size=1000, dirichlet_alpha=0.5, seed=42):
        self.js_sigma = js_sigma              # ceiling = honest mean + js_sigma * honest std
        self.n_shadow = n_shadow
        self.shadow_epochs = shadow_epochs
        self.shadow_size = shadow_size
        self.probe_size = probe_size
        self.dirichlet_alpha = dirichlet_alpha
        self.rng = np.random.default_rng(seed)
        self.js_ceiling = None
        self.arch = None
        self.input_dim = None
        self.num_classes = None
        self.X_probe = None
        self.calibrated = False

    def calibrate(self, X_ref, y_ref, archs, input_dim, num_classes):
        """`archs` is accepted only for interface parity with FingerprintVerifier
        (so FederatedServer.fit() can calibrate either defence the same way); in
        distribution mode every client shares one architecture, so only the first
        is used. Trains honest shadow clients under Dirichlet-skewed label splits
        (mirroring how real clients partition data) and sets a divergence ceiling
        from how far their predicted-class distribution strays from their own
        (honestly declared) label histogram."""
        self.arch = archs[0] if archs else "mlp"
        self.input_dim = input_dim
        self.num_classes = num_classes

        n = len(X_ref)
        probe_n = max(1, min(self.probe_size, n // 2))
        perm = self.rng.permutation(n)
        probe_idx, shadow_idx = perm[:probe_n], perm[probe_n:]
        self.X_probe = X_ref[probe_idx]
        Xs, ys = X_ref[shadow_idx], y_ref[shadow_idx]

        shadow_clients = partition_dirichlet(
            Xs, ys, num_clients=self.n_shadow, alpha=self.dirichlet_alpha,
            random_state=int(self.rng.integers(0, 1_000_000)),
        )

        divs = []
        for X_c, y_c in shadow_clients:
            take = min(self.shadow_size, len(X_c))
            if take < 10:
                continue
            idx = self.rng.choice(len(X_c), size=take, replace=False)
            declared_hist = label_histogram(y_c[idx], num_classes)  # honest: declared == actual
            model = build_model(self.arch, input_dim, num_classes)
            model.fit(X_c[idx], y_c[idx], epochs=self.shadow_epochs,
                      batch_size=32, verbose=0)
            pred_dist = self._predicted_distribution(model.get_weights())
            if pred_dist is not None:
                divs.append(_js_divergence(pred_dist, declared_hist))

        if divs:
            divs = np.array(divs)
            self.js_ceiling = float(divs.mean() + self.js_sigma * divs.std())
        else:
            self.js_ceiling = float("inf")
        self.calibrated = True
        print(f"[defence] distribution verifier calibrated: "
              f"JS-divergence ceiling={self.js_ceiling:.3f} ({len(divs)} shadow clients, "
              f"honest mean={np.mean(divs) if len(divs) else 0:.3f} "
              f"std={np.std(divs) if len(divs) else 0:.3f})")

    def _predicted_distribution(self, weights):
        """Mean predicted probability per class over the probe set (soft), not a
        hard argmax-vote histogram -- empirically less noisy across honest shadow
        clients at small data scale. Returns None for non-finite output (e.g. a
        sign-flipped BatchNorm model) rather than letting NaN silently propagate
        -- NaN comparisons are always False in Python, so an unguarded NaN
        divergence would never exceed any ceiling and the corrupted update would
        pass unflagged. Uses the shared probe-model cache; see
        `FingerprintVerifier._fingerprint` for why that matters."""
        probs = probe(self.arch, weights, self.input_dim, self.num_classes,
                      self.X_probe)
        return None if probs is None else probs.mean(axis=0)

    def flag(self, update):
        """True if the class distribution the update actually behaves like
        diverges too far from the label histogram it declared (i.e. likely a
        LabelHistSpoof)."""
        declared_hist = update.metadata.get("label_hist")
        if declared_hist is None or self.js_ceiling is None:
            return False
        pred_dist = self._predicted_distribution(update.weights)
        if pred_dist is None:
            return True  # non-finite output is itself maximally suspicious
        return js_divergence(pred_dist, declared_hist) > self.js_ceiling


# =========================================================================== #
# Fingerprint-space verifiers
# =========================================================================== #
# Harvested from `legacy/defense_fingerprint.py`, which was the only place these
# existed. Both verifiers above decide with a hand-set THRESHOLD on one scalar
# (a probe-accuracy floor, an NLL ceiling, a JS ceiling). The two below decide
# in the full fingerprint space instead, and that difference is the point:
#
#   ArchitectureMetaClassifier  supervised. Learns the map from behaviour to
#                               architecture, then flags an update whose
#                               behaviour names a different architecture than
#                               the one it declared. This is the
#                               Ateniese / Ganju / Suri meta-classifier
#                               paradigm the report cites and, until now, had
#                               no implementation of anywhere in the live tree.
#
#   NoveltyVerifier             unsupervised. Fits a one-class model per
#                               architecture over honest fingerprints and flags
#                               anything outside it. Needs no attack examples,
#                               which is the realistic deployment assumption.
#
# NEITHER OF THESE IS A RESULT ON ITS OWN. A detection rate means nothing
# without a false-alarm rate on honest clients, measured leave-one-seed-out.
# Score them with `run_context.detection_table`, which returns both or raises.

class ArchitectureMetaClassifier:
    """Infer an update's architecture from how it BEHAVES, then check the claim.

    The server holds no ground truth about a client's architecture, only the
    client's declaration and the update itself. This fits a classifier from
    honest shadow fingerprints to architecture names, so at verification time it
    can ask what the update behaves like and compare that against what was
    declared.

    THE SHAPE-FEATURE TRAP. Fingerprints carry shape-dependent features
    (`n_params`, `delta_norm`, ...). Including them makes this task trivial AND
    meaningless: the server already knows the declared architecture's parameter
    count, and an ADAPTIVE spoofer's shape matches its declaration by
    construction, because it trained the architecture it declared. A
    meta-classifier fed `n_params` scores near-perfectly on naive spoofers and
    learns nothing that catches the attack that matters. The default feature set
    is therefore the shape-agnostic one, and `include_shape=True` exists only to
    demonstrate the leak.
    """

    def __init__(self, features=None, n_estimators: int = 200, seed: int = 42,
                 include_shape: bool = False):
        from lab_probe import SHAPE_AGNOSTIC_FEATURES, FEATURE_COLUMNS

        self.features = list(features) if features is not None else list(
            FEATURE_COLUMNS if include_shape else SHAPE_AGNOSTIC_FEATURES)
        self.n_estimators = n_estimators
        self.seed = seed
        self.include_shape = include_shape
        self.scaler = None
        self.clf = None
        self.classes_ = None

    def _matrix(self, fingerprints):
        """list[dict] or 2-D array -> (n, n_features) with NaNs filled.

        A dead model's fingerprint is all-NaN by design (see
        `lab_probe.extract_fingerprint`), and sklearn will not fit through that.
        NaNs become 0 AFTER scaling, which puts a dead update at the honest
        mean on every feature rather than at an arbitrary extreme. That is the
        conservative choice: it makes the meta-classifier LESS likely to flag a
        wrecked model, so any detection it does report is not just detecting
        NaN. Distinguishing "we detected a spoof" from "we detected a NaN" is
        the whole point of DEFENSE_NOTES.md section 4.1.
        """
        if isinstance(fingerprints, np.ndarray) and fingerprints.ndim == 2:
            X = np.asarray(fingerprints, dtype=np.float64)
        else:
            X = np.array([[float(fp.get(k, np.nan)) for k in self.features]
                          for fp in fingerprints], dtype=np.float64)
        return X

    def fit(self, fingerprints, archs):
        """Fit on honest shadow fingerprints and the architectures that made them."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        X = self._matrix(fingerprints)
        y = np.asarray(archs)
        if len(X) != len(y):
            raise ValueError(f"{len(X)} fingerprints against {len(y)} labels")
        if len(np.unique(y)) < 2:
            raise ValueError(
                "a meta-classifier needs at least two architectures to separate; "
                f"got only {np.unique(y).tolist()}")

        self.scaler = StandardScaler()
        Xs = np.nan_to_num(self.scaler.fit_transform(np.nan_to_num(X, nan=0.0)),
                           nan=0.0, posinf=0.0, neginf=0.0)
        self.clf = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.seed).fit(Xs, y)
        self.classes_ = list(self.clf.classes_)
        print(f"[defence] meta-classifier fitted on {len(X)} honest fingerprints "
              f"over {self.classes_} using {len(self.features)} "
              f"{'shape-inclusive' if self.include_shape else 'shape-agnostic'} "
              f"features")
        return self

    def _transform(self, fingerprints):
        X = np.nan_to_num(self._matrix(fingerprints), nan=0.0)
        return np.nan_to_num(self.scaler.transform(X), nan=0.0, posinf=0.0,
                             neginf=0.0)

    def inferred_arch(self, fingerprint):
        """Which architecture does this update behave like?"""
        if self.clf is None:
            raise RuntimeError("fit the meta-classifier before using it")
        return self.clf.predict(self._transform([fingerprint]))[0]

    def declared_probability(self, fingerprint, declared: str) -> float:
        """How much probability the classifier puts on the DECLARED architecture.

        A soft score is more useful than the hard argmax for a threshold sweep,
        and it is what an ROC over this defence needs.
        """
        if self.clf is None:
            raise RuntimeError("fit the meta-classifier before using it")
        if declared not in self.classes_:
            return 0.0
        proba = self.clf.predict_proba(self._transform([fingerprint]))[0]
        return float(proba[self.classes_.index(declared)])

    def flag(self, fingerprint, declared: str, threshold: float = None) -> bool:
        """True if the update does not behave like the architecture it declared.

        With `threshold=None` this is the hard test, inferred != declared. With
        a threshold it flags when the classifier's confidence in the declaration
        falls below it, which is the tunable version.
        """
        if threshold is None:
            return str(self.inferred_arch(fingerprint)) != str(declared)
        return self.declared_probability(fingerprint, declared) < threshold


class NoveltyVerifier:
    """One-class novelty detection per architecture, over honest fingerprints.

    The realistic deployment assumption: the server can observe honest clients
    but has no labelled attacks to train against. An IsolationForest per
    declared architecture learns the honest behaviour cloud, and anything
    outside it is flagged.

    `contamination` is the fraction of the honest calibration set the forest is
    told to treat as outliers, and it sets the false-alarm rate almost directly.
    It is therefore the knob to sweep, not a constant to tune once and forget:
    reporting a detection rate at one contamination value without the matching
    false-alarm rate is how this project previously convinced itself a defence
    worked.
    """

    def __init__(self, contamination: float = 0.1, seed: int = 42, features=None):
        from lab_probe import SHAPE_AGNOSTIC_FEATURES

        self.contamination = contamination
        self.seed = seed
        self.features = list(features) if features is not None else list(
            SHAPE_AGNOSTIC_FEATURES)
        self.scaler = None
        self.forests = {}

    def fit(self, fingerprints, archs):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        meta = ArchitectureMetaClassifier(features=self.features, seed=self.seed)
        X = np.nan_to_num(meta._matrix(fingerprints), nan=0.0)
        archs = np.asarray(archs)

        self.scaler = StandardScaler().fit(X)
        Xs = np.nan_to_num(self.scaler.transform(X), nan=0.0, posinf=0.0, neginf=0.0)
        self._meta = meta

        for arch in np.unique(archs):
            mask = (archs == arch)
            if mask.sum() < 2:
                print(f"[defence] only {int(mask.sum())} honest fingerprint(s) for "
                      f"'{arch}', skipping its novelty detector")
                continue
            self.forests[str(arch)] = IsolationForest(
                contamination=self.contamination,
                random_state=self.seed).fit(Xs[mask])
        print(f"[defence] novelty detectors fitted for {sorted(self.forests)} "
              f"at contamination={self.contamination}")
        return self

    def flag(self, fingerprint, declared: str) -> bool:
        """True if the update falls outside the declared architecture's honest cloud.

        An architecture with no fitted detector returns False rather than True:
        a verifier that cannot judge must abstain, not accuse. Flagging on
        absence of evidence would make the false-alarm rate depend on how many
        shadow clients happened to be available, which is not a property of the
        defence.
        """
        forest = self.forests.get(str(declared))
        if forest is None or self.scaler is None:
            return False
        X = np.nan_to_num(self._meta._matrix([fingerprint]), nan=0.0)
        Xs = np.nan_to_num(self.scaler.transform(X), nan=0.0, posinf=0.0, neginf=0.0)
        return bool(forest.predict(Xs)[0] == -1)
