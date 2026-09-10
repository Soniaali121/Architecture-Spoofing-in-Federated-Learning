"""
DEFENCE (MNIST standalone): behavioural fingerprinting / declared-metadata verification.

The server does not trust the declared architecture family. Before aggregation it
fingerprints each submitted update (its behaviour on a public probe set) and checks
it against the honest distribution of the cluster the client DECLARED.

Primary decision (robust, round-stable): an update whose probe-set behaviour is
inconsistent with a *functional* member of the declared cluster is quarantined.
Poisoning payloads (sign-flip / boost / label-flip) collapse probe accuracy and
inflate the loss, so they fall outside the honest cluster's behaviour band while
honest non-IID clients stay inside it.

Diagnostics also fitted (the Ateniese/Ganju/Suri meta-classifier paradigm used as a
verification primitive, and the IsolationForest the reference code imported but never
used): an inferred-family meta-classifier and a per-family novelty detector.

Run:  python defense_fingerprint.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix)

import fl_common as fl

# --------------------------------------------------------------------------- #
# Configuration (kept in step with attack_spoofing.py)
# --------------------------------------------------------------------------- #
SEED = 42
N_CLIENTS = 10
ALPHA = 0.5
ROUNDS = 5
EPOCHS = 2
HOME_ASSIGNMENT = ["mlp"] * 5 + ["dnn"] * 5
SPOOFERS = [0, 1]
VICTIM_FAMILY = "dnn"
PAYLOAD = "sign_flip"

N_SHADOW_PER_FAMILY = 6        # honest shadow clients per family for calibration
SHADOW_EPOCHS = 2
CONTAMINATION = 0.1            # IsolationForest expected honest-outlier fraction
ACC_MARGIN = 0.5              # accuracy floor = ACC_MARGIN * honest mean probe acc
NLL_SIGMA = 4.0              # nll ceiling = honest mean + NLL_SIGMA * honest std


class FingerprintDefense:
    """Server-side verifier. Interface used by fl.simulate: filter(records, gw)."""

    ACC_IDX = fl.FINGERPRINT_FEATURES.index("probe_accuracy")
    NLL_IDX = fl.FINGERPRINT_FEATURES.index("probe_nll")

    def __init__(self, probe, alpha=ALPHA, per_client=2000, seed=SEED):
        self.probe = probe
        self.alpha = alpha
        self.per_client = per_client
        self.seed = seed
        self.acc_floor = {}          # family -> minimum plausible probe accuracy
        self.nll_ceiling = {}        # family -> maximum plausible probe nll
        self.scaler = StandardScaler()
        self.iforest = {}            # family -> IsolationForest (diagnostic)
        self.family_clf = None       # fingerprint -> family (diagnostic)
        self._calibrate()

    # -- server-held public data (FedMD-style), disjoint from client partitions #
    def _shadow_pool(self):
        (X, y), _ = tf.keras.datasets.mnist.load_data()
        X = X.reshape(-1, fl.INPUT_SHAPE).astype("float32") / 255.0
        y = y.astype("int64")
        return X[-12000:], y[-12000:]

    def _calibrate(self):
        X_pool, y_pool = self._shadow_pool()
        X_probe, y_probe = self.probe

        feats, fams = [], []
        acc_by_fam = {f: [] for f in fl.FAMILIES}
        nll_by_fam = {f: [] for f in fl.FAMILIES}

        for family in fl.FAMILIES:
            g0 = [np.asarray(w) for w in fl.ARCH_FACTORIES[family]().get_weights()]
            # non-IID shadow clients so the honest cloud covers natural variation
            parts = fl.dirichlet_partition(y_pool, N_SHADOW_PER_FAMILY, self.alpha,
                                           seed=self.seed + hash(family) % 997)
            for idx in parts:
                idx = idx[:self.per_client]
                if len(idx) < 100:
                    continue
                w = fl.local_train(family, g0, X_pool[idx], y_pool[idx], SHADOW_EPOCHS)
                fp = fl.fingerprint(family, w, g0, X_probe, y_probe)
                feats.append(fp)
                fams.append(family)
                acc_by_fam[family].append(fp[self.ACC_IDX])
                nll_by_fam[family].append(fp[self.NLL_IDX])

        feats = np.array(feats)

        # robust one-sided behaviour band per family (the actual decision rule)
        for f in fl.FAMILIES:
            a = np.array(acc_by_fam[f])
            n = np.array(nll_by_fam[f])
            self.acc_floor[f] = max(ACC_MARGIN * a.mean(), a.mean() - 3 * a.std())
            self.nll_ceiling[f] = n.mean() + NLL_SIGMA * (n.std() + 1e-6) + 2.0

        # diagnostics only
        self.scaler.fit(feats)
        feats_s = self.scaler.transform(feats)
        for f in fl.FAMILIES:
            mask = np.array(fams) == f
            self.iforest[f] = IsolationForest(
                contamination=CONTAMINATION, random_state=SEED).fit(feats_s[mask])
        self.family_clf = RandomForestClassifier(
            n_estimators=200, random_state=SEED).fit(feats_s, fams)

        print(f"[defence] calibrated on {len(feats)} honest shadow fingerprints. "
              f"Accuracy floors: "
              + ", ".join(f"{f}>={self.acc_floor[f]:.2f}" for f in fl.FAMILIES))

    # -- runtime verification ------------------------------------------------ #
    def filter(self, records, global_weights):
        keep = []
        for i, rec in enumerate(records):
            acc = rec.fp[self.ACC_IDX]
            nll = rec.fp[self.NLL_IDX]
            spoof_flag = (acc < self.acc_floor[rec.declared]) or \
                         (nll > self.nll_ceiling[rec.declared])
            if not spoof_flag:
                keep.append(i)
        return keep

    def inferred_family(self, fp):
        """Diagnostic: which family does this update behave like?"""
        return self.family_clf.predict(self.scaler.transform(fp.reshape(1, -1)))[0]


def main():
    fl.set_global_seeds(SEED)
    client_data, test, probe = fl.load_federated_data(
        n_clients=N_CLIENTS, alpha=ALPHA, seed=SEED)

    honest = fl.build_honest_clients(client_data, HOME_ASSIGNMENT)
    base = fl.simulate(honest, ROUNDS, EPOCHS, test, probe, verbose=False)

    print("########## ATTACK WITHOUT DEFENCE ##########")
    spoofed = fl.build_spoofed_clients(
        client_data, HOME_ASSIGNMENT, SPOOFERS, VICTIM_FAMILY, payload=PAYLOAD)
    undef = fl.simulate(spoofed, ROUNDS, EPOCHS, test, probe, verbose=False)

    print("########## ATTACK WITH FINGERPRINTING DEFENCE ##########")
    defense = FingerprintDefense(probe)
    spoofed2 = fl.build_spoofed_clients(
        client_data, HOME_ASSIGNMENT, SPOOFERS, VICTIM_FAMILY, payload=PAYLOAD)
    defd = fl.simulate(spoofed2, ROUNDS, EPOCHS, test, probe, defense=defense)

    # ----- detection quality (over all client-rounds) ----------------------- #
    y_true = [int(spoof) for (_, _, _, spoof) in defd.detections]
    y_pred = [int(flag) for (_, _, flag, _) in defd.detections]
    print("\n########## DETECTION QUALITY ##########")
    print(f"Precision : {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall    : {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1        : {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print("Confusion matrix [rows=true honest/spoof, cols=pred keep/flag]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    # ----- recovery of the victim cluster ----------------------------------- #
    print("\n########## VICTIM (DNN) CLUSTER ACCURACY ##########")
    print(f"{'round':<6}{'baseline':<12}{'no defence':<14}{'with defence':<14}")
    for r in range(ROUNDS):
        print(f"{r + 1:<6}{base.history['dnn'][r]:<12.4f}"
              f"{undef.history['dnn'][r]:<14.4f}{defd.history['dnn'][r]:<14.4f}")

    print(f"\nFinal DNN accuracy  baseline={base.history['dnn'][-1]:.4f}  "
          f"no-defence={undef.history['dnn'][-1]:.4f}  "
          f"with-defence={defd.history['dnn'][-1]:.4f}")

    # ----- plot ------------------------------------------------------------- #
    rounds_axis = range(1, ROUNDS + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(rounds_axis, base.history["dnn"], "o-", label="baseline (honest)")
    plt.plot(rounds_axis, undef.history["dnn"], "x--", label="attack, no defence")
    plt.plot(rounds_axis, defd.history["dnn"], "s-", label="attack, fingerprint defence")
    plt.xlabel("Communication round")
    plt.ylabel("DNN cluster global accuracy")
    plt.title("Fingerprinting defence recovers the victim cluster")
    plt.legend()
    plt.tight_layout()
    plt.savefig("defense_result.png", dpi=130)
    print("\nSaved plot -> defense_result.png")


if __name__ == "__main__":
    main()
