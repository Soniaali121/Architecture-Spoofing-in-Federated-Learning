"""
Shared core for the Architecture-Spoofing-in-Clustered-FL experiments.

Two runnable scripts import from this module:
  * attack_spoofing.py     - the architecture-spoofing attack (+ poison payload)
  * defense_fingerprint.py - the behavioural-fingerprinting defence

THREAT MODEL (what this project is actually about)
--------------------------------------------------
Setting: model-heterogeneous / resource-aware Clustered FL. The server keeps one
global model per *architecture family*:
    "mlp" -> low-capacity cluster,   "dnn" -> high-capacity cluster.

Each client DECLARES a family. The server TRUSTS that declaration to place the
client in a cluster and hand it that cluster's global model. This declared-
metadata channel is the untrusted surface the project targets.

  * honest client : declared_family == home_family, trains normally.
  * spoofer       : declared_family == victim family (!= home_family). It gets
                    inside a cluster it does not belong to, then delivers a
                    poisoning payload from within (sign-flip / boost / label-flip).

Spoofing is the ENTRY mechanism (the novel contribution). Poisoning is the
PAYLOAD (the damage after entry). The defence (defense_fingerprint.py) verifies
the declared family against a behavioural fingerprint inferred from the update.

Swap the dataset by replacing load_federated_data() only.
"""

import numpy as np
from dataclasses import dataclass, field

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import accuracy_score


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_global_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


# --------------------------------------------------------------------------- #
# Data  (standard public dataset: MNIST, flattened to a tabular feature vector)
# --------------------------------------------------------------------------- #
INPUT_SHAPE = 784
NUM_CLASSES = 10


def dirichlet_partition(y: np.ndarray, n_clients: int, alpha: float, seed: int):
    """Split indices of `y` across clients with a Dirichlet(alpha) label skew.

    Small alpha -> strongly non-IID (each client sees few classes).
    Large alpha -> close to IID. This gives the non-IID setting the report
    leans on, which is what makes malicious updates hard to isolate.
    """
    rng = np.random.default_rng(seed)
    n_classes = int(y.max()) + 1
    idx_by_class = [np.where(y == c)[0] for c in range(n_classes)]
    for idx in idx_by_class:
        rng.shuffle(idx)

    client_idx = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        cut_points = (np.cumsum(proportions) * len(idx_by_class[c])).astype(int)[:-1]
        for i, chunk in enumerate(np.split(idx_by_class[c], cut_points)):
            client_idx[i].extend(chunk.tolist())

    return [np.array(ci, dtype=np.int64) for ci in client_idx]


def load_federated_data(n_clients: int = 10, alpha: float = 0.5, seed: int = 42,
                        probe_size: int = 500, train_subset: int | None = 20000):
    """Return (client_data, (X_test, y_test), (X_probe, y_probe)).

    * client_data : list of (X, y) tuples, one per client (non-IID via Dirichlet).
    * probe set   : a small balanced batch the SERVER holds, used only by the
                    defence to fingerprint a client's behaviour (public data,
                    as assumed by FedMD-style knowledge-distillation FL).

    To use your IDS-IoT-2024 CSV or NSL-KDD instead, replace the body of this
    function so it returns the same three objects.
    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, INPUT_SHAPE).astype("float32") / 255.0
    X_test = X_test.reshape(-1, INPUT_SHAPE).astype("float32") / 255.0
    y_train = y_train.astype("int64")
    y_test = y_test.astype("int64")

    rng = np.random.default_rng(seed)
    if train_subset is not None and train_subset < len(X_train):
        keep = rng.choice(len(X_train), size=train_subset, replace=False)
        X_train, y_train = X_train[keep], y_train[keep]

    probe_idx = rng.choice(len(X_test), size=probe_size, replace=False)
    X_probe, y_probe = X_test[probe_idx], y_test[probe_idx]

    parts = dirichlet_partition(y_train, n_clients, alpha, seed)
    client_data = [(X_train[idx], y_train[idx]) for idx in parts]
    return client_data, (X_test, y_test), (X_probe, y_probe)


# --------------------------------------------------------------------------- #
# Architecture families (the two clusters). Kept from the reference code.
# --------------------------------------------------------------------------- #
def create_mlp_model(input_shape: int = INPUT_SHAPE, num_classes: int = NUM_CLASSES):
    """Low-capacity cluster."""
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def create_dnn_model(input_shape: int = INPUT_SHAPE, num_classes: int = NUM_CLASSES):
    """High-capacity cluster (deeper/wider than the MLP)."""
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


ARCH_FACTORIES = {"mlp": create_mlp_model, "dnn": create_dnn_model}
FAMILIES = ("mlp", "dnn")

# Declared metadata the server sees (the spoofable channel). In a real system
# these would be self-reported; here only `family` drives cluster assignment.
ARCH_META = {
    "mlp": {"family": "mlp", "declared_depth": 2, "declared_capacity": "low"},
    "dnn": {"family": "dnn", "declared_depth": 4, "declared_capacity": "high"},
}


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #
@dataclass
class Client:
    cid: int
    X: np.ndarray
    y: np.ndarray
    home_family: str                 # the cluster the client honestly belongs to
    declared_family: str             # what it tells the server (== home if honest)
    malicious: bool = False
    payload: str = "none"            # 'none' | 'sign_flip' | 'boost' | 'label_flip'
    boost: float = 10.0

    @property
    def is_spoofing(self) -> bool:
        """Ground-truth spoof label: declared cluster != true home cluster."""
        return self.declared_family != self.home_family


# --------------------------------------------------------------------------- #
# Local training + poison payloads
# --------------------------------------------------------------------------- #
def local_train(family: str, global_weights, X, y, epochs: int, batch_size: int = 32,
                malicious: bool = False, payload: str = "none", boost: float = 10.0):
    """Train the *declared* cluster's architecture locally and return its weights.

    The update always matches the declared cluster's shape (the server handed
    the client that model), so shape-checking alone cannot catch a spoofer.
    Malicious behaviour is expressed through the poison payload, not the shape.
    """
    model = ARCH_FACTORIES[family]()
    if global_weights is not None:
        model.set_weights(global_weights)

    y_local = y
    if malicious and payload == "label_flip":            # data-poisoning variant
        y_local = (y + 1) % NUM_CLASSES

    model.fit(X, y_local, epochs=epochs, batch_size=batch_size, verbose=0)
    w = [np.asarray(wi) for wi in model.get_weights()]

    if malicious and payload == "sign_flip":             # untargeted model poison
        w = [-wi for wi in w]
    elif malicious and payload == "boost":               # model-replacement (Bagdasaryan)
        gw = [np.asarray(g) for g in global_weights]
        w = [g + boost * (wi - g) for g, wi in zip(gw, w)]

    return w


def fedavg(weight_list):
    """Coordinate-wise mean of a list of same-shape weight sets (one cluster)."""
    return [np.mean([w[layer] for w in weight_list], axis=0)
            for layer in range(len(weight_list[0]))]


def evaluate(model, X, y) -> float:
    probs = model.predict(X, verbose=0)
    return accuracy_score(y, probs.argmax(axis=1))


# --------------------------------------------------------------------------- #
# Behavioural fingerprint  (the primitive the defence verifies against)
# --------------------------------------------------------------------------- #
FINGERPRINT_FEATURES = [
    "probe_confidence",   # mean top softmax on the server probe set
    "probe_accuracy",     # accuracy on the server probe set
    "probe_nll",          # negative log-likelihood on the probe set
    "probe_entropy",      # mean predictive entropy
    "delta_norm",         # L2 norm of (update - global) : magnitude of the step
    "weight_norm",        # L2 norm of the submitted weights
]


def fingerprint(declared_family: str, weights, global_weights, X_probe, y_probe) -> np.ndarray:
    """Infer a client's *behaviour* from the update it submitted.

    The server loads the client's returned weights into a model of the family
    the client DECLARED, then probes it on public data. Honest members of a
    cluster share a tight fingerprint distribution; a spoofer's malicious update
    lands outside it. Returns a vector aligned with FINGERPRINT_FEATURES.
    """
    model = ARCH_FACTORIES[declared_family]()
    model.set_weights([np.asarray(w) for w in weights])
    probs = model.predict(X_probe, verbose=0)

    eps = 1e-9
    probs = np.clip(probs, eps, 1.0)
    confidence = float(probs.max(axis=1).mean())
    accuracy = float(accuracy_score(y_probe, probs.argmax(axis=1)))
    nll = float(-np.log(probs[np.arange(len(y_probe)), y_probe]).mean())
    entropy = float(-(probs * np.log(probs)).sum(axis=1).mean())

    delta_norm = float(np.sqrt(sum(np.sum((np.asarray(w) - np.asarray(g)) ** 2)
                                   for w, g in zip(weights, global_weights))))
    weight_norm = float(np.sqrt(sum(np.sum(np.asarray(w) ** 2) for w in weights)))

    fp = np.array([confidence, accuracy, nll, entropy, delta_norm, weight_norm],
                  dtype=np.float64)
    return np.nan_to_num(fp, nan=1e9, posinf=1e9, neginf=1e9)


# --------------------------------------------------------------------------- #
# Federated simulation loop (shared by both scripts)
# --------------------------------------------------------------------------- #
@dataclass
class RoundRecord:
    cid: int
    declared: str
    fp: np.ndarray
    weights: list
    is_spoofing: bool


@dataclass
class SimResult:
    history: dict = field(default_factory=dict)   # family -> [acc per round]
    detections: list = field(default_factory=list)  # (round, cid, flagged, is_spoof)


def simulate(clients, rounds: int, epochs: int, test, probe,
             families=FAMILIES, defense=None, verbose: bool = True) -> SimResult:
    """Run Clustered FL.

    Clients are placed in the cluster they DECLARE. Each round the server may run
    an optional `defense` with interface:
        defense.filter(round_records, global_weights) -> list[int]
    returning the indices of the updates to KEEP (spoofers are quarantined).
    """
    X_test, y_test = test
    X_probe, y_probe = probe

    global_models = {f: ARCH_FACTORIES[f]() for f in families}
    global_weights = {f: [np.asarray(w) for w in global_models[f].get_weights()]
                      for f in families}

    result = SimResult(history={f: [] for f in families})

    for r in range(rounds):
        if verbose:
            print(f"\n=== Round {r + 1}/{rounds} ===")

        records = []
        for c in clients:
            fam = c.declared_family                        # server trusts the declaration
            w = local_train(fam, global_weights[fam], c.X, c.y, epochs,
                            malicious=c.malicious, payload=c.payload, boost=c.boost)
            fp = fingerprint(fam, w, global_weights[fam], X_probe, y_probe)
            records.append(RoundRecord(c.cid, fam, fp, w, c.is_spoofing))

        # --- defence decides which updates to accept -------------------------
        if defense is not None:
            accepted = set(defense.filter(records, global_weights))
        else:
            accepted = set(range(len(records)))

        for i, rec in enumerate(records):
            flagged = i not in accepted
            result.detections.append((r, rec.cid, flagged, rec.is_spoofing))
            if verbose and rec.is_spoofing:
                tag = "QUARANTINED" if flagged else "ACCEPTED"
                print(f"  spoofer client {rec.cid} (declared {rec.declared}) -> {tag}")

        # --- per-cluster FedAvg over accepted updates ------------------------
        for f in families:
            cluster = [records[i].weights for i in sorted(accepted)
                       if records[i].declared == f]
            if cluster:
                global_weights[f] = fedavg(cluster)
                global_models[f].set_weights(global_weights[f])

        for f in families:
            acc = evaluate(global_models[f], X_test, y_test)
            result.history[f].append(acc)
            if verbose:
                print(f"  {f.upper()} cluster global accuracy: {acc:.4f}")

    return result


# --------------------------------------------------------------------------- #
# Client-roster builders used by both scripts
# --------------------------------------------------------------------------- #
def build_honest_clients(client_data, assignment):
    """assignment: list of family strings, one per client (their true home)."""
    return [Client(cid=i, X=X, y=y, home_family=fam, declared_family=fam)
            for i, ((X, y), fam) in enumerate(zip(client_data, assignment))]


def build_spoofed_clients(client_data, assignment, spoofers, victim_family,
                          payload="sign_flip", boost=10.0):
    """`spoofers`: dict {client_id: True}. Each spoofer declares `victim_family`
    (a cluster that is NOT its home) and delivers `payload` from inside it."""
    clients = build_honest_clients(client_data, assignment)
    for cid in spoofers:
        c = clients[cid]
        c.declared_family = victim_family        # infiltrate the victim cluster
        c.malicious = True
        c.payload = payload
        c.boost = boost
    return clients
