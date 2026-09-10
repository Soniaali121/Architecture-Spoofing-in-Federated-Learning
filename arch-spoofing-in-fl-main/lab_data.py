"""Dataset bundles for the fingerprinting experiments.

One `Bundle` is a dataset already split three ways: per-client partitions, a
held-out test set, and a server-held probe set. The probe set is what every
behavioural fingerprint is measured on, and keeping its construction here (once)
is what stops a caller from accidentally probing a client on data it trained on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from data import DEFAULT_CSV, label_histogram, partition_dirichlet, partition_iid

# TensorFlow is imported lazily, inside the Keras loaders that need it. It used
# to be at module scope, which meant the torch pipeline paid a multi-second, few
# hundred MB import to call `partition_label_hists`. Holding two deep-learning
# runtimes in one process on a machine with 0.3 GB free is how a run becomes a
# silent OOM kill rather than an error.


class Bundle:
    """One dataset, already split into clients / test / server reference."""

    def __init__(self, name, client_data, X_test, y_test, X_probe, y_probe,
                 num_classes, input_shape):
        self.name = name
        self.client_data = client_data
        self.X_test, self.y_test = X_test, y_test
        self.X_probe, self.y_probe = X_probe, y_probe
        self.num_classes = num_classes
        self.input_shape = tuple(int(x) for x in input_shape)

    def __repr__(self):
        sizes = [len(y) for _, y in self.client_data]
        return (f"<Bundle {self.name} shape={self.input_shape} "
                f"classes={self.num_classes} clients={sizes} "
                f"test={len(self.y_test)} probe={len(self.y_probe)}>")


def _normalise_images(X):
    X = X.astype(np.float32)
    return X / 255.0 if X.max() > 1.5 else X


def _load_raw(name: str, seed: int):
    """Return (X_train, y_train, X_test, y_test, num_classes, input_shape).

    KERAS LAYOUT, (H, W, C). See `_load_raw_torch` for the (C, H, W) version.
    """
    import tensorflow as tf

    name = name.lower()

    if name == "mnist":
        (Xtr, ytr), (Xte, yte) = tf.keras.datasets.mnist.load_data()
        Xtr, Xte = _normalise_images(Xtr)[..., None], _normalise_images(Xte)[..., None]
        return Xtr, ytr.astype(np.int64), Xte, yte.astype(np.int64), 10, Xtr.shape[1:]

    if name in ("cifar10", "cifar"):
        (Xtr, ytr), (Xte, yte) = tf.keras.datasets.cifar10.load_data()
        Xtr, Xte = _normalise_images(Xtr), _normalise_images(Xte)
        return (Xtr, ytr.ravel().astype(np.int64), Xte, yte.ravel().astype(np.int64),
                10, Xtr.shape[1:])

    if name == "femnist":
        return _load_femnist(seed)

    if name == "ids":
        from data import prepare_fl_data
        fl = prepare_fl_data(csv_path=DEFAULT_CSV, num_clients=2,
                             partition="iid", random_state=seed, server_ref_frac=0.0)
        return (fl.X_train.astype(np.float32), fl.y_train.astype(np.int64),
                fl.X_test.astype(np.float32), fl.y_test.astype(np.int64),
                fl.num_classes, (fl.X_train.shape[1],))

    raise ValueError(f"Unknown dataset '{name}'. "
                     f"Use one of: mnist, cifar10, femnist, ids.")


def _load_femnist(seed: int, max_writers: int = 40, min_samples: int = 40,
                  max_scan: int = 25_000):
    """FEMNIST via HuggingFace `flwrlabs/femnist` (writer_id = natural client).

    `datasets` is imported lazily so the rest of the module works without it.
    FEMNIST's writer split is the canonical *natural* non-IID partition in the FL
    literature, which is why it is worth the extra dependency: unlike synthetic
    Dirichlet skew, nobody can argue the partition was chosen to flatter us.

    Returns pooled train and test arrays, because the test and probe splits want
    pooled data, and records the writer of each TRAINING row on
    `_load_femnist.writer_of_train`. `data.partition_natural_permuted` uses that
    to make each writer one client, which is the whole reason FEMNIST is worth an
    extra dependency: the non-IID structure is real rather than imposed.

    A partition that ignores those writer labels produces clients of identical
    size drawn from one pooled distribution, which is an IID split of FEMNIST
    rather than FEMNIST's own partition.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "FEMNIST needs the `datasets` package (pip install datasets). "
            "It is installed in the Docker image; skip femnist otherwise."
        ) from exc

    rng = np.random.default_rng(seed)
    ds = load_dataset("flwrlabs/femnist", split="train", streaming=True)

    by_writer: Dict[str, list] = {}
    for scanned, ex in enumerate(ds, start=1):
        img = ex["image"]
        arr = (np.array(img.convert("L"), dtype=np.uint8) if hasattr(img, "convert")
               else np.asarray(img, dtype=np.uint8))
        if arr.ndim == 3:
            arr = arr[..., 0]
        by_writer.setdefault(str(ex["writer_id"]), []).append((arr, int(ex["character"])))
        if scanned >= max_scan:
            break

    eligible = [w for w, v in by_writer.items() if len(v) >= min_samples]
    if len(eligible) < 2:
        raise RuntimeError(f"FEMNIST: only {len(eligible)} writers with "
                           f">={min_samples} samples after {max_scan} examples.")
    rng.shuffle(eligible)
    eligible = eligible[:max_writers]

    X = np.stack([p[0] for w in eligible for p in by_writer[w]])
    y = np.array([p[1] for w in eligible for p in by_writer[w]], dtype=np.int64)
    # Which writer produced each row, in the same order as X and y.
    writer = np.array([w for w in eligible for _ in by_writer[w]])
    X = _normalise_images(X)[..., None]
    y = np.clip(y, 0, 61)

    cut = int(0.8 * len(X))
    perm = rng.permutation(len(X))
    tr, te = perm[:cut], perm[cut:]
    _load_femnist.writer_of_train = writer[tr]
    return X[tr], y[tr], X[te], y[te], 62, X.shape[1:]


def load_bundle(dataset: str, num_clients: int = 6, partition: str = "dirichlet",
                alpha: float = 0.5, seed: int = 42, max_train: Optional[int] = 8000,
                max_test: Optional[int] = 2000, probe_size: int = 1000) -> Bundle:
    """Load a dataset and split it into clients, a test set and a probe set.

    Split order mirrors `data.prepare_fl_data`: the probe set is carved out of
    the TRAINING pool and never given to a client, and is disjoint from the test
    set. Fingerprinting a client on data it might have trained on, or on the same
    data used to report accuracy, would make every downstream number meaningless.
    """
    Xtr, ytr, Xte, yte, num_classes, input_shape = _load_raw(dataset, seed)
    rng = np.random.default_rng(seed)

    if max_train is not None and len(Xtr) > max_train:
        keep = rng.choice(len(Xtr), size=max_train, replace=False)
        Xtr, ytr = Xtr[keep], ytr[keep]
    if max_test is not None and len(Xte) > max_test:
        keep = rng.choice(len(Xte), size=max_test, replace=False)
        Xte, yte = Xte[keep], yte[keep]

    probe_n = min(probe_size, len(Xtr) // 4)
    perm = rng.permutation(len(Xtr))
    probe_idx, pool_idx = perm[:probe_n], perm[probe_n:]
    X_probe, y_probe = Xtr[probe_idx], ytr[probe_idx]
    X_pool, y_pool = Xtr[pool_idx], ytr[pool_idx]

    if partition == "iid":
        client_data = partition_iid(X_pool, y_pool, num_clients=num_clients,
                                    random_state=seed)
    elif partition == "dirichlet":
        client_data = partition_dirichlet(X_pool, y_pool, num_clients=num_clients,
                                          alpha=alpha, random_state=seed)
    else:
        raise ValueError(f"Unknown partition '{partition}'")

    return Bundle(dataset, client_data, Xte, yte, X_probe, y_probe,
                  num_classes, input_shape)


def partition_label_hists(bundle: Bundle) -> np.ndarray:
    """(n_clients, num_classes) label histograms: the distribution-clustering
    feature, computable without training anything."""
    return np.vstack([label_histogram(y, bundle.num_classes)
                      for _, y in bundle.client_data])


# =========================================================================== #
# Torch bundles
# =========================================================================== #
# Same `Bundle` type, same three-way split, one difference: images come out in
# TORCH-NATIVE (C, H, W) layout rather than Keras (H, W, C), and `input_shape`
# reports (C, H, W) to match. The torch architectures in `models.py` expect
# that, so nothing in the pipeline permutes.
#
# These loaders never import TensorFlow. MNIST is read straight out of the .npz
# that Keras has already cached, because that file is a plain NumPy archive and
# re-downloading it would be gratuitous; torchvision fetches it otherwise. On a
# machine that routinely has 0.3 GB free, keeping TF out of the torch process is
# the difference between a run and a silent OOM kill.

TORCH_DATA_ROOT = Path("data") / "torchvision"
KERAS_MNIST_NPZ = Path.home() / ".keras" / "datasets" / "mnist.npz"


def _to_nchw(X: np.ndarray) -> np.ndarray:
    """(N, H, W) or (N, H, W, C) -> (N, C, H, W)."""
    if X.ndim == 3:
        return X[:, None, :, :]
    if X.ndim == 4:
        return np.transpose(X, (0, 3, 1, 2))
    raise ValueError(f"cannot interpret an image array of shape {X.shape}")


def _load_raw_torch(name: str, seed: int):
    """Return (X_train, y_train, X_test, y_test, num_classes, input_shape)."""
    name = name.lower()

    if name == "mnist":
        if KERAS_MNIST_NPZ.exists():
            with np.load(KERAS_MNIST_NPZ, allow_pickle=True) as f:
                Xtr, ytr, Xte, yte = (f["x_train"], f["y_train"],
                                      f["x_test"], f["y_test"])
        else:
            from torchvision import datasets

            TORCH_DATA_ROOT.mkdir(parents=True, exist_ok=True)
            tr = datasets.MNIST(str(TORCH_DATA_ROOT), train=True, download=True)
            te = datasets.MNIST(str(TORCH_DATA_ROOT), train=False, download=True)
            Xtr, ytr = tr.data.numpy(), tr.targets.numpy()
            Xte, yte = te.data.numpy(), te.targets.numpy()
        Xtr, Xte = _to_nchw(_normalise_images(Xtr)), _to_nchw(_normalise_images(Xte))
        return Xtr, ytr.astype(np.int64), Xte, yte.astype(np.int64), 10, Xtr.shape[1:]

    if name in ("cifar10", "cifar"):
        from torchvision import datasets

        TORCH_DATA_ROOT.mkdir(parents=True, exist_ok=True)
        tr = datasets.CIFAR10(str(TORCH_DATA_ROOT), train=True, download=True)
        te = datasets.CIFAR10(str(TORCH_DATA_ROOT), train=False, download=True)
        Xtr = _to_nchw(_normalise_images(np.asarray(tr.data)))
        Xte = _to_nchw(_normalise_images(np.asarray(te.data)))
        return (Xtr, np.asarray(tr.targets, dtype=np.int64),
                Xte, np.asarray(te.targets, dtype=np.int64), 10, Xtr.shape[1:])

    if name == "femnist":
        Xtr, ytr, Xte, yte, C, _ = _load_femnist(seed)
        # `_load_femnist` returns Keras (N, H, W, 1); restate it as (N, 1, H, W).
        Xtr, Xte = _to_nchw(Xtr[..., 0]), _to_nchw(Xte[..., 0])
        return Xtr, ytr, Xte, yte, C, Xtr.shape[1:]

    if name == "ids":
        from data import prepare_fl_data

        fl = prepare_fl_data(csv_path=DEFAULT_CSV, num_clients=2,
                             partition="iid", random_state=seed, server_ref_frac=0.0)
        return (fl.X_train.astype(np.float32), fl.y_train.astype(np.int64),
                fl.X_test.astype(np.float32), fl.y_test.astype(np.int64),
                fl.num_classes, (fl.X_train.shape[1],))

    raise ValueError(f"Unknown dataset '{name}'. Use mnist, cifar10, femnist or ids.")


def load_bundle_torch(dataset: str, num_clients: int = 12,
                      partition: str = "permuted", alpha: float = 0.2,
                      seed: int = 0, max_train: Optional[int] = 8000,
                      max_test: Optional[int] = 2000,
                      probe_size: int = 1000, n_groups: int = 2,
                      concentration: float = 20.0, overlap: float = 0.3) -> Bundle:
    """Torch-layout counterpart of `load_bundle`.

    Split ORDER is fixed and matters: the probe set is carved out of the
    TRAINING pool first, so it is never given to a client and never overlaps the
    test set. Fingerprinting a client on data it might have trained on, or on
    the rows used to report accuracy, makes every downstream number meaningless
    in a way no later assertion can detect.

    THE DEFAULTS ARE A MEASURED OPERATING POINT, NOT A GUESS
    --------------------------------------------------------
    `partition="clustered"` with `concentration=20, overlap=0.3`. Swept on the
    Intrusion Detection System (IDS) data, 12 clients, 2 planted groups, 6
    rounds, reporting the ratio of cross-group to within-group Jensen-Shannon
    (JS) divergence of the planted grouping, how many rounds the server
    recovered that grouping exactly, and accuracy on each cluster's own
    distribution:

        conc  overlap   planted ratio   exact rounds   own-dist accuracy
          20      0.0            23.4            5/6               1.000
          20      0.3             4.5            4/6               0.752
          20      0.5             2.7            0/6               0.983
          20      0.7             2.3            0/6               0.988
           5      0.0             3.4            6/6               0.992
           5      0.3             4.1            4/6               0.733
           5      0.5             1.6            0/6               0.814

    The usable window is narrow and both edges are traps.

    At overlap=0 the groups have disjoint label supports, the server recovers
    them almost perfectly, and accuracy SATURATES at 1.000. A saturated metric
    has no headroom, so every attack strength registers the same drop and the
    experiment cannot distinguish them. That exact failure has already happened
    once in this project, where infiltration read 100% for every knowledge level
    and was read as success.

    At overlap>=0.5 the separation ratio falls below about 2.7 and clustering
    fails completely, 0 rounds out of 6. There is then no cluster to infiltrate,
    so an infiltration rate measured there describes nothing.

    overlap=0.3 sits between them: the structure is real and recoverable, and
    accuracy at 0.752 has room to move in both directions.
    """
    Xtr, ytr, Xte, yte, num_classes, input_shape = _load_raw_torch(dataset, seed)
    rng = np.random.default_rng(seed)

    if max_train is not None and len(Xtr) > max_train:
        keep = rng.choice(len(Xtr), size=max_train, replace=False)
        Xtr, ytr = Xtr[keep], ytr[keep]
    if max_test is not None and len(Xte) > max_test:
        keep = rng.choice(len(Xte), size=max_test, replace=False)
        Xte, yte = Xte[keep], yte[keep]

    probe_n = min(probe_size, len(Xtr) // 4)
    perm = rng.permutation(len(Xtr))
    probe_idx, pool_idx = perm[:probe_n], perm[probe_n:]
    X_probe, y_probe = Xtr[probe_idx], ytr[probe_idx]
    X_pool, y_pool = Xtr[pool_idx], ytr[pool_idx]

    groups = None
    concept = None
    if partition == "iid":
        client_data = partition_iid(X_pool, y_pool, num_clients=num_clients,
                                    random_state=seed)
    elif partition == "dirichlet":
        client_data = partition_dirichlet(X_pool, y_pool, num_clients=num_clients,
                                          alpha=alpha, random_state=seed)
    elif partition == "permuted":
        from data import partition_concept_permuted

        client_data, groups, concept = partition_concept_permuted(
            X_pool, y_pool, num_clients=num_clients, n_groups=n_groups,
            random_state=seed)
    elif partition == "natural":
        from data import partition_natural_permuted

        shard = getattr(_load_femnist, "writer_of_train", None)
        if shard is None:
            raise ValueError(
                "partition='natural' needs per-row source labels, which only the "
                "FEMNIST loader provides (writer_id). Use dataset='femnist'.")
        # The pool was subsampled by max_train before this point, so the writer
        # labels have to be cut the same way or they will not line up.
        client_data, groups, concept = partition_natural_permuted(
            X_pool, y_pool, shard[:len(y_pool)], num_clients=num_clients,
            n_groups=n_groups, random_state=seed)
    elif partition == "rotated":
        from data import partition_concept_rotated

        client_data, groups, concept = partition_concept_rotated(
            X_pool, y_pool, num_clients=num_clients, n_groups=n_groups,
            random_state=seed)
    elif partition == "clustered":
        from data import partition_clustered

        client_data, groups = partition_clustered(
            X_pool, y_pool, num_clients=num_clients, n_groups=n_groups,
            concentration=concentration, overlap=overlap, random_state=seed)
    else:
        raise ValueError(
            f"Unknown partition '{partition}'. Use permuted (Sattler), rotated "
            f"(IFCA), iid, dirichlet, or clustered (non-standard, dormant).")

    b = Bundle(dataset.lower(), client_data, Xte, yte, X_probe, y_probe,
               num_classes, input_shape)
    # The planted group of each client, or None for the partitioners that plant
    # nothing. This is the ground truth the analysis scores against; the server
    # never sees it. `partition="dirichlet"` leaves it None ON PURPOSE, because
    # Dirichlet does not create groups and pretending it does is exactly the
    # error that makes an infiltration rate meaningless.
    b.groups = groups
    # Per-group view of the held-out set, so each cluster is scored against the
    # concept it was actually trained for. Under concept shift all groups share
    # one label marginal, so subsetting by class (what `cluster_test_sets` does
    # for label-distribution shift) cannot distinguish them and every cluster
    # would be graded against group 0's labels.
    b.group_test = ({g: concept(Xte, yte, g) for g in range(n_groups)}
                    if concept is not None else None)
    # The callable itself, for the `relabel` attack: an attacker that applies
    # the target group's concept to its own data trains the target's task, so
    # its update is a genuine member of the target cluster with no internal
    # inconsistency for a verifier to find. Oracle knowledge of the target
    # concept, so results using it are an upper bound on attacker power.
    b.concept_fn = concept
    return b


def probe_prior(bundle: Bundle) -> np.ndarray:
    """The probe set's own class distribution.

    Needed by the prior estimators. PACC and LogitGap both compare a model's
    behaviour against the distribution of the inputs it was measured on, and
    assuming a uniform probe set when it is not uniform biases every estimate in
    a direction nobody would notice.
    """
    return label_histogram(bundle.y_probe, bundle.num_classes)


def smoke(dataset: str = "ids", torch_layout: bool = True) -> int:
    """Load a bundle and assert the invariants the rest of the tree relies on.

    Run with `python lab_data.py --dataset ids`.
    """
    loader = load_bundle_torch if torch_layout else load_bundle
    b = loader(dataset, num_clients=12, partition="dirichlet", alpha=0.2, seed=0)
    print(b)

    problems = []
    for i, (X, y) in enumerate(b.client_data):
        if len(y) == 0:
            problems.append(f"client {i} has no samples")
        if not np.all(np.isfinite(X)):
            problems.append(f"client {i} has non-finite features")
        if len(y) and (y.min() < 0 or y.max() >= b.num_classes):
            problems.append(f"client {i} has labels outside [0, {b.num_classes})")
    if not (np.all(np.isfinite(b.X_test)) and np.all(np.isfinite(b.X_probe))):
        problems.append("test or probe features are non-finite")
    if len(b.y_probe) == 0:
        problems.append("probe set is empty")

    hists = partition_label_hists(b)
    print(f"probe prior:      {np.round(probe_prior(b), 3)}")
    print(f"client 0 hist:    {np.round(hists[0], 3)}")
    print(f"mean client hist: {np.round(hists.mean(axis=0), 3)}")
    print(f"client sizes:     {[len(y) for _, y in b.client_data]}")

    if problems:
        for p in problems:
            print(f"[FAIL] {p}")
        return 1
    print("[ok] all invariants hold")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Dataset smoke test.")
    ap.add_argument("--dataset", default="ids")
    ap.add_argument("--keras-layout", action="store_true",
                    help="use the Keras (H, W, C) loader instead of the torch one")
    args = ap.parse_args()
    raise SystemExit(smoke(args.dataset, torch_layout=not args.keras_layout))
