"""Dataset loading and client partitioning."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


DEFAULT_CSV = "data/example_ids_iot.csv"
DEFAULT_TARGET = "Attack_Category_x"


@dataclass
class FLData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    client_data: list  # list[(X_i, y_i)]
    num_classes: int
    input_dim: int
    label_encoder: LabelEncoder
    # Server-held reference data, carved out of the training pool and never
    # given to any client. This is what server-side machinery (e.g. the
    # fingerprinting defence's shadow models and probe set) is allowed to see.
    # It is deliberately NOT the test set: calibrating on X_test and then
    # reporting accuracy on X_test would train the defence on the evaluation
    # data. Always present so client partitions are identical across the
    # baseline/attack/defence conditions, whether or not a defence is used.
    X_ref: np.ndarray = None
    y_ref: np.ndarray = None
    scaler: StandardScaler = None


def load_ids_iot(
    csv_path=DEFAULT_CSV,
    target_col=DEFAULT_TARGET,
    test_size=0.2,
    random_state=42,
    scale=True,
):
    """Load the IDS-IoT CSV into a scaled, encoded train/test split.

    Features are every column except `target_col`, restricted to numeric dtypes
    (the real IDS-IoT export carries identifier/string columns that are not
    model inputs). The target is dropped BY NAME rather than by position, so a
    dataset whose label column isn't last cannot silently leak the label into X.

    The scaler is fitted on the training split only and then applied to the test
    split, so no test-set statistics leak into training. The synthetic example
    CSV is already roughly standardised, so scaling is close to a no-op there;
    it matters for the real dataset, whose raw features span wildly different
    magnitudes.
    """
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not in {csv_path}. "
            f"Available: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}"
        )

    y = df[target_col].values
    features = df.drop(columns=[target_col])

    numeric = features.select_dtypes(include=[np.number])
    dropped = [c for c in features.columns if c not in numeric.columns]
    if dropped:
        print(f"[data] dropped {len(dropped)} non-numeric feature column(s): {dropped}")
    if numeric.shape[1] == 0:
        raise ValueError(f"No numeric feature columns found in {csv_path}")

    X = numeric.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(X)):
        n_bad = int((~np.isfinite(X)).sum())
        print(f"[data] replaced {n_bad} non-finite feature value(s) with 0.0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state,
        stratify=_stratify_or_none(y_encoded),
    )

    scaler = None
    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, num_classes, label_encoder, scaler


def _stratify_or_none(y, min_per_class=2):
    """Stratify only when every class has enough members for it to be legal."""
    counts = np.bincount(np.asarray(y, dtype=int))
    counts = counts[counts > 0]
    return y if len(counts) and counts.min() >= min_per_class else None


def partition_iid(X_train, y_train, num_clients=10, client_frac=0.2, random_state=0):
    """Each client gets an independent random subsample (matches original CFL.py)."""
    client_data = []
    holdout = 1.0 - client_frac
    for i in range(num_clients):
        X_client, _, y_client, _ = train_test_split(
            X_train, y_train, test_size=holdout, random_state=random_state + i
        )
        client_data.append((X_client, y_client))
    return client_data


def partition_dirichlet(X_train, y_train, num_clients=10, alpha=0.5, random_state=42):
    """
    Non-IID label skew via Dirichlet (for distribution-based CFL).
    Smaller alpha => more skewed label distributions across clients.
    """
    rng = np.random.default_rng(random_state)
    labels = np.unique(y_train)
    client_indices = [[] for _ in range(num_clients)]

    for label in labels:
        idx = np.where(y_train == label)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        # Ensure every client gets at least a chance; split by cumulative proportions.
        splits = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        chunks = np.split(idx, splits)
        for client_id, chunk in enumerate(chunks):
            client_indices[client_id].extend(chunk.tolist())

    client_data = []
    for idxs in client_indices:
        idxs = np.array(idxs, dtype=int)
        if len(idxs) == 0:
            # Fallback: give a tiny random sample so training doesn't break.
            idxs = rng.choice(len(y_train), size=min(32, len(y_train)), replace=False)
        client_data.append((X_train[idxs], y_train[idxs]))
    return client_data


# =========================================================================== #
# Concept shift: how the Clustered Federated Learning (CFL) literature actually
# creates cluster structure
# =========================================================================== #
# Both of the methods this project implements plant clusters by CONCEPT SHIFT,
# not by giving groups different label proportions:
#
#   Sattler, Mueller and Samek (2020)  a different random LABEL PERMUTATION per
#                                      group. Same images, relabelled.
#   Ghosh et al. (2020), IFCA          MNIST ROTATED by a different multiple of
#                                      90 degrees per group.
#
# In both, every group has the SAME label marginal distribution. The groups
# disagree about what an input MEANS, not about how often each class appears.
#
# WHY THAT MATTERS, MEASURED. Under concept shift two groups asked to label the
# same image differently produce genuinely OPPOSED output-layer gradients, so
# the cosine similarity the server clusters on carries a large, clean signal.
# Under label-proportion shift every group is solving the same problem with
# different class weights, so their gradients nearly coincide. Measured on
# `partition_clustered` this session, the within-group minus cross-group cosine
# margin was 0.0085 on the Intrusion Detection System (IDS) data and 0.0003 on
# MNIST, and Sattler's criterion never fired on MNIST at all.
#
# So the weak clustering signal we spent this session fighting was a property of
# our non-standard generator, not of the method.


def _group_assignment(num_clients, n_groups, rng):
    """Balanced, shuffled group labels. Shared by both concept-shift generators."""
    groups = np.array([i % n_groups for i in range(num_clients)], dtype=int)
    rng.shuffle(groups)
    return groups


def partition_natural_permuted(X_train, y_train, shard_of, num_clients=12,
                               n_groups=2, random_state=42):
    """NATURAL non-IID clients, with concept shift planted on top.

    `shard_of` gives the real-world source of each row (for FEMNIST, the writer
    who produced it). Each distinct source becomes one client, so the label
    heterogeneity across clients is whatever the world produced rather than
    anything we chose. A different random label permutation is then applied per
    GROUP, exactly as in `partition_concept_permuted`.

    This is the strongest available setting: an examiner can argue that a
    synthetic partition was chosen to flatter the attack, but not that a set of
    real people's handwriting was.

    The two heterogeneity sources are deliberately layered rather than merged.
    The writer split makes clients differ in HOW MUCH of each class they hold;
    the permutation makes groups differ in WHAT A LABEL MEANS. Only the second
    defines the clusters, so the first acts as realistic nuisance variation that
    the server has to see through.
    """
    rng = np.random.default_rng(random_state)
    y_train = np.asarray(y_train)
    shard_of = np.asarray(shard_of)
    num_classes = int(np.max(y_train)) + 1

    sources = list(rng.permutation(np.unique(shard_of)))
    if len(sources) < num_clients:
        raise ValueError(
            f"only {len(sources)} natural shards available but {num_clients} "
            f"clients requested; raise max_writers or lower num_clients.")
    sources = sources[:num_clients]
    n_groups = max(1, min(int(n_groups), num_clients))

    perms = [np.arange(num_classes)]
    while len(perms) < n_groups:
        cand = rng.permutation(num_classes)
        if not any(np.array_equal(cand, p) for p in perms):
            perms.append(cand)

    groups = _group_assignment(num_clients, n_groups, rng)
    client_data = []
    for i, src in enumerate(sources):
        idx = np.where(shard_of == src)[0]
        client_data.append((X_train[idx], perms[groups[i]][y_train[idx]]))

    def concept(X, y, group, from_group=None):
        y = np.asarray(y)
        if from_group is not None:
            y = np.argsort(perms[int(from_group)])[y]
        return X, perms[int(group)][y]

    return client_data, groups, concept


def partition_concept_permuted(X_train, y_train, num_clients=12, n_groups=2,
                               client_frac=0.2, random_state=42):
    """Sattler's setup: one random LABEL PERMUTATION per group.

    Returns `(client_data, group_of_client)`. Every client sees an independent
    and identically distributed (IID) sample of the data, so all groups share
    the same label marginal. What differs is the mapping from input to label:
    group A may call a 3 a 3 while group B calls it a 7.

    Group 0 deliberately keeps the IDENTITY permutation. It makes one group's
    labels the true ones, so a cluster model can be read against the real task,
    and it matches the usual presentation in which the permuted groups are
    departures from a reference.

    A permutation is drawn until it differs from every one already used. With
    `n_groups` small against `num_classes` a collision is unlikely, but a
    repeated permutation silently merges two groups into one concept, and the
    resulting "failure to separate two clusters" would be correct behaviour
    reported as a defect.
    """
    rng = np.random.default_rng(random_state)
    y_train = np.asarray(y_train)
    num_classes = int(np.max(y_train)) + 1
    n_groups = max(1, min(int(n_groups), num_clients))

    perms = [np.arange(num_classes)]
    while len(perms) < n_groups:
        cand = rng.permutation(num_classes)
        if not any(np.array_equal(cand, p) for p in perms):
            perms.append(cand)

    groups = _group_assignment(num_clients, n_groups, rng)
    base = partition_iid(X_train, y_train, num_clients=num_clients,
                         client_frac=client_frac, random_state=random_state)

    client_data = [(Xc, perms[groups[i]][np.asarray(yc)])
                   for i, (Xc, yc) in enumerate(base)]

    def concept(X, y, group, from_group=None):
        """Re-express `(X, y)` in group `group`'s concept.

        `from_group=None` treats `y` as TRUE labels, which is the evaluation
        case: a held-out set carries true labels and each cluster must be scored
        against its own permutation of them. Without this, every cluster is
        graded against group 0's labels, a correctly-permuted cluster reads as
        ~0% accurate, and the mean sits at chance.

        `from_group=g` treats `y` as already being in group g's concept and
        inverts that first. This is the ATTACK case and it is easy to get wrong:
        a client's own labels are already permuted, so applying the target's
        permutation directly composes the two into
        `perm_target[perm_home[y_true]]`, a THIRD concept belonging to neither
        group. Measured that way, an attacker fully adopting the target's task
        showed 0.0 infiltration at every strength, which looks like the attack
        failing rather than the transform being wrong.
        """
        y = np.asarray(y)
        if from_group is not None:
            y = np.argsort(perms[int(from_group)])[y]     # undo the source concept
        return X, perms[int(group)][y]

    return client_data, groups, concept


def partition_concept_rotated(X_train, y_train, num_clients=12, n_groups=2,
                              client_frac=0.2, random_state=42):
    """IFCA's setup: each group's images ROTATED by a different multiple of 90 degrees.

    Returns `(client_data, group_of_client)`. Labels are untouched, so the label
    marginal is identical across groups by construction. Group 0 is unrotated.

    Image data only. Accepts channels-first `(C, H, W)` or bare `(H, W)` and
    requires the last two axes to be equal, because a 90 degree rotation of a
    non-square image changes its shape and would not fit the model. Raises
    rather than silently reshaping: a wrong axis choice here rotates along the
    batch dimension and produces noise that still trains to a plausible-looking
    accuracy.
    """
    rng = np.random.default_rng(random_state)
    X_train = np.asarray(X_train)
    if X_train.ndim < 3:
        raise ValueError(
            f"partition_concept_rotated needs image-shaped data, got "
            f"{X_train.shape}. Use partition_concept_permuted for tabular data "
            f"such as the Intrusion Detection System (IDS) set.")
    if X_train.shape[-1] != X_train.shape[-2]:
        raise ValueError(
            f"rotation requires square images, got {X_train.shape[-2:]}; a 90 "
            f"degree turn would change the input shape the model expects.")

    n_groups = max(1, min(int(n_groups), num_clients))
    groups = _group_assignment(num_clients, n_groups, rng)
    base = partition_iid(X_train, y_train, num_clients=num_clients,
                         client_frac=client_frac, random_state=random_state)

    client_data = []
    for i, (Xc, yc) in enumerate(base):
        k = int(groups[i]) % 4          # 0, 90, 180, 270 degrees
        # Rotate the two trailing spatial axes, whatever leads them.
        client_data.append((np.rot90(np.asarray(Xc), k=k, axes=(-2, -1)).copy(), yc))

    def concept(X, y, group, from_group=None):
        """Rotate `X` into group `group`'s frame.

        `from_group=None` treats `X` as unrotated. `from_group=g` treats it as
        already in group g's frame and turns by the DIFFERENCE, so an attacker
        re-expressing its own data does not compose two rotations into a third
        orientation belonging to neither group.
        """
        k = int(group) % 4 - (int(from_group) % 4 if from_group is not None else 0)
        return np.rot90(np.asarray(X), k=k % 4, axes=(-2, -1)).copy(), y

    return client_data, groups, concept


def partition_clustered(X_train, y_train, num_clients=12, n_groups=2,
                        concentration=20.0, overlap=0.0, random_state=42):
    """Label-DISTRIBUTION shift. The partitioner behind `cfl-distribution.ipynb`.

    Gives each group a different label PROFILE, which is a different kind of
    heterogeneity from the concept shift that Sattler and the Iterative
    Federated Clustering Algorithm (IFCA) evaluate on. Use
    `partition_concept_permuted` or `partition_concept_rotated` for that one.

    Partition into groups of clients that share a label profile.

    STATUS, corrected 25 Aug 2026
    -----------------------------
    This docstring previously called the function dormant and reported a head
    margin of 0.0003 on MNIST "against a criterion that never fired at all".
    Both statements are now superseded by measurement, and the earlier numbers
    came from a run at `overlap=0.0` with untuned thresholds:

        at concentration=20, overlap=0.3, 12 clients, 5 seeds, MNIST
          separation ratio      4.51   (against a measured random-split null of 0.99)
          head AUC              1.000  (full-vector 0.885, margin 0.0002)
          head margin           0.7545
          criterion             fires at round 3 to 5 and recovers the
                                planted grouping exactly

    The full-vector margin really is near zero, and that is the BatchNorm result
    rather than a property of this partitioner: the running statistics track
    inputs, are near-identical across clients, and dominate the update's
    magnitude. Scope the similarity to the head (`SattlerCosineClusterer(
    scope="head")`) and the signal is there. Tune the thresholds on an
    attacker-free warmup with `tune_thresholds`; the published constants do not
    fire on this data.

    Returns `(client_data, group_of_client)`, where `group_of_client[i]` is the
    planted group index of client i. That array is the ground truth the analysis
    scores against, and the server never sees it.

    WHY THIS EXISTS, AND WHY `partition_dirichlet` IS NOT ENOUGH
    -----------------------------------------------------------
    Dirichlet partitioning does not create clusters. It creates N individually
    skewed clients with no group structure, and Clustered Federated Learning
    (CFL) experiments run on it are asking a server to recover structure that
    was never planted.

    Measured by exhaustive search over all 2-way splits of 12 clients, scoring
    each with the ratio of cross-group to within-group Jensen-Shannon (JS)
    divergence. On the Intrusion Detection System (IDS) data:

        alpha    best possible 2-way split    median split
        0.05                         2.41            0.99
        0.10                         2.36            0.99
        0.20                         1.78            0.99
        1.00                         1.74            0.98

    and replicated on MNIST, which is what `cfl-distribution.ipynb` Part 1 runs:

        alpha    best possible 2-way split    median split
        0.05                         1.65            0.99
        0.20                         1.43            0.99
        1.00                         1.46            0.99

    The BEST split available anywhere in Dirichlet data is barely above a median
    random split, at every concentration and on both datasets. This partitioner
    scores 4.51 at overlap=0.3 and 17.31 at overlap=0.0 on the same metric. So on
    Dirichlet data there is essentially no grouping to find, and a clusterer
    scoring 1.25 there is recovering about 70% of the little that exists rather
    than failing.

    This is why the CFL literature plants structure explicitly: Sattler uses
    label-swapped MNIST, the Iterative Federated Clustering Algorithm (IFCA)
    uses rotated MNIST. Neither evaluates on plain Dirichlet.

    PARAMETERS
        n_groups       how many planted clusters.
        concentration  how tightly clients hug their group's profile. Large
                       values make within-group clients nearly identical, which
                       raises the separation ratio; small values blur the groups
                       together. This is the knob that sets how hard the
                       clustering problem is, and it should be swept rather than
                       fixed, because "the attack works" at concentration=100 is
                       a much weaker claim than at concentration=5.
        overlap        fraction of each group's mass spread uniformly over ALL
                       classes rather than its own block. overlap=0 gives
                       disjoint supports (easy, and unrealistic); raising it
                       makes the groups share classes, which is the realistic
                       case and the one where placement is contestable.
    """
    rng = np.random.default_rng(random_state)
    y_train = np.asarray(y_train)
    classes = np.unique(y_train)
    num_classes = int(classes.max()) + 1
    n_groups = max(1, min(int(n_groups), num_clients))

    # Each group owns a contiguous block of classes. Contiguous rather than
    # interleaved so that a group's profile is readable at a glance when it is
    # printed, which matters when debugging why a split went the way it did.
    blocks = np.array_split(classes, n_groups)

    profiles = []
    for block in blocks:
        p = np.full(num_classes, overlap / num_classes, dtype=np.float64)
        if len(block):
            p[block] += (1.0 - overlap) / len(block)
        profiles.append(p / p.sum())

    group_of_client = np.array([i % n_groups for i in range(num_clients)], dtype=int)
    rng.shuffle(group_of_client)

    # Index pools per class, drawn down as clients take from them.
    pools = {c: list(rng.permutation(np.where(y_train == c)[0])) for c in range(num_classes)}
    per_client = max(1, len(y_train) // num_clients)

    client_data = []
    for i in range(num_clients):
        target = rng.dirichlet(concentration * profiles[group_of_client[i]] + 1e-3)
        counts = np.floor(target * per_client).astype(int)

        picked = []
        for c in range(num_classes):
            want = int(counts[c])
            if want <= 0:
                continue
            pool = pools[c]
            take = min(want, len(pool))
            if take:
                picked.extend(pool[:take])
                del pool[:take]
            if take < want and len(np.where(y_train == c)[0]):
                # The pool is exhausted. Sample the shortfall WITH replacement
                # from the class globally rather than silently under-filling,
                # so the achieved histogram still matches the planted profile.
                # Without this, later clients drift toward whichever classes
                # happen to be left over and the planted structure erodes in
                # client order, which is invisible in any per-client statistic.
                extra = rng.choice(np.where(y_train == c)[0], size=want - take,
                                   replace=True)
                picked.extend(extra.tolist())

        if not picked:
            picked = rng.choice(len(y_train), size=min(32, len(y_train)),
                                replace=False).tolist()
        idx = np.array(picked, dtype=int)
        rng.shuffle(idx)
        client_data.append((X_train[idx], y_train[idx]))

    return client_data, group_of_client


def label_histogram(y, num_classes):
    hist = np.bincount(y, minlength=num_classes).astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def prepare_fl_data(
    csv_path=DEFAULT_CSV,
    num_clients=10,
    partition="iid",
    dirichlet_alpha=0.5,
    test_size=0.2,
    random_state=42,
    server_ref_frac=0.1,
    scale=True,
):
    """Build client partitions plus a held-out test set and a server reference set.

    Split order matters and is: full data -> (train pool, test); train pool ->
    (server reference, client pool); client pool -> per-client partitions. The
    server reference set is carved out unconditionally so that the client data
    is byte-identical whether or not a defence is enabled -- otherwise the
    baseline and defended runs would be training on different data and the
    comparison between them would be confounded.

    Set `server_ref_frac=0.0` to disable the reference split (client data then
    covers the whole training pool, but any defence has no legitimate data to
    calibrate on).
    """
    X_train, X_test, y_train, y_test, num_classes, label_encoder, scaler = load_ids_iot(
        csv_path=csv_path, test_size=test_size, random_state=random_state, scale=scale
    )

    X_ref, y_ref = None, None
    if server_ref_frac and server_ref_frac > 0.0:
        X_client_pool, X_ref, y_client_pool, y_ref = train_test_split(
            X_train, y_train, test_size=server_ref_frac, random_state=random_state,
            stratify=_stratify_or_none(y_train),
        )
    else:
        X_client_pool, y_client_pool = X_train, y_train

    if partition == "iid":
        client_data = partition_iid(
            X_client_pool, y_client_pool, num_clients=num_clients,
            random_state=random_state,
        )
    elif partition == "dirichlet":
        client_data = partition_dirichlet(
            X_client_pool, y_client_pool, num_clients=num_clients,
            alpha=dirichlet_alpha, random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown partition '{partition}'. Use 'iid' or 'dirichlet'.")

    return FLData(
        X_train=X_client_pool,
        y_train=y_client_pool,
        X_test=X_test,
        y_test=y_test,
        client_data=client_data,
        num_classes=num_classes,
        input_dim=X_client_pool.shape[1],
        label_encoder=label_encoder,
        X_ref=X_ref,
        y_ref=y_ref,
        scaler=scaler,
    )
