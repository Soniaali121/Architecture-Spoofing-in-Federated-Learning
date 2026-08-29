"""Cluster-agnostic aggregation operators, and the flat-vector representation
every one of them works in.

Two halves.

The FIRST half is `fedavg` / `fedavg_weighted`, which operate on the Keras
representation of a model: a list of per-layer arrays.

The SECOND half is the flat-vector representation the torch pipeline uses, plus
the Byzantine-robust aggregators and the update geometry that clustering and the
steering attacks are built on. A model becomes one 1-D vector; every aggregator,
every similarity metric and every steering operation then works in that one
space. The last audit found the probe-model pattern written out five times
before anybody noticed; flattening is this build's equivalent temptation, so it
is written once, here, next to the things that consume it.

NO TORCH IMPORT
---------------
The flat helpers are duck-typed: a "state" is any mapping from name to something
array-like, and a tensor is recognised by having `.detach()`. That keeps this
module importable in a notebook that only wants to score some vectors, and keeps
it usable from both the Keras and the torch side without a second copy.
"""

from typing import Dict, List, Mapping, Sequence

import numpy as np


def fedavg(weight_lists: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Plain Federated Averaging over a list of client weight sets."""
    if not weight_lists:
        raise ValueError("fedavg requires at least one client's weights")

    n_layers = len(weight_lists[0])
    return [
        np.mean([client_weights[layer] for client_weights in weight_lists], axis=0)
        for layer in range(n_layers)
    ]


def fedavg_weighted(weight_lists: List[List[np.ndarray]], sample_counts: List[int]) -> List[np.ndarray]:
    """Sample-size weighted FedAvg."""
    if not weight_lists:
        raise ValueError("fedavg_weighted requires at least one client's weights")
    if len(weight_lists) != len(sample_counts):
        raise ValueError("sample_counts must match number of clients")

    total = float(sum(sample_counts))
    if total <= 0:
        return fedavg(weight_lists)

    n_layers = len(weight_lists[0])
    averaged = []
    for layer in range(n_layers):
        acc = None
        for weights, n in zip(weight_lists, sample_counts):
            contrib = (n / total) * np.asarray(weights[layer])
            acc = contrib if acc is None else acc + contrib
        averaged.append(acc)
    return averaged


# =========================================================================== #
# Flat-vector representation
# =========================================================================== #
# SORTED KEY ORDER IS LOAD-BEARING.
#
# A torch `state_dict()` iterates in module-registration order. That order is
# stable for one model class in one process, but it is NOT stable across a model
# built fresh versus one reconstructed from a checkpoint, nor across versions
# that reorder a module's children. Flattening under one order and unflattening
# under another gives a model with every tensor intact and every tensor in the
# wrong place: it still runs, still trains, still returns a plausible accuracy,
# and is entirely wrong.
#
# Plausible-looking wrong numbers rather than an error is the failure mode this
# project keeps hitting (see HANDOFF section 6), so the order is pinned.

def _to_array(t) -> np.ndarray:
    """Tensor or array to float64 numpy, without importing torch."""
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.asarray(t, dtype=np.float64)


def sorted_keys(state: Mapping) -> List[str]:
    """The canonical order. Every function here goes through this."""
    return sorted(state.keys())


def flatten(state: Mapping) -> np.ndarray:
    """Concatenate every tensor, in sorted key order, into one float64 vector."""
    keys = sorted_keys(state)
    if not keys:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate([_to_array(state[k]).ravel() for k in keys])


def unflatten(vec, template: Mapping) -> Dict:
    """Inverse of `flatten`. `template` supplies keys, shapes, dtypes, devices.

    Integer entries are ROUNDED, not truncated. BatchNorm's
    `num_batches_tracked` comes back as 4.999999999 after an averaging
    aggregator has touched it, and `int(4.999...)` is 4.
    """
    vec = np.asarray(vec, dtype=np.float64).ravel()
    keys = sorted_keys(template)

    expected = sum(int(np.asarray(template[k]).size if not hasattr(template[k], "numel")
                       else template[k].numel()) for k in keys)
    if vec.size != expected:
        raise ValueError(
            f"flat vector has {vec.size} entries, template needs {expected}. "
            "This normally means the vector came from a different architecture.")

    out: Dict = {}
    offset = 0
    for k in keys:
        ref = template[k]
        shape = tuple(ref.shape)
        n = int(np.prod(shape)) if shape else 1
        chunk = vec[offset:offset + n].reshape(shape)
        offset += n

        if hasattr(ref, "detach"):                       # torch tensor
            import torch

            if ref.dtype.is_floating_point:
                out[k] = torch.as_tensor(chunk, dtype=ref.dtype, device=ref.device)
            else:
                out[k] = torch.as_tensor(np.rint(chunk), dtype=ref.dtype,
                                         device=ref.device)
        else:                                            # numpy array
            ref_arr = np.asarray(ref)
            out[k] = (chunk.astype(ref_arr.dtype) if np.issubdtype(ref_arr.dtype, np.floating)
                      else np.rint(chunk).astype(ref_arr.dtype))
    return out


def learnable_mask(template: Mapping) -> np.ndarray:
    """Boolean mask, True where the flat vector holds a floating-point entry.

    Apply this before any cosine similarity, norm, median or trimmed mean.
    `num_batches_tracked` is a monotonically increasing counter that every
    client holds at the same large value. Leaving it in a cosine similarity adds
    one identical, dominant, non-informative coordinate to every client, which
    drags every pairwise similarity toward 1 and makes the clusterer look like
    it is separating clients when it is doing nothing of the kind.
    """
    keys = sorted_keys(template)
    if not keys:
        return np.zeros(0, dtype=bool)

    parts = []
    for k in keys:
        ref = template[k]
        n = int(ref.numel()) if hasattr(ref, "numel") else int(np.asarray(ref).size)
        if hasattr(ref, "dtype") and hasattr(ref.dtype, "is_floating_point"):
            is_float = bool(ref.dtype.is_floating_point)          # torch
        else:
            is_float = bool(np.issubdtype(np.asarray(ref).dtype, np.floating))
        parts.append(np.full(n, is_float, dtype=bool))
    return np.concatenate(parts)


def stack(vectors: Sequence) -> np.ndarray:
    """(n_clients, n_params) from a sequence of flat vectors.

    Raises rather than broadcasting when the lengths disagree. Two
    architectures with different parameter counts silently stacking is exactly
    the shape confusion the architecture-spoofing study is about, and it must
    not happen by accident inside the machinery that studies it.
    """
    if len(vectors) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    lengths = {np.asarray(v).size for v in vectors}
    if len(lengths) != 1:
        raise ValueError(f"cannot stack vectors of differing lengths: {sorted(lengths)}")
    return np.vstack([np.asarray(v, dtype=np.float64).ravel() for v in vectors])


# =========================================================================== #
# Update geometry
# =========================================================================== #
def cosine(a, b, eps: float = 1e-12) -> float:
    """Cosine similarity of two flat vectors. 0.0 when either is degenerate."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > eps else 0.0


def cosine_matrix(M) -> np.ndarray:
    """(n, n) pairwise cosine similarity of the rows of `M`.

    Degenerate rows get similarity 0 against everything, rather than NaN. A NaN
    here propagates into the clusterer's distance matrix and sklearn raises
    somewhere far from the cause, or silently sorts around it.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.size == 0:
        return np.zeros((0, 0))
    row_norms = np.linalg.norm(M, axis=1, keepdims=True)
    safe = np.where(row_norms == 0, 1.0, row_norms)
    U = M / safe
    S = np.clip(U @ U.T, -1.0, 1.0)
    dead = (row_norms.ravel() == 0)
    if dead.any():
        S[dead, :] = 0.0
        S[:, dead] = 0.0
    return S


def norms(M) -> np.ndarray:
    """Row-wise L2 norms of a stacked update matrix."""
    M = np.asarray(M, dtype=np.float64)
    return np.linalg.norm(np.atleast_2d(M), axis=1)


# =========================================================================== #
# Byzantine-robust aggregators
# =========================================================================== #
# The report claims that spoofing defeats Byzantine-robust defences and has no
# experimental support for it (HANDOFF section 10). These are that support.
#
# Every one of them takes a stacked (n_clients, n_params) array and returns
# either an aggregated vector or the indices it selected, so the experiment can
# record WHICH clients an aggregator kept, not just what it produced. Whether
# the attacker survives selection is the result; the aggregated vector is only
# how the damage propagates.
#
# `byzfl` (MIT) is used when installed and these serve as the cross-check;
# `test_robust_aggregators` below compares the two on random inputs.

def krum_scores(M, f: int) -> np.ndarray:
    """Krum score per client (Blanchard et al. 2017).

    For each client, the sum of the `n - f - 2` smallest squared distances to
    the other clients. Low is good: a client close to many others scores low.
    """
    M = np.asarray(M, dtype=np.float64)
    n = len(M)
    if n == 0:
        return np.zeros(0)
    # Squared euclidean distance matrix, via the expansion, which is far cheaper
    # than an explicit double loop at this parameter count.
    sq = np.sum(M ** 2, axis=1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (M @ M.T), 0.0)

    k = max(1, n - int(f) - 2)
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        others = np.delete(D[i], i)
        scores[i] = np.sort(others)[:k].sum()
    return scores


def krum(M, f: int = 1):
    """Select the single lowest-scoring client. Returns (vector, [index])."""
    M = np.asarray(M, dtype=np.float64)
    if len(M) == 0:
        raise ValueError("krum requires at least one client")
    idx = int(np.argmin(krum_scores(M, f)))
    return M[idx].copy(), [idx]


def multi_krum(M, f: int = 1, m: int = None):
    """Average the `m` lowest-scoring clients. Returns (vector, indices)."""
    M = np.asarray(M, dtype=np.float64)
    n = len(M)
    if n == 0:
        raise ValueError("multi_krum requires at least one client")
    m = int(m if m is not None else max(1, n - int(f)))
    m = min(m, n)
    order = np.argsort(krum_scores(M, f))[:m]
    return M[order].mean(axis=0), sorted(int(i) for i in order)


def coordinate_median(M):
    """Per-coordinate median. Returns (vector, None): it selects no client."""
    M = np.asarray(M, dtype=np.float64)
    if len(M) == 0:
        raise ValueError("coordinate_median requires at least one client")
    return np.median(M, axis=0), None


def trimmed_mean(M, f: int = 1):
    """Per coordinate, drop the `f` largest and `f` smallest, mean the rest."""
    M = np.asarray(M, dtype=np.float64)
    n = len(M)
    if n == 0:
        raise ValueError("trimmed_mean requires at least one client")
    f = int(f)
    if 2 * f >= n:
        # Trimming everything would leave nothing. Fall back to the median,
        # which is what trimmed mean converges to, rather than raising in the
        # middle of a corpus run.
        return np.median(M, axis=0), None
    S = np.sort(M, axis=0)
    return S[f:n - f].mean(axis=0), None


def bulyan(M, f: int = 1):
    """Multi-Krum to a shortlist, then a coordinate-wise trimmed mean on it.

    Selects `theta = n - 2f` by Multi-Krum, then per coordinate averages the
    `beta = theta - 2f` values closest to that coordinate's median. Bulyan needs
    `n >= 4f + 3`; below that it degrades to Multi-Krum rather than raising.
    """
    M = np.asarray(M, dtype=np.float64)
    n, f = len(M), int(f)
    if n == 0:
        raise ValueError("bulyan requires at least one client")

    theta = n - 2 * f
    beta = theta - 2 * f
    if n < 4 * f + 3 or theta < 1 or beta < 1:
        return multi_krum(M, f=f)

    _, selected = multi_krum(M, f=f, m=theta)
    S = M[selected]
    med = np.median(S, axis=0)
    # Per coordinate, keep the beta values nearest the median.
    order = np.argsort(np.abs(S - med), axis=0)[:beta]
    kept = np.take_along_axis(S, order, axis=0)
    return kept.mean(axis=0), sorted(int(i) for i in selected)


def foolsgold_weights(H, kappa: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    """Per-client weights FoolsGold would assign (Fung et al., RAID 2020).

    Reimplemented from the published algorithm; the authors' repository carries
    no licence, so nothing is adapted from it.

    `H` is (n_clients, n_params) of CUMULATIVE historical updates, one row per
    client. FoolsGold's premise is that clients pursuing a shared objective
    produce unusually similar update HISTORIES, so it down-weights whoever looks
    too much like somebody else.

    USED HERE AS AN OBSERVATION, NOT A DEFENCE. Nothing in this project deploys
    it. It is the sharpest available characterisation of a membership spoof's
    signature: an attacker that adopts the target group's task should, by
    construction, start resembling that group's members. Whether that
    resemblance is anomalous enough to be down-weighted is a measurement about
    the attack, and it is worth knowing in either direction.

    Returns weights in [0, 1]; low means "this client looks like a colluder".

    The steps are the paper's, and the pardoning step in the middle is the one
    that is easy to drop: without it an honest client gets punished merely for
    resembling a Sybil that is imitating it.
    """
    H = np.asarray(H, dtype=np.float64)
    n = len(H)
    if n < 2:
        return np.ones(n)

    cs = cosine_matrix(H)
    np.fill_diagonal(cs, 0.0)
    v = cs.max(axis=1)

    # Pardoning: scale down i's similarity to j when j is more suspicious than
    # i, so the imitated client is not penalised for the imitator's behaviour.
    for i in range(n):
        for j in range(n):
            if i != j and v[j] > v[i] and v[j] > eps:
                cs[i, j] *= v[i] / v[j]

    wv = np.clip(1.0 - cs.max(axis=1), 0.0, 1.0)
    top = wv.max()
    if top > eps:
        wv = wv / top
    wv = np.clip(wv, eps, 1.0 - eps)
    # Logit rescale, which sharpens the separation between kept and dropped.
    wv = kappa * (np.log(wv / (1.0 - wv)) + 0.5)
    return np.clip(wv, 0.0, 1.0)


ROBUST_AGGREGATORS = {
    "fedavg": lambda M, f=1: (np.asarray(M, dtype=np.float64).mean(axis=0), None),
    "krum": krum,
    "multikrum": multi_krum,
    "median": lambda M, f=1: coordinate_median(M),
    "trimmed": trimmed_mean,
    "bulyan": bulyan,
}


def aggregate_flat(name: str, M, f: int = 1):
    """Dispatch by name. Returns (aggregated_vector, selected_indices_or_None)."""
    if name not in ROBUST_AGGREGATORS:
        raise KeyError(f"Unknown aggregator '{name}'. "
                       f"Known: {sorted(ROBUST_AGGREGATORS)}")
    return ROBUST_AGGREGATORS[name](M, f)


# =========================================================================== #
# Self-tests
# =========================================================================== #
# Run with `python aggregation.py`. These live here rather than in a test file
# because the invariants they check are invariants OF THIS MODULE, and a reader
# deciding whether to trust `flatten` should not have to go looking.

def _selftest_flat() -> List[str]:
    fails = []

    # Round trip through a numpy "state", including an integer buffer.
    state = {
        "b.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "a.bias": np.array([0.5, -1.5], dtype=np.float32),
        "b.num_batches_tracked": np.array(7, dtype=np.int64),
    }
    vec = flatten(state)
    # Sorted order means a.bias comes first, NOT b.weight, even though b was
    # inserted first. If this assertion fails the whole tree is unsafe.
    if not np.allclose(vec[:2], [0.5, -1.5]):
        fails.append("flatten did not use sorted key order")

    back = unflatten(vec, state)
    for k in state:
        if not np.array_equal(np.asarray(back[k]), np.asarray(state[k])):
            fails.append(f"round trip changed {k}")

    # Insertion order must not matter.
    if not np.array_equal(flatten(dict(reversed(list(state.items())))), vec):
        fails.append("flatten depends on insertion order")

    # Integer buffers round rather than truncate.
    noisy = vec.copy()
    mask = learnable_mask(state)
    noisy[~mask] -= 1e-9
    if int(np.asarray(unflatten(noisy, state)["b.num_batches_tracked"])) != 7:
        fails.append("integer buffer truncated instead of rounding")

    if int((~mask).sum()) != 1:
        fails.append("learnable_mask did not exclude exactly the integer buffer")

    try:
        unflatten(np.zeros(3), state)
        fails.append("unflatten accepted a wrongly sized vector")
    except ValueError:
        pass

    try:
        stack([np.zeros(4), np.zeros(5)])
        fails.append("stack accepted mismatched lengths")
    except ValueError:
        pass

    return fails


def _selftest_geometry() -> List[str]:
    fails = []
    a = np.array([1.0, 0.0, 0.0])
    if abs(cosine(a, a) - 1.0) > 1e-12:
        fails.append("cosine(a, a) != 1")
    if abs(cosine(a, np.array([0.0, 1.0, 0.0]))) > 1e-12:
        fails.append("orthogonal cosine != 0")
    if abs(cosine(a, -a) + 1.0) > 1e-12:
        fails.append("cosine(a, -a) != -1")
    if cosine(np.zeros(3), a) != 0.0:
        fails.append("degenerate cosine is not 0")

    rng = np.random.default_rng(0)
    M = rng.normal(size=(5, 11))
    S = cosine_matrix(M)
    if not np.allclose(S, S.T) or not np.allclose(np.diag(S), 1.0):
        fails.append("cosine_matrix is not symmetric with unit diagonal")
    for i in range(5):
        for j in range(5):
            if abs(S[i, j] - cosine(M[i], M[j])) > 1e-10:
                fails.append("cosine_matrix disagrees with cosine")
                break

    dead = cosine_matrix(np.vstack([np.zeros(3), np.ones(3), np.ones(3)]))
    if not np.isfinite(dead).all() or dead[0].any():
        fails.append("cosine_matrix mishandled a zero row")
    return fails


def _selftest_aggregators() -> List[str]:
    """Hand-checkable cases, plus a cross-check against byzfl when installed."""
    fails = []

    # Four honest clients clustered near 0, one obvious outlier at 100.
    M = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [100.0, 100.0]])

    _, sel = krum(M, f=1)
    if sel[0] == 4:
        fails.append("krum selected the outlier")

    _, sel = multi_krum(M, f=1, m=4)
    if 4 in sel:
        fails.append("multi_krum kept the outlier")

    med, _ = coordinate_median(M)
    if not np.allclose(med, [0.1, 0.1]):
        fails.append(f"coordinate_median gave {med}, expected [0.1, 0.1]")

    tm, _ = trimmed_mean(M, f=1)
    if not np.allclose(tm, [0.1 / 3 + 0.0 / 3 + 0.1 / 3, 0.1]) and tm[0] > 1.0:
        fails.append(f"trimmed_mean did not exclude the outlier: {tm}")

    # Degenerate settings must degrade, not raise, mid corpus run.
    for name in ROBUST_AGGREGATORS:
        try:
            aggregate_flat(name, M[:2], f=1)
        except Exception as exc:                          # noqa: BLE001
            fails.append(f"{name} raised on 2 clients with f=1: {exc}")

    try:
        import byzfl                                       # noqa: F401
        fails.extend(_crosscheck_byzfl())
    except ImportError:
        print("[selftest] byzfl not installed, skipping the cross-check. "
              "Install it before reporting any robust-aggregation result: "
              "pip install byzfl")
    return fails


def _crosscheck_byzfl() -> List[str]:
    """Compare our aggregators against byzfl's on random inputs.

    Ours are the ones the experiment runs, because they also report WHICH
    clients were selected, which byzfl does not expose. This check exists so a
    byzfl API change, or a misreading of a paper on our side, fails loudly
    instead of producing a defensible-looking number.
    """
    import byzfl

    fails = []
    rng = np.random.default_rng(0)
    M = rng.normal(size=(7, 13))
    f = 2

    pairs = [
        ("median", getattr(byzfl, "Median", None), lambda: coordinate_median(M)[0]),
        ("trimmed", getattr(byzfl, "TrMean", None), lambda: trimmed_mean(M, f)[0]),
        ("krum", getattr(byzfl, "Krum", None), lambda: krum(M, f)[0]),
    ]
    for name, cls, ours in pairs:
        if cls is None:
            print(f"[selftest] byzfl has no aggregator matching '{name}', skipped")
            continue
        try:
            theirs = np.asarray(cls(f=f)(M) if name != "median" else cls()(M),
                                dtype=np.float64)
        except Exception as exc:                          # noqa: BLE001
            print(f"[selftest] byzfl '{name}' would not run ({exc}); "
                  f"check the byzfl API before trusting the cross-check")
            continue
        if not np.allclose(theirs, ours(), atol=1e-8):
            fails.append(f"{name} disagrees with byzfl by "
                         f"{np.abs(theirs - ours()).max():.3e}")
    return fails


if __name__ == "__main__":
    all_fails = _selftest_flat() + _selftest_geometry() + _selftest_aggregators()
    if all_fails:
        print(f"\n{len(all_fails)} FAILURE(S):")
        for f_ in all_fails:
            print(f"  [FAIL] {f_}")
        raise SystemExit(1)
    print("[ok] flat representation, geometry and aggregators all pass")
