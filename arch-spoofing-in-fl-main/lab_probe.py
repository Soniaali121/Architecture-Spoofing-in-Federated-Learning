"""Probing an update: load submitted weights and see how the model BEHAVES.

This is the single home for "set these weights into a model of this
architecture and run it over some data". Five places needed it and each had its
own copy: the fingerprint extractor, the distribution experiment loop, the
attacker (twice, to read a target prior out of the global model and its own
prior out of its own model), and both verifiers in `defense.py`.

The two in `defense.py` mattered beyond tidiness. They each called
`models.build_model` per update, which is exactly the cost `_probe_model`
exists to avoid, and `flag()` runs once per client per round.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

import models
from metrics import expected_calibration_error, js_divergence

_PROBE_MODELS: Dict[tuple, object] = {}
_TORCH_PROBE_MODELS: Dict[tuple, object] = {}


def _probe_model(arch: str, input_shape, num_classes):
    """Reuse one probe model per (arch, shape, classes).

    Rebuilding a Keras model for every update is the dominant cost of corpus
    generation, and a fresh model forces `predict` to retrace its tf.function
    each time. The weights are overwritten per call, so reuse is safe.
    """
    # `input_shape` arrives as a tuple from the lab bundles and as a plain int
    # from the main pipeline's `input_dim`; both have to hash to one key.
    key = (arch, tuple(np.atleast_1d(input_shape).tolist()), int(num_classes))
    if key not in _PROBE_MODELS:
        _PROBE_MODELS[key] = models.build_model(arch, input_shape, num_classes)
    return _PROBE_MODELS[key]


def clear_probe_cache():
    """Drop every cached probe model, on both backends.

    For Keras this must be called alongside `tf.keras.backend.clear_session()`:
    the cached models belong to the session being cleared, and reusing them
    afterwards references freed state. For torch it simply releases the models
    so the CUDA allocator can reclaim their blocks.
    """
    _PROBE_MODELS.clear()
    _TORCH_PROBE_MODELS.clear()


def probe(arch: str, weights, input_shape, num_classes, X) -> Optional[np.ndarray]:
    """Softmax outputs of `weights` over `X`, or None if they are not finite.

    None rather than an array of NaN, because NaN comparisons are always False
    and an unguarded NaN silently passes every threshold it is tested against.
    Callers decide what a dead model means for them: the verifiers treat it as
    maximally suspicious, the fingerprint extractor records it as a worst-case
    row so the NaN rate stays measurable.
    """
    model = _probe_model(arch, input_shape, num_classes)
    model.set_weights([np.asarray(w) for w in weights])
    probs = model.predict(X, verbose=0, batch_size=256)
    return probs if np.all(np.isfinite(probs)) else None


def mean_output_prior(arch: str, weights, input_shape, num_classes, X) -> np.ndarray:
    """Mean predicted class probability of `weights` over `X`.

    Returns a uniform distribution for a dead model rather than NaN, so callers
    that feed this straight into a divergence get a defined, maximally
    uninformative answer instead of a silent pass.
    """
    probs = probe(arch, weights, input_shape, num_classes, X)
    if probs is None:
        return np.ones(int(num_classes)) / float(num_classes)
    return probs.mean(axis=0)


def prior_and_accuracy(arch: str, weights, input_shape, num_classes,
                       X, y) -> Tuple[np.ndarray, float]:
    """One forward pass, both metrics the distribution experiment records.

    The output prior is the imitation-fidelity signal; the accuracy is what the
    disguise cost. Computing them together halves the predict calls in the
    distribution generator's inner loop.
    """
    probs = probe(arch, weights, input_shape, num_classes, X)
    if probs is None:
        return np.ones(int(num_classes)) / float(num_classes), 0.0
    return probs.mean(axis=0), float((np.argmax(probs, axis=1) == y).mean())


# --------------------------------------------------------------------------- #
# Fingerprint features
# --------------------------------------------------------------------------- #
# Behavioural features live in a FIXED-DIMENSION probe-output space, so they are
# comparable across architectures. Shape-dependent ones are tagged: including a
# parameter or layer count in an architecture classifier makes the task trivial
# AND meaningless, because the server already knows the declared shape and an
# adaptive spoofer's shape matches by construction. The notebook excludes these
# for the headline result and uses them only to demonstrate the leak.
SHAPE_AGNOSTIC_FEATURES = [
    "probe_accuracy", "probe_nll", "probe_confidence", "probe_entropy",
    "probe_entropy_std", "probe_margin", "probe_ece", "pred_dist_gini",
    "js_pred_vs_declared", "js_pred_vs_true",
]
SHAPE_DEPENDENT_FEATURES = [
    "delta_norm", "weight_norm", "delta_norm_per_param",
    "weight_norm_per_param", "delta_cosine", "n_params",
]
FEATURE_COLUMNS = SHAPE_AGNOSTIC_FEATURES + SHAPE_DEPENDENT_FEATURES


def extract_fingerprint(arch: str, weights, global_weights, bundle,
                        declared_hist=None, true_hist=None) -> dict:
    """Probe an update and describe how it BEHAVES.

    The update is loaded into a model of `arch` and run over the server's probe
    set. Non-finite output (e.g. a sign-flipped BatchNorm net, whose negative
    variances poison the forward pass) is recorded as `finite_output=False` with
    worst-case values rather than being dropped. How often that happens is
    itself a result, since it is the signal the current defence mostly rides on.
    """
    probs = probe(arch, weights, bundle.input_shape, bundle.num_classes,
                  bundle.X_probe)

    n_params = int(sum(np.asarray(w).size for w in weights))
    delta_norm = float(np.sqrt(sum(np.sum((np.asarray(w) - np.asarray(g)) ** 2)
                                   for w, g in zip(weights, global_weights))))
    weight_norm = float(np.sqrt(sum(np.sum(np.asarray(w) ** 2) for w in weights)))

    flat_w = np.concatenate([np.asarray(w).ravel() for w in weights])
    flat_g = np.concatenate([np.asarray(g).ravel() for g in global_weights])
    denom = np.linalg.norm(flat_w) * np.linalg.norm(flat_g)
    delta_cosine = float(np.dot(flat_w, flat_g) / denom) if denom > 0 else 0.0

    row = {
        "finite_output": probs is not None,
        "n_params": n_params,
        "delta_norm": delta_norm,
        "weight_norm": weight_norm,
        "delta_norm_per_param": delta_norm / max(n_params, 1) ** 0.5,
        "weight_norm_per_param": weight_norm / max(n_params, 1) ** 0.5,
        "delta_cosine": delta_cosine,
    }

    if probs is None:
        # Worst-case behavioural fingerprint; keeps the row in the corpus so the
        # NaN rate is measurable instead of silently vanishing.
        row.update({k: np.nan for k in SHAPE_AGNOSTIC_FEATURES})
        row["probe_accuracy"], row["probe_nll"] = 0.0, 1e9
        row["pred_dist"] = ""
        return row

    eps = 1e-9
    probs = np.clip(probs, eps, 1.0)
    y = bundle.y_probe
    top = np.sort(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    entropy = -(probs * np.log(probs)).sum(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == y).astype(np.float64)
    pred_dist = probs.mean(axis=0)

    row.update({
        "probe_accuracy": float(correct.mean()),
        "probe_nll": float(-np.log(probs[np.arange(len(y)), y]).mean()),
        "probe_confidence": float(conf.mean()),
        "probe_entropy": float(entropy.mean()),
        "probe_entropy_std": float(entropy.std()),
        "probe_margin": float((top[:, -1] - top[:, -2]).mean()),
        "probe_ece": expected_calibration_error(conf, correct),
        # Gini of the mean predicted class distribution: how lopsided the
        # model's output prior is, independent of which class it favours.
        "pred_dist_gini": float(1.0 - np.sum(pred_dist ** 2)),
        "pred_dist": ";".join(f"{p:.6g}" for p in pred_dist),
        "js_pred_vs_declared": (js_divergence(pred_dist, declared_hist)
                                if declared_hist is not None else np.nan),
        "js_pred_vs_true": (js_divergence(pred_dist, true_hist)
                            if true_hist is not None else np.nan),
    })
    return row


# =========================================================================== #
# Torch
# =========================================================================== #
# Same contract as the Keras helpers above: one home for "load these weights
# into a model of this architecture and see how it behaves", one cached model
# per (arch, shape, classes), and None rather than NaN for a dead model.
#
# The torch path takes a FLAT VECTOR rather than a list of per-layer arrays,
# because that is what `aggregation.flatten` produces and what every clusterer,
# aggregator and steering attack works in.
#
# ONE ADDITION THE KERAS SIDE DOES NOT HAVE: `mean_logits_torch`. The torch
# architectures end in a bare Linear emitting logits, with no softmax inside the
# module, so the pre-softmax values are readable. The LogitGap prior estimator
# needs them, and a baked-in softmax destroys them irrecoverably.

def _torch_probe_model(arch: str, input_shape, num_classes: int):
    """Reuse one model per (arch, shape, classes). Weights are overwritten per
    call, so reuse is safe."""
    from models import build_torch_model

    key = (arch, tuple(np.atleast_1d(input_shape).tolist()), int(num_classes))
    if key not in _TORCH_PROBE_MODELS:
        _TORCH_PROBE_MODELS[key] = build_torch_model(arch, input_shape, num_classes)
    return _TORCH_PROBE_MODELS[key]


def load_torch_weights(arch: str, vec, input_shape, num_classes: int):
    """Load a flat vector into the cached model for this architecture.

    Returned in EVAL mode. Probing in train mode would update BatchNorm's
    running statistics from the probe set, which mutates the very weights being
    measured and makes the measurement depend on how many times it has been
    taken.
    """
    from aggregation import unflatten

    model = _torch_probe_model(arch, input_shape, num_classes)
    model.load_state_dict(unflatten(vec, model.state_dict()))
    model.eval()
    return model


def probe_logits_torch(arch: str, vec, input_shape, num_classes: int, X,
                       batch_size: int = 512) -> Optional[np.ndarray]:
    """Raw pre-softmax outputs over `X`, or None if they are not finite."""
    import torch

    from seeding import DEVICE

    model = load_torch_weights(arch, vec, input_shape, num_classes)
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=DEVICE)

    chunks = []
    with torch.no_grad():
        for start in range(0, len(Xt), batch_size):
            out = model(Xt[start:start + batch_size])
            if not torch.isfinite(out).all():
                return None
            chunks.append(out.detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64) if chunks else None


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax in float64."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def probe_torch(arch: str, vec, input_shape, num_classes: int, X,
                batch_size: int = 512) -> Optional[np.ndarray]:
    """Softmax probabilities over `X`, or None if the model is dead."""
    logits = probe_logits_torch(arch, vec, input_shape, num_classes, X, batch_size)
    return None if logits is None else _softmax(logits)


def mean_output_prior_torch(arch: str, vec, input_shape, num_classes: int, X,
                            batch_size: int = 512) -> np.ndarray:
    """Mean predicted class probability over `X`.

    This is Classify-and-Count, the NAIVE prevalence estimator, kept as the
    baseline the estimators in `attacks.py` have to beat rather than because it
    is good. The whole "the output-prior leak is only two percentage points"
    finding was measured with this, and quantification methods exist precisely
    to correct the bias that makes it under-read a prior shift.

    Returns uniform for a dead model rather than NaN, so a caller feeding this
    straight into a divergence gets a defined, maximally uninformative answer
    instead of a silent pass.
    """
    probs = probe_torch(arch, vec, input_shape, num_classes, X, batch_size)
    if probs is None:
        return np.ones(int(num_classes)) / float(num_classes)
    return probs.mean(axis=0)


def mean_logits_torch(arch: str, vec, input_shape, num_classes: int, X,
                      batch_size: int = 512) -> np.ndarray:
    """Mean pre-softmax output per class over `X`.

    What the LogitGap estimator reads. A model trained under class prior p has
    logits approximately `z_balanced + log p`, so the DIFFERENCE between two
    models' mean logits on the same inputs estimates the log-ratio of their
    training priors. Returns zeros for a dead model, which the estimator reads
    as "no evidence" once exponentiated.
    """
    logits = probe_logits_torch(arch, vec, input_shape, num_classes, X, batch_size)
    if logits is None:
        return np.zeros(int(num_classes), dtype=np.float64)
    return logits.mean(axis=0)


def prior_and_accuracy_torch(arch: str, vec, input_shape, num_classes: int, X, y,
                             batch_size: int = 512) -> Tuple[np.ndarray, float]:
    """One forward pass, both metrics the experiment records.

    The output prior is the imitation-fidelity signal; the accuracy is what the
    disguise cost. Computing them together halves the forward passes in the
    generator's inner loop.
    """
    probs = probe_torch(arch, vec, input_shape, num_classes, X, batch_size)
    if probs is None:
        return np.ones(int(num_classes)) / float(num_classes), 0.0
    return probs.mean(axis=0), float((np.argmax(probs, axis=1) == np.asarray(y)).mean())


def posteriors_by_true_class(probs, y, num_classes: int) -> np.ndarray:
    """(num_classes, num_classes) matrix M, M[k, j] = mean posterior for class j
    over examples whose TRUE label is k.

    The confusion structure PACC inverts. Rows for classes absent from the probe
    set are left uniform rather than zero, so the matrix stays invertible.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    M = np.full((num_classes, num_classes), 1.0 / num_classes, dtype=np.float64)
    for k in range(num_classes):
        mask = (y == k)
        if mask.any():
            M[k] = probs[mask].mean(axis=0)
    return M


def extract_fingerprint_torch(arch: str, vec, global_vec, bundle,
                              declared_hist=None, true_hist=None) -> dict:
    """Torch counterpart of `extract_fingerprint`, over flat vectors.

    Emits the SAME column names, so a corpus row from either backend is scored
    by the same analysis code and the same meta-classifier.
    """
    from aggregation import cosine

    probs = probe_torch(arch, vec, bundle.input_shape, bundle.num_classes,
                        bundle.X_probe)

    vec = np.asarray(vec, dtype=np.float64).ravel()
    gvec = (np.asarray(global_vec, dtype=np.float64).ravel()
            if global_vec is not None else np.zeros_like(vec))

    n_params = int(vec.size)
    delta_norm = float(np.linalg.norm(vec - gvec))
    weight_norm = float(np.linalg.norm(vec))

    row = {
        "finite_output": probs is not None,
        "n_params": n_params,
        "delta_norm": delta_norm,
        "weight_norm": weight_norm,
        "delta_norm_per_param": delta_norm / max(n_params, 1) ** 0.5,
        "weight_norm_per_param": weight_norm / max(n_params, 1) ** 0.5,
        "delta_cosine": cosine(vec, gvec),
    }

    if probs is None:
        # Worst-case behavioural fingerprint. The row stays in the corpus so the
        # non-finite rate remains measurable rather than silently vanishing.
        row.update({k: np.nan for k in SHAPE_AGNOSTIC_FEATURES})
        row["probe_accuracy"], row["probe_nll"] = 0.0, 1e9
        row["pred_dist"] = ""
        return row

    eps = 1e-9
    probs = np.clip(probs, eps, 1.0)
    y = bundle.y_probe
    top = np.sort(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    entropy = -(probs * np.log(probs)).sum(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == y).astype(np.float64)
    pred_dist = probs.mean(axis=0)

    row.update({
        "probe_accuracy": float(correct.mean()),
        "probe_nll": float(-np.log(probs[np.arange(len(y)), y]).mean()),
        "probe_confidence": float(conf.mean()),
        "probe_entropy": float(entropy.mean()),
        "probe_entropy_std": float(entropy.std()),
        "probe_margin": float((top[:, -1] - top[:, -2]).mean()),
        "probe_ece": expected_calibration_error(conf, correct),
        "pred_dist_gini": float(1.0 - np.sum(pred_dist ** 2)),
        "pred_dist": ";".join(f"{p:.6g}" for p in pred_dist),
        "js_pred_vs_declared": (js_divergence(pred_dist, declared_hist)
                                if declared_hist is not None else np.nan),
        "js_pred_vs_true": (js_divergence(pred_dist, true_hist)
                            if true_hist is not None else np.nan),
    })
    return row
