"""Client-side local training and update messages, for both backends.

`ClientUpdate` is the one message type both stacks submit. It carries the model
in whichever representation its backend produced:

  weights   list of per-layer arrays. What Keras `get_weights()` returns, and
            what `aggregation.fedavg` and the existing verifiers consume.
  vec       one flat float64 vector in sorted-key order. What the torch path
            produces, and what every clusterer, robust aggregator and steering
            attack in the new pipeline works in. See `aggregation.flatten`.

Exactly one of the two is populated. `as_vector()` gives a flat vector either
way, so code that only needs the geometry does not have to care which backend
built the update.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from data import label_histogram
from models import build_model


@dataclass
class ClientUpdate:
    client_id: int
    weights: List[np.ndarray] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    true_arch: Optional[str] = None
    vec: Optional[np.ndarray] = None

    @property
    def arch(self) -> str:
        """The DECLARED architecture, which is what the server acts on."""
        return self.metadata.get("arch", self.true_arch)

    def as_vector(self) -> np.ndarray:
        """Flat float64 vector, whichever representation was populated.

        For the Keras path this concatenates the per-layer arrays in the order
        Keras returned them, which is that backend's canonical order. It is NOT
        interchangeable with a torch vector from the same nominal architecture:
        the two stacks lay parameters out differently, and comparing across them
        would produce a number rather than an error. Never mix backends inside
        one federation.
        """
        if self.vec is not None:
            return np.asarray(self.vec, dtype=np.float64).ravel()
        if not self.weights:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate([np.asarray(w, dtype=np.float64).ravel()
                               for w in self.weights])


# =========================================================================== #
# Keras
# =========================================================================== #
def train_client(
    client_id,
    X_client,
    y_client,
    arch_name,
    global_weights,
    input_dim,
    num_classes,
    epochs=10,
    batch_size=32,
    true_arch=None,
):
    """Train a local Keras model from the cluster's current weights.

    arch_name is the architecture actually trained (== the declared/spoofed arch).
    true_arch records the client's real home architecture for ground-truth/eval; it
    differs from arch_name only when the client is spoofing.
    """
    local_model = build_model(arch_name, input_dim, num_classes)
    local_model.set_weights(global_weights)
    fit_history = local_model.fit(X_client, y_client, epochs=epochs,
                                   batch_size=batch_size, verbose=0)

    metadata = {
        "arch": arch_name,
        "label_hist": label_histogram(y_client, num_classes),
        "n_samples": int(len(y_client)),
        # Local-training observability: how well this client's own data fit its
        # own model, independent of the server-side probe/fingerprint checks.
        "train_loss": float(fit_history.history["loss"][-1]),
        "train_accuracy": float(fit_history.history["accuracy"][-1])
        if "accuracy" in fit_history.history else None,
    }
    return ClientUpdate(
        client_id=client_id,
        weights=local_model.get_weights(),
        metadata=metadata,
        true_arch=true_arch if true_arch is not None else arch_name,
    )


# =========================================================================== #
# Torch
# =========================================================================== #
# Batching is done by hand rather than through a DataLoader. The client shards
# here are a few hundred to a few thousand rows already resident in memory, so a
# DataLoader buys nothing and costs a worker-process spawn per client per round,
# which on Windows is expensive and on a machine with 0.3 GB free is a hazard.

def make_optimizer(model):
    """Build the optimiser the architecture declares through its `opt_spec`."""
    import torch

    spec = getattr(model, "opt_spec", None)
    if spec is None:
        return torch.optim.Adam(model.parameters(), lr=1e-3)
    if spec.kind == "adam":
        return torch.optim.Adam(model.parameters(), lr=spec.lr)
    if spec.kind == "sgd":
        return torch.optim.SGD(model.parameters(), lr=spec.lr, momentum=spec.momentum)
    raise ValueError(f"unknown optimiser kind {spec.kind!r}")


def _tensors(X, y):
    import torch

    from seeding import DEVICE

    return (torch.as_tensor(np.asarray(X), dtype=torch.float32, device=DEVICE),
            torch.as_tensor(np.asarray(y), dtype=torch.long, device=DEVICE))


def train_client_torch(client_id, X_client, y_client, arch_name, global_vec,
                       input_shape, num_classes, epochs=2, batch_size=32,
                       true_arch=None, seed=None) -> ClientUpdate:
    """Warm-start from the cluster's current weights, train locally, submit.

    `arch_name` is the architecture actually TRAINED, which for a spoofer is the
    one it declares rather than the one it owns. `true_arch` records the real
    home architecture for ground truth.

    `global_vec` may be None, in which case the client starts from a fresh
    initialisation. That happens only in the first round of a cluster that has
    just been created by a bipartition.
    """
    import torch
    import torch.nn as nn

    from aggregation import flatten, unflatten
    from models import build_torch_model
    from seeding import DEVICE

    model = build_torch_model(arch_name, input_shape, num_classes)
    if global_vec is not None:
        model.load_state_dict(unflatten(global_vec, model.state_dict()))
    if seed is not None:
        torch.manual_seed(int(seed))

    Xt, yt = _tensors(X_client, y_client)
    optimizer = make_optimizer(model)
    criterion = nn.CrossEntropyLoss()

    n = len(yt)
    model.train()
    last_loss, last_correct, last_total = float("nan"), 0, 0

    for _ in range(int(epochs)):
        perm = torch.randperm(n, device=DEVICE)
        epoch_loss, correct, total = 0.0, 0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            # A batch of one breaks BatchNorm in train mode: it cannot compute a
            # variance. Dropping it costs at most one sample per epoch and is
            # the standard remedy.
            if len(idx) < 2 and n >= 2:
                continue
            xb, yb = Xt[idx], yt[idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach()) * len(idx)
            correct += int((logits.detach().argmax(1) == yb).sum())
            total += len(idx)
        if total:
            last_loss, last_correct, last_total = epoch_loss / total, correct, total

    metadata = {
        "arch": arch_name,
        "label_hist": label_histogram(np.asarray(y_client), num_classes),
        "n_samples": int(n),
        "train_loss": float(last_loss),
        "train_accuracy": float(last_correct / last_total) if last_total else None,
    }
    return ClientUpdate(
        client_id=client_id,
        metadata=metadata,
        true_arch=true_arch if true_arch is not None else arch_name,
        vec=flatten(model.state_dict()),
    )


def evaluate_torch(model, X, y, batch_size: int = 512):
    """Return (mean loss, accuracy) on a held-out set.

    A model whose weights have gone non-finite returns `(inf, 0.0)` rather than
    NaN. NaN comparisons are always False, so an unguarded NaN silently passes
    every threshold it is tested against, which is how a wrecked update once
    sailed through a verifier untouched.
    """
    import torch
    import torch.nn as nn

    model.eval()
    Xt, yt = _tensors(X, y)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss, correct, n = 0.0, 0, len(yt)
    if n == 0:
        return float("inf"), 0.0

    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb, yb = Xt[start:start + batch_size], yt[start:start + batch_size]
            logits = model(xb)
            if not torch.isfinite(logits).all():
                return float("inf"), 0.0
            total_loss += float(criterion(logits, yb))
            correct += int((logits.argmax(1) == yb).sum())
    return total_loss / n, correct / n


def evaluate_vec(vec, arch_name, input_shape, num_classes, X, y, batch_size=512):
    """`evaluate_torch` starting from a flat vector rather than a live model.

    Goes through the probe-model cache, so scoring every cluster model against
    every client (which is what IFCA-style assignment costs, K times per client
    per round) does not rebuild an architecture per call.
    """
    from lab_probe import load_torch_weights

    model = load_torch_weights(arch_name, vec, input_shape, num_classes)
    return evaluate_torch(model, X, y, batch_size=batch_size)
