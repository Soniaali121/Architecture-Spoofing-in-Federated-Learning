"""Model architecture factories and registries, for both backends.

TWO REGISTRIES, ONE SET OF NAMES
--------------------------------
`ARCH_REGISTRY` holds the Keras factories the original pipeline uses.
`TORCH_ARCH_REGISTRY` holds torch equivalents under the SAME names, so a result
from either stack can be put beside the other without a translation table.

The Keras imports are deferred into the factory functions. They used to be at
module scope, which meant that anything importing `models` paid a TensorFlow
import: several seconds and a few hundred MB. The torch pipeline does not want
TF resident at all. On a machine that routinely has 0.3 GB free, holding two
deep-learning runtimes in one process is how a run becomes a silent OOM kill.

TWO DELIBERATE DIFFERENCES IN THE TORCH MODELS
----------------------------------------------
1. Every torch model ends in a bare `nn.Linear` emitting LOGITS. There is no
   softmax inside the module. `nn.CrossEntropyLoss` wants logits, and the
   LogitGap prior estimator reads the pre-softmax values, which a baked-in
   softmax destroys. `lab_probe` applies the softmax where probabilities are
   wanted. The Keras models keep their softmax output layer, unchanged.
2. Torch image input is `(C, H, W)`, not Keras `(H, W, C)`. `lab_data`'s torch
   bundles produce that layout, so nothing in the pipeline permutes.
"""

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


# =========================================================================== #
# Keras
# =========================================================================== #
def create_mlp_model(input_shape, num_classes):
    from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                         Input)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def create_dnn_model(input_shape, num_classes):
    from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                         Input)
    from tensorflow.keras.models import Sequential
    try:  # AdamW is a class; the string alias is not registered in all builds
        from tensorflow.keras.optimizers import AdamW
    except ImportError:
        from tensorflow.keras.optimizers.experimental import AdamW

    model = Sequential()
    model.add(Input(shape=(input_shape,)))

    for units in (512, 256, 128, 64):
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=AdamW(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# Supervised architectures used in FL rounds. `lab_models` registers seven more
# into this dict at import time; it is an extension point, not a fixed list.
#
# The reference code's unsupervised autoencoder is not carried over: nothing in
# the FL pipeline ever built one. It is still in `aimens-code.py` verbatim.
ARCH_REGISTRY = {
    "mlp": create_mlp_model,
    "dnn": create_dnn_model,
}


def build_model(arch_name, input_shape, num_classes):
    """Build a KERAS model by name. See `build_torch_model` for the other stack."""
    if arch_name not in ARCH_REGISTRY:
        raise KeyError(f"Unknown architecture '{arch_name}'. Known: {list(ARCH_REGISTRY)}")
    return ARCH_REGISTRY[arch_name](input_shape, num_classes)


# =========================================================================== #
# Torch
# =========================================================================== #
def _shape(input_shape) -> Tuple[int, ...]:
    """Accept an int (tabular width) or an iterable, return a tuple."""
    if isinstance(input_shape, (int, np.integer)):
        return (int(input_shape),)
    return tuple(int(x) for x in np.atleast_1d(input_shape).tolist())


class _OptSpec:
    """How an architecture is trained. Read by `client.make_optimizer`.

    The optimiser differs per architecture and those choices are load-bearing:
    the dense nets use Adam 1e-3 and the CNNs use SGD with momentum, matching
    the settings the existing corpus was generated with. Do not harmonise them
    without regenerating.
    """

    def __init__(self, kind: str, lr: float, momentum: float = 0.0):
        self.kind, self.lr, self.momentum = kind, lr, momentum

    def __repr__(self):
        return f"<{self.kind} lr={self.lr} momentum={self.momentum}>"


def _torch_nn():
    import torch.nn as nn

    return nn


def _dense_stack_torch(input_shape, num_classes, units, batchnorm, dropout):
    """Dense net that accepts the dataset's native shape and flattens itself.

    Flattening inside the model, rather than requiring pre-flattened input, is
    what lets dense and convolutional architectures share one input tensor, so
    they can be compared as members of the same federation.
    """
    import torch.nn as nn

    shape = _shape(input_shape)
    in_features = int(np.prod(shape))

    layers: List = [nn.Flatten()]
    prev = in_features
    for n in units:
        layers.append(nn.Linear(prev, n))
        layers.append(nn.ReLU())
        if batchnorm:
            layers.append(nn.BatchNorm1d(n))
        if dropout:
            layers.append(nn.Dropout(dropout))
        prev = n
    layers.append(nn.Linear(prev, num_classes))
    return nn.Sequential(*layers)


def _conv_net_torch(input_shape, num_classes, blocks, head):
    """Conv blocks, then a dense head. `blocks` is [(filters, kernel, pool), ...].

    The flattened width after the trunk is measured with a dry forward pass of a
    single zero tensor rather than derived by hand. Hand-derived shapes are a
    standard source of silent off-by-one errors the moment a pooling layer is
    added or removed.
    """
    import torch
    import torch.nn as nn

    shape = _shape(input_shape)
    if len(shape) == 2:                       # (H, W) -> (1, H, W)
        shape = (1,) + shape

    trunk: List = []
    prev = shape[0]
    for filters, kernel, pool in blocks:
        trunk.append(nn.Conv2d(prev, filters, kernel, padding="same"))
        trunk.append(nn.ReLU())
        if pool:
            trunk.append(nn.MaxPool2d(2, 2))
        prev = filters
    trunk_seq = nn.Sequential(*trunk)

    with torch.no_grad():
        flat_width = trunk_seq(torch.zeros(1, *shape)).flatten(1).shape[1]

    dense: List = [nn.Flatten()]
    prev_w = flat_width
    for n in head:
        dense.append(nn.Linear(prev_w, n))
        dense.append(nn.ReLU())
        prev_w = n
    dense.append(nn.Linear(prev_w, num_classes))

    return nn.Sequential(trunk_seq, nn.Sequential(*dense))


def _tag(model, spec: _OptSpec):
    model.opt_spec = spec
    return model


def create_mlp_flat_torch(input_shape, num_classes):
    """Low-capacity dense net."""
    return _tag(_dense_stack_torch(input_shape, num_classes, [128, 64],
                                   batchnorm=True, dropout=0.2),
                _OptSpec("adam", 1e-3))


def create_dnn_flat_torch(input_shape, num_classes):
    """High-capacity dense net."""
    return _tag(_dense_stack_torch(input_shape, num_classes, [512, 256, 128, 64],
                                   batchnorm=True, dropout=0.3),
                _OptSpec("adam", 1e-3))


def create_mlp_plain_torch(input_shape, num_classes):
    """Dense net with NO BatchNorm and NO Dropout.

    Included deliberately. Sign-flipping a BatchNorm network produces negative
    variances and hence non-finite output, which any verifier detects trivially.
    Having a BatchNorm-free architecture in the pool separates "we detected a
    spoof" from "we detected a NaN", which is the distinction DEFENSE_NOTES.md
    section 4.1 turns on.
    """
    return _tag(_dense_stack_torch(input_shape, num_classes, [256, 128],
                                   batchnorm=False, dropout=0.0),
                _OptSpec("adam", 1e-3))


def create_mnist_cnn_torch(input_shape, num_classes):
    """McMahan et al. FedAvg MNIST CNN: Conv32-5, pool, Conv64-5, pool, 512."""
    return _tag(_conv_net_torch(input_shape, num_classes,
                                [(32, 5, True), (64, 5, True)], [512]),
                _OptSpec("sgd", 0.01, 0.9))


def create_leaf_cnn_torch(input_shape, num_classes):
    """LEAF FEMNIST CNN (Caldas et al.): same trunk, 2048-unit head."""
    return _tag(_conv_net_torch(input_shape, num_classes,
                                [(32, 5, True), (64, 5, True)], [2048]),
                _OptSpec("sgd", 0.004, 0.9))


def create_cifar_cnn_torch(input_shape, num_classes):
    """FedAvg-paper style CIFAR CNN: two conv blocks, a conv, a dense head."""
    return _tag(_conv_net_torch(input_shape, num_classes,
                                [(32, 3, True), (64, 3, True), (64, 3, False)], [64]),
                _OptSpec("sgd", 0.01, 0.9))


def create_cifar_cnn_deep_torch(input_shape, num_classes):
    """Deeper CIFAR CNN, for architecture-contrast within one federation."""
    return _tag(_conv_net_torch(input_shape, num_classes,
                                [(32, 3, False), (32, 3, True), (64, 3, False),
                                 (64, 3, True), (128, 3, True)], [256]),
                _OptSpec("sgd", 0.01, 0.9))


# Complete at definition time, deliberately. The Keras registry needs
# `import lab_models` to be populated, and forgetting that produced
# `KeyError: Unknown architecture 'mlp_flat'` from a module with no obvious
# connection to architectures (HANDOFF section 7).
TORCH_ARCH_REGISTRY: Dict[str, Callable] = {
    "mlp": create_mlp_flat_torch,
    "dnn": create_dnn_flat_torch,
    "mlp_flat": create_mlp_flat_torch,
    "dnn_flat": create_dnn_flat_torch,
    "mlp_plain": create_mlp_plain_torch,
    "mnist_cnn": create_mnist_cnn_torch,
    "leaf_cnn": create_leaf_cnn_torch,
    "cifar_cnn": create_cifar_cnn_torch,
    "cifar_cnn_deep": create_cifar_cnn_deep_torch,
}

# Architecture pools per dataset. More than two is deliberate: a two-class
# meta-classifier can look strong by luck, whereas a four-way one has to
# actually separate the fingerprint space.
ARCH_POOLS: Dict[str, List[str]] = {
    "mnist": ["mlp_flat", "dnn_flat", "mnist_cnn", "leaf_cnn"],
    "cifar10": ["mlp_flat", "dnn_flat", "cifar_cnn", "cifar_cnn_deep"],
    "femnist": ["mlp_flat", "mnist_cnn", "leaf_cnn"],
    "ids": ["mlp_flat", "dnn_flat", "mlp_plain"],
}


def build_torch_model(arch_name: str, input_shape, num_classes: int):
    """Build a TORCH model by name, already moved to `seeding.DEVICE`."""
    from seeding import DEVICE

    if arch_name not in TORCH_ARCH_REGISTRY:
        raise KeyError(f"Unknown torch architecture '{arch_name}'. "
                       f"Known: {sorted(TORCH_ARCH_REGISTRY)}")
    model = TORCH_ARCH_REGISTRY[arch_name](input_shape, num_classes)
    return model.to(DEVICE) if DEVICE is not None else model


def head_indices(model) -> "np.ndarray":
    """Flat indices of the final Linear layer's weight and bias.

    WHY CLUSTERING WANTS THIS. Under concept shift (a different label
    permutation or rotation per group, which is how the Clustered Federated
    Learning literature plants clusters) every client sees the SAME input
    distribution, so the representation-learning gradients in the trunk are
    near-identical across the whole federation. The task disagreement lives
    almost entirely in the classifier head. Measured on label-permuted data,
    within-group minus cross-group cosine margin at round 4:

        dataset   full vector   head only   head cross-group cosine
        MNIST          0.0001       0.875                    -0.171
        IDS            0.0032       1.388                    -0.499

    The head is 0.6% of the MNIST parameters and 3.1% of the IDS ones, and it
    carries essentially all of the signal. On the full vector the cross-group
    cosine is +0.9997 and nothing cancels, so Sattler's split criterion can
    never fire; on the head it is NEGATIVE, which is the opposed-gradient
    condition the criterion is built to detect.

    Restricting similarity to the head follows FedRep and FedPer, which split a
    network into a shared representation and a personalised head, and matches
    the common practice of computing client similarity on classifier updates.

    Indices are into the vector `aggregation.flatten` produces, which orders
    `state_dict` keys by SORTED NAME rather than by network depth. The head is
    therefore not at the end of the vector, and slicing the tail selects trunk
    parameters instead.
    """
    import numpy as np
    import torch.nn as nn

    last_linear = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear = name
    if last_linear is None:
        raise ValueError("no nn.Linear layer found, cannot locate a classifier head")

    wanted = {f"{last_linear}.weight", f"{last_linear}.bias"}
    state = model.state_dict()
    idx, offset = [], 0
    for key in sorted(state.keys()):
        n = state[key].numel()
        if key in wanted:
            idx.append(np.arange(offset, offset + n))
        offset += n
    if not idx:
        raise ValueError(f"could not locate {wanted} in the state_dict")
    return np.concatenate(idx)


def final_bias_key(model) -> str:
    """Key of the final Linear layer's bias in `state_dict`, in sorted order.

    The prior-imitation attack edits this one tensor and the LogitGap estimator
    reads it, and both have to find it without assuming it is the last entry.
    Under the sorted key order that `aggregation.flatten` pins, it is
    emphatically NOT last: `1.6.bias` sorts before `1.6.weight`, and for the
    CNNs the head sorts before the trunk.
    """
    import torch.nn as nn

    last_linear = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear = name
    if last_linear is None:
        raise ValueError("model has no nn.Linear layer, cannot locate a final bias")
    return f"{last_linear}.bias"


# =========================================================================== #
# Self-test
# =========================================================================== #
# Run with `python models.py`. Checks the torch registry only: it builds every
# architecture, round-trips it through the flat representation, and confirms the
# final bias is findable. The Keras side is exercised by the existing pipeline.

_SELFTEST_CASES = [
    ("mlp", (24,), 7), ("dnn", (24,), 7), ("mlp_flat", (24,), 7),
    ("dnn_flat", (24,), 7), ("mlp_plain", (24,), 7),
    ("mnist_cnn", (1, 28, 28), 10), ("leaf_cnn", (1, 28, 28), 62),
    ("cifar_cnn", (3, 32, 32), 10), ("cifar_cnn_deep", (3, 32, 32), 10),
]


def _selftest() -> List[str]:
    import torch

    from aggregation import flatten, learnable_mask, sorted_keys, unflatten

    fails = []
    if set(a for a, _, _ in _SELFTEST_CASES) != set(TORCH_ARCH_REGISTRY):
        fails.append("a torch architecture is not covered by the self-test")

    for arch, shape, classes in _SELFTEST_CASES:
        torch.manual_seed(0)
        model = build_torch_model(arch, shape, classes)
        sd = model.state_dict()

        # Non-trivial values: a freshly initialised model has BatchNorm buffers
        # at exactly 0 and 1 and its counters at 0, so it round-trips through a
        # great many wrong implementations.
        perturbed = {k: (v + torch.randn_like(v) * 0.1 if v.dtype.is_floating_point
                         else v + 17) for k, v in sd.items()}
        restored = unflatten(flatten(perturbed), sd)
        for k in perturbed:
            if restored[k].dtype != perturbed[k].dtype:
                fails.append(f"{arch}: dtype drift on {k}")
            elif not torch.equal(restored[k], perturbed[k]):
                fails.append(f"{arch}: value drift on {k}")

        # Insertion order must not matter.
        reversed_sd = {k: sd[k] for k in reversed(list(sd))}
        if not np.array_equal(flatten(sd), flatten(reversed_sd)):
            fails.append(f"{arch}: flatten depends on insertion order")

        # The final bias must be locatable and correctly sized.
        key = final_bias_key(model)
        if key not in sd:
            fails.append(f"{arch}: final_bias_key returned a missing key {key!r}")
        elif tuple(sd[key].shape) != (classes,):
            fails.append(f"{arch}: final bias has shape {tuple(sd[key].shape)}, "
                         f"expected ({classes},)")

        # The mask must exclude exactly the integer buffers.
        n_int = sum(v.numel() for v in sd.values() if not v.dtype.is_floating_point)
        if int((~learnable_mask(sd)).sum()) != n_int:
            fails.append(f"{arch}: learnable_mask does not match the integer buffers")

        # A forward pass must produce finite logits of the right width.
        model.eval()
        with torch.no_grad():
            from seeding import DEVICE

            out = model(torch.zeros(2, *shape, device=DEVICE))
        if tuple(out.shape) != (2, classes):
            fails.append(f"{arch}: forward gave {tuple(out.shape)}, "
                         f"expected (2, {classes})")
        elif not torch.isfinite(out).all():
            fails.append(f"{arch}: forward produced non-finite logits")

        n_params = sum(v.numel() for v in sd.values())
        print(f"  {arch:<16} shape={str(shape):<12} params={n_params:>9,} "
              f"final_bias={key}")
    return fails


if __name__ == "__main__":
    from seeding import describe_device

    print(describe_device())
    problems = _selftest()
    if problems:
        print(f"\n{len(problems)} FAILURE(S):")
        for p in problems:
            print(f"  [FAIL] {p}")
        raise SystemExit(1)
    print("[ok] every torch architecture builds, round-trips and runs forward")
