"""Architectures for the fingerprinting experiments, plus GPU configuration.

NOTHING IN THE EXISTING CODEBASE IS MODIFIED. These architectures are
*registered* into `models.ARCH_REGISTRY` (a module-level dict, i.e. an extension
point) rather than edited in. Verified: `models.build_model` forwards its shape
argument untouched, and `fl_loop`/`client`/`defense` only ever pass `input_dim`
through to it, so a shape TUPLE flows end-to-end and CNNs work through main's
unmodified `FederatedServer`.

CNNs adapted from Amin's branch (`arch-spoofing-in-fl-adding-cnns/models.py`):
the McMahan FedAvg MNIST CNN, the LEAF/FEMNIST CNN (Caldas et al.) and the two
CIFAR CNNs. Adapted so every factory accepts the *same* input shape tuple and
flattens internally when it is a dense net. That is what lets one dataset tensor
feed every architecture in a run, which is what makes cross-architecture
fingerprint comparison possible at all.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

# Keep TF quiet before it is imported anywhere else.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam

import models


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #
def configure_gpu(verbose: bool = True):
    """Enable memory growth so an 8 GB laptop GPU is not fully pre-allocated.

    Safe to call on CPU-only builds, where it simply finds no GPUs. That is what
    happens on native Windows, since TF's Windows wheels have no GPU support
    compiled in (see Dockerfile for the full explanation and the container that
    fixes it).
    """
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:          # already initialised
            if verbose:
                print(f"[gpu] memory_growth skipped: {exc}")
    if verbose:
        print(f"[gpu] TF {tf.__version__} devices: "
              f"{[g.name for g in gpus] if gpus else 'CPU only'}")
    return gpus


configure_gpu(verbose=False)


# --------------------------------------------------------------------------- #
# Architectures
# --------------------------------------------------------------------------- #
def _shape(input_shape) -> Tuple[int, ...]:
    if isinstance(input_shape, (int, np.integer)):
        return (int(input_shape),)
    return tuple(int(x) for x in input_shape)


def _dense_stack(input_shape, num_classes, units, optimizer, batchnorm, dropout):
    """Dense net that accepts the dataset's native shape and flattens itself.

    Flattening inside the model (rather than requiring pre-flattened input) is
    what lets dense and convolutional architectures share one input tensor, so
    they can be compared as members of the same federation.
    """
    layers = [Input(shape=_shape(input_shape)), Flatten()]
    for n in units:
        layers.append(Dense(n, activation="relu"))
        if batchnorm:
            layers.append(BatchNormalization())
        if dropout:
            layers.append(Dropout(dropout))
    layers.append(Dense(num_classes, activation="softmax"))
    model = Sequential(layers)
    model.compile(optimizer=optimizer(), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def create_mlp_flat(input_shape, num_classes):
    """Low-capacity dense net (mirrors main's `mlp`, shape-tolerant)."""
    return _dense_stack(input_shape, num_classes, [128, 64],
                        lambda: Adam(1e-3), batchnorm=True, dropout=0.2)


def create_dnn_flat(input_shape, num_classes):
    """High-capacity dense net (mirrors main's `dnn`, shape-tolerant).

    Note the optimiser differs from `models.create_dnn_model`, which uses AdamW.
    The two are not interchangeable: this one is the architecture the corpus was
    generated with, so do not "harmonise" them without regenerating.
    """
    return _dense_stack(input_shape, num_classes, [512, 256, 128, 64],
                        lambda: Adam(1e-3), batchnorm=True, dropout=0.3)


def create_mlp_plain(input_shape, num_classes):
    """Dense net with NO BatchNorm/Dropout.

    Included deliberately: sign-flipping a BatchNorm network produces negative
    variances and hence non-finite output, which any verifier detects trivially.
    Having a BatchNorm-free architecture in the pool separates "we detected a
    spoof" from "we detected a NaN". See DEFENSE_NOTES.md section 4.1.
    """
    return _dense_stack(input_shape, num_classes, [256, 128],
                        lambda: Adam(1e-3), batchnorm=False, dropout=0.0)


def _cnn(input_shape, num_classes, blocks, head, lr):
    shape = _shape(input_shape)
    if len(shape) == 2:                       # (H, W) -> (H, W, 1)
        shape = shape + (1,)
    layers = [Input(shape=shape)]
    for filters, kernel, pool in blocks:
        layers.append(Conv2D(filters, kernel, padding="same", activation="relu"))
        if pool:
            layers.append(MaxPooling2D(2, 2))
    layers.append(Flatten())
    for n in head:
        layers.append(Dense(n, activation="relu"))
    layers.append(Dense(num_classes, activation="softmax"))
    model = Sequential(layers)
    model.compile(optimizer=SGD(learning_rate=lr, momentum=0.9),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def create_mnist_cnn(input_shape, num_classes):
    """McMahan et al. FedAvg MNIST CNN: Conv32-5 -> pool -> Conv64-5 -> pool -> 512."""
    return _cnn(input_shape, num_classes,
                [(32, 5, True), (64, 5, True)], [512], lr=0.01)


def create_leaf_cnn(input_shape, num_classes):
    """LEAF FEMNIST CNN (Caldas et al.): same trunk, 2048-unit head."""
    return _cnn(input_shape, num_classes,
                [(32, 5, True), (64, 5, True)], [2048], lr=0.004)


def create_cifar_cnn(input_shape, num_classes):
    """FedAvg-paper style CIFAR CNN: two conv blocks + conv + dense head."""
    return _cnn(input_shape, num_classes,
                [(32, 3, True), (64, 3, True), (64, 3, False)], [64], lr=0.01)


def create_cifar_cnn_deep(input_shape, num_classes):
    """Deeper CIFAR CNN, for architecture-contrast within one federation."""
    return _cnn(input_shape, num_classes,
                [(32, 3, False), (32, 3, True), (64, 3, False),
                 (64, 3, True), (128, 3, True)], [256], lr=0.01)


LAB_ARCHITECTURES = {
    "mlp_flat": create_mlp_flat,
    "dnn_flat": create_dnn_flat,
    "mlp_plain": create_mlp_plain,
    "mnist_cnn": create_mnist_cnn,
    "leaf_cnn": create_leaf_cnn,
    "cifar_cnn": create_cifar_cnn,
    "cifar_cnn_deep": create_cifar_cnn_deep,
}


def register_architectures(verbose: bool = False) -> List[str]:
    """Add the lab architectures to `models.ARCH_REGISTRY`.

    Registration, not modification: `models.py` is untouched and its existing
    `mlp`/`dnn` entries keep working for the tabular IDS pipeline.
    """
    models.ARCH_REGISTRY.update(LAB_ARCHITECTURES)
    if verbose:
        print(f"[arch] registry: {sorted(models.ARCH_REGISTRY)}")
    return sorted(LAB_ARCHITECTURES)


register_architectures()


# Architecture pools per dataset. More than two architectures is deliberate:
# a two-class meta-classifier can look strong by luck, whereas a 4-way one has
# to actually separate the fingerprint space.
ARCH_POOLS: Dict[str, List[str]] = {
    "mnist": ["mlp_flat", "dnn_flat", "mnist_cnn", "leaf_cnn"],
    "cifar10": ["mlp_flat", "dnn_flat", "cifar_cnn", "cifar_cnn_deep"],
    "femnist": ["mlp_flat", "mnist_cnn", "leaf_cnn"],
    "ids": ["mlp_flat", "dnn_flat", "mlp_plain"],
}
