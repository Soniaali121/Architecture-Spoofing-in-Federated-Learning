"""Global seeding and device selection, for both backends.

Every source of randomness in a run has to be pinned or the numbers aren't
comparable across conditions: Python's `random`, NumPy's legacy global RNG, and
whichever framework is doing the training (which drives weight initialisation,
dropout masks, and shuffling inside the fit loop).

Call `set_global_seeds(seed)` once at the START of a run, before any model is
built. For multi-seed experiments, run the whole pipeline once per seed and
aggregate across runs; do not reseed mid-run, which correlates conditions in
ways that are hard to reason about.

WHY THE BACKEND IS A PARAMETER
------------------------------
This module used to import TensorFlow unconditionally, which costs several
seconds and a few hundred MB even when the caller only wanted to shuffle a
partition. The torch pipeline does not want TF loaded at all: on a machine that
routinely has 0.3 GB free, holding two deep-learning runtimes in one process is
how a run turns into a silent OOM kill. The default therefore seeds only what is
already imported, and the caller asks explicitly for anything else.

DEVICE is resolved here, once, and imported from here. Calling
`torch.cuda.is_available()` in several places is how a codebase ends up with
half its tensors on the GPU and half on the CPU, and the resulting error names
the operation that failed rather than the module that guessed wrong.
"""

import os
import random
import sys

import numpy as np


def _torch_device():
    """Resolve the torch device without making torch a hard dependency."""
    try:
        import torch
    except ImportError:
        return None
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = _torch_device()


def set_global_seeds(seed: int = 42, deterministic_ops: bool = False,
                     backend: str = "auto") -> None:
    """Pin Python, NumPy, and the chosen framework's RNGs to `seed`.

    `backend` is one of:
      auto    seed whichever frameworks are ALREADY imported. The default,
              because it never pays an import cost the caller did not ask for.
      torch   seed torch, importing it if needed.
      tf      seed TensorFlow, importing it if needed.
      both    seed both. Only for a process that genuinely runs both, which on
              this machine is a memory hazard.
      none    Python and NumPy only.

    `deterministic_ops=True` additionally asks the framework for bitwise
    reproducibility on the same hardware. It is off by default because it
    disables optimised kernels and can slow training noticeably; seeding alone
    is enough to make runs reproducible in practice here.
    """
    if backend not in ("auto", "torch", "tf", "both", "none"):
        raise ValueError(f"backend must be auto/torch/tf/both/none, got {backend!r}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    want_torch = backend in ("torch", "both") or (
        backend == "auto" and "torch" in sys.modules)
    want_tf = backend in ("tf", "both") or (
        backend == "auto" and "tensorflow" in sys.modules)

    if want_torch:
        _seed_torch(seed, deterministic_ops)
    if want_tf:
        _seed_tf(seed, deterministic_ops)


def _seed_torch(seed: int, deterministic_ops: bool) -> None:
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_ops:
        # CUBLAS reads this when the CUDA context is created, so setting it
        # after training has begun does nothing for the current process. Set it
        # anyway: it is correct for the next one.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _seed_tf(seed: int, deterministic_ops: bool) -> None:
    if deterministic_ops:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"

    import tensorflow as tf

    tf.random.set_seed(seed)
    if deterministic_ops and hasattr(tf.config.experimental, "enable_op_determinism"):
        tf.config.experimental.enable_op_determinism()


def describe_device() -> str:
    """One line naming the compute device, for run manifests and notebook headers."""
    if DEVICE is None:
        return "torch not installed"
    if DEVICE.type == "cuda":
        import torch

        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"cuda: {name} ({total:.1f} GB), torch {torch.__version__}"

    import torch

    return (f"cpu, torch {torch.__version__} "
            f"(no CUDA device visible to torch)")


def release_memory(backend: str = "auto") -> None:
    """Drop cached allocations between configurations.

    Torch has no global graph the way Keras did, so this matters far less than
    the Keras `clear_session` it replaces. It is kept because the probe-model
    cache holds one live model per architecture and the CUDA caching allocator
    retains freed blocks, and this machine runs with very little headroom.
    """
    import gc

    if backend in ("auto", "torch", "both") and "torch" in sys.modules:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if backend in ("auto", "tf", "both") and "tensorflow" in sys.modules:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    print(describe_device())
    set_global_seeds(0, backend="torch")
    print("[ok] seeded python, numpy and torch")
