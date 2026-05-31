"""
src/algorithms/__init__.py
==========================
Exposes the REGISTRY and the ``get_algorithm`` factory.

Importing this package triggers the registration of all algorithm subclasses.
"""

from src.algorithms.base import REGISTRY, BaseAlgorithm, ScalerBundle  # noqa: F401

# Import subclasses — side-effect: each registers itself in REGISTRY
from src.algorithms.pi_lstm   import PILSTMAlgorithm   # noqa: F401
from src.algorithms.neural_ode import NeuralODEAlgorithm  # noqa: F401

# Raman variants reuse the same classes but with different config keys
# (use_raman=true in their YAML), so they share the same REGISTRY name with
# a suffix for human-readable identification.  To make them loadable by name
# we create aliases:
REGISTRY["pi_lstm_raman"]    = PILSTMAlgorithm
REGISTRY["neural_ode_raman"] = NeuralODEAlgorithm

# REAKT+ encoder variants — same model classes, different Raman encoder and dim
REGISTRY["pi_lstm_v4"]    = PILSTMAlgorithm
REGISTRY["pi_lstm_v5"]    = PILSTMAlgorithm
REGISTRY["neural_ode_v4"] = NeuralODEAlgorithm
REGISTRY["neural_ode_v5"] = NeuralODEAlgorithm


def get_algorithm(name: str) -> type:
    """
    Return the algorithm class for ``name``.

    Parameters
    ----------
    name : str
        One of: ``"pi_lstm"``, ``"pi_lstm_raman"``, ``"pi_lstm_v4"``,
        ``"pi_lstm_v5"``, ``"neural_ode"``, ``"neural_ode_raman"``,
        ``"neural_ode_v4"``, ``"neural_ode_v5"``

    Raises
    ------
    KeyError if the name is not registered.
    """
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown algorithm '{name}'. "
            f"Available: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[name]


__all__ = [
    "REGISTRY",
    "BaseAlgorithm",
    "ScalerBundle",
    "PILSTMAlgorithm",
    "NeuralODEAlgorithm",
    "get_algorithm",
]
