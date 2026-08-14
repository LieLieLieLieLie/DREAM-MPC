"""DREAM-MPC reference implementation.

The controller is imported lazily so result-only plotting does not require the
numerical solver stack to be installed in the rendering environment.
"""

from .config import METHODS, ExperimentConfig

__all__ = ["METHODS", "ExperimentConfig", "DreamController"]


def __getattr__(name: str):
    if name == "DreamController":
        from .controllers import DreamController

        return DreamController
    raise AttributeError(name)
