"""
C/CUDA backend for PVM.

This backend delegates VRAM allocation, the training loop, forward/backward
passes and kernel launches to a native shared library (libpvm.so) via ctypes.

Build the library first:
    cd pvmcuda_pkg/backend_c && make        # see Makefile / CMakeLists.txt

The module exposes the same public API as backend_python:
    PVM_object              the sequence-learning model
    Readout                 readout / classification head
    SyntheticDataProvider   synthetic stimulus generator
"""

import ctypes
import os

_LIB_NAME = "libpvm.so"
_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_LIB_DIR, _LIB_NAME)


def _load_lib():
    if not os.path.isfile(_LIB_PATH):
        raise FileNotFoundError(
            f"C/CUDA backend library not found at {_LIB_PATH}. "
            "Build it first (see backend_c/README)."
        )
    return ctypes.CDLL(_LIB_PATH)


# ---------------------------------------------------------------------------
# Stub classes – same interface as backend_python so run.py works unchanged.
# Replace the bodies once the native library is implemented.
# ---------------------------------------------------------------------------

class PVM_object:
    """Drop-in replacement for backend_python.sequence_learner.PVM_object."""

    def __init__(self, specs=None, name="pvm"):
        self._lib = _load_lib()
        self.specs = specs
        self.name = name
        # TODO: call into native lib to allocate VRAM, build the network, etc.
        raise NotImplementedError("C/CUDA backend is not yet implemented")

    def get_input_shape(self):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError


class Readout:
    """Drop-in replacement for backend_python.readout.Readout."""

    def __init__(self, PVMObject=None, representation_size=100, heatmap_block_size=None):
        raise NotImplementedError("C/CUDA backend is not yet implemented")

    def load(self, path):
        raise NotImplementedError

    def set_pvm(self, PVMObject):
        raise NotImplementedError


class SyntheticDataProvider:
    """Drop-in replacement for backend_python.synthetic_data.SyntheticDataProvider."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("C/CUDA backend is not yet implemented")
