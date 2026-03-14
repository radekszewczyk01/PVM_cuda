"""
C/CUDA backend for PVM.

This backend delegates VRAM allocation, the training loop, forward/backward
passes and kernel launches to a native shared library (libpvm.so) via ctypes.

Build the library first:
    cd pvmcuda_pkg/backend_c && make

The module exposes the same public API as backend_python:
    PVM_object              the sequence-learning model
    Readout                 readout / classification head
    SyntheticDataProvider   synthetic stimulus generator
"""

import ctypes
import ctypes.util
import json
import os
import numpy as np

_LIB_NAME = "libpvm.so"
_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_LIB_DIR, _LIB_NAME)

_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.isfile(_LIB_PATH):
        raise FileNotFoundError(
            f"C/CUDA backend library not found at {_LIB_PATH}. "
            "Build it first: cd pvmcuda_pkg/backend_c && make"
        )
    _lib = ctypes.CDLL(_LIB_PATH)
    _setup_prototypes(_lib)
    return _lib


def _setup_prototypes(lib):
    """Declare argtypes and restypes for all API functions."""
    c_void_p = ctypes.c_void_p
    c_int = ctypes.c_int
    c_float = ctypes.c_float
    c_char_p = ctypes.c_char_p
    POINTER = ctypes.POINTER

    # -- PVM Object --
    lib.pvm_api_create.argtypes = [c_char_p, c_char_p]
    lib.pvm_api_create.restype = c_void_p

    lib.pvm_api_destroy.argtypes = [c_void_p]
    lib.pvm_api_destroy.restype = None

    lib.pvm_api_get_input_shape.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int), POINTER(c_int)]
    lib.pvm_api_get_input_shape.restype = c_int

    lib.pvm_api_push_input.argtypes = [c_void_p, POINTER(c_float), c_int, c_int, c_int]
    lib.pvm_api_push_input.restype = c_int

    lib.pvm_api_forward.argtypes = [c_void_p]
    lib.pvm_api_forward.restype = c_int

    lib.pvm_api_backward.argtypes = [c_void_p]
    lib.pvm_api_backward.restype = c_int

    lib.pvm_api_update_learning_rate.argtypes = [c_void_p, c_float]
    lib.pvm_api_update_learning_rate.restype = c_int

    lib.pvm_api_pop_prediction.argtypes = [c_void_p, POINTER(c_float), c_int]
    lib.pvm_api_pop_prediction.restype = c_int

    lib.pvm_api_pop_layer.argtypes = [c_void_p, ctypes.POINTER(ctypes.c_ubyte), c_int]
    lib.pvm_api_pop_layer.restype = c_int

    lib.pvm_api_freeze_learning.argtypes = [c_void_p]
    lib.pvm_api_freeze_learning.restype = c_int

    lib.pvm_api_unfreeze_learning.argtypes = [c_void_p]
    lib.pvm_api_unfreeze_learning.restype = c_int

    lib.pvm_api_save.argtypes = [c_void_p, c_char_p]
    lib.pvm_api_save.restype = c_int

    lib.pvm_api_load.argtypes = [c_char_p]
    lib.pvm_api_load.restype = c_void_p

    lib.pvm_api_get_step.argtypes = [c_void_p]
    lib.pvm_api_get_step.restype = c_int

    lib.pvm_api_set_step.argtypes = [c_void_p, c_int]
    lib.pvm_api_set_step.restype = None

    lib.pvm_api_get_name.argtypes = [c_void_p]
    lib.pvm_api_get_name.restype = c_char_p

    lib.pvm_api_get_uniq_id.argtypes = [c_void_p]
    lib.pvm_api_get_uniq_id.restype = c_char_p

    lib.pvm_api_get_device.argtypes = [c_void_p]
    lib.pvm_api_get_device.restype = c_char_p

    lib.pvm_api_get_learning_rate.argtypes = [c_void_p]
    lib.pvm_api_get_learning_rate.restype = c_float

    lib.pvm_api_get_num_layers.argtypes = [c_void_p]
    lib.pvm_api_get_num_layers.restype = c_int

    lib.pvm_api_get_layer_shape.argtypes = [c_void_p, c_int]
    lib.pvm_api_get_layer_shape.restype = c_int

    lib.pvm_api_get_total_units.argtypes = [c_void_p]
    lib.pvm_api_get_total_units.restype = c_int

    lib.pvm_api_get_time_stamp.argtypes = [c_void_p]
    lib.pvm_api_get_time_stamp.restype = c_char_p

    lib.pvm_api_get_config_int.argtypes = [c_void_p, c_char_p]
    lib.pvm_api_get_config_int.restype = c_int

    lib.pvm_api_get_config_float.argtypes = [c_void_p, c_char_p]
    lib.pvm_api_get_config_float.restype = c_float

    lib.pvm_api_get_graph_length.argtypes = [c_void_p]
    lib.pvm_api_get_graph_length.restype = c_int

    lib.pvm_api_get_layer_ptr.argtypes = [c_void_p, c_int]
    lib.pvm_api_get_layer_ptr.restype = c_int

    lib.pvm_api_get_block_size.argtypes = [c_void_p, c_int]
    lib.pvm_api_get_block_size.restype = c_int

    # -- Readout Object --
    lib.readout_api_create.argtypes = [c_void_p, c_int, c_int]
    lib.readout_api_create.restype = c_void_p

    lib.readout_api_destroy.argtypes = [c_void_p]
    lib.readout_api_destroy.restype = None

    lib.readout_api_copy_data.argtypes = [c_void_p]
    lib.readout_api_copy_data.restype = c_int

    lib.readout_api_forward.argtypes = [c_void_p]
    lib.readout_api_forward.restype = c_int

    lib.readout_api_train.argtypes = [c_void_p, POINTER(c_float), c_int, c_int]
    lib.readout_api_train.restype = c_int

    lib.readout_api_get_heatmap.argtypes = [c_void_p, POINTER(c_float)]
    lib.readout_api_get_heatmap.restype = c_int

    lib.readout_api_update_learning_rate.argtypes = [c_void_p, c_float]
    lib.readout_api_update_learning_rate.restype = c_int

    lib.readout_api_set_pvm.argtypes = [c_void_p, c_void_p]
    lib.readout_api_set_pvm.restype = None

    lib.readout_api_save.argtypes = [c_void_p, c_char_p]
    lib.readout_api_save.restype = c_int

    lib.readout_api_load.argtypes = [c_char_p]
    lib.readout_api_load.restype = c_void_p

    lib.readout_api_get_shape.argtypes = [c_void_p]
    lib.readout_api_get_shape.restype = c_int

    lib.readout_api_mlp_set_learning_rate.argtypes = [c_void_p, c_float]
    lib.readout_api_mlp_set_learning_rate.restype = c_int


def _float_ptr(arr):
    """Get a ctypes float pointer from a contiguous numpy float32 array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _ubyte_ptr(arr):
    """Get a ctypes unsigned byte pointer from a contiguous numpy uint8 array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))


# ---------------------------------------------------------------------------
# Minimal MLP proxy so that manager.py can call ReadoutObject.mlp.set_learning_rate()
# ---------------------------------------------------------------------------

class _MLPProxy:
    """Proxy that delegates set_learning_rate to the C readout API."""

    def __init__(self, readout_handle, lib):
        self._handle = readout_handle
        self._lib = lib

    def set_learning_rate(self, rate):
        self._lib.readout_api_mlp_set_learning_rate(self._handle, ctypes.c_float(rate))


# ---------------------------------------------------------------------------
# Graph block proxy for manager.py compatibility
# ---------------------------------------------------------------------------

class _GraphBlockProxy:
    """Lightweight proxy for PVMObject.graph[i] dict access used by manager."""

    def __init__(self, pvm_handle, block_id, lib):
        self._handle = pvm_handle
        self._id = block_id
        self._lib = lib

    def __getitem__(self, key):
        if key == 'id':
            return self._id
        if key == 'size':
            return self._lib.pvm_api_get_block_size(self._handle, self._id)
        raise KeyError(key)


class _GraphProxy:
    """List-like proxy for PVMObject.graph that returns block proxies."""

    def __init__(self, pvm_handle, lib):
        self._handle = pvm_handle
        self._lib = lib
        self._length = lib.pvm_api_get_graph_length(pvm_handle)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx < 0:
                idx += self._length
            if idx < 0 or idx >= self._length:
                raise IndexError(f"graph index {idx} out of range")
            return _GraphBlockProxy(self._handle, idx, self._lib)
        raise TypeError(f"graph indices must be integers, not {type(idx).__name__}")


# ---------------------------------------------------------------------------
# PVM_object  --  drop-in replacement for backend_python.sequence_learner.PVM_object
# ---------------------------------------------------------------------------

class PVM_object:
    """C/CUDA backend PVM_object, matching the Python backend API."""

    def __init__(self, specs=None, name="pvm"):
        self._lib = _load_lib()
        self._handle = None
        self.specs = specs

        if specs is not None:
            config_json = json.dumps(specs).encode('utf-8')
            name_b = name.encode('utf-8') if isinstance(name, str) else name
            self._handle = self._lib.pvm_api_create(config_json, name_b)
            if not self._handle:
                raise RuntimeError("pvm_api_create returned NULL")
            self._sync_attrs()

    def _sync_attrs(self):
        """Pull scalar attributes from the C object for Python-side access."""
        raw = self._lib.pvm_api_get_name(self._handle)
        self.name = raw.decode('utf-8') if raw else "pvm"
        raw = self._lib.pvm_api_get_uniq_id(self._handle)
        self.uniq_id = raw.decode('utf-8') if raw else ""
        raw = self._lib.pvm_api_get_time_stamp(self._handle)
        self.time_stamp = raw.decode('utf-8') if raw else ""
        raw = self._lib.pvm_api_get_device(self._handle)
        self.device = raw.decode('utf-8') if raw else ""
        self.learning_rate = float(self._lib.pvm_api_get_learning_rate(self._handle))
        self.graph = _GraphProxy(self._handle, self._lib)

    @property
    def step(self):
        if self._handle:
            return self._lib.pvm_api_get_step(self._handle)
        return 0

    @step.setter
    def step(self, value):
        if self._handle:
            self._lib.pvm_api_set_step(self._handle, int(value))

    def get_input_shape(self):
        w = ctypes.c_int()
        h = ctypes.c_int()
        ch = ctypes.c_int()
        self._lib.pvm_api_get_input_shape(self._handle,
                                          ctypes.byref(w),
                                          ctypes.byref(h),
                                          ctypes.byref(ch))
        return (w.value, h.value, ch.value)

    def update_learning_rate(self, override_rate=None):
        rate = ctypes.c_float(-1.0) if override_rate is None else ctypes.c_float(override_rate)
        self._lib.pvm_api_update_learning_rate(self._handle, rate)
        self.learning_rate = float(self._lib.pvm_api_get_learning_rate(self._handle))

    def push_input_gpu(self, frame):
        if frame.dtype == np.uint8:
            frame = (frame.astype(np.float32) / 255.0)
        frame = np.ascontiguousarray(frame, dtype=np.float32)
        h, w = frame.shape[0], frame.shape[1]
        ch = frame.shape[2] if frame.ndim == 3 else 1
        self._lib.pvm_api_push_input(self._handle, _float_ptr(frame), h, w, ch)

    def forward_gpu(self):
        self._lib.pvm_api_forward(self._handle)

    def backward_gpu(self):
        self._lib.pvm_api_backward(self._handle)

    def pop_prediction(self, delta_step=0):
        shape = self.get_input_shape()
        buf = np.empty((shape[0], shape[1], shape[2]), dtype=np.float32)
        self._lib.pvm_api_pop_prediction(self._handle, _float_ptr(buf), int(delta_step))
        return buf

    def pop_layer(self, layer=0):
        num_layers = self._lib.pvm_api_get_num_layers(self._handle)
        if layer >= num_layers:
            # Return empty for layers beyond what exists
            return np.zeros((1, 1), dtype=np.uint8)
        ls = self._lib.pvm_api_get_layer_shape(self._handle, layer)
        hbs = self._lib.pvm_api_get_config_int(self._handle, b"hidden_block_size")
        size = ls * hbs
        buf = np.zeros((size, size), dtype=np.uint8)
        self._lib.pvm_api_pop_layer(self._handle, _ubyte_ptr(buf), layer)
        return buf

    def freeze_learning(self):
        self._lib.pvm_api_freeze_learning(self._handle)

    def unfreeze_learning(self):
        self._lib.pvm_api_unfreeze_learning(self._handle)

    def save(self, outfile):
        path = outfile.encode('utf-8') if isinstance(outfile, str) else outfile
        self._lib.pvm_api_save(self._handle, path)

    def load(self, filename):
        path = filename.encode('utf-8') if isinstance(filename, str) else filename
        self._handle = self._lib.pvm_api_load(path)
        if not self._handle:
            raise RuntimeError(f"pvm_api_load failed for {filename}")
        self._sync_attrs()
        # Reconstruct specs dict from config
        if self.specs is None:
            self.specs = self._reconstruct_specs()

    def _reconstruct_specs(self):
        """Build a specs dict from the C-side config for manager.py compatibility."""
        specs = {}
        num_layers = self._lib.pvm_api_get_num_layers(self._handle)
        layer_shapes = []
        for i in range(num_layers):
            layer_shapes.append(str(self._lib.pvm_api_get_layer_shape(self._handle, i)))
        specs['layer_shapes'] = layer_shapes

        int_keys = ['input_block_size', 'hidden_block_size', 'lateral_radius',
                     'context_exclude_self', 'fan_in_square_size', 'fan_in_radius',
                     'feed_context_in_complex_layer', 'send_context_two_layers_back',
                     'last_layer_context_to_all', 'polynomial',
                     'delay_each_layer_learning', 'delay_final_learning_rate',
                     'delay_intermediate_learning_rate', 'opt_abs_diff', 'ignore_depth',
                     'input_channels']
        for k in int_keys:
            val = self._lib.pvm_api_get_config_int(self._handle, k.encode('utf-8'))
            specs[k] = str(val)

        float_keys = ['initial_learning_rate', 'final_learning_rate',
                       'intermediate_learning_rate', 'momentum']
        for k in float_keys:
            val = self._lib.pvm_api_get_config_float(self._handle, k.encode('utf-8'))
            specs[k] = str(val)

        return specs

    def display_unit(self, i, delta_step=0):
        """Debug display - not implemented in C backend."""
        pass

    def __del__(self):
        if hasattr(self, '_handle') and self._handle and hasattr(self, '_lib') and self._lib:
            self._lib.pvm_api_destroy(self._handle)
            self._handle = None


# ---------------------------------------------------------------------------
# Readout  --  drop-in replacement for backend_python.readout.Readout
# ---------------------------------------------------------------------------

class Readout:
    """C/CUDA backend Readout, matching the Python backend API."""

    def __init__(self, PVMObject=None, representation_size=100, heatmap_block_size=None):
        self._lib = _load_lib()
        self._handle = None
        self.PVMObject = PVMObject

        if PVMObject is not None:
            hbs = heatmap_block_size if heatmap_block_size is not None else 0
            self._handle = self._lib.readout_api_create(
                PVMObject._handle, int(representation_size), int(hbs))
            if not self._handle:
                raise RuntimeError("readout_api_create returned NULL")
            self.shape = self._lib.readout_api_get_shape(self._handle)
            self.mlp = _MLPProxy(self._handle, self._lib)

    def copy_data(self):
        self._lib.readout_api_copy_data(self._handle)

    def forward(self):
        self._lib.readout_api_forward(self._handle)

    def train(self, label):
        label = np.ascontiguousarray(label, dtype=np.float32)
        h, w = label.shape[0], label.shape[1]
        self._lib.readout_api_train(self._handle, _float_ptr(label), h, w)

    def get_heatmap(self):
        shape = self.shape
        buf = np.zeros((shape, shape), dtype=np.float32)
        self._lib.readout_api_get_heatmap(self._handle, _float_ptr(buf))
        return buf

    def update_learning_rate(self, override_rate=None):
        rate = ctypes.c_float(-1.0) if override_rate is None else ctypes.c_float(override_rate)
        self._lib.readout_api_update_learning_rate(self._handle, rate)

    def set_pvm(self, PVMObject):
        self.PVMObject = PVMObject
        self._lib.readout_api_set_pvm(self._handle, PVMObject._handle)
        self.shape = self._lib.readout_api_get_shape(self._handle)

    def save(self, outfile):
        path = outfile.encode('utf-8') if isinstance(outfile, str) else outfile
        self._lib.readout_api_save(self._handle, path)

    def load(self, filename):
        path = filename.encode('utf-8') if isinstance(filename, str) else filename
        self._handle = self._lib.readout_api_load(path)
        if not self._handle:
            return False
        self.shape = self._lib.readout_api_get_shape(self._handle)
        self.mlp = _MLPProxy(self._handle, self._lib)
        return True

    def __del__(self):
        if hasattr(self, '_handle') and self._handle and hasattr(self, '_lib') and self._lib:
            self._lib.readout_api_destroy(self._handle)
            self._handle = None


# ---------------------------------------------------------------------------
# SyntheticDataProvider  --  delegates to the Python version since it has
# no GPU code (just numpy/cv2 for generating synthetic stimuli).
# ---------------------------------------------------------------------------

class SyntheticDataProvider:
    """SyntheticDataProvider for the C backend.

    Since this class only generates CPU-side data (numpy arrays, cv2 shapes),
    we reuse the Python implementation directly.
    """

    def __init__(self, *args, **kwargs):
        from pvmcuda_pkg.backend_python.synthetic_data import SyntheticDataProvider as _PySDP
        self._impl = _PySDP(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._impl, name)
