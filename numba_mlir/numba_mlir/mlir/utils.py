# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import atexit
import ctypes
import numba_mlir
import os
import sys
import warnings
import llvmlite.binding as ll
from .. import mlir_compiler


def load_lib(name):
    runtime_search_paths = [os.path.dirname(numba_mlir.__file__)]

    try:
        runtime_search_paths += os.environ["PYTHONPATH"].split(os.pathsep)
    except KeyError:
        pass

    if sys.platform.startswith("linux"):
        lib_name = f"lib{name}.so"
    elif sys.platform.startswith("darwin"):
        lib_name = f"lib{name}.dylib"
    elif sys.platform.startswith("win"):
        lib_name = f"{name}.dll"
    else:
        assert False, "unsupported platform"

    saved_errors = []
    for path in runtime_search_paths:
        lib_path = lib_name if len(path) == 0 else os.path.join(path, lib_name)
        try:
            return ctypes.CDLL(lib_path)
        except Exception as e:
            saved_errors.append(f'CDLL("{lib_path}"): {str(e)}')

    raise ValueError(f'load_lib("{name}") failed:\n' + "\n".join(saved_errors))


def mlir_func_name(name):
    return "_mlir_ciface_" + name


_registered_cfuncs = []
_compiler_context = None


def _get_compiler_context():
    global _compiler_context
    if _compiler_context is None:
        from . import compiler_context

        _compiler_context = compiler_context.global_compiler_context

    return _compiler_context


def register_cfunc(name, cfunc):
    global _registered_cfuncs
    ptr = ctypes.cast(cfunc, ctypes.c_void_p)
    _registered_cfuncs.append(ptr)
    ll.add_symbol(name, ptr.value)
    mlir_compiler.register_symbol(_get_compiler_context(), name, ptr.value)


def readenv(name, ctor, default):
    value = os.environ.get(name)
    if value is None:
        return default() if callable(default) else default
    try:
        return ctor(value)
    except Exception:
        warnings.warn(
            "environ %s defined but failed to parse '%s'" % (name, value),
            RuntimeWarning,
        )
        return default
