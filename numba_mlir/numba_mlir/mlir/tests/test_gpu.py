# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from numpy.testing import assert_equal, assert_allclose
import numpy as np
import math
import numba
import itertools

from numba_mlir.mlir.utils import readenv
from numba_mlir.kernel import *
from numba_mlir.mlir.passes import (
    print_pass_ir,
    get_print_buffer,
    is_print_buffer_empty,
)

from .utils import JitfuncCache, parametrize_function_variants
from .utils import njit_cached as njit

kernel_cache = JitfuncCache(kernel)
kernel_cached = kernel_cache.cached_decorator

GPU_TESTS_ENABLED = readenv("NUMBA_MLIR_ENABLE_GPU_TESTS", int, 0)
DPCTL_TESTS_ENABLED = readenv("NUMBA_MLIR_ENABLE_DPCTL_TESTS", int, 0)

if DPCTL_TESTS_ENABLED:
    import dpctl
    import dpctl.tensor as dpt


def require_gpu(func):
    return pytest.mark.skipif(not GPU_TESTS_ENABLED, reason="GPU tests disabled")(func)


def require_dpctl(func):
    return pytest.mark.skipif(
        not DPCTL_TESTS_ENABLED, reason="DPCTL interop tests disabled"
    )(func)


_test_values = [
    True,
    False,
    -3,
    -2,
    -1,
    0,
    1,
    2,
    3,
    -2.5,
    -1.0,
    -0.5,
    -0.0,
    0.0,
    0.5,
    1.0,
    2.5,
]


@pytest.mark.smoke
@require_gpu
def test_simple1():
    def func(a, b, c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        c[i, j, k] = a[i, j, k] + b[i, j, k]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([[[1, 2, 3], [4, 5, 6]]], np.float32)
    b = np.array([[[7, 8, 9], [10, 11, 12]]], np.float32)

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@pytest.mark.smoke
@require_gpu
def test_simple2():
    get_id = get_global_id

    def func(a, b, c):
        i = get_id(0)
        j = get_id(1)
        k = get_id(2)
        c[i, j, k] = a[i, j, k] + b[i, j, k]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([[[1, 2, 3], [4, 5, 6]]], np.float32)
    b = np.array([[[7, 8, 9], [10, 11, 12]]], np.float32)

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
def test_simple3():
    def func(a, b):
        i = get_global_id(0)
        b[i, 0] = a[i, 0]
        b[i, 1] = a[i, 1]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([[1, 2], [3, 4], [5, 6]], np.float32)

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape[0], DEFAULT_LOCAL_SIZE](a, sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[a.shape[0], DEFAULT_LOCAL_SIZE](a, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("val", _test_values)
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_scalar(val, dtype):
    get_id = get_global_id

    def func(a, b, c):
        i = get_id(0)
        c[i] = a[i] + b

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.arange(-6, 6, dtype=dtype)

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, val, sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, val, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
def test_empty_kernel():
    def func(a):
        pass

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([[[1, 2, 3], [4, 5, 6]]], np.float32)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 0, ir


@require_gpu
def test_list_args():
    def func(a, b, c):
        i = get_global_id(0)
        c[i] = a[i] + b[i]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([1, 2, 3, 4, 5, 6], np.float32)
    b = np.array([7, 8, 9, 10, 11, 12], np.float32)

    sim_res = np.zeros(a.shape, a.dtype)

    dims = [a.shape[0]]
    sim_func[dims, []](a, b, sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[dims, []](a, b, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
def test_slice():
    def func(a, b):
        i = get_global_id(0)
        b1 = b[i]
        j = get_global_id(1)
        b2 = b1[j]
        k = get_global_id(2)
        b2[k] = a[i, j, k]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.arange(3 * 4 * 5).reshape((3, 4, 5))

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
def test_inner_loop():
    def func(a, b, c):
        i = get_global_id(0)
        res = 0.0
        for j in range(a[i]):
            res = res + b[j]
        c[i] = res

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([1, 2, 3, 4], np.int32)
    b = np.array([5, 6, 7, 8, 9], np.float32)

    sim_res = np.zeros(a.shape, b.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, sim_res)

    gpu_res = np.zeros(a.shape, b.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


def _test_unary(func, dtype, ir_pass, ir_check):
    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype)

    sim_res = np.zeros(a.shape, dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, sim_res)

    gpu_res = np.zeros(a.shape, dtype)

    with print_pass_ir([], [ir_pass]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, gpu_res)
        ir = get_print_buffer()
        assert ir_check(ir), ir

    assert_allclose(gpu_res, sim_res, rtol=1e-5)


def _test_binary(func, dtype, ir_pass, ir_check):
    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([11, 12, 13, 14, 15], dtype)
    b = np.array([1, 2, 3, 4, 5], dtype)

    sim_res = np.zeros(a.shape, dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, sim_res)

    gpu_res = np.zeros(a.shape, dtype)

    with print_pass_ir([], [ir_pass]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, gpu_res)
        ir = get_print_buffer()
        assert ir_check(ir), ir

    assert_allclose(gpu_res, sim_res, rtol=1e-5)


@require_gpu
@pytest.mark.parametrize("op", ["sqrt", "log", "sin", "cos", "floor"])
def test_math_funcs_unary(op):
    f = eval(f"math.{op}")

    def func(a, b):
        i = get_global_id(0)
        b[i] = f(a[i])

    _test_unary(
        func, np.float32, "GPUToSpirvPass", lambda ir: ir.count(f"CL.{op}") == 1
    )


@require_gpu
@pytest.mark.parametrize("op", ["+", "-", "*", "/", "//", "%", "**"])
@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_gpu_ops_binary(op, dtype):
    f = eval(f"lambda a, b: a {op} b")
    inner = kernel_func(f)

    def func(a, b, c):
        i = get_global_id(0)
        c[i] = inner(a[i], b[i])

    _test_binary(
        func,
        dtype,
        "ConvertParallelLoopToGpu",
        lambda ir: ir.count(f"gpu.launch blocks") == 1,
    )


_test_shapes = [
    (1,),
    (7,),
    (1, 1),
    (7, 13),
    (1, 1, 1),
    (7, 13, 23),
]


@require_gpu
@pytest.mark.parametrize("shape", _test_shapes)
def test_get_global_id(shape):
    def func1(c):
        i = get_global_id(0)
        c[i] = i

    def func2(c):
        i = get_global_id(0)
        j = get_global_id(1)
        c[i, j] = i + j * 100

    def func3(c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        c[i, j, k] = i + j * 100 + k * 10000

    func = [func1, func2, func3][len(shape) - 1]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    dtype = np.int32

    sim_res = np.zeros(shape, dtype)
    sim_func[shape, DEFAULT_LOCAL_SIZE](sim_res)

    gpu_res = np.zeros(shape, dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[shape, DEFAULT_LOCAL_SIZE](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("shape", _test_shapes)
@pytest.mark.parametrize("lsize", [(1, 1, 1), (2, 4, 8)])
def test_get_local_id(shape, lsize):
    def func1(c):
        i = get_global_id(0)
        li = get_local_id(0)
        c[i] = li

    def func2(c):
        i = get_global_id(0)
        j = get_global_id(1)
        li = get_local_id(0)
        lj = get_local_id(1)
        c[i, j] = li + lj * 100

    def func3(c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        li = get_local_id(0)
        lj = get_local_id(1)
        lk = get_local_id(2)
        c[i, j, k] = li + lj * 100 + lk * 10000

    func = [func1, func2, func3][len(shape) - 1]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    dtype = np.int32

    if len(lsize) > len(shape):
        lsize = tuple(lsize[: len(shape)])

    sim_res = np.zeros(shape, dtype)
    sim_func[shape, lsize](sim_res)

    gpu_res = np.zeros(shape, dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[shape, lsize](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("shape", _test_shapes)
@pytest.mark.parametrize("lsize", [(1, 1, 1), (2, 4, 8)])
def test_get_group_id(shape, lsize):
    def func1(c):
        i = get_global_id(0)
        li = get_group_id(0)
        c[i] = li

    def func2(c):
        i = get_global_id(0)
        j = get_global_id(1)
        li = get_group_id(0)
        lj = get_group_id(1)
        c[i, j] = li + lj * 100

    def func3(c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        li = get_group_id(0)
        lj = get_group_id(1)
        lk = get_group_id(2)
        c[i, j, k] = li + lj * 100 + lk * 10000

    func = [func1, func2, func3][len(shape) - 1]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    dtype = np.int32

    if len(lsize) > len(shape):
        lsize = tuple(lsize[: len(shape)])

    sim_res = np.zeros(shape, dtype)
    sim_func[shape, lsize](sim_res)

    gpu_res = np.zeros(shape, dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[shape, lsize](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("shape", _test_shapes)
def test_get_global_size(shape):
    def func1(c):
        i = get_global_id(0)
        w = get_global_size(0)
        c[i] = w

    def func2(c):
        i = get_global_id(0)
        j = get_global_id(1)
        w = get_global_size(0)
        h = get_global_size(1)
        c[i, j] = w + h * 100

    def func3(c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        w = get_global_size(0)
        h = get_global_size(1)
        d = get_global_size(2)
        c[i, j, k] = w + h * 100 + d * 10000

    func = [func1, func2, func3][len(shape) - 1]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    dtype = np.int32

    sim_res = np.zeros(shape, dtype)
    sim_func[shape, DEFAULT_LOCAL_SIZE](sim_res)

    gpu_res = np.zeros(shape, dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[shape, DEFAULT_LOCAL_SIZE](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("shape", _test_shapes)
@pytest.mark.parametrize("lsize", [(1, 1, 1), (2, 4, 8)])
def test_get_local_size(shape, lsize):
    def func1(c):
        i = get_global_id(0)
        w = get_local_size(0)
        c[i] = w

    def func2(c):
        i = get_global_id(0)
        j = get_global_id(1)
        w = get_local_size(0)
        h = get_local_size(1)
        c[i, j] = w + h * 100

    def func3(c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        w = get_local_size(0)
        h = get_local_size(1)
        d = get_local_size(2)
        c[i, j, k] = w + h * 100 + d * 10000

    func = [func1, func2, func3][len(shape) - 1]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    dtype = np.int32

    if len(lsize) > len(shape):
        lsize = tuple(lsize[: len(shape)])

    sim_res = np.zeros(shape, dtype)
    sim_func[shape, lsize](sim_res)

    gpu_res = np.zeros(shape, dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[shape, lsize](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


_atomic_dtypes = ["int32", "int64", "float32"]
_atomic_funcs = [atomic.add, atomic.sub]


def _check_atomic_ir(ir):
    return (
        ir.count("spirv.AtomicIAdd") == 1
        or ir.count("spirv.AtomicISub") == 1
        or ir.count("spirv.EXT.AtomicFAdd") == 1
    )


def _test_atomic(func, dtype, ret_size):
    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype)

    sim_res = np.zeros([ret_size], dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, sim_res)

    gpu_res = np.zeros([ret_size], dtype)

    with print_pass_ir([], ["GPUToSpirvPass"]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, gpu_res)
        ir = get_print_buffer()
        assert _check_atomic_ir(ir), ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("dtype", _atomic_dtypes)
@pytest.mark.parametrize("atomic_op", _atomic_funcs)
def test_atomics(dtype, atomic_op):
    def func(a, b):
        i = get_global_id(0)
        atomic_op(b, 0, a[i])

    _test_atomic(func, dtype, 1)


@require_gpu
@pytest.mark.xfail(reason="Only direct func calls work for now")
def test_atomics_modname():
    def func(a, b):
        i = get_global_id(0)
        atomic.add(b, 0, a[i])

    _test_atomic(func, "int32", 1)


@require_gpu
@pytest.mark.parametrize("dtype", _atomic_dtypes)
@pytest.mark.parametrize("atomic_op", _atomic_funcs)
def test_atomics_offset(dtype, atomic_op):
    def func(a, b):
        i = get_global_id(0)
        atomic_op(b, i % 2, a[i])

    _test_atomic(func, dtype, 2)


@require_gpu
@pytest.mark.parametrize("atomic_op", _atomic_funcs)
def test_atomics_different_types1(atomic_op):
    dtype = "int32"

    def func(a, b):
        i = get_global_id(0)
        atomic_op(b, 0, a[i] + 1)

    _test_atomic(func, dtype, 1)


@require_gpu
@pytest.mark.parametrize("atomic_op", _atomic_funcs)
def test_atomics_different_types2(atomic_op):
    dtype = "int32"

    def func(a, b):
        i = get_global_id(0)
        atomic_op(b, 0, 1)

    _test_atomic(func, dtype, 1)


@require_gpu
@pytest.mark.parametrize("funci", [1, 2])
def test_atomics_multidim(funci):
    atomic_op = atomic.add
    dtype = "int32"

    def func1(a, b):
        i = get_global_id(0)
        j = get_global_id(1)
        atomic_op(b, (i % 2, 0), a[i, j])

    def func2(a, b):
        i = get_global_id(0)
        j = get_global_id(1)
        atomic_op(b, (i % 2, j % 2), a[i, j])

    func = func1 if funci == 1 else func2

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype)

    sim_res = np.zeros((2, 2), dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, sim_res)

    gpu_res = np.zeros((2, 2), dtype)

    with print_pass_ir([], ["GPUToSpirvPass"]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, gpu_res)
        ir = get_print_buffer()
        assert _check_atomic_ir(ir), ir

    assert_equal(gpu_res, sim_res)


@require_gpu
def test_fastmath():
    def func(a, b, c, res):
        i = get_global_id(0)
        res[i] = a[i] * b[i] + c[i]

    sim_func = kernel_sim(func)
    a = np.array([1, 2, 3, 4], np.float32)
    b = np.array([5, 6, 7, 8], np.float32)
    c = np.array([9, 10, 11, 12], np.float32)

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, c, sim_res)

    with print_pass_ir([], ["GPUToSpirvPass"]):
        gpu_res = np.zeros(a.shape, a.dtype)
        gpu_func = kernel(fastmath=False)(func)
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, c, gpu_res)
        ir = get_print_buffer()
        assert ir.count("spirv.CL.fma") == 0, ir
        assert_equal(gpu_res, sim_res)

    with print_pass_ir([], ["GPUToSpirvPass"]):
        gpu_res = np.zeros(a.shape, a.dtype)
        gpu_func = kernel(fastmath=True)(func)
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, c, gpu_res)
        ir = get_print_buffer()
        assert ir.count("spirv.CL.fma") == 1, ir
        assert_equal(gpu_res, sim_res)


@require_gpu
def test_input_load_cse():
    def func(c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        c[i, j, k] = i + 10 * j + 100 * k

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.array([[[1, 2, 3], [4, 5, 6]]], np.float32)
    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](sim_res)

    gpu_res = np.zeros(a.shape, a.dtype)

    with print_pass_ir(["SerializeSPIRVPass"], []):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](gpu_res)
        ir = get_print_buffer()
        assert (
            ir.count(
                'spirv.Load "Input" %__builtin_var_GlobalInvocationId___addr : vector<3xi64>'
            )
            == 1
        ), ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("op", [barrier, mem_fence])
@pytest.mark.parametrize("flags", [CLK_LOCAL_MEM_FENCE, CLK_GLOBAL_MEM_FENCE])
@pytest.mark.parametrize("global_size", [1, 2, 27, 67, 101])
@pytest.mark.parametrize("local_size", [1, 2, 7, 17, 33])
def test_barrier_ops(op, flags, global_size, local_size):
    atomic_add = atomic.add

    def func(a, b):
        i = get_global_id(0)
        v = a[i]
        op(flags)
        b[i] = a[i]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.arange(global_size, dtype=np.int64)

    sim_res = np.zeros(global_size, a.dtype)
    sim_func[global_size, local_size](a, sim_res)

    gpu_res = np.zeros(global_size, a.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[global_size, local_size](a, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("global_size", [1, 2, 4, 27, 67, 101])
@pytest.mark.parametrize("local_size", [1, 2, 7, 17, 33])
def test_barrier1(global_size, local_size):
    atomic_add = atomic.add

    def func(a, b):
        i = get_global_id(0)
        off = i // local_size
        atomic_add(a, off, i)
        barrier(CLK_GLOBAL_MEM_FENCE)
        b[i] = a[off]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    count = (global_size + local_size - 1) // local_size
    a = np.array([0] * count, np.int64)

    sim_res = np.zeros(global_size, a.dtype)
    sim_func[global_size, local_size](a.copy(), sim_res)

    gpu_res = np.zeros(global_size, a.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[global_size, local_size](a.copy(), gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_equal(gpu_res, sim_res)


@require_gpu
@pytest.mark.parametrize("blocksize", [1, 10, 17, 64, 67, 101])
def test_local_memory(blocksize):
    local_array = local.array

    def func(A):
        lm = local_array(shape=blocksize, dtype=np.float32)
        i = get_global_id(0)

        # preload
        lm[i] = A[i]
        # barrier local or global will both work as we only have one work group
        barrier(CLK_LOCAL_MEM_FENCE)  # local mem fence
        # write
        A[i] += lm[blocksize - 1 - i]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    arr = np.arange(blocksize).astype(np.float32)

    sim_res = arr.copy()
    sim_func[blocksize, blocksize](sim_res)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_res = arr.copy()
        gpu_func[blocksize, blocksize](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_allclose(sim_res, gpu_res)


@require_gpu
@pytest.mark.parametrize("blocksize", [1, 10, 17, 64, 67, 101])
def test_private_memory(blocksize):
    private_array = private.array

    def func(A):
        i = get_global_id(0)
        prvt_mem = private_array(shape=1, dtype=np.float32)
        prvt_mem[0] = i
        barrier(CLK_LOCAL_MEM_FENCE)  # local mem fence
        A[i] = prvt_mem[0] * 2

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    arr = np.zeros(blocksize).astype(np.float32)

    sim_res = arr.copy()
    sim_func[blocksize, blocksize](sim_res)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_res = arr.copy()
        gpu_func[blocksize, blocksize](gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_allclose(sim_res, gpu_res)


@require_gpu
@pytest.mark.parametrize("group_op", [group.reduce_add])
@pytest.mark.parametrize("global_size", [1, 2, 4, 27, 67, 101])
@pytest.mark.parametrize("local_size", [1, 2, 7, 17, 33])
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32])
def test_group_func(group_op, global_size, local_size, dtype):
    def func(a, b):
        i = get_global_id(0)
        v = group_op(a[i])
        b[i] = v

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.arange(global_size, dtype=dtype)

    sim_res = np.zeros(global_size, a.dtype)
    sim_func[global_size, local_size](a, sim_res)

    gpu_res = np.zeros(global_size, a.dtype)

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[global_size, local_size](a, gpu_res)
        ir = get_print_buffer()
        assert ir.count("gpu.launch blocks") == 1, ir

    assert_allclose(gpu_res, sim_res)


def _from_host(arr, buffer):
    ret = dpt.usm_ndarray(arr.shape, dtype=arr.dtype, buffer=buffer)
    ret.usm_data.copy_from_host(arr.reshape((-1)).view("|u1"))
    return ret


def _to_host(src, dst):
    src.usm_data.copy_to_host(dst.reshape((-1)).view("|u1"))


@pytest.mark.smoke
@require_dpctl
def test_dpctl_simple1():
    def func(a, b, c):
        i = get_global_id(0)
        c[i] = a[i] + b[i]

    sim_func = kernel_sim(func)
    gpu_func = kernel_cached(func)

    a = np.arange(1024, dtype=np.float32)
    b = np.arange(1024, dtype=np.float32) * 3

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, sim_res)

    da = _from_host(a, buffer="device")
    db = _from_host(b, buffer="shared")

    gpu_res = np.zeros(a.shape, a.dtype)
    dgpu_res = _from_host(gpu_res, buffer="device")

    filter_string = dgpu_res.device.sycl_device.filter_string
    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func[a.shape, DEFAULT_LOCAL_SIZE](da, db, dgpu_res)
        ir = get_print_buffer()
        assert (
            ir.count(
                f'numba_util.env_region #gpu_runtime.region_desc<device = "{filter_string}">'
            )
            > 0
        ), ir
        assert ir.count("gpu.launch blocks") == 1, ir

    _to_host(dgpu_res, gpu_res)
    assert_equal(gpu_res, sim_res)


@pytest.mark.smoke
@require_dpctl
def test_parfor_simple1():
    def py_func(a, b, c):
        for i in numba.prange(len(a)):
            c[i] = a[i] + b[i]

    gpu_func = njit(py_func)

    a = np.arange(1024, dtype=np.float32)
    b = np.arange(1024, dtype=np.float32) * 3

    sim_res = np.zeros(a.shape, a.dtype)
    py_func(a, b, sim_res)

    da = _from_host(a, buffer="device")
    db = _from_host(b, buffer="shared")

    gpu_res = np.zeros(a.shape, a.dtype)
    dgpu_res = _from_host(gpu_res, buffer="device")

    filter_string = dgpu_res.device.sycl_device.filter_string
    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        gpu_func(da, db, dgpu_res)
        ir = get_print_buffer()
        assert (
            ir.count(
                f'numba_util.env_region #gpu_runtime.region_desc<device = "{filter_string}">'
            )
            > 0
        ), ir
        assert ir.count("gpu.launch blocks") == 1, ir

    _to_host(dgpu_res, gpu_res)
    assert_equal(gpu_res, sim_res)


@pytest.mark.smoke
@require_dpctl
def test_cfd_simple1():
    def py_func(a, b):
        b[:] = a * 2

    jit_func = njit(py_func)

    a = np.arange(1024, dtype=np.float32)

    sim_res = np.zeros(a.shape, a.dtype)
    py_func(a, sim_res)

    da = _from_host(a, buffer="device")

    gpu_res = np.zeros(a.shape, a.dtype)
    dgpu_res = _from_host(gpu_res, buffer="device")

    filter_string = dgpu_res.device.sycl_device.filter_string
    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        jit_func(da, dgpu_res)
        ir = get_print_buffer()
        assert (
            ir.count(
                f'numba_util.env_region #gpu_runtime.region_desc<device = "{filter_string}">'
            )
            > 0
        ), ir
        assert ir.count("gpu.launch blocks") == 1, ir

    _to_host(dgpu_res, gpu_res)
    assert_equal(gpu_res, sim_res)


@pytest.mark.smoke
@require_dpctl
def test_cfd_simple2():
    def py_func(a, b, c):
        c[:] = a + b

    jit_func = njit(py_func)

    a = np.arange(1024, dtype=np.float32)
    b = np.arange(1024, dtype=np.float32) * 3

    sim_res = np.zeros(a.shape, a.dtype)
    py_func(a, b, sim_res)

    da = _from_host(a, buffer="device")
    db = _from_host(b, buffer="shared")

    gpu_res = np.zeros(a.shape, a.dtype)
    dgpu_res = _from_host(gpu_res, buffer="device")

    filter_string = dgpu_res.device.sycl_device.filter_string
    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        jit_func(da, db, dgpu_res)
        ir = get_print_buffer()
        assert (
            ir.count(
                f'numba_util.env_region #gpu_runtime.region_desc<device = "{filter_string}">'
            )
            > 0
        ), ir
        assert ir.count("gpu.launch blocks") > 0, ir

    _to_host(dgpu_res, gpu_res)
    assert_equal(gpu_res, sim_res)


@require_dpctl
def test_cfd_indirect():
    def py_func1(a, b):
        b[:] = a * 2

    jit_func1 = njit(py_func1)

    def py_func2(a, b):
        jit_func1(a, b)

    jit_func2 = njit(py_func2)

    a = np.arange(1024, dtype=np.float32)

    sim_res = np.zeros(a.shape, a.dtype)
    py_func2(a, sim_res)

    da = _from_host(a, buffer="device")

    gpu_res = np.zeros(a.shape, a.dtype)
    dgpu_res = _from_host(gpu_res, buffer="device")

    filter_string = dgpu_res.device.sycl_device.filter_string
    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        jit_func2(da, dgpu_res)
        ir = get_print_buffer()
        assert (
            ir.count(
                f'numba_util.env_region #gpu_runtime.region_desc<device = "{filter_string}">'
            )
            > 0
        ), ir
        assert ir.count("gpu.launch blocks") > 0, ir

    _to_host(dgpu_res, gpu_res)
    assert_equal(gpu_res, sim_res)


@require_dpctl
def test_cfd_reshape():
    def py_func1(a):
        b = a.reshape(17, 23, 56)
        b[:] = 42

    jit_func1 = njit(py_func1)

    def py_func2(a):
        jit_func1(a)

    jit_func2 = njit(py_func2)

    a = np.arange(17 * 23 * 56, dtype=np.float32).reshape(23, 56, 17).copy()

    sim_res = np.zeros(a.shape, a.dtype)
    py_func2(sim_res)

    da = _from_host(a, buffer="device")

    gpu_res = np.zeros(a.shape, a.dtype)
    dgpu_res = _from_host(gpu_res, buffer="device")

    filter_string = dgpu_res.device.sycl_device.filter_string
    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        jit_func2(dgpu_res)
        ir = get_print_buffer()
        assert (
            ir.count(
                f'numba_util.env_region #gpu_runtime.region_desc<device = "{filter_string}">'
            )
            > 0
        ), ir
        assert ir.count("gpu.launch blocks") > 0, ir
    _to_host(dgpu_res, gpu_res)
    assert_equal(gpu_res, sim_res)


@pytest.mark.smoke
@require_dpctl
@pytest.mark.parametrize("size", [1, 7, 16, 64, 65, 256, 512, 1024 * 1024])
def test_cfd_reduce1(size):
    if size == 1:
        # TODO: Handle gpu array access outside the loops
        pytest.xfail()

    py_func = lambda a: a.sum()
    jit_func = njit(py_func)

    a = np.arange(size, dtype=np.float32)

    da = _from_host(a, buffer="device")

    filter_string = da.device.sycl_device.filter_string
    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        assert_allclose(jit_func(da), py_func(a), rtol=1e-5)
        ir = get_print_buffer()
        assert (
            ir.count(
                f'numba_util.env_region #gpu_runtime.region_desc<device = "{filter_string}">'
            )
            > 0
        ), ir
        assert ir.count("gpu.launch blocks") == 1, ir


_shapes = (1, 7, 16, 25, 64, 65)


@require_dpctl
@parametrize_function_variants(
    "py_func",
    [
        "lambda a: a.sum()",
        "lambda a: a.sum(axis=0)",
        "lambda a: a.sum(axis=1)",
    ],
)
@pytest.mark.parametrize("shape", itertools.product(_shapes, _shapes))
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32])
def test_cfd_reduce2(py_func, shape, dtype):
    if shape[0] == 1 or shape[1] == 1:
        # TODO: Handle gpu array access outside the loops
        pytest.xfail()

    jit_func = njit(py_func)

    a = np.arange(math.prod(shape), dtype=dtype).reshape(shape).copy()

    da = _from_host(a, buffer="device")
    assert_allclose(jit_func(da), py_func(a), rtol=1e-5)


@require_dpctl
@pytest.mark.skip(reason="Not implemented")
@pytest.mark.parametrize(
    "a,b",
    [
        (
            np.array([[1, 2, 3], [4, 5, 6]], np.float32),
            np.array([[1, 2], [3, 4], [5, 6]], np.float32),
        ),
    ],
)
@parametrize_function_variants(
    "py_func",
    [
        "lambda a, b, c: np.dot(a, b, c)",
        "lambda a, b, c: np.dot(a, b, out=c)",
    ],
)
def test_cfd_dot(a, b, py_func):
    jit_func = njit(py_func)

    tmp = np.dot(a, b)
    res_py = np.zeros_like(tmp)
    gpu_res = np.zeros_like(tmp)

    py_func(a, b, res_py)

    da = _from_host(a, buffer="device")
    db = _from_host(b, buffer="device")
    dgpu_res = _from_host(gpu_res, buffer="device")

    with print_pass_ir([], ["ConvertParallelLoopToGpu"]):
        jit_func(da, db, dgpu_res)
        ir = get_print_buffer()
        assert (
            ir.count(
                f'numba_util.env_region #gpu_runtime.region_desc<device = "{filter_string}">'
            )
            > 0
        ), ir
        assert ir.count("gpu.launch blocks") == 1, ir

    _to_host(dgpu_res, gpu_res)
    assert_equal(res_py, gpu_res)
