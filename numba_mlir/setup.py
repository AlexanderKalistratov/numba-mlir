# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import subprocess
from setuptools import find_packages, setup
import versioneer
import numpy

root_dir = os.path.dirname(os.path.abspath(__file__))

# CMAKE =======================================================================

if int(os.environ.get("NUMBA_MLIR_SETUP_RUN_CMAKE", 1)):
    LLVM_PATH = os.environ["LLVM_PATH"]
    LLVM_DIR = os.path.join(LLVM_PATH, "lib", "cmake", "llvm")
    MLIR_DIR = os.path.join(LLVM_PATH, "lib", "cmake", "mlir")
    TBB_DIR = os.path.join(os.environ["TBB_PATH"], "lib", "cmake", "tbb")
    NUMBA_MLIR_USE_MKL = os.environ.get("NUMBA_MLIR_USE_MKL")
    CMAKE_INSTALL_PREFIX = os.path.join(root_dir, "..")

    cmake_build_dir = os.path.join(CMAKE_INSTALL_PREFIX, "numba_mlir_cmake_build")
    cmake_cmd = [
        "cmake",
        root_dir,
    ]

    cmake_cmd += ["-GNinja"]

    NUMPY_INCLUDE_DIR = numpy.get_include()

    cmake_cmd += [
        CMAKE_INSTALL_PREFIX,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLVM_DIR=" + LLVM_DIR,
        "-DMLIR_DIR=" + MLIR_DIR,
        "-DTBB_DIR=" + TBB_DIR,
        "-DCMAKE_INSTALL_PREFIX=" + CMAKE_INSTALL_PREFIX,
        "-DPython3_NumPy_INCLUDE_DIRS=" + NUMPY_INCLUDE_DIR,
        "-DPython3_FIND_STRATEGY=LOCATION",
        "-DNUMBA_MLIR_ENABLE_NUMBA_FE=ON",
        "-DNUMBA_MLIR_ENABLE_TBB_SUPPORT=ON",
        "-DLLVM_ENABLE_ZSTD=OFF",
    ]

    if NUMBA_MLIR_USE_MKL is not None:
        cmake_cmd += ["-DNUMBA_MLIR_USE_MKL=" + NUMBA_MLIR_USE_MKL]

    # DPNP
    try:
        from dpnp import get_include as dpnp_get_include

        DPNP_LIBRARY_DIR = os.path.join(dpnp_get_include(), "..", "..")
        DPNP_INCLUDE_DIR = dpnp_get_include()
        cmake_cmd += [
            "-DDPNP_LIBRARY_DIR=" + DPNP_LIBRARY_DIR,
            "-DDPNP_INCLUDE_DIR=" + DPNP_INCLUDE_DIR,
            "-DNUMBA_MLIR_USE_DPNP=ON",
        ]
        print("Found DPNP at", DPNP_LIBRARY_DIR)
    except ImportError:
        print("DPNP not found")

    # GPU/L0
    LEVEL_ZERO_DIR = os.getenv("LEVEL_ZERO_DIR", None)
    if LEVEL_ZERO_DIR is None:
        print("LEVEL_ZERO_DIR is not set")
    else:
        print("LEVEL_ZERO_DIR is", LEVEL_ZERO_DIR)
        cmake_cmd += [
            "-DNUMBA_MLIR_ENABLE_IGPU_DIALECT=ON",
        ]

    try:
        os.mkdir(cmake_build_dir)
    except FileExistsError:
        pass

    subprocess.check_call(
        cmake_cmd, stderr=subprocess.STDOUT, shell=False, cwd=cmake_build_dir
    )
    subprocess.check_call(
        ["cmake", "--build", ".", "--config", "Release"], cwd=cmake_build_dir
    )
    subprocess.check_call(
        ["cmake", "--install", ".", "--config", "Release"], cwd=cmake_build_dir
    )

# =============================================================================

packages = find_packages(where=root_dir, include=["numba_mlir", "numba_mlir.*"])

metadata = dict(
    name="numba-mlir",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=packages,
    install_requires=["numba>=0.56,<0.57"],
    include_package_data=True,
)

setup(**metadata)
