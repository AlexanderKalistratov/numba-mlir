// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"

#include <functional>
#include <memory>

namespace mlir {
class ModuleOp;
}

namespace llvm {
template <typename T> class Expected;
class JITEventListener;

namespace orc {
class LLJIT;
class MangleAndInterner;
} // namespace orc
} // namespace llvm

namespace numba {
struct ExecutionEngineOptions {
  /// `jitCodeGenOptLevel`, when provided, is used as the optimization level for
  /// target code generation.
  llvm::Optional<llvm::CodeGenOpt::Level> jitCodeGenOptLevel = std::nullopt;

  /// If `enableObjectCache` is set, the JIT compiler will create one to store
  /// the object generated for the given module. The contents of the cache can
  /// be dumped to a file via the `dumpToObjectfile` method.
  bool enableObjectCache = false;

  /// If enable `enableGDBNotificationListener` is set, the JIT compiler will
  /// notify the llvm's global GDB notification listener.
  bool enableGDBNotificationListener = true;

  /// If `enablePerfNotificationListener` is set, the JIT compiler will notify
  /// the llvm's global Perf notification listener.
  bool enablePerfNotificationListener = true;

  /// Register symbols with this ExecutionEngine.
  std::function<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)> symbolMap;

  /// If `transformer` is provided, it will be called on the LLVM module during
  /// JIT-compilation and can be used, e.g., for reporting or optimization.
  std::function<llvm::Error(llvm::Module &)> transformer;

  /// If `lateTransformer` is provided, it will be called on the LLVM module
  /// just before final code generation and can be used, e.g., for reporting or
  /// optimization.
  std::function<llvm::Error(llvm::Module &)> lateTransformer;

  /// If `asmPrinter` is provided, it will be called to print resulted assembly
  /// just before final code generation.
  std::function<void(llvm::StringRef)> asmPrinter;
};

class ExecutionEngine {
  class SimpleObjectCache;

public:
  using ModuleHandle = void *;

  ExecutionEngine(ExecutionEngineOptions options);
  ~ExecutionEngine();

  /// Compiles given module, adds it to execution engine and run its contructors
  /// if any.
  llvm::Expected<ModuleHandle> loadModule(mlir::ModuleOp m);

  /// Runs module desctructors and removes it from execution engine.
  void releaseModule(ModuleHandle handle);

  /// Looks up the original function with the given name and returns a
  /// pointer to it. Propagates errors in case of failure.
  llvm::Expected<void *> lookup(ModuleHandle handle,
                                llvm::StringRef name) const;

  /// Dump object code to output file `filename`.
  void dumpToObjectFile(llvm::StringRef filename);

private:
  /// Ordering of llvmContext and jit is important for destruction purposes: the
  /// jit must be destroyed before the context.
  llvm::LLVMContext llvmContext;

  /// Underlying LLJIT.
  std::unique_ptr<llvm::orc::LLJIT> jit;

  /// Underlying cache.
  std::unique_ptr<SimpleObjectCache> cache;

  /// GDB notification listener.
  llvm::JITEventListener *gdbListener;

  /// Perf notification listener.
  llvm::JITEventListener *perfListener;

  /// Callback to get additional symbol definitions.
  std::function<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)> symbolMap;

  /// If `transformer` is provided, it will be called on the LLVM module during
  /// JIT-compilation and can be used, e.g., for reporting or optimization.
  std::function<llvm::Error(llvm::Module &)> transformer;

  /// Id for unique module name generation.
  int uniqueNameCounter = 0;
};
} // namespace numba
