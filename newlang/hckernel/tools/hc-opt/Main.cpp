// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <hckernel/InitHCDialects.hpp>
#include <hckernel/InitHCPasses.hpp>

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  hckernel::registerAllPasses();

  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  hckernel::registerAllDialects(registry);
  registerAllExtensions(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "HC modular optimizer driver\n", registry));
}
