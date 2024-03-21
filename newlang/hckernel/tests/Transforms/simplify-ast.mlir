// RUN: hc-opt -allow-unregistered-dialect -split-input-file %s --hc-simplify-ast-pass | FileCheck %s


// CHECK-LABEL: py_ast.func()
//       CHECK: py_ast.return
//   CHECK-NOT: py_ast.return
py_ast.module {
  py_ast.func() {
    py_ast.return
    py_ast.return
  }
}
