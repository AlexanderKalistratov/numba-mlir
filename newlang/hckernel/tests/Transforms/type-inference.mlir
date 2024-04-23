// RUN: hc-opt -split-input-file %s --hc-py-type-inference-pass | FileCheck %s


typing.type_resolver ["py_ir.loadvar", "CurrentGroup"] {
  %0 = typing.make_ident "CurrentGroup" []
  typing.type_resolver_return %0
}

//       CHECK: ![[ID:.*]] = !typing<ident "CurrentGroup">
// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG:.*]]: ![[ID]]):
//       CHECK:  %[[R:.*]] = py_ir.getattr %[[ARG]] : ![[ID]] attr "foo" -> !py_ir.undefined
//       CHECK:  py_ir.return %[[R]] : !py_ir.undefined

py_ir.module {
  %0 = py_ir.loadvar "CurrentGroup" : !py_ir.undefined
  %1 = py_ir.func "func" (group:%0) : !py_ir.undefined capture () -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined):
    %2 = py_ir.getattr %arg0 : !py_ir.undefined attr "foo" -> !py_ir.undefined
    py_ir.return %2 : !py_ir.undefined
  }
  %3 = py_ir.call %1 : !py_ir.undefined  () -> !py_ir.undefined
}
