// RUN: hc-opt -allow-unregistered-dialect -split-input-file %s --hc-convert-py-ast-to-ir-pass | FileCheck %s

// CHECK-LABEL: py_ir.module
py_ast.module {
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  %[[R:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK:  py_ir.return %[[R]] : !py_ir.undefined
py_ast.module {
  py_ast.func "func"() {
    %0 = py_ast.name "A"
    py_ast.return %0
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  %[[R:.*]] = py_ir.none
//       CHECK:  py_ir.return %[[R]] : none
py_ast.module {
  py_ast.func "func"() {
    %0 = py_ast.constant #py_ast.none
    py_ast.return %0
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[R:.*]] = py_ir.constant 42 : i64
//       CHECK:  py_ir.func "func"
//       CHECK:  py_ir.return %[[R]] : i64
py_ast.module {
  py_ast.func "func"() {
    %0 = py_ast.constant 42 : i64
    py_ast.return %0
  }
}


// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG1:.*]]: !py_ir.undefined, %[[ARG2:.*]]: !py_ir.undefined, %[[ARG3:.*]]: !py_ir.undefined):
//       CHECK:  py_ir.storevar "a" %[[ARG1]] : !py_ir.undefined
//       CHECK:  py_ir.storevar "b" %[[ARG2]] : !py_ir.undefined
//       CHECK:  py_ir.storevar "c" %[[ARG3]] : !py_ir.undefined
//       CHECK:  %[[R:.*]] = py_ir.none
//       CHECK:  py_ir.return %[[R]] : none
py_ast.module {
  %0 = py_ast.arg "a"
  %1 = py_ast.arg "b"
  %2 = py_ast.arg "c"
  py_ast.func "func"(%0, %1, %2) {
    %3 = py_ast.constant #py_ast.none
    py_ast.return %3
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG1:.*]]: !py_ir<ident "Foo">):
//       CHECK:  py_ir.storevar "a" %[[ARG1]] : !py_ir<ident "Foo">
//       CHECK:  %[[R:.*]] = py_ir.none
//       CHECK:  py_ir.return %[[R]] : none
py_ast.module {
  %0 = py_ast.name "Foo"
  %1 = py_ast.arg "a" : %0
  py_ast.func "func"(%1) {
    %3 = py_ast.constant #py_ast.none
    py_ast.return %3
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG1:.*]]: !py_ir<subscript !py_ir<ident "Foo">[!py_ir<ident "Bar">]>):
//       CHECK:  py_ir.storevar "a" %[[ARG1]] : !py_ir<subscript !py_ir<ident "Foo">[!py_ir<ident "Bar">]>
//       CHECK:  %[[R:.*]] = py_ir.none
//       CHECK:  py_ir.return %[[R]] : none
py_ast.module {
  %0 = py_ast.name "Foo"
  %1 = py_ast.name "Bar"
  %2 = py_ast.subscript %0 [%1]
  %3 = py_ast.arg "a" : %2
  py_ast.func "func"(%3) {
    %4 = py_ast.constant #py_ast.none
    py_ast.return %4
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG1:.*]]: !py_ir<const 42 : i64>):
//       CHECK:  py_ir.storevar "a" %[[ARG1]] : !py_ir<const 42 : i64>
//       CHECK:  %[[R:.*]] = py_ir.none
//       CHECK:  py_ir.return %[[R]] : none
py_ast.module {
  %0 = py_ast.constant 42 : i64
  %1 = py_ast.arg "a" : %0
  py_ast.func "func"(%1) {
    %3 = py_ast.constant #py_ast.none
    py_ast.return %3
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  %[[C1:.*]] = py_ir.loadvar "Cond" : !py_ir.undefined
//       CHECK:  %[[C2:.*]] = py_ir.cast %[[C1]] : !py_ir.undefined to i1
//       CHECK:  cf.cond_br %[[C2]], ^bb1, ^bb2
//       CHECK:  ^bb1
//       CHECK:  "test.test1"() : () -> ()
//       CHECK:  cf.br ^bb2
//       CHECK:  ^bb2
//       CHECK:  %[[R:.*]] = py_ir.none
//       CHECK:  py_ir.return %[[R]] : none
py_ast.module {
  py_ast.func "func"() {
    %0 = py_ast.name "Cond"
    py_ast.if %0 {
      "test.test1"() : () -> ()
    } {}
    %1 = py_ast.constant #py_ast.none
    py_ast.return %1
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  %[[C1:.*]] = py_ir.loadvar "Cond" : !py_ir.undefined
//       CHECK:  %[[C2:.*]] = py_ir.cast %[[C1]] : !py_ir.undefined to i1
//       CHECK:  cf.cond_br %[[C2]], ^bb1, ^bb2
//       CHECK:  ^bb1
//       CHECK:  "test.test1"() : () -> ()
//       CHECK:  cf.br ^bb3
//       CHECK:  ^bb2
//       CHECK:  "test.test2"() : () -> ()
//       CHECK:  cf.br ^bb3
//       CHECK:  ^bb3
//       CHECK:  %[[R:.*]] = py_ir.none
//       CHECK:  py_ir.return %[[R]] : none
py_ast.module {
  py_ast.func "func"() {
    %0 = py_ast.name "Cond"
    py_ast.if %0 {
      "test.test1"() : () -> ()
    } {
      "test.test2"() : () -> ()
    }
    %1 = py_ast.constant #py_ast.none
    py_ast.return %1
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  %[[V:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:  py_ir.storevar "A" %[[V]] : !py_ir.undefined
//       CHECK:  %[[R:.*]] = py_ir.none
//       CHECK:  py_ir.return %[[R]] : none
py_ast.module {
  py_ast.func "func"() {
    %0 = py_ast.name "A"
    %1 = py_ast.name "B"
    py_ast.assign(%0) = %1
    %2 = py_ast.constant #py_ast.none
    py_ast.return %2
  }
}
