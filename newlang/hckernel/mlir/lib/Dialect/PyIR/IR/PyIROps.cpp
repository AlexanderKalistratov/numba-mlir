// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>

#include <llvm/ADT/TypeSwitch.h>

void hc::py_ir::PyIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hc/Dialect/PyIR/IR/PyIROps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "hc/Dialect/PyIR/IR/PyIROpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "hc/Dialect/PyIR/IR/PyIROpsAttributes.cpp.inc"
      >();
}

mlir::OpFoldResult hc::py_ir::ConstantOp::fold(FoldAdaptor /*adaptor*/) {
  return getValue();
}

void hc::py_ir::PyFuncOp::build(::mlir::OpBuilder &odsBuilder,
                                ::mlir::OperationState &odsState,
                                mlir::Type resultType, llvm::StringRef name,
                                llvm::ArrayRef<llvm::StringRef> argNames,
                                mlir::ValueRange annotations,
                                mlir::ValueRange decorators) {
  odsState.addAttribute(getNameAttrName(odsState.name),
                        odsBuilder.getStringAttr(name));
  odsState.addAttribute(getArgNamesAttrName(odsState.name),
                        odsBuilder.getStrArrayAttr(argNames));
  odsState.addOperands(annotations);
  odsState.addOperands(decorators);
  odsState.addTypes(resultType);

  int32_t segmentSizes[2] = {};
  segmentSizes[0] = static_cast<int32_t>(annotations.size());
  segmentSizes[1] = static_cast<int32_t>(decorators.size());
  odsState.addAttribute(getOperandSegmentSizeAttr(),
                        odsBuilder.getDenseI32ArrayAttr(segmentSizes));

  mlir::Region *region = odsState.addRegion();

  mlir::OpBuilder::InsertionGuard g(odsBuilder);

  llvm::SmallVector<mlir::Type> types(
      annotations.size(), UndefinedType::get(odsBuilder.getContext()));
  llvm::SmallVector<mlir::Location> locs(annotations.size(),
                                         odsBuilder.getUnknownLoc());
  odsBuilder.createBlock(region, {}, types, locs);
}

bool hc::py_ir::CastOp::areCastCompatible(mlir::TypeRange inputs,
                                          mlir::TypeRange outputs) {
  (void)inputs;
  (void)outputs;
  assert(inputs.size() == 1 && "expected one input");
  assert(outputs.size() == 1 && "expected one output");
  return true;
}

static bool parseArgList(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &argsOperands,
    mlir::ArrayAttr &args_namesAttr) {
  if (parser.parseLParen())
    return true;

  auto *context = parser.getContext();
  llvm::SmallVector<mlir::Attribute> names;
  if (parser.parseOptionalRParen()) {
    std::string name;
    while (true) {
      name.clear();
      if (!parser.parseOptionalKeywordOrString(&name)) {
        if (parser.parseColon())
          return true;
      }
      names.push_back(mlir::StringAttr::get(context, name));

      argsOperands.push_back({});
      if (parser.parseOperand(argsOperands.back()))
        return true;

      if (!parser.parseOptionalRParen())
        break;

      if (parser.parseComma())
        return true;
    }
  }

  assert(names.size() == argsOperands.size());
  args_namesAttr = mlir::ArrayAttr::get(context, names);
  return false;
}

template <typename Op>
static void printArgList(mlir::OpAsmPrinter &printer, Op /*op*/,
                         mlir::ValueRange args, mlir::ArrayAttr argsNames) {
  assert(args.size() == argsNames.size());
  printer << '(';
  bool first = true;
  for (auto &&[arg, name] : llvm::zip(args, argsNames)) {
    if (first) {
      first = false;
    } else {
      printer << ", ";
    }
    auto nameStr =
        (name ? name.cast<mlir::StringAttr>().getValue() : llvm::StringRef());
    if (!nameStr.empty())
      printer << nameStr << ':';
    printer.printOperand(arg);
  }
  printer << ')';
}

#include "hc/Dialect/PyIR/IR/PyIROpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "hc/Dialect/PyIR/IR/PyIROps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/Dialect/PyIR/IR/PyIROpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "hc/Dialect/PyIR/IR/PyIROpsTypes.cpp.inc"

#include "hc/Dialect/PyIR/IR/PyIROpsEnums.cpp.inc"
