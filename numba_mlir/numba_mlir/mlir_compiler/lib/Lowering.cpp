// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Lowering.hpp"

#include <algorithm>
#include <array>
#include <unordered_map>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Complex/IR/Complex.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ExecutionEngine/Orc/Mangling.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/TargetSelect.h>

#include "numba/Dialect/gpu_runtime/IR/GpuRuntimeOps.hpp"
#include "numba/Dialect/ntensor/IR/NTensorOps.hpp"
#include "numba/Dialect/numba_util/Dialect.hpp"
#include "numba/Dialect/plier/Dialect.hpp"

#include "numba/Compiler/Compiler.hpp"
#include "numba/Compiler/PipelineRegistry.hpp"
#include "numba/ExecutionEngine/ExecutionEngine.hpp"
#include "numba/Utils.hpp"

#include "PyTypeConverter.hpp"
#include "pipelines/BasePipeline.hpp"
#include "pipelines/LowerToGpu.hpp"
#include "pipelines/LowerToGpuTypeConversion.hpp"
#include "pipelines/LowerToLlvm.hpp"
#include "pipelines/ParallelToTbb.hpp"
#include "pipelines/PlierToLinalg.hpp"
#include "pipelines/PlierToLinalgTypeConversion.hpp"
#include "pipelines/PlierToScf.hpp"
#include "pipelines/PlierToStd.hpp"
#include "pipelines/PlierToStdTypeConversion.hpp"
#include "pipelines/PreLowSimplifications.hpp"

namespace py = pybind11;
namespace {

class dummy_complex : public py::object {
public:
  static bool check_(py::handle h) {
    return h.ptr() != nullptr && PyComplex_Check(h.ptr());
  }
};

class CallbackOstream : public llvm::raw_ostream {
public:
  using Func = std::function<void(llvm::StringRef)>;

  CallbackOstream(Func func = nullptr)
      : raw_ostream(/*unbuffered=*/false), callback(std::move(func)), pos(0u) {}

  ~CallbackOstream() override { flush(); }

  void write_impl(const char *ptr, size_t size) override {
    if (callback)
      callback(llvm::StringRef(ptr, size));
    pos += size;
  }

  uint64_t current_pos() const override { return pos; }

  void setCallback(Func func) { callback = std::move(func); }

private:
  Func callback;
  uint64_t pos;
};

static llvm::SmallVector<std::pair<int, py::object>>
getBlocks(py::handle func) {
  llvm::SmallVector<std::pair<int, py::object>> ret;
  auto blocks = func.cast<py::dict>();
  ret.reserve(blocks.size());
  for (auto [id, block] : blocks)
    ret.emplace_back(id.cast<int>(), block.cast<py::object>());

  auto pred = [](auto lhs, auto rhs) { return lhs.first < rhs.first; };

  llvm::sort(ret, pred);
  return ret;
}

static py::list getBody(py::handle block) {
  return block.attr("body").cast<py::list>();
}

struct InstHandles {
  InstHandles() {
    auto mod = py::module::import("numba.core.ir");
    Assign = mod.attr("Assign");
    Del = mod.attr("Del");
    Return = mod.attr("Return");
    Branch = mod.attr("Branch");
    Jump = mod.attr("Jump");
    SetItem = mod.attr("SetItem");
    StaticSetItem = mod.attr("StaticSetItem");

    Arg = mod.attr("Arg");
    Expr = mod.attr("Expr");
    Var = mod.attr("Var");
    Const = mod.attr("Const");
    Global = mod.attr("Global");
    FreeVar = mod.attr("FreeVar");

    auto ops = py::module::import("operator");

    for (auto elem : llvm::zip(plier::getOperators(), opsHandles)) {
      auto name = std::get<0>(elem).name;
      if (py::hasattr(ops, name.data())) {
        std::get<1>(elem) = ops.attr(name.data());
      } else {
        llvm::SmallVector<char> storage;
        auto str = (name + "_").toNullTerminatedStringRef(storage);
        std::get<1>(elem) = ops.attr(str.data());
      }
    }
  }

  py::handle Assign;
  py::handle Del;
  py::handle Return;
  py::handle Branch;
  py::handle Jump;
  py::handle SetItem;
  py::handle StaticSetItem;

  py::handle Arg;
  py::handle Expr;
  py::handle Var;
  py::handle Const;
  py::handle Global;
  py::handle FreeVar;

  std::array<py::handle, plier::OperatorsCount> opsHandles;
};

struct PlierLowerer final {
  PlierLowerer(mlir::MLIRContext &context, PyTypeConverter &conv)
      : ctx(context), builder(&ctx), typeConverter(conv) {
    ctx.loadDialect<gpu_runtime::GpuRuntimeDialect>();
    ctx.loadDialect<mlir::func::FuncDialect>();
    ctx.loadDialect<numba::ntensor::NTensorDialect>();
    ctx.loadDialect<numba::util::NumbaUtilDialect>();
    ctx.loadDialect<plier::PlierDialect>();
  }

  mlir::func::FuncOp lower(const py::object &compilationContext,
                           mlir::ModuleOp mod, const py::object &funcIr) {
    auto newFunc = createFunc(compilationContext, mod);
    lowerFuncBody(funcIr);
    return newFunc;
  }

private:
  mlir::MLIRContext &ctx;
  mlir::OpBuilder builder;
  std::vector<mlir::Block *> blocks;
  std::unordered_map<int, mlir::Block *> blocksMap;
  InstHandles insts;
  mlir::func::FuncOp func;
  std::unordered_map<std::string, mlir::Value> varsMap;
  struct BlockInfo {
    struct PhiDesc {
      mlir::Block *destBlock = nullptr;
      std::string varName;
      unsigned argIndex = 0;
    };
    llvm::SmallVector<PhiDesc, 2> outgoingPhiNodes;
  };
  py::handle currentInstr;
  py::handle typemap;
  py::handle funcNameResolver;

  std::unordered_map<mlir::Block *, BlockInfo> blockInfos;

  PyTypeConverter &typeConverter;

  mlir::func::FuncOp createFunc(const py::object &compilationContext,
                                mlir::ModuleOp mod) {
    assert(!func);
    typemap = compilationContext["typemap"];
    funcNameResolver = compilationContext["resolve_func"];
    auto name = compilationContext["fnname"]().cast<std::string>();
    auto typ = getFuncType(compilationContext["fnargs"],
                           compilationContext["restype"]);
    func = mlir::func::FuncOp::create(builder.getUnknownLoc(), name, typ);
    if (compilationContext["fastmath"]().cast<bool>())
      func->setAttr(numba::util::attributes::getFastmathName(),
                    builder.getUnitAttr());

    if (compilationContext["force_inline"]().cast<bool>())
      func->setAttr(numba::util::attributes::getForceInlineName(),
                    builder.getUnitAttr());

    func->setAttr(numba::util::attributes::getOptLevelName(),
                  builder.getI64IntegerAttr(
                      compilationContext["opt_level"]().cast<int64_t>()));
    auto maxConcurrency = compilationContext["max_concurrency"]().cast<int>();
    if (maxConcurrency > 0)
      func->setAttr(numba::util::attributes::getMaxConcurrencyName(),
                    builder.getI64IntegerAttr(maxConcurrency));

    mod.push_back(func);
    return func;
  }

  mlir::Type getObjType(py::handle obj) const {
    if (auto type = typeConverter.convertType(ctx, obj))
      return type;

    numba::reportError(llvm::Twine("Unhandled type: ") +
                       py::str(obj).cast<std::string>());
  }

  mlir::Type getType(py::handle inst) const {
    auto type = typemap(inst);
    return getObjType(type);
  }

  void lowerFuncBody(py::handle funcIr) {
    auto irBlocks = getBlocks(funcIr.attr("blocks"));
    assert(!irBlocks.empty());
    blocks.reserve(irBlocks.size());
    for (auto [i, irBlock] : llvm::enumerate(irBlocks)) {
      auto block = (0 == i ? func.addEntryBlock() : func.addBlock());
      blocks.emplace_back(block);
      blocksMap[irBlock.first] = block;
    }

    for (auto [i, irBlock] : llvm::enumerate(irBlocks))
      lowerBlock(blocks[i], irBlock.second);

    fixupPhis();
  }

  void lowerBlock(mlir::Block *bb, py::handle irBlock) {
    assert(nullptr != bb);
    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(bb);
    for (auto it : getBody(irBlock))
      lowerInst(it);
  }

  void lowerInst(py::handle inst) {
    currentInstr = inst;
    if (py::isinstance(inst, insts.Assign)) {
      auto target = inst.attr("target");
      auto val = lowerAssign(inst, target);
      storevar(val, target);
    } else if (py::isinstance(inst, insts.SetItem)) {
      setitem(inst.attr("target"), inst.attr("index"), inst.attr("value"));
    } else if (py::isinstance(inst, insts.StaticSetItem)) {
      staticSetitem(inst.attr("target"), inst.attr("index"),
                    inst.attr("value"));
    } else if (py::isinstance(inst, insts.Del)) {
      delvar(inst.attr("value"));
    } else if (py::isinstance(inst, insts.Return)) {
      retvar(inst.attr("value"));
    } else if (py::isinstance(inst, insts.Branch)) {
      branch(inst.attr("cond"), inst.attr("truebr"), inst.attr("falsebr"));
    } else if (py::isinstance(inst, insts.Jump)) {
      jump(inst.attr("target"));
    } else {
      numba::reportError(llvm::Twine("lower_inst not handled: \"") +
                         py::str(inst.get_type()).cast<std::string>() + "\"");
    }
    currentInstr = nullptr;
  }

  mlir::Value lowerAssign(py::handle inst, py::handle target) {
    auto value = inst.attr("value");
    if (py::isinstance(value, insts.Arg)) {
      auto index = value.attr("index").cast<std::size_t>();
      auto args = func.getFunctionBody().front().getArguments();
      if (index >= args.size())
        numba::reportError(llvm::Twine("Invalid arg index: \"") +
                           llvm::Twine(index) + "\"");

      return args[index];
    }

    if (py::isinstance(value, insts.Expr))
      return lowerExpr(value);

    if (py::isinstance(value, insts.Var))
      return loadvar(value);

    if (py::isinstance(value, insts.Const))
      return getConst(value.attr("value"));

    if (py::isinstance(value, insts.Global) ||
        py::isinstance(value, insts.FreeVar)) {
      auto constVal = getConstOrNull(value.attr("value"));
      if (constVal)
        return constVal;
      auto name = value.attr("name").cast<std::string>();
      return builder.create<plier::GlobalOp>(getCurrentLoc(), name);
    }

    numba::reportError(llvm::Twine("lower_assign not handled: \"") +
                       py::str(value.get_type()).cast<std::string>() + "\"");
  }

  mlir::Value lowerExpr(py::handle expr) {
    auto op = expr.attr("op").cast<std::string>();
    using func_t = mlir::Value (PlierLowerer::*)(py::handle);
    const std::pair<mlir::StringRef, func_t> handlers[] = {
        {"binop", &PlierLowerer::lowerBinop},
        {"inplace_binop", &PlierLowerer::lowerInplceBinop},
        {"unary", &PlierLowerer::lowerUnary},
        {"cast", &PlierLowerer::lowerCast},
        {"call", &PlierLowerer::lowerCall},
        {"phi", &PlierLowerer::lowerPhi},
        {"build_tuple", &PlierLowerer::lowerBuildTuple},
        {"getitem", &PlierLowerer::lowerGetitem},
        {"static_getitem", &PlierLowerer::lowerStaticGetitem},
        {"getiter", &PlierLowerer::lowerSimple<plier::GetiterOp>},
        {"iternext", &PlierLowerer::lowerSimple<plier::IternextOp>},
        {"pair_first", &PlierLowerer::lowerSimple<plier::PairfirstOp>},
        {"pair_second", &PlierLowerer::lowerSimple<plier::PairsecondOp>},
        {"getattr", &PlierLowerer::lowerGetattr},
        {"exhaust_iter", &PlierLowerer::lowerExhaustIter},
    };
    for (auto &h : handlers)
      if (h.first == op)
        return (this->*h.second)(expr);

    numba::reportError(llvm::Twine("lower_expr not handled: \"") + op + "\"");
  }

  template <typename T> mlir::Value lowerSimple(py::handle inst) {
    auto value = loadvar(inst.attr("value"));
    return builder.create<T>(getCurrentLoc(), value);
  }

  mlir::Value lowerCast(py::handle inst) {
    auto value = loadvar(inst.attr("value"));
    auto resType = getType(currentInstr.attr("target"));
    return builder.create<plier::CastOp>(getCurrentLoc(), resType, value);
  }

  mlir::Value lowerGetitem(py::handle inst) {
    auto value = loadvar(inst.attr("value"));
    auto index = loadvar(inst.attr("index"));
    return builder.create<plier::GetItemOp>(getCurrentLoc(), value, index);
  }

  mlir::Value lowerStaticIndex(mlir::Location loc, py::handle obj) {
    if (obj.is_none()) {
      auto type = mlir::NoneType::get(builder.getContext());
      return builder.create<numba::util::UndefOp>(loc, type);
    }
    if (py::isinstance<py::int_>(obj)) {
      auto index = obj.cast<int64_t>();
      return builder.create<mlir::arith::ConstantIndexOp>(loc, index);
    }
    if (py::isinstance<py::slice>(obj)) {
      auto start = lowerStaticIndex(loc, obj.attr("start"));
      auto stop = lowerStaticIndex(loc, obj.attr("stop"));
      auto step = lowerStaticIndex(loc, obj.attr("step"));
      return builder.create<plier::BuildSliceOp>(loc, start, stop, step);
    }
    if (py::isinstance<py::iterable>(obj)) {
      auto len = py::len(obj);
      llvm::SmallVector<mlir::Value> args(len);
      llvm::SmallVector<mlir::Type> types(len);
      for (auto [i, val] : llvm::enumerate(obj)) {
        auto arg = lowerStaticIndex(loc, val);
        args[i] = arg;
        types[i] = arg.getType();
      }

      auto tupleType = builder.getTupleType(types);
      return builder.create<plier::BuildTupleOp>(loc, tupleType, args);
    }
    numba::reportError(llvm::Twine("Unhandled index type: ") +
                       py::str(obj.get_type()).cast<std::string>());
  }

  mlir::Value lowerStaticGetitem(py::handle inst) {
    auto value = loadvar(inst.attr("value"));
    auto loc = getCurrentLoc();
    auto indexVar = lowerStaticIndex(loc, inst.attr("index"));
    return builder.create<plier::GetItemOp>(loc, value, indexVar);
  }

  mlir::Value lowerBuildTuple(py::handle inst) {
    auto items = inst.attr("items").cast<py::list>();
    mlir::SmallVector<mlir::Value> args;
    for (auto item : items)
      args.push_back(loadvar(item));

    return builder.create<plier::BuildTupleOp>(getCurrentLoc(), args);
  }

  mlir::Value lowerPhi(py::handle expr) {
    auto incomingVals = expr.attr("incoming_values").cast<py::list>();
    auto incomingBlocks = expr.attr("incoming_blocks").cast<py::list>();
    assert(incomingVals.size() == incomingBlocks.size());

    auto currentBlock = builder.getBlock();
    assert(nullptr != currentBlock);

    auto argIndex = currentBlock->getNumArguments();
    auto loc = builder.getUnknownLoc();
    auto arg =
        currentBlock->addArgument(getType(currentInstr.attr("target")), loc);

    for (auto i : llvm::seq<size_t>(0, incomingVals.size())) {
      auto var = incomingVals[i].attr("name").cast<std::string>();
      auto block = blocksMap.find(incomingBlocks[i].cast<int>())->second;
      blockInfos[block].outgoingPhiNodes.push_back(
          {currentBlock, std::move(var), argIndex});
    }

    return arg;
  }

  mlir::Value lowerCall(py::handle expr) {
    auto pyPunc = expr.attr("func");
    auto func = loadvar(pyPunc);
    auto args = expr.attr("args").cast<py::list>();
    auto kws = expr.attr("kws").cast<py::list>();
    auto vararg = expr.attr("vararg");

    auto varargVar = (vararg.is_none() ? mlir::Value() : loadvar(vararg));

    mlir::SmallVector<mlir::Value> argsList;
    argsList.reserve(args.size());
    for (auto a : args)
      argsList.push_back(loadvar(a));

    mlir::SmallVector<std::pair<std::string, mlir::Value>> kwargsList;
    for (auto a : kws) {
      auto item = a.cast<py::tuple>();
      auto name = item[0];
      auto valName = item[1];
      kwargsList.push_back({name.cast<std::string>(), loadvar(valName)});
    }

    auto pyFuncName = funcNameResolver(typemap(pyPunc));
    if (pyFuncName.is_none())
      numba::reportError(llvm::Twine("Can't resolve function: ") +
                         py::str(typemap(pyPunc)).cast<std::string>());

    auto funcName = pyFuncName.cast<std::string>();

    return builder.create<plier::PyCallOp>(getCurrentLoc(), func, funcName,
                                           argsList, varargVar, kwargsList);
  }

  mlir::Value lowerBinop(py::handle expr) {
    auto op = expr.attr("fn");
    auto lhsName = expr.attr("lhs");
    auto rhsName = expr.attr("rhs");
    auto lhs = loadvar(lhsName);
    auto rhs = loadvar(rhsName);
    auto opName = resolveOp(op);
    return builder.create<plier::BinOp>(getCurrentLoc(), lhs, rhs, opName);
  }

  mlir::Value lowerInplceBinop(py::handle expr) {
    auto op = expr.attr("immutable_fn");
    auto lhsName = expr.attr("lhs");
    auto rhsName = expr.attr("rhs");
    auto lhs = loadvar(lhsName);
    auto rhs = loadvar(rhsName);
    auto opName = resolveOp(op);
    return builder.create<plier::BinOp>(getCurrentLoc(), lhs, rhs, opName);
  }

  mlir::Value lowerUnary(py::handle expr) {
    auto op = expr.attr("fn");
    auto valName = expr.attr("value");
    auto val = loadvar(valName);
    auto opName = resolveOp(op);
    return builder.create<plier::UnaryOp>(getCurrentLoc(), val, opName);
  }

  llvm::StringRef resolveOp(py::handle op) {
    for (auto elem : llvm::zip(plier::getOperators(), insts.opsHandles))
      if (op.is(std::get<1>(elem)))
        return std::get<0>(elem).op;

    numba::reportError(llvm::Twine("resolve_op not handled: \"") +
                       py::str(op).cast<std::string>() + "\"");
  }

  mlir::Value lowerGetattr(py::handle inst) {
    auto value = loadvar(inst.attr("value"));
    auto name = inst.attr("attr").cast<std::string>();
    return builder.create<plier::GetattrOp>(getCurrentLoc(), value, name);
  }

  mlir::Value lowerExhaustIter(py::handle inst) {
    auto value = loadvar(inst.attr("value"));
    auto count = inst.attr("count").cast<int64_t>();
    return builder.create<plier::ExhaustIterOp>(getCurrentLoc(), value, count);
  }

  void setitem(py::handle target, py::handle index, py::handle value) {
    builder.create<plier::SetItemOp>(getCurrentLoc(), loadvar(target),
                                     loadvar(index), loadvar(value));
  }

  void staticSetitem(py::handle target, py::handle index, py::handle value) {
    auto loc = getCurrentLoc();
    builder.create<plier::SetItemOp>(
        loc, loadvar(target), lowerStaticIndex(loc, index), loadvar(value));
  }

  void storevar(mlir::Value val, py::handle inst) {
    auto type = getType(inst);
    if (val.getDefiningOp()) {
      val.setType(type);
    } else {
      // TODO: unify
      val = builder.create<plier::CastOp>(getCurrentLoc(), type, val);
    }
    varsMap[inst.attr("name").cast<std::string>()] = val;
  }

  mlir::Value loadvar(py::handle inst) {
    auto it = varsMap.find(inst.attr("name").cast<std::string>());
    assert(varsMap.end() != it);
    return it->second;
  }

  void delvar(py::handle inst) {
    auto var = loadvar(inst);
    builder.create<plier::DelOp>(getCurrentLoc(), var);
  }

  void retvar(py::handle inst) {
    auto var = loadvar(inst);
    auto funcType = func.getFunctionType();
    auto retType = funcType.getResult(0);
    auto varType = var.getType();
    if (retType != varType)
      var = builder.create<plier::CastOp>(getCurrentLoc(), retType, var);

    builder.create<mlir::func::ReturnOp>(getCurrentLoc(), var);
  }

  void branch(py::handle cond, py::handle tr, py::handle fl) {
    auto c = loadvar(cond);
    auto trBlock = blocksMap.find(tr.cast<int>())->second;
    auto flBlock = blocksMap.find(fl.cast<int>())->second;
    auto condVal = builder.create<plier::CastOp>(
        getCurrentLoc(), mlir::IntegerType::get(&ctx, 1), c);
    builder.create<mlir::cf::CondBranchOp>(getCurrentLoc(), condVal, trBlock,
                                           flBlock);
  }

  void jump(py::handle target) {
    auto block = blocksMap.find(target.cast<int>())->second;
    builder.create<mlir::cf::BranchOp>(getCurrentLoc(), std::nullopt, block);
  }

  mlir::Value getConstOrNull(py::handle val) {
    auto getVal = [&](mlir::Attribute attr) {
      return builder.create<plier::ConstOp>(getCurrentLoc(), attr);
    };
    if (py::isinstance<py::int_>(val)) {
      auto type = mlir::IntegerType::get(builder.getContext(), 64,
                                         mlir::IntegerType::Signed);
      auto attr = builder.getIntegerAttr(type, val.cast<int64_t>());
      return getVal(attr);
    }

    if (py::isinstance<py::float_>(val))
      return getVal(builder.getF64FloatAttr(val.cast<double>()));

    if (py::isinstance<dummy_complex>(val)) {
      auto c = val.cast<std::complex<double>>();
      auto type = mlir::ComplexType::get(builder.getF64Type());
      auto attr = mlir::complex::NumberAttr::get(type, c.real(), c.imag());
      return getVal(attr);
    }

    if (py::isinstance<py::none>(val))
      return getVal(builder.getUnitAttr());

    return {};
  }

  mlir::Value getConst(py::handle val) {
    auto ret = getConstOrNull(val);
    if (!ret)
      numba::reportError(llvm::Twine("get_const unhandled type \"") +
                         py::str(val.get_type()).cast<std::string>() + "\"");
    return ret;
  }

  mlir::FunctionType getFuncType(py::handle fnargs, py::handle restype) {
    auto ret = getObjType(restype());
    llvm::SmallVector<mlir::Type> args;
    for (auto arg : fnargs())
      args.push_back(getObjType(arg));

    return mlir::FunctionType::get(&ctx, args, {ret});
  }

  mlir::Location getCurrentLoc() {
    return builder.getUnknownLoc(); // TODO
  }

  void fixupPhis() {
    auto buildArgList = [&](mlir::Block *block, auto &outgoingPhiNodes,
                            auto &list) {
      for (auto &o : outgoingPhiNodes) {
        if (o.destBlock == block) {
          auto argIndex = o.argIndex;
          if (list.size() <= argIndex)
            list.resize(argIndex + 1);

          auto it = varsMap.find(o.varName);
          assert(varsMap.end() != it);
          auto argType = block->getArgument(argIndex).getType();
          auto val = builder.create<plier::CastOp>(builder.getUnknownLoc(),
                                                   argType, it->second);
          list[argIndex] = val;
        }
      }
    };
    for (auto bb : blocks) {
      auto it = blockInfos.find(bb);
      if (blockInfos.end() != it) {
        auto &info = it->second;
        auto term = bb->getTerminator();
        if (nullptr == term)
          numba::reportError("broken ir: block without terminator");

        builder.setInsertionPointToEnd(bb);

        if (auto op = mlir::dyn_cast<mlir::cf::BranchOp>(term)) {
          auto dest = op.getDest();
          mlir::SmallVector<mlir::Value> args;
          buildArgList(dest, info.outgoingPhiNodes, args);
          op.erase();
          builder.create<mlir::cf::BranchOp>(builder.getUnknownLoc(), dest,
                                             args);
        } else if (auto op = mlir::dyn_cast<mlir::cf::CondBranchOp>(term)) {
          auto trueDest = op.getTrueDest();
          auto falseDest = op.getFalseDest();
          auto cond = op.getCondition();
          mlir::SmallVector<mlir::Value> trueArgs;
          mlir::SmallVector<mlir::Value> falseArgs;
          buildArgList(trueDest, info.outgoingPhiNodes, trueArgs);
          buildArgList(falseDest, info.outgoingPhiNodes, falseArgs);
          op.erase();
          builder.create<mlir::cf::CondBranchOp>(builder.getUnknownLoc(), cond,
                                                 trueDest, trueArgs, falseDest,
                                                 falseArgs);
        } else {
          numba::reportError(llvm::Twine("Unhandled terminator: ") +
                             term->getName().getStringRef());
        }
      }
    }
  }
};

numba::CompilerContext::Settings getSettings(py::handle settings,
                                             CallbackOstream &os) {
  numba::CompilerContext::Settings ret;
  ret.verify = settings["verify"].cast<bool>();
  ret.passStatistics = settings["pass_statistics"].cast<bool>();
  ret.passTimings = settings["pass_timings"].cast<bool>();
  ret.irDumpStderr = settings["ir_printing"].cast<bool>();
  ret.diagDumpStderr = settings["diag_printing"].cast<bool>();

  auto printBefore = settings["print_before"].cast<py::list>();
  auto printAfter = settings["print_after"].cast<py::list>();
  if (!printBefore.empty() || !printAfter.empty()) {
    auto callback = settings["print_callback"].cast<py::function>();
    auto getList = [](py::list src) {
      llvm::SmallVector<std::string, 1> res(src.size());
      for (auto [i, val] : llvm::enumerate(src))
        res[i] = py::str(val).cast<std::string>();

      return res;
    };
    os.setCallback([callback](llvm::StringRef text) {
      callback(py::str(text.data(), text.size()));
    });
    using S = numba::CompilerContext::Settings::IRPrintingSettings;
    ret.irPrinting = S{getList(printBefore), getList(printAfter), &os};
  }
  return ret;
}

struct ModuleSettings {
  bool enableGpuPipeline = false;
};

static void createPipeline(numba::PipelineRegistry &registry,
                           PyTypeConverter &converter,
                           const ModuleSettings &settings) {
  converter.addConversion(
      [](mlir::MLIRContext &ctx, py::handle obj) -> llvm::Optional<mlir::Type> {
        return plier::PyType::get(&ctx, py::str(obj).cast<std::string>());
      });

  registerBasePipeline(registry);

  registerLowerToLLVMPipeline(registry);

  registerPlierToScfPipeline(registry);

  populateStdTypeConverter(converter);
  registerPlierToStdPipeline(registry);

  populateArrayTypeConverter(converter);
  registerPlierToLinalgPipeline(registry);

  registerPreLowSimpleficationsPipeline(registry);

  registerParallelToTBBPipeline(registry);

  if (settings.enableGpuPipeline) {
#ifdef NUMBA_MLIR_ENABLE_IGPU_DIALECT
    populateGpuTypeConverter(converter);
    registerLowerToGPUPipeline(registry);
    // TODO(nbpatel): Add Gpu->GpuRuntime & GpuRuntimetoLlvm Transformation
#else
    numba::reportError("Numba-MLIR was compiled without GPU support");
#endif
  }
}

struct Module {
  mlir::MLIRContext context;
  numba::PipelineRegistry registry;
  mlir::ModuleOp module;
  PyTypeConverter typeConverter;

  Module(const ModuleSettings &settings) {
    createPipeline(registry, typeConverter, settings);
  }
};

static void runCompiler(Module &mod, const py::object &compilationContext) {
  auto &context = mod.context;
  auto &module = mod.module;
  auto &registry = mod.registry;

  CallbackOstream printStream;
  auto settings =
      getSettings(compilationContext["compiler_settings"], printStream);
  numba::CompilerContext compiler(context, settings, registry);
  compiler.run(module);
}

static auto getLLModulePrinter(py::handle printer) {
  return [func = printer.cast<py::function>()](llvm::Module &m) -> llvm::Error {
    std::string str;
    llvm::raw_string_ostream os(str);
    m.print(os, nullptr);
    os.flush();

    func(py::str(str));
    return llvm::Error::success();
  };
}

static auto getPrinter(py::handle printer) {
  return [func = printer.cast<py::function>()](llvm::StringRef str) {
    func(py::str(str.data(), str.size()));
  };
}

struct GlobalCompilerContext {
  GlobalCompilerContext(const py::dict &settings)
      : executionEngine(getOpts(settings)) {}

  llvm::llvm_shutdown_obj s;
  llvm::SmallVector<std::pair<std::string, void *>, 0> symbolList;
  numba::ExecutionEngine executionEngine;

private:
  numba::ExecutionEngineOptions getOpts(const py::dict &settings) const {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    numba::ExecutionEngineOptions opts;
    opts.symbolMap =
        [this](llvm::orc::MangleAndInterner m) -> llvm::orc::SymbolMap {
      llvm::orc::SymbolMap ret;
      for (auto [name, ptr] : symbolList) {
        auto jitPtr = llvm::JITEvaluatedSymbol::fromPointer(ptr);
        ret.insert({m(name), jitPtr});
      }
      return ret;
    };
    opts.jitCodeGenOptLevel = llvm::CodeGenOpt::Level::Aggressive;

    auto llvmPrinter = settings["llvm_printer"];
    if (!llvmPrinter.is_none())
      opts.transformer = getLLModulePrinter(llvmPrinter);

    auto optimizedPrinter = settings["optimized_printer"];
    if (!optimizedPrinter.is_none())
      opts.lateTransformer = getLLModulePrinter(optimizedPrinter);

    auto asmPrinter = settings["asm_printer"];
    if (!asmPrinter.is_none())
      opts.asmPrinter = getPrinter(asmPrinter);

    return opts;
  }
};
} // namespace

py::capsule initCompiler(py::dict settings) {
  auto debugType = settings["debug_type"].cast<py::list>();
  auto debugTypeSize = debugType.size();
  if (debugTypeSize != 0) {
    llvm::DebugFlag = true;
    llvm::BumpPtrAllocator alloc;
    auto types = alloc.Allocate<const char *>(debugTypeSize);
    llvm::StringSaver strSaver(alloc);
    for (size_t i = 0; i < debugTypeSize; ++i) {
      types[i] = strSaver.save(debugType[i].cast<std::string>()).data();
    }
    llvm::setCurrentDebugTypes(types, static_cast<unsigned>(debugTypeSize));
  }

  auto context = std::make_unique<GlobalCompilerContext>(settings);
  return py::capsule(context.release(), [](void *ptr) {
    delete static_cast<GlobalCompilerContext *>(ptr);
  });
}

template <typename T>
static bool getDictVal(py::dict &dict, const char *str, T &&def) {
  auto key = py::str(str);
  if (dict.contains(key))
    return dict[key].cast<T>();

  return def;
}

py::capsule createModule(py::dict settings) {
  ModuleSettings modSettings;
  modSettings.enableGpuPipeline =
      getDictVal(settings, "enable_gpu_pipeline", false);

  auto mod = std::make_unique<Module>(modSettings);
  {
    mlir::OpBuilder builder(&mod->context);
    mod->module = mlir::ModuleOp::create(builder.getUnknownLoc());
  }
  py::capsule capsule(mod.get(),
                      [](void *ptr) { delete static_cast<Module *>(ptr); });
  mod.release();
  return capsule;
}

py::capsule lowerFunction(const py::object &compilationContext,
                          const py::capsule &pyMod, const py::object &funcIr) {
  auto mod = static_cast<Module *>(pyMod);
  auto &context = mod->context;
  auto &module = mod->module;
  auto func = PlierLowerer(context, mod->typeConverter)
                  .lower(compilationContext, module, funcIr);
  return py::capsule(func.getOperation()); // no dtor, func owned by the module.
}

py::capsule compileModule(const py::capsule &compiler,
                          const py::object &compilationContext,
                          const py::capsule &pyMod) {
  auto context = static_cast<GlobalCompilerContext *>(compiler);
  assert(context);
  auto mod = static_cast<Module *>(pyMod);
  assert(mod);

  runCompiler(*mod, compilationContext);
  mlir::registerLLVMDialectTranslation(*mod->module->getContext());
  auto res = context->executionEngine.loadModule(mod->module);
  if (!res)
    numba::reportError(llvm::Twine("Failed to load MLIR module:\n") +
                       llvm::toString(res.takeError()));

  return py::capsule(static_cast<void *>(res.get()));
}

void registerSymbol(const py::capsule &compiler, const py::str &name,
                    const py::int_ &ptr) {
  auto context = static_cast<GlobalCompilerContext *>(compiler);
  assert(context);

  auto ptrValue = reinterpret_cast<void *>(ptr.cast<intptr_t>());
  context->symbolList.emplace_back(name.cast<std::string>(), ptrValue);
}

py::int_ getFunctionPointer(const py::capsule &compiler,
                            const py::capsule &module, py::str funcName) {
  auto context = static_cast<GlobalCompilerContext *>(compiler);
  assert(context);
  auto handle = static_cast<numba::ExecutionEngine::ModuleHandle *>(module);
  assert(handle);

  auto name = funcName.cast<std::string>();
  auto res = context->executionEngine.lookup(handle, name);
  if (!res)
    numba::reportError(llvm::Twine("Failed to get function pointer:\n") +
                       llvm::toString(res.takeError()));

  return py::int_(reinterpret_cast<intptr_t>(res.get()));
}

void releaseModule(const py::capsule &compiler, const py::capsule &module) {
  auto context = static_cast<GlobalCompilerContext *>(compiler);
  assert(context);
  auto handle = static_cast<numba::ExecutionEngine::ModuleHandle *>(module);
  assert(handle);

  context->executionEngine.releaseModule(handle);
}

py::str moduleStr(const py::capsule &pyMod) {
  auto mod = static_cast<Module *>(pyMod);
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  mod->module.print(ss);
  ss.flush();
  return py::str(ss.str());
}
