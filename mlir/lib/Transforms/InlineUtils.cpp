// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Transforms/InlineUtils.hpp"

#include "numba/Dialect/numba_util/Dialect.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/InliningUtils.h>

namespace {
static bool mustInline(mlir::func::CallOp call, mlir::func::FuncOp func) {
  auto attr = mlir::StringAttr::get(
      call.getContext(), numba::util::attributes::getForceInlineName());
  return call->hasAttr(attr) || func->hasAttr(attr);
}

struct ForceInline : public mlir::OpRewritePattern<mlir::func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    assert(mod);

    auto func = mod.lookupSymbol<mlir::func::FuncOp>(op.getCallee());
    if (!func)
      return mlir::failure();

    if (!mustInline(op, func))
      return mlir::failure();

    auto loc = op.getLoc();
    auto reg =
        rewriter.create<mlir::scf::ExecuteRegionOp>(loc, op.getResultTypes());
    auto newCall = [&]() -> mlir::Operation * {
      auto &regBlock = reg.getRegion().emplaceBlock();
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&regBlock);
      auto call = rewriter.clone(*op);
      rewriter.create<mlir::scf::YieldOp>(loc, call->getResults());
      return call;
    }();

    mlir::InlinerInterface inlinerInterface(op->getContext());
    auto parent = op->getParentOp();
    rewriter.startRootUpdate(parent);
    auto res =
        mlir::inlineCall(inlinerInterface, newCall, func, &func.getRegion());
    if (mlir::succeeded(res)) {
      assert(newCall->getUsers().empty());
      rewriter.eraseOp(newCall);
      rewriter.replaceOp(op, reg.getResults());
      rewriter.finalizeRootUpdate(parent);
    } else {
      rewriter.eraseOp(reg);
      rewriter.cancelRootUpdate(parent);
    }
    return res;
  }
};

struct ForceInlinePass
    : public mlir::PassWrapper<ForceInlinePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ForceInlinePass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  virtual mlir::LogicalResult initialize(mlir::MLIRContext *context) override {
    mlir::RewritePatternSet p(context);
    p.insert<ForceInline>(context);
    patterns = std::move(p);
    return mlir::success();
  }

  virtual void runOnOperation() override {
    auto mod = getOperation();
    (void)mlir::applyPatternsAndFoldGreedily(mod, patterns);

    mod->walk([&](mlir::func::CallOp call) {
      auto func = mod.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
      if (func && mustInline(call, func)) {
        call.emitError("Couldn't inline force-inline call");
        signalPassFailure();
      }
    });
  }

private:
  mlir::FrozenRewritePatternSet patterns;
};
} // namespace

std::unique_ptr<mlir::Pass> numba::createForceInlinePass() {
  return std::make_unique<ForceInlinePass>();
}
