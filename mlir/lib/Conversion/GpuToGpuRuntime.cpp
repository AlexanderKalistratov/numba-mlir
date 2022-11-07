// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "imex/Conversion/GpuToGpuRuntime.hpp"

#include "imex/Dialect/gpu_runtime/IR/GpuRuntimeOps.hpp"
#include "imex/Dialect/imex_util/Dialect.hpp"
#include "imex/Dialect/imex_util/Utils.hpp"
#include "imex/Transforms/ScalarOpsConversion.hpp"
#include "imex/Transforms/TypeConversion.hpp"

#include <mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h>
#include <mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h>
#include <mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h>
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h>
#include <mlir/Conversion/MathToSPIRV/MathToSPIRV.h>
#include <mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Target/SPIRV/Serialization.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/SmallBitVector.h>

namespace {
static mlir::gpu::Processor getProcessor(unsigned val) {
  const mlir::gpu::Processor mapping[] = {
      mlir::gpu::Processor::BlockX,  mlir::gpu::Processor::BlockY,
      mlir::gpu::Processor::BlockZ,  mlir::gpu::Processor::ThreadX,
      mlir::gpu::Processor::ThreadY, mlir::gpu::Processor::ThreadZ,
  };
  if (val >= std::size(mapping))
    return mlir::gpu::Processor::Sequential;

  return mapping[val];
}
struct ParallelLoopGPUMappingPass
    : public mlir::PassWrapper<ParallelLoopGPUMappingPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParallelLoopGPUMappingPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    func->walk([&](imex::util::EnvironmentRegionOp envOp) {
      if (!envOp.getEnvironment().isa<gpu_runtime::GPURegionDescAttr>())
        return;

      auto &region = envOp.getRegion();

      mlir::OpBuilder builder(&getContext());
      auto identityMap = builder.getDimIdentityMap();
      llvm::SmallVector<mlir::gpu::ParallelLoopDimMappingAttr> mapping;
      for (auto &op : llvm::make_early_inc_range(region.front())) {
        auto parallel = mlir::dyn_cast<mlir::scf::ParallelOp>(op);
        if (!parallel || parallel->hasAttr(mlir::gpu::getMappingAttrName()))
          continue;

        auto numLoops = parallel.getNumLoops();
        mapping.resize(numLoops);
        for (auto i : llvm::seq(0u, numLoops))
          mapping[i] = builder.getAttr<mlir::gpu::ParallelLoopDimMappingAttr>(
              getProcessor(i), identityMap, identityMap);

        if (mlir::failed(mlir::gpu::setMappingAttr(parallel, mapping))) {
          signalPassFailure();
          return;
        }
      }
    });
  }
};

struct InsertGPUAllocs
    : public mlir::PassWrapper<InsertGPUAllocs,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertGPUAllocs)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto &funcBody = func.getBody();
    if (funcBody.empty()) {
      return;
    } else if (!llvm::hasSingleElement(funcBody)) {
      func.emitError("Function must have exactly one block");
      signalPassFailure();
      return;
    }

    struct AccessType {
      mlir::Attribute env;
      bool hostRead = false;
      bool hostWrite = false;
      bool deviceRead = false;
      bool deviceWrite = false;
    };

    llvm::SmallMapVector<mlir::Operation *, AccessType, 8> gpuBufferAllocs;
    llvm::SmallMapVector<unsigned, AccessType, 8> gpuBufferParams;
    auto &aliases = getAnalysis<mlir::BufferViewFlowAnalysis>();

    auto getMemref = [](mlir::Operation *op)
        -> llvm::Optional<mlir::SmallVector<mlir::Value, 4>> {
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
        return {{load.getMemref()}};
      } else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
        return {{store.getMemref()}};
      } else if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
        mlir::SmallVector<mlir::Value, 4> ret;
        for (auto arg : call.operands()) {
          if (arg.getType().isa<mlir::MemRefType>())
            ret.emplace_back(arg);
        }
        return std::move(ret);
      } else {
        op->emitError("Uhhandled mem op in gpu region");
        return llvm::None;
      }
    };

    auto hasMemAccess = [](mlir::Operation *op) -> bool {
      if (auto memInterface =
              mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
        if (memInterface.hasEffect<mlir::MemoryEffects::Read>() ||
            memInterface.hasEffect<mlir::MemoryEffects::Write>())
          return true;
      }
      if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
        for (auto arg : call.operands()) {
          if (arg.getType().isa<mlir::MemRefType>())
            return true;
        }
      }
      return false;
    };

    auto gpuAccessibleArg = [&]() -> llvm::SmallVector<bool> {
      auto gpuAttr = func->getAttr(gpu_runtime::getGpuAccessibleAttrName())
                         .dyn_cast_or_null<mlir::ArrayAttr>();
      if (!gpuAttr)
        return {};

      auto range = gpuAttr.getAsValueRange<mlir::BoolAttr>();
      return {range.begin(), range.end()};
    }();

    auto isGpuAccessibleArg = [&](unsigned i) {
      if (gpuAccessibleArg.empty())
        return false;

      assert(i < gpuAccessibleArg.size());
      return gpuAccessibleArg[i];
    };

    if (func.walk([&](mlir::Operation *op) {
              if (!op->getParentOfType<mlir::gpu::LaunchOp>())
                return mlir::WalkResult::advance();

              if (!hasMemAccess(op))
                return mlir::WalkResult::advance();

              auto memref = getMemref(op);
              if (!memref)
                return mlir::WalkResult::interrupt();

              for (auto mem : *memref) {
                while (auto parentView =
                           mem.getDefiningOp<mlir::ViewLikeOpInterface>())
                  mem = parentView.getViewSource();

                for (auto alias : aliases.resolve(mem)) {
                  auto op = alias.getDefiningOp();
                  if (op) {
                    if (mlir::isa<mlir::scf::SCFDialect>(op->getDialect()) ||
                        mlir::isa<mlir::ViewLikeOpInterface,
                                  mlir::arith::SelectOp, mlir::func::CallOp>(
                            op))
                      // Ignore
                      continue;
                    if (mlir::isa<mlir::memref::AllocOp,
                                  mlir::memref::GetGlobalOp>(op)) {
                      gpuBufferAllocs.insert({op, {}});
                    } else {
                      op->emitError("Unhandled memref producer");
                      return mlir::WalkResult::interrupt();
                    }

                  } else {
                    auto block = alias.getParentBlock();
                    auto blockArgs = block->getArguments();
                    auto it = llvm::find(blockArgs, alias);
                    assert(it != blockArgs.end());
                    auto index = static_cast<unsigned>(it - blockArgs.begin());
                    if (!isGpuAccessibleArg(index))
                      gpuBufferParams.insert({index, {}});
                  }
                }
              }

              return mlir::WalkResult::advance();
            })
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }

    auto getEnv = [](mlir::Operation *op) -> mlir::FailureOr<mlir::Attribute> {
      assert(op && "Invalid op");
      auto region = op->getParentOfType<imex::util::EnvironmentRegionOp>();
      if (!region ||
          !region.getEnvironment().isa<gpu_runtime::GPURegionDescAttr>())
        return mlir::failure();

      return region.getEnvironment();
    };

    auto getAccessType =
        [&](mlir::Value memref) -> mlir::FailureOr<AccessType> {
      AccessType ret;
      for (auto mem : aliases.resolve(memref)) {
        for (auto user : mem.getUsers()) {
          if (mlir::isa<mlir::func::ReturnOp>(user)) {
            ret.hostRead = true;
            ret.hostWrite = true;
            continue;
          }

          if (auto copy = mlir::dyn_cast<mlir::memref::CopyOp>(user)) {
            if (copy.getSource() == mem)
              ret.hostRead = true;

            if (copy.getTarget() == mem)
              ret.hostWrite = true;

            continue;
          }

          if (auto memInterface =
                  mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
            bool onDevice = user->getParentOfType<mlir::gpu::LaunchOp>();
            if (memInterface.hasEffect<mlir::MemoryEffects::Read>())
              (onDevice ? ret.deviceRead : ret.hostRead) = true;

            if (memInterface.hasEffect<mlir::MemoryEffects::Write>())
              (onDevice ? ret.deviceWrite : ret.hostWrite) = true;

            if (onDevice) {
              auto env = getEnv(user);
              if (mlir::succeeded(env)) {

                assert(*env && "Invalid device");
                if (!ret.env) {
                  ret.env = *env;
                } else if (ret.env != *env) {
                  return user->emitError("Device conflict: ")
                         << ret.env << " and " << *env;
                }
              }
            }

            continue;
          }
          if (mlir::isa<mlir::func::CallOp>(user)) {
            bool onDevice = user->getParentOfType<mlir::gpu::LaunchOp>();
            (onDevice ? ret.deviceRead : ret.hostRead) = true;
            (onDevice ? ret.deviceWrite : ret.hostWrite) = true;

            if (onDevice) {
              auto env = getEnv(user);
              if (mlir::succeeded(env)) {

                assert(*env && "Invalid device");
                if (!ret.env) {
                  ret.env = *env;
                } else if (ret.env != *env) {
                  return user->emitError("Device conflict: ")
                         << ret.env << " and " << *env;
                }
              }
            }

            continue;
          }
        }
      }
      return ret;
    };

    for (auto &it : gpuBufferAllocs) {
      auto op = it.first;
      assert(op->getNumResults() == 1);
      auto access = getAccessType(op->getResult(0));
      if (mlir::failed(access))
        return signalPassFailure();

      it.second = *access;
      if (mlir::isa<mlir::memref::GetGlobalOp>(op))
        it.second.hostWrite = true;
    }

    auto &block = funcBody.front();
    for (auto &it : gpuBufferParams) {
      auto param = block.getArgument(it.first);
      auto access = getAccessType(param);
      if (mlir::failed(access))
        return signalPassFailure();

      it.second = *access;

      it.second.hostRead = true;
      it.second.hostWrite = true;
    }

    auto term = block.getTerminator();
    assert(term);

    llvm::SmallVector<mlir::Value> dims;
    llvm::SmallPtrSet<mlir::Operation *, 8> filter;
    mlir::OpBuilder builder(func);
    auto createGpuAlloc = [&](mlir::Value src, const AccessType &access) {
      auto loc = src.getLoc();
      filter.clear();
      dims.clear();
      auto memrefType = src.getType().cast<mlir::MemRefType>();
      auto rank = static_cast<unsigned>(memrefType.getRank());
      for (auto i : llvm::seq(0u, rank)) {
        if (memrefType.isDynamicDim(i)) {
          auto dimOp = builder.create<mlir::memref::DimOp>(loc, src, i);
          dims.push_back(dimOp);
          filter.insert(dimOp);
        }
      }

      auto allocType = memrefType;
      if (!allocType.getLayout().isIdentity())
        allocType = mlir::MemRefType::get(allocType.getShape(),
                                          allocType.getElementType(),
                                          allocType.getMemorySpace());

      bool hostShared = access.hostRead || access.hostWrite;
      auto results = imex::util::wrapEnvRegion(
                         builder, src.getLoc(), access.env, memrefType,
                         [&](mlir::OpBuilder &b, mlir::Location loc) {
                           auto gpuAlloc = builder.create<mlir::gpu::AllocOp>(
                               loc, allocType, /*asyncToken*/ nullptr,
                               /*asyncDependencies*/ llvm::None, dims,
                               /*symbolOperands*/ llvm::None, hostShared);
                           auto allocResult = gpuAlloc.getMemref();
                           if (allocType != memrefType)
                             allocResult = builder.create<mlir::memref::CastOp>(
                                 loc, memrefType, allocResult);

                           if (access.hostWrite && access.deviceRead) {
                             auto copy = builder.create<mlir::memref::CopyOp>(
                                 loc, src, allocResult);
                             filter.insert(copy);
                           }
                           return allocResult;
                         })
                         .front();

      src.replaceAllUsesExcept(results, filter);

      builder.setInsertionPoint(term);
      imex::util::wrapEnvRegion(
          builder, src.getLoc(), access.env, llvm::None,
          [&](mlir::OpBuilder &b, mlir::Location loc) {
            if (access.hostRead && access.deviceWrite)
              builder.create<mlir::memref::CopyOp>(loc, results, src);

            builder.create<mlir::gpu::DeallocOp>(loc, llvm::None, results);
            return llvm::None;
          });
    };

    for (auto [op, access] : gpuBufferAllocs) {
      if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
        builder.setInsertionPoint(alloc);
        bool hostShared = access.hostRead || access.hostWrite;
        auto results = imex::util::wrapEnvRegion(
            builder, op->getLoc(), access.env, alloc.getType(),
            [&](mlir::OpBuilder &b, mlir::Location loc) {
              auto gpuAlloc = builder.create<mlir::gpu::AllocOp>(
                  loc, alloc.getType(), /*asyncToken*/ nullptr,
                  /*asyncDependencies*/ llvm::None, alloc.getDynamicSizes(),
                  alloc.getSymbolOperands(), hostShared);
              return gpuAlloc.getResults();
            });
        alloc->replaceAllUsesWith(results);
        alloc.erase();
      } else if (auto getGlobal =
                     mlir::dyn_cast<mlir::memref::GetGlobalOp>(op)) {
        builder.setInsertionPointAfter(getGlobal);
        createGpuAlloc(getGlobal.getResult(), access);
      } else {
        llvm_unreachable("Invalid alloc type");
      }
    }

    for (auto it : gpuBufferParams) {
      auto param = block.getArgument(it.first);
      auto access = it.second;
      builder.setInsertionPointToStart(&block);
      createGpuAlloc(param, access);
    }
  }
};

struct ConvertGPUDeallocsPass
    : public mlir::PassWrapper<ConvertGPUDeallocsPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertGPUDeallocsPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto op = getOperation();

    mlir::OpBuilder builder(&getContext());
    op->walk([&](mlir::gpu::DeallocOp dealloc) {
      if (dealloc.getAsyncToken()) {
        dealloc->emitError("Cannot convert gpu.dealloc with async tokens");
        signalPassFailure();
        return;
      }
      builder.setInsertionPoint(dealloc);
      builder.create<mlir::memref::DeallocOp>(dealloc->getLoc(),
                                              dealloc.getMemref());
      dealloc->erase();
    });
  }
};

static void setInsertionPointToStart(mlir::OpBuilder &builder,
                                     mlir::Value val) {
  if (auto parentOp = val.getDefiningOp()) {
    builder.setInsertionPointAfter(parentOp);
  } else {
    builder.setInsertionPointToStart(val.getParentBlock());
  }
}

static mlir::Value getFlatIndex(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value memref, mlir::ValueRange indices) {
  auto memrefType = memref.getType().cast<mlir::MemRefType>();
  auto rank = static_cast<unsigned>(memrefType.getRank());
  assert(indices.size() == rank);
  if (memrefType.getLayout().isIdentity()) {
    auto shape = memrefType.getShape();
    auto expr =
        mlir::makeCanonicalStridedLayoutExpr(shape, builder.getContext());
    llvm::SmallVector<mlir::Value> applyOperands;
    if (rank != 0) {
      applyOperands.reserve(rank * 2);
      applyOperands.assign(indices.begin(), indices.end());
      mlir::OpBuilder::InsertionGuard g(builder);
      setInsertionPointToStart(builder, memref);
      mlir::Value size;
      for (auto i : llvm::seq(0u, rank - 1)) {
        auto dimInd = rank - i - 1;
        auto dim =
            builder.createOrFold<mlir::memref::DimOp>(loc, memref, dimInd);
        if (i != 0) {
          size = builder.createOrFold<mlir::arith::MulIOp>(loc, size, dim);
        } else {
          size = dim;
        }

        applyOperands.emplace_back(size);
      }
    }
    auto affineMap = mlir::AffineMap::get(
        rank, static_cast<unsigned>(applyOperands.size()) - rank, expr);
    assert(affineMap.getNumDims() == indices.size());
    return builder.createOrFold<mlir::AffineApplyOp>(loc, affineMap,
                                                     applyOperands);
  } else {
    auto affineMap = memrefType.getLayout().getAffineMap();
    assert(affineMap.getNumDims() == indices.size());
    llvm::SmallVector<mlir::Value> applyOperands;
    if (rank != 0) {
      mlir::OpBuilder::InsertionGuard g(builder);
      setInsertionPointToStart(builder, memref);
      applyOperands.reserve(rank * 2 + 1);
      applyOperands.assign(indices.begin(), indices.end());

      auto numSymbols = affineMap.getNumSymbols();
      if (numSymbols > 0) {
        applyOperands.emplace_back(
            builder.createOrFold<imex::util::ExtractMemrefMetadataOp>(loc,
                                                                      memref));
        --numSymbols;
        assert(numSymbols <= rank);
        for (auto i : llvm::seq(0u, numSymbols)) {
          applyOperands.emplace_back(
              builder.createOrFold<imex::util::ExtractMemrefMetadataOp>(
                  loc, memref, i));
        }
      }
    }
    return builder.createOrFold<mlir::AffineApplyOp>(loc, affineMap,
                                                     applyOperands);
  }
}

static mlir::Value getFlatIndex(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value memref,
                                llvm::ArrayRef<mlir::OpFoldResult> indices) {
  llvm::SmallVector<mlir::Value> vals(indices.size());
  for (auto it : llvm::enumerate(indices)) {
    auto i = it.index();
    auto val = it.value();
    if (auto attr = val.dyn_cast<mlir::Attribute>()) {
      auto ind = attr.cast<mlir::IntegerAttr>().getValue().getSExtValue();
      vals[i] = builder.create<mlir::arith::ConstantIndexOp>(loc, ind);
    } else {
      vals[i] = val.get<mlir::Value>();
    }
  }
  return getFlatIndex(builder, loc, memref, vals);
}

static mlir::Value getFlatMemref(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Value memref) {
  auto memrefType = memref.getType().cast<mlir::MemRefType>();
  auto resultType = mlir::MemRefType::get(mlir::ShapedType::kDynamicSize,
                                          memrefType.getElementType());
  mlir::OpBuilder::InsertionGuard g(builder);
  setInsertionPointToStart(builder, memref);
  mlir::OpFoldResult offset = builder.getIndexAttr(0);
  mlir::OpFoldResult size =
      builder.createOrFold<imex::util::UndefOp>(loc, builder.getIndexType());
  mlir::OpFoldResult stride = builder.getIndexAttr(1);
  return builder.createOrFold<mlir::memref::ReinterpretCastOp>(
      loc, resultType, memref, offset, size, stride);
}

static bool needFlatten(mlir::Value val) {
  auto type = val.getType().cast<mlir::MemRefType>();
  return !type.getLayout().isIdentity() || (type.getRank() > 1);
}

struct FlattenLoad : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->getParentOfType<mlir::gpu::LaunchOp>())
      return mlir::failure();

    auto memref = op.getMemref();
    if (!needFlatten(memref))
      return mlir::failure();

    auto loc = op.getLoc();
    auto flatIndex = getFlatIndex(rewriter, loc, memref, op.getIndices());
    auto flatMemref = getFlatMemref(rewriter, loc, memref);
    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, flatMemref,
                                                      flatIndex);
    return mlir::success();
  }
};

struct FlattenStore : public mlir::OpRewritePattern<mlir::memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->getParentOfType<mlir::gpu::LaunchOp>())
      return mlir::failure();

    auto memref = op.getMemref();
    if (!needFlatten(memref))
      return mlir::failure();

    auto loc = op.getLoc();
    auto flatIndex = getFlatIndex(rewriter, loc, memref, op.getIndices());
    auto flatMemref = getFlatMemref(rewriter, loc, memref);
    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, op.getValue(),
                                                       flatMemref, flatIndex);
    return mlir::success();
  }
};

struct FlattenSubview : public mlir::OpRewritePattern<mlir::memref::SubViewOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::SubViewOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->getParentOfType<mlir::gpu::LaunchOp>())
      return mlir::failure();

    auto memref = op.getSource();
    if (!needFlatten(memref))
      return mlir::failure();

    auto offsets = op.getMixedOffsets();
    auto sizes = op.getMixedSizes();
    auto strides = op.getMixedStrides();

    auto srcType = memref.getType().cast<mlir::MemRefType>();
    auto dstType = mlir::memref::SubViewOp::inferResultType(srcType, offsets,
                                                            sizes, strides)
                       .cast<mlir::MemRefType>();

    int64_t resultOffset; // TODO: remove
    llvm::SmallVector<int64_t, 4> resultStrides;
    if (mlir::failed(
            mlir::getStridesAndOffset(dstType, resultStrides, resultOffset)))
      return mlir::failure();

    auto loc = op.getLoc();
    mlir::OpFoldResult flatIndex = getFlatIndex(rewriter, loc, memref, offsets);
    mlir::OpFoldResult flatSize =
        rewriter.create<imex::util::UndefOp>(loc, rewriter.getIndexType())
            .getResult();
    mlir::OpFoldResult flatStride = rewriter.getIndexAttr(1);
    auto flatMemref = getFlatMemref(rewriter, loc, memref);
    auto flatMemrefType = flatMemref.getType().cast<mlir::MemRefType>();
    assert(flatMemrefType.getLayout().isIdentity());
    auto flatSubview = rewriter.createOrFold<mlir::memref::SubViewOp>(
        loc, flatMemref, flatIndex, flatSize, flatStride);
    auto dstFlatType = flatSubview.getType();
    if (dstFlatType != flatMemrefType)
      flatSubview = rewriter.createOrFold<mlir::memref::CastOp>(
          loc, dstFlatType, flatSubview);

    auto offset = rewriter.getIndexAttr(0);

    for (auto i : llvm::seq<size_t>(0, strides.size())) {
      if (mlir::ShapedType::isDynamicStrideOrOffset(resultStrides[i])) {
        auto stride = strides[i];
        if (auto c = stride.dyn_cast<mlir::Attribute>()) {
          auto val = c.dyn_cast<mlir::IntegerAttr>().getValue().getSExtValue();
          stride = rewriter.create<mlir::arith::ConstantIndexOp>(loc, val)
                       .getResult();
        }

        auto origStride = [&]() {
          mlir::OpBuilder::InsertionGuard g(rewriter);
          setInsertionPointToStart(rewriter, memref);
          return rewriter.createOrFold<imex::util::ExtractMemrefMetadataOp>(
              loc, memref, i);
        }();
        auto newStride = rewriter.createOrFold<mlir::arith::MulIOp>(
            loc, stride.get<mlir::Value>(), origStride);
        strides[i] = newStride;
      }
    }

    auto resultType = op.getType().cast<mlir::MemRefType>();
    auto srcRank = static_cast<unsigned>(srcType.getRank());
    auto resultRank = static_cast<unsigned>(resultType.getRank());
    mlir::Value result;
    if (srcRank == resultRank) {
      result = rewriter.createOrFold<mlir::memref::ReinterpretCastOp>(
          loc, resultType, flatSubview, offset, sizes, strides);
    } else {
      assert(resultRank < srcRank);
      llvm::SmallVector<mlir::OpFoldResult> filteredSizes;
      llvm::SmallVector<mlir::OpFoldResult> filteredStrides;
      filteredSizes.reserve(resultRank);
      filteredStrides.reserve(resultRank);

      auto droppedDims = op.getDroppedDims();
      for (auto i : llvm::seq(0u, srcRank)) {
        if (!droppedDims[i]) {
          filteredSizes.emplace_back(sizes[i]);
          filteredStrides.emplace_back(strides[i]);
        }
      }
      result = rewriter.createOrFold<mlir::memref::ReinterpretCastOp>(
          loc, resultType, flatSubview, offset, filteredSizes, filteredStrides);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct UnstrideMemrefsPass
    : public mlir::PassWrapper<UnstrideMemrefsPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnstrideMemrefsPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<imex::util::ImexUtilDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<FlattenLoad, FlattenStore, FlattenSubview>(ctx);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

static llvm::Optional<mlir::Value> getGpuStream(mlir::OpBuilder &builder,
                                                mlir::Operation *op) {
  assert(op);
  auto func = op->getParentOfType<mlir::FunctionOpInterface>();
  if (!func)
    return {};

  if (!llvm::hasSingleElement(func.getFunctionBody()))
    return {};

  mlir::Attribute device;
  if (auto envRegion = op->getParentOfType<imex::util::EnvironmentRegionOp>())
    if (auto desc = envRegion.getEnvironment()
                        .dyn_cast<gpu_runtime::GPURegionDescAttr>())
      device = desc.getDevice();

  auto &block = func.getFunctionBody().front();
  auto ops = block.getOps<gpu_runtime::CreateGpuStreamOp>();
  for (auto streamOp : ops)
    if (streamOp.getDeviceAttr() == device)
      return streamOp.getResult();

  mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(&block);
  auto loc = builder.getUnknownLoc();
  mlir::Value stream =
      builder.create<gpu_runtime::CreateGpuStreamOp>(loc, device);
  builder.setInsertionPoint(block.getTerminator());
  builder.create<gpu_runtime::DestroyGpuStreamOp>(loc, stream);
  return stream;
}

class ConvertSubviewOp
    : public mlir::OpConversionPattern<mlir::memref::SubViewOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::SubViewOp op,
                  mlir::memref::SubViewOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto dstType = op.getType().cast<mlir::MemRefType>();
    if (!dstType.hasRank() || dstType.getRank() != 1)
      return mlir::failure();

    auto intType = getTypeConverter()->convertType(rewriter.getIndexType());
    if (!intType)
      return mlir::failure();

    auto loc = op.getLoc();
    auto getValue = [&](mlir::OpFoldResult src) -> mlir::Value {
      if (auto val = src.dyn_cast<mlir::Value>())
        return val;

      auto attr = src.get<mlir::Attribute>();
      return rewriter.create<mlir::spirv::ConstantOp>(loc, intType, attr);
    };

    auto offset =
        getValue(op.isDynamicOffset(0)
                     ? mlir::OpFoldResult(adaptor.getOffsets()[0])
                     : mlir::OpFoldResult(adaptor.getStaticOffsets()[0]));
    auto stride =
        getValue(op.isDynamicStride(0)
                     ? mlir::OpFoldResult(adaptor.getStrides()[0])
                     : mlir::OpFoldResult(adaptor.getStaticStrides()[0]));
    auto finalOffset = rewriter.createOrFold<mlir::spirv::IMulOp>(
        loc, intType, offset, stride);

    auto ptr = rewriter
                   .create<mlir::spirv::InBoundsPtrAccessChainOp>(
                       loc, adaptor.getSource(), finalOffset, llvm::None)
                   .getResult();

    rewriter.replaceOp(op, ptr);
    return mlir::success();
  }
};

template <typename T>
class ConvertCastOp : public mlir::OpConversionPattern<T> {
public:
  using mlir::OpConversionPattern<T>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSource());
    return mlir::success();
  }
};

template <typename Op>
class ConvertBitcastOp : public mlir::OpConversionPattern<Op> {
public:
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = this->getTypeConverter();
    assert(converter && "Invalid type converter");

    auto resType = converter->convertType(op.getResult().getType());
    if (!resType)
      return mlir::failure();

    auto src = adaptor.getSource();
    auto srcType = src.getType();
    if (srcType == resType) {
      rewriter.replaceOp(op, src);
      return mlir::success();
    }

    rewriter.replaceOpWithNewOp<mlir::spirv::BitcastOp>(op, resType, src);
    return mlir::success();
  }
};

static llvm::Optional<unsigned> getTypeSize(mlir::Type type) {
  if (type.isIntOrFloat())
    return type.getIntOrFloatBitWidth() / 8;

  if (auto vec = type.dyn_cast<mlir::VectorType>()) {
    if (!vec.hasStaticShape())
      return llvm::None;

    auto elemSize = getTypeSize(vec.getElementType());
    if (!elemSize)
      return llvm::None;

    return static_cast<unsigned>(vec.getNumElements()) * *elemSize;
  }
  return llvm::None;
}

class ConvertLoadOp : public mlir::OpConversionPattern<mlir::memref::LoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::memref::LoadOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memrefType = op.getMemref().getType().cast<mlir::MemRefType>();
    auto typeSize = getTypeSize(memrefType.getElementType());
    if (!typeSize)
      return mlir::failure();

    if (memrefType.getRank() == 0) {
      auto memoryAccess = mlir::spirv::MemoryAccessAttr::get(
          op.getContext(), mlir::spirv::MemoryAccess::Aligned);
      auto alignment = rewriter.getI32IntegerAttr(*typeSize);
      rewriter.replaceOpWithNewOp<mlir::spirv::LoadOp>(op, adaptor.getMemref(),
                                                       memoryAccess, alignment);
      return mlir::success();
    } else if (memrefType.hasRank() && memrefType.getRank() == 1) {
      auto loc = op.getLoc();
      auto ptr = rewriter.create<mlir::spirv::InBoundsPtrAccessChainOp>(
          loc, adaptor.getMemref(), adaptor.getIndices().front(), llvm::None);

      auto memoryAccess = mlir::spirv::MemoryAccessAttr::get(
          op.getContext(), mlir::spirv::MemoryAccess::Aligned);
      auto alignment = rewriter.getI32IntegerAttr(*typeSize);
      rewriter.replaceOpWithNewOp<mlir::spirv::LoadOp>(op, ptr, memoryAccess,
                                                       alignment);

      return mlir::success();
    } else {
      return mlir::failure();
    }
  }
};

class ConvertStoreOp : public mlir::OpConversionPattern<mlir::memref::StoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op,
                  mlir::memref::StoreOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memrefType = op.getMemref().getType().cast<mlir::MemRefType>();
    if (!memrefType.hasRank() || memrefType.getRank() != 1)
      return mlir::failure();

    auto typeSize = getTypeSize(memrefType.getElementType());
    if (!typeSize)
      return mlir::failure();

    auto loc = op.getLoc();
    auto ptr = rewriter.create<mlir::spirv::InBoundsPtrAccessChainOp>(
        loc, adaptor.getMemref(), adaptor.getIndices().front(), llvm::None);

    auto memoryAccess = mlir::spirv::MemoryAccessAttr::get(
        op.getContext(), mlir::spirv::MemoryAccess::Aligned);
    auto alignment = rewriter.getI32IntegerAttr(*typeSize);
    rewriter.replaceOpWithNewOp<mlir::spirv::StoreOp>(
        op, ptr, adaptor.getValue(), memoryAccess, alignment);

    return mlir::success();
  }
};

template <typename Op>
static mlir::Value lowerIntAtomic(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value ptr, mlir::Value val) {
  return builder.create<Op>(loc, ptr, mlir::spirv::Scope::Device,
                            mlir::spirv::MemorySemantics::None, val);
}

static mlir::Value lowerFloatAddAtomic(mlir::OpBuilder &builder,
                                       mlir::Location loc, mlir::Value ptr,
                                       mlir::Value val) {
  return builder.create<mlir::spirv::EXTAtomicFAddOp>(
      loc, val.getType(), ptr, mlir::spirv::Scope::Device,
      mlir::spirv::MemorySemantics::None, val);
}

static mlir::Value lowerFloatSubAtomic(mlir::OpBuilder &builder,
                                       mlir::Location loc, mlir::Value ptr,
                                       mlir::Value val) {
  auto neg = builder.create<mlir::spirv::FNegateOp>(loc, val).getResult();
  return builder.create<mlir::spirv::EXTAtomicFAddOp>(
      loc, neg.getType(), ptr, mlir::spirv::Scope::Device,
      mlir::spirv::MemorySemantics::None, neg);
}

class ConvertAtomicOps : public mlir::OpConversionPattern<mlir::func::CallOp> {
public:
  ConvertAtomicOps(mlir::TypeConverter &typeConverter,
                   mlir::MLIRContext *context)
      : mlir::OpConversionPattern<mlir::func::CallOp>(typeConverter, context,
                                                      /*benefit*/ 10) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op, mlir::func::CallOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.operands();
    if (operands.size() != 2)
      return mlir::failure();

    if (op.getNumResults() != 1)
      return mlir::failure();

    auto ptr = operands[0];
    auto ptrType = ptr.getType().dyn_cast<mlir::spirv::PointerType>();
    if (!ptrType)
      return mlir::failure();

    auto val = operands[1];
    auto valType = val.getType();
    if (ptrType.getPointeeType() != valType)
      return mlir::failure();

    bool isInt;
    if (valType.isSignlessInteger())
      isInt = true;
    else if (valType.isa<mlir::FloatType>())
      isInt = false;
    else
      return mlir::failure();

    auto funcName = op.getCallee();

    using func_t = mlir::Value (*)(mlir::OpBuilder &, mlir::Location,
                                   mlir::Value, mlir::Value);

    struct Desc {
      mlir::StringRef name;
      func_t intOp;
      func_t floatOp;
    };

    const Desc handlers[] = {
        {"atomic_add", &lowerIntAtomic<mlir::spirv::AtomicIAddOp>,
         &lowerFloatAddAtomic},
        {"atomic_sub", &lowerIntAtomic<mlir::spirv::AtomicISubOp>,
         &lowerFloatSubAtomic},
    };

    auto handler = [&]() -> func_t {
      for (auto &h : handlers) {
        if (funcName.consume_front(h.name))
          return (isInt ? h.intOp : h.floatOp);
      }
      return nullptr;
    }();

    if (handler) {
      auto res = handler(rewriter, op.getLoc(), ptr, val);
      if (res) {
        rewriter.replaceOp(op, res);
        return mlir::success();
      }
    }

    return mlir::failure();
  }
};

class ConvertBarrierOp
    : public mlir::OpConversionPattern<gpu_runtime::GPUBarrierOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::GPUBarrierOp op,
                  gpu_runtime::GPUBarrierOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto scope = mlir::spirv::Scope::Workgroup;
    mlir::spirv::MemorySemantics semantics;
    if (adaptor.getFlags() == gpu_runtime::FenceFlags::global) {
      semantics = mlir::spirv::MemorySemantics::SequentiallyConsistent |
                  mlir::spirv::MemorySemantics::CrossWorkgroupMemory;
    } else if (adaptor.getFlags() == gpu_runtime::FenceFlags::local) {
      semantics = mlir::spirv::MemorySemantics::SequentiallyConsistent |
                  mlir::spirv::MemorySemantics::WorkgroupMemory;
    } else {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<mlir::spirv::ControlBarrierOp>(op, scope, scope,
                                                               semantics);
    return mlir::success();
  }
};

class ConvertMemFenceOp
    : public mlir::OpConversionPattern<gpu_runtime::GPUMemFenceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::GPUMemFenceOp op,
                  gpu_runtime::GPUMemFenceOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto scope = mlir::spirv::Scope::Workgroup;
    mlir::spirv::MemorySemantics semantics;
    if (adaptor.getFlags() == gpu_runtime::FenceFlags::global) {
      semantics = mlir::spirv::MemorySemantics::SequentiallyConsistent |
                  mlir::spirv::MemorySemantics::CrossWorkgroupMemory;
    } else if (adaptor.getFlags() == gpu_runtime::FenceFlags::local) {
      semantics = mlir::spirv::MemorySemantics::SequentiallyConsistent |
                  mlir::spirv::MemorySemantics::WorkgroupMemory;
    } else {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<mlir::spirv::MemoryBarrierOp>(op, scope,
                                                              semantics);
    return mlir::success();
  }
};

static llvm::Optional<mlir::spirv::StorageClass>
convertStorageClass(mlir::Attribute src) {
  // TODO: Fix storage class upstream
  //  auto attr = src.dyn_cast_or_null<gpu_runtime::StorageClassAttr>();
  //  if (!attr)
  //    return llvm::None;

  //  auto sc = attr.getValue();
  //  if (sc == gpu_runtime::StorageClass::local)
  //    return mlir::spirv::StorageClass::Workgroup;

  if (auto attr = src.dyn_cast_or_null<mlir::IntegerAttr>())
    if (attr.getInt() == mlir::gpu::GPUDialect::getPrivateAddressSpace())
      return mlir::spirv::StorageClass::Workgroup;

  return llvm::None;
}

static mlir::spirv::StorageClass
convertStorageClass(mlir::Attribute src, mlir::spirv::StorageClass def) {
  auto ret = convertStorageClass(src);
  if (ret)
    return *ret;

  return def;
}

class ConvertGlobalOp
    : public mlir::OpConversionPattern<mlir::memref::GlobalOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::GlobalOp op,
                  mlir::memref::GlobalOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memrefType = op.getType();
    if (!memrefType.hasStaticShape())
      return mlir::failure();

    auto storageClass = convertStorageClass(memrefType.getMemorySpace());
    if (!storageClass)
      return mlir::failure();

    auto converter = getTypeConverter();
    assert(converter);

    auto elemType = converter->convertType(memrefType.getElementType());
    if (!elemType)
      return mlir::failure();

    auto elemCount = memrefType.getNumElements();
    auto newType = mlir::spirv::ArrayType::get(elemType, elemCount);
    auto ptrType = mlir::spirv::PointerType::get(newType, *storageClass);

    rewriter.replaceOpWithNewOp<mlir::spirv::GlobalVariableOp>(
        op, ptrType, adaptor.getSymName());
    return mlir::success();
  }
};

class ConvertGetGlobalOp
    : public mlir::OpConversionPattern<mlir::memref::GetGlobalOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::GetGlobalOp op,
                  mlir::memref::GetGlobalOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memrefType = op.getType().dyn_cast<mlir::MemRefType>();
    if (!memrefType)
      return mlir::failure();

    auto storageClass = convertStorageClass(memrefType.getMemorySpace());
    if (!storageClass)
      return mlir::failure();

    auto converter = getTypeConverter();
    assert(converter);
    auto resType = converter->convertType(memrefType);
    if (!resType)
      return mlir::failure();

    auto elemType = converter->convertType(memrefType.getElementType());
    if (!elemType)
      return mlir::failure();

    auto elemCount = memrefType.getNumElements();
    auto newType = mlir::spirv::ArrayType::get(elemType, elemCount);
    auto ptrType = mlir::spirv::PointerType::get(newType, *storageClass);

    auto loc = op->getLoc();
    mlir::Value res = rewriter.create<mlir::spirv::AddressOfOp>(
        loc, ptrType, adaptor.getName());
    if (res.getType() != resType)
      res = rewriter.create<mlir::spirv::BitcastOp>(loc, resType, res);

    rewriter.replaceOp(op, res);
    return mlir::success();
  }
};

template <typename SpirvOp, bool Subgroup>
static void genReduceOp(mlir::Operation *srcOp, mlir::PatternRewriter &rewriter,
                        mlir::Value arg) {
  auto type = arg.getType();
  auto ctx = srcOp->getContext();
  auto s =
      Subgroup ? mlir::spirv::Scope::Subgroup : mlir::spirv::Scope::Workgroup;
  auto scope = mlir::spirv::ScopeAttr::get(ctx, s);
  auto groupOp = mlir::spirv::GroupOperationAttr::get(
      ctx, mlir::spirv::GroupOperation::Reduce);
  rewriter.replaceOpWithNewOp<SpirvOp>(srcOp, type, scope, groupOp, arg,
                                       mlir::Value{});
}

class ConvertAllReduceOp
    : public mlir::OpConversionPattern<mlir::gpu::AllReduceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::AllReduceOp op,
                  mlir::gpu::AllReduceOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto reduceOp = adaptor.getOp();
    if (!reduceOp)
      return mlir::failure();

    auto val = adaptor.getValue();
    auto valType = val.getType();
    if (!valType.isIntOrFloat())
      return mlir::failure();

    using funcptr_t =
        void (*)(mlir::Operation *, mlir::PatternRewriter &, mlir::Value);

    using ReduceType = mlir::gpu::AllReduceOperation;
    struct Handler {
      ReduceType op;
      funcptr_t floatFunc;
      funcptr_t intFunc;
    };

    namespace spv = mlir::spirv;
    const Handler handlers[] = {
        {ReduceType::ADD, &genReduceOp<spv::GroupNonUniformFAddOp, false>,
         &genReduceOp<spv::GroupNonUniformIAddOp, false>},
    };

    for (auto &h : handlers) {
      if (h.op == *reduceOp) {
        auto func = (valType.isa<mlir::FloatType>() ? h.floatFunc : h.intFunc);
        func(op, rewriter, val);
        return mlir::success();
      }
    }

    return mlir::success();
  }
};

class ConvertSubgroupReduceOp
    : public mlir::OpConversionPattern<mlir::gpu::SubgroupReduceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::SubgroupReduceOp op,
                  mlir::gpu::SubgroupReduceOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto reduceOp = adaptor.getOp();
    //    if (!reduceOp)
    //      return mlir::failure();

    auto val = adaptor.getValue();
    auto valType = val.getType();
    if (!valType.isIntOrFloat())
      return mlir::failure();

    using funcptr_t =
        void (*)(mlir::Operation *, mlir::PatternRewriter &, mlir::Value);

    using ReduceType = mlir::gpu::AllReduceOperation;
    struct Handler {
      ReduceType op;
      funcptr_t floatFunc;
      funcptr_t intFunc;
    };

    namespace spv = mlir::spirv;
    const Handler handlers[] = {
        {ReduceType::ADD, &genReduceOp<spv::GroupNonUniformFAddOp, true>,
         &genReduceOp<spv::GroupNonUniformIAddOp, true>},
    };

    for (auto &h : handlers) {
      if (h.op == reduceOp) {
        auto func = (valType.isa<mlir::FloatType>() ? h.floatFunc : h.intFunc);
        func(op, rewriter, val);
        return mlir::success();
      }
    }

    return mlir::success();
  }
};

// TODO: something better
class ConvertFunc : public mlir::OpConversionPattern<mlir::func::FuncOp> {
public:
  using mlir::OpConversionPattern<mlir::func::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::FuncOp op,
                  mlir::func::FuncOp::Adaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op.getBody().empty())
      return mlir::failure();

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ConvertAssert : public mlir::OpConversionPattern<mlir::cf::AssertOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::AssertOp op, mlir::cf::AssertOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::spirv::KHRAssumeTrueOp>(op,
                                                              adaptor.getArg());
    return mlir::success();
  }
};

class ConvertUndef : public mlir::OpConversionPattern<imex::util::UndefOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::util::UndefOp op, imex::util::UndefOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter);

    auto resType = converter->convertType(op.getType());
    if (!resType)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::spirv::UndefOp>(op, resType);
    return mlir::success();
  }
};

// Upstream lowers them to i64, bu we need i32.
template <typename SourceOp, mlir::spirv::BuiltIn builtin>
class SingleDimLaunchConfigConversion
    : public mlir::OpConversionPattern<SourceOp> {
public:
  SingleDimLaunchConfigConversion(mlir::TypeConverter &typeConverter,
                                  mlir::MLIRContext *context)
      : mlir::OpConversionPattern<SourceOp>(typeConverter, context,
                                            /*benefit*/ 10) {}

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        this->template getTypeConverter<mlir::SPIRVTypeConverter>();
    auto indexType = typeConverter->getIndexType();
    auto i32Type = rewriter.getI32Type();

    auto spirvBuiltin =
        mlir::spirv::getBuiltinVariableValue(op, builtin, i32Type, rewriter);
    if (indexType != i32Type)
      spirvBuiltin = rewriter.create<mlir::arith::ExtSIOp>(
          op->getLoc(), indexType, spirvBuiltin);

    rewriter.replaceOp(op, spirvBuiltin);
    return mlir::success();
  }
};

struct GPUToSpirvPass
    : public mlir::PassWrapper<GPUToSpirvPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUToSpirvPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::AffineDialect>();
    registry.insert<mlir::spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    auto module = getOperation();

    llvm::SmallVector<mlir::Operation *, 1> kernelModules;
    mlir::OpBuilder builder(context);
    module.walk([&builder, &kernelModules](mlir::gpu::GPUModuleOp moduleOp) {
      // For each kernel module (should be only 1 for now, but that is not a
      // requirement here), clone the module for conversion because the
      // gpu.launch function still needs the kernel module.
      builder.setInsertionPoint(moduleOp.getOperation());
      kernelModules.push_back(builder.clone(*moduleOp.getOperation()));
    });

    for (auto kernelModule : kernelModules) {
      auto targetAttr = mlir::spirv::lookupTargetEnvOrDefault(kernelModule);
      auto target = mlir::SPIRVConversionTarget::get(targetAttr);

      mlir::SPIRVConversionOptions options;
      options.use64bitIndex = true;

      mlir::SPIRVTypeConverter typeConverter(targetAttr, options);
      mlir::RewritePatternSet patterns(context);

      typeConverter.addConversion([&typeConverter](mlir::MemRefType type)
                                      -> llvm::Optional<mlir::Type> {
        auto srcElemType = type.getElementType();
        if (!srcElemType.isIntOrFloat() && !srcElemType.isa<mlir::VectorType>())
          return mlir::Type(nullptr);

        auto elemType = typeConverter.convertType(srcElemType);
        if (!elemType)
          return mlir::Type(nullptr);

        auto sc = convertStorageClass(
            type.getMemorySpace(), mlir::spirv::StorageClass::CrossWorkgroup);

        return mlir::spirv::PointerType::get(elemType, sc);
      });

      mlir::ScfToSPIRVContext scfToSpirvCtx;
      mlir::populateSCFToSPIRVPatterns(typeConverter, scfToSpirvCtx, patterns);
      mlir::populateGPUToSPIRVPatterns(typeConverter, patterns);
      mlir::populateFuncToSPIRVPatterns(typeConverter, patterns);
      mlir::cf::populateControlFlowToSPIRVPatterns(typeConverter, patterns);
      mlir::arith::populateArithToSPIRVPatterns(typeConverter, patterns);
      mlir::populateMathToSPIRVPatterns(typeConverter, patterns);

      patterns
          .insert<ConvertSubviewOp, ConvertCastOp<mlir::memref::CastOp>,
                  ConvertCastOp<mlir::memref::ReinterpretCastOp>,
                  ConvertBitcastOp<imex::util::BitcastOp>,
                  ConvertBitcastOp<imex::util::MemrefBitcastOp>, ConvertLoadOp,
                  ConvertStoreOp, ConvertAtomicOps, ConvertFunc, ConvertAssert,
                  ConvertBarrierOp, ConvertMemFenceOp, ConvertUndef,
                  ConvertGlobalOp, ConvertGetGlobalOp, ConvertAllReduceOp,
                  ConvertSubgroupReduceOp>(typeConverter, context);

      patterns.add<
          SingleDimLaunchConfigConversion<mlir::gpu::SubgroupIdOp,
                                          mlir::spirv::BuiltIn::SubgroupId>,
          SingleDimLaunchConfigConversion<mlir::gpu::NumSubgroupsOp,
                                          mlir::spirv::BuiltIn::NumSubgroups>,
          SingleDimLaunchConfigConversion<mlir::gpu::SubgroupSizeOp,
                                          mlir::spirv::BuiltIn::SubgroupSize>>(
          typeConverter, patterns.getContext());

      if (failed(
              applyFullConversion(kernelModule, *target, std::move(patterns))))
        return signalPassFailure();
    }
  }
};

template <typename Op, typename F>
static mlir::LogicalResult createGpuKernelLoad(mlir::PatternRewriter &builder,
                                               Op &&op, F &&func) {
  auto mod = op->template getParentOfType<mlir::ModuleOp>();
  if (!mod)
    return mlir::failure();

  auto gpuMod = mod.template lookupSymbol<mlir::gpu::GPUModuleOp>(
      op.getKernelModuleName());
  if (!gpuMod)
    return mlir::failure();

  auto gpuKernel =
      gpuMod.template lookupSymbol<mlir::gpu::GPUFuncOp>(op.getKernelName());
  if (!gpuKernel)
    return mlir::failure();

  auto stream = getGpuStream(builder, op);
  if (!stream)
    return mlir::failure();

  auto loc = op.getLoc();
  auto module =
      builder.create<gpu_runtime::LoadGpuModuleOp>(loc, *stream, gpuMod);
  auto kernel =
      builder.create<gpu_runtime::GetGpuKernelOp>(loc, module, gpuKernel);
  auto newOp = func(builder, loc, *stream, kernel);
  builder.replaceOp(op, newOp.getResults());
  return mlir::success();
}

struct ExpandLaunchOp : public mlir::OpRewritePattern<mlir::gpu::LaunchFuncOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::LaunchFuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    return createGpuKernelLoad(
        rewriter, op,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value stream,
            mlir::Value kernel) {
          return builder.create<gpu_runtime::LaunchGpuKernelOp>(
              loc, stream, kernel, op.getGridSizeOperandValues(),
              op.getBlockSizeOperandValues(), op.getKernelOperands());
        });
  }
};

struct ExpandAllocOp : public mlir::OpRewritePattern<mlir::gpu::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::AllocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto stream = getGpuStream(rewriter, op);
    if (!stream)
      return mlir::failure();

    auto hostShared = op.getHostShared();
    mlir::Type token =
        op.getAsyncToken() ? op.getAsyncToken().getType() : nullptr;
    rewriter.replaceOpWithNewOp<gpu_runtime::GPUAllocOp>(
        op, op.getType(), token, op.getAsyncDependencies(), *stream,
        op.getDynamicSizes(), op.getSymbolOperands(), hostShared);

    return mlir::success();
  }
};

struct ExpandDeallocOp : public mlir::OpRewritePattern<mlir::gpu::DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::DeallocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto stream = getGpuStream(rewriter, op);
    if (!stream)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<gpu_runtime::GPUDeallocOp>(
        op, op.getResultTypes(), op.getAsyncDependencies(), op.getMemref(),
        *stream);

    return mlir::success();
  }
};

struct ExpandSuggestBlockSizeOp
    : public mlir::OpRewritePattern<gpu_runtime::GPUSuggestBlockSizeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::GPUSuggestBlockSizeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getKernel())
      return mlir::failure();

    assert(op.getKernelRef());
    return createGpuKernelLoad(
        rewriter, op,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value stream,
            mlir::Value kernel) {
          return builder.create<gpu_runtime::GPUSuggestBlockSizeOp>(
              loc, stream, op.getGridSize(), kernel);
        });
  }
};

struct AbiAttrsPass
    : public mlir::PassWrapper<AbiAttrsPass,
                               mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AbiAttrsPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    auto gpuModule = getOperation();
    auto *context = &getContext();
    auto attrName =
        mlir::StringAttr::get(context, mlir::spirv::getEntryPointABIAttrName());
    auto abi = mlir::spirv::getEntryPointABIAttr(llvm::None, context);
    for (auto gpuFunc : gpuModule.getOps<mlir::gpu::GPUFuncOp>()) {
      if (!mlir::gpu::GPUDialect::isKernel(gpuFunc) ||
          gpuFunc->getAttr(attrName))
        continue;

      gpuFunc->setAttr(attrName, abi);
    }
  }
};

static mlir::spirv::TargetEnvAttr defaultCapsMapper(mlir::gpu::GPUModuleOp op) {
  auto context = op.getContext();
  namespace spirv = mlir::spirv;
  spirv::Capability caps[] = {
      // clang-format off
      spirv::Capability::Addresses,
      spirv::Capability::AtomicFloat32AddEXT,
      spirv::Capability::ExpectAssumeKHR,
      spirv::Capability::Float16,
      spirv::Capability::Float16Buffer,
      spirv::Capability::Float64,
      spirv::Capability::GenericPointer,
      spirv::Capability::GroupNonUniformArithmetic,
      spirv::Capability::Groups,
      spirv::Capability::Int16,
      spirv::Capability::Int64,
      spirv::Capability::Int8,
      spirv::Capability::Kernel,
      spirv::Capability::Linkage,
      spirv::Capability::Vector16,
      // clang-format on
  };
  spirv::Extension exts[] = {spirv::Extension::SPV_EXT_shader_atomic_float_add,
                             spirv::Extension::SPV_KHR_expect_assume};
  llvm::sort(caps);
  llvm::sort(exts);
  auto triple =
      spirv::VerCapExtAttr::get(spirv::Version::V_1_0, caps, exts, context);
  auto attr = spirv::TargetEnvAttr::get(
      triple, spirv::Vendor::Unknown, spirv::DeviceType::Unknown,
      spirv::TargetEnvAttr::kUnknownDeviceID,
      spirv::getDefaultResourceLimits(context));
  return attr;
}

struct SetSPIRVCapabilitiesPass
    : public mlir::PassWrapper<SetSPIRVCapabilitiesPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetSPIRVCapabilitiesPass)

  SetSPIRVCapabilitiesPass(
      std::function<mlir::spirv::TargetEnvAttr(mlir::gpu::GPUModuleOp)> m)
      : mapper(std::move(m)) {
    if (!mapper)
      mapper = &defaultCapsMapper;
  }

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    assert(mapper && "Invalid mapper");
    namespace spirv = mlir::spirv;
    auto op = getOperation();
    op->walk([&](mlir::gpu::GPUModuleOp op) {
      if (auto attr = mapper(op))
        op->setAttr(spirv::getTargetEnvAttrName(), attr);
    });
  }

private:
  std::function<mlir::spirv::TargetEnvAttr(mlir::gpu::GPUModuleOp)> mapper;
};

struct SerializeSPIRVPass
    : public mlir::PassWrapper<SerializeSPIRVPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SerializeSPIRVPass)

  void runOnOperation() override {
    auto mod = getOperation();

    namespace gpu = mlir::gpu;
    namespace spirv = mlir::spirv;
    llvm::SmallVector<uint32_t, 0> spvBinary;
    for (auto gpuMod : mod.getOps<gpu::GPUModuleOp>()) {
      auto name = gpuMod.getName();
      auto isSameMod = [&](spirv::ModuleOp spvMod) -> bool {
        auto spvModName = spvMod.getName();
        return spvModName->consume_front("__spv__") && spvModName == name;
      };
      auto spvMods = mod.getOps<spirv::ModuleOp>();
      auto it = llvm::find_if(spvMods, isSameMod);
      if (it == spvMods.end()) {
        gpuMod.emitError() << "Unable to find corresponding SPIR-V module";
        signalPassFailure();
        return;
      }
      auto spvMod = *it;

      spvBinary.clear();
      if (mlir::failed(spirv::serialize(spvMod, spvBinary))) {
        spvMod.emitError() << "Failed to serialize SPIR-V module";
        signalPassFailure();
        return;
      }

      auto spvData =
          llvm::StringRef(reinterpret_cast<const char *>(spvBinary.data()),
                          spvBinary.size() * sizeof(uint32_t));
      auto spvAttr = mlir::StringAttr::get(&getContext(), spvData);
      gpuMod->setAttr(gpu::getDefaultGpuBinaryAnnotation(), spvAttr);
      spvMod->erase();
    }
  }
};

struct GPUExPass
    : public mlir::PassWrapper<GPUExPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUExPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<gpu_runtime::GpuRuntimeDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<ExpandLaunchOp, ExpandAllocOp, ExpandDeallocOp,
                    ExpandSuggestBlockSizeOp>(ctx);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

struct TileParallelOp : public mlir::OpRewritePattern<mlir::scf::ParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ParallelOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Process only loops inside gpu region.
    auto envOp = op->getParentOfType<imex::util::EnvironmentRegionOp>();
    if (!envOp || !envOp.getEnvironment().isa<gpu_runtime::GPURegionDescAttr>())
      return mlir::failure();

    // Process only outermost loops without mappings.
    if (op->getParentOfType<mlir::scf::ParallelOp>() ||
        op->hasAttr(mlir::gpu::getMappingAttrName()))
      return mlir::failure();

    // Reductions is not supported yet.
    if (!op.getBody()->getOps<mlir::scf::ReduceOp>().empty())
      return mlir::failure();

    auto oldLowerBounds = op.getLowerBound();
    auto oldUpperBounds = op.getUpperBound();
    auto oldSteps = op.getStep();
    auto oldLoopsCount = static_cast<unsigned>(oldSteps.size());

    const unsigned maxLoops = 3;
    // Only unit step is supported and iteration must start from 0.
    unsigned numLoops = 0;
    for (auto [start, step] : llvm::zip(oldLowerBounds.take_front(maxLoops),
                                        oldSteps.take_front(maxLoops)))
      if (mlir::isConstantIntValue(start, 0) &&
          mlir::isConstantIntValue(step, 1))
        ++numLoops;

    // No suitable loops.
    if (numLoops == 0)
      return mlir::failure();

    auto loc = op->getLoc();
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);

    std::array<mlir::Value, 3> globalSize;
    globalSize.fill(one);
    llvm::copy(oldUpperBounds.take_front(numLoops), globalSize.begin());

    llvm::Optional<mlir::Value> stream;
    auto localSize =
        rewriter
            .create<gpu_runtime::GPUSuggestBlockSizeOp>(loc, stream, globalSize)
            ->getResults();

    llvm::SmallVector<mlir::Value> newLowerBounds;
    llvm::SmallVector<mlir::Value> newUpperBounds;
    llvm::SmallVector<mlir::Value> newSteps;

    // Insert grid vars.
    for (auto i : llvm::seq(0u, maxLoops)) {
      newLowerBounds.emplace_back(zero);
      newSteps.emplace_back(one);
      if (i < numLoops) {
        auto oldUpperBound = oldUpperBounds[i];
        mlir::Value newUpperBound = rewriter.create<mlir::arith::CeilDivUIOp>(
            loc, oldUpperBound, localSize[i]);
        newUpperBounds.emplace_back(newUpperBound);
      } else {
        newUpperBounds.emplace_back(one);
      }
    }

    // Insert block vars.
    for (auto i : llvm::seq(0u, maxLoops)) {
      newLowerBounds.emplace_back(zero);
      newSteps.emplace_back(one);
      if (i < numLoops) {
        newUpperBounds.emplace_back(localSize[i]);
      } else {
        newUpperBounds.emplace_back(one);
      }
    }

    for (auto i : llvm::seq(numLoops, oldLoopsCount)) {
      newLowerBounds.emplace_back(oldLowerBounds[i]);
      newUpperBounds.emplace_back(oldUpperBounds[i]);
      newSteps.emplace_back(oldSteps[i]);
    }

    auto initVals = op.getInitVals();
    auto newOp = rewriter.create<mlir::scf::ParallelOp>(
        loc, newLowerBounds, newUpperBounds, newSteps, initVals);
    mlir::Block *originalBlock = op.getBody();
    mlir::Block *newBlock = newOp.getBody();

    mlir::Value inBounds;
    llvm::SmallVector<mlir::Value> argMapping(oldLoopsCount);
    {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(newBlock);
      for (auto i : llvm::seq(0u, oldLoopsCount)) {
        if (i < numLoops) {
          mlir::Value gridId = newBlock->getArgument(i);
          mlir::Value blockId = newBlock->getArgument(i + maxLoops);
          mlir::Value blockSize = localSize[i];
          mlir::Value gridSize = globalSize[i];
          mlir::Value val =
              rewriter.create<mlir::arith::MulIOp>(loc, gridId, blockSize);
          val = rewriter.create<mlir::arith::AddIOp>(loc, val, blockId);
          argMapping[i] = val;
          mlir::Value in = rewriter.createOrFold<mlir::arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::slt, val, gridSize);
          if (0 == i) {
            inBounds = in;
          } else {
            inBounds =
                rewriter.createOrFold<mlir::arith::AndIOp>(loc, inBounds, in);
          }
        } else {
          argMapping[i] = newBlock->getArgument(i + maxLoops * 2 - numLoops);
        }
      }

      assert(inBounds);
      auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, llvm::None, inBounds);
      newBlock = ifOp.thenBlock();
    }
    rewriter.eraseOp(newBlock->getTerminator()); // Erase exisitng yield.
    rewriter.mergeBlocks(originalBlock, newBlock, argMapping);
    rewriter.replaceOp(op, newOp->getResults());

    auto newLoopsCount = static_cast<unsigned>(newSteps.size());
    auto identityMap = rewriter.getDimIdentityMap();
    llvm::SmallVector<mlir::gpu::ParallelLoopDimMappingAttr> mapping(
        newLoopsCount);
    for (auto i : llvm::seq(0u, newLoopsCount))
      mapping[i] = rewriter.getAttr<mlir::gpu::ParallelLoopDimMappingAttr>(
          getProcessor(i), identityMap, identityMap);

    return mlir::gpu::setMappingAttr(newOp, mapping);
  }
};

struct TileParallelLoopsForGPUPass
    : public mlir::PassWrapper<TileParallelLoopsForGPUPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileParallelLoopsForGPUPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<gpu_runtime::GpuRuntimeDialect>();
    registry.insert<imex::util::ImexUtilDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<TileParallelOp>(ctx);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

// Some manual fp conversion, denormals and nan/infs are not supported.
static mlir::Value f64Tof32(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value src) {
  auto i64 = builder.getI64Type();
  auto srcI64 = builder.create<imex::util::BitcastOp>(loc, i64, src);

  auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, i64);
  auto absMask = builder.create<mlir::arith::ConstantIntOp>(
      loc, static_cast<int64_t>(0x7FFFFFFFFFFFFFFFULL), i64);

  mlir::Value absVal =
      builder.create<mlir::arith::AndIOp>(loc, srcI64, absMask);
  mlir::Value isZero = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, absVal, zero);

  auto signShift = builder.create<mlir::arith::ConstantIntOp>(loc, 63, i64);
  auto expShift = builder.create<mlir::arith::ConstantIntOp>(loc, 52, i64);
  auto expMask = builder.create<mlir::arith::ConstantIntOp>(loc, 0x7FF, i64);
  auto manMask = builder.create<mlir::arith::ConstantIntOp>(
      loc, static_cast<int64_t>(0x000FFFFFFFFFFFFFULL), i64);
  auto b = builder.create<mlir::arith::ConstantIntOp>(loc, 1023 - 127, i64);
  auto _ff = builder.create<mlir::arith::ConstantIntOp>(loc, 0xFF, i64);
  auto _29 = builder.create<mlir::arith::ConstantIntOp>(loc, 29, i64);
  auto _23 = builder.create<mlir::arith::ConstantIntOp>(loc, 23, i64);
  auto _31 = builder.create<mlir::arith::ConstantIntOp>(loc, 31, i64);

  mlir::Value sign =
      builder.create<mlir::arith::ShRUIOp>(loc, srcI64, signShift);
  mlir::Value exponent =
      builder.create<mlir::arith::ShRUIOp>(loc, srcI64, expShift);
  exponent = builder.create<mlir::arith::AndIOp>(loc, exponent, expMask);
  mlir::Value mantissa =
      builder.create<mlir::arith::AndIOp>(loc, srcI64, manMask);
  exponent = builder.create<mlir::arith::SubIOp>(loc, exponent, b);

  exponent = builder.create<mlir::arith::AndIOp>(loc, exponent, _ff);
  mantissa = builder.create<mlir::arith::ShRUIOp>(loc, mantissa, _29);

  exponent = builder.create<mlir::arith::ShLIOp>(loc, exponent, _23);
  sign = builder.create<mlir::arith::ShLIOp>(loc, sign, _31);

  mlir::Value res = mantissa;
  res = builder.create<mlir::arith::OrIOp>(loc, res, exponent);
  res = builder.create<mlir::arith::OrIOp>(loc, res, sign);

  res = builder.create<mlir::arith::SelectOp>(loc, isZero, srcI64, res);

  res = builder.create<mlir::arith::TruncIOp>(loc, builder.getI32Type(), res);
  res = builder.create<mlir::arith::BitcastOp>(loc, builder.getF32Type(), res);
  return res;
}

// Some manual fp conversion, denormals and nan/infs are not supported.
static mlir::Value f32Tof64(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value src, mlir::Type resType) {
  auto i32 = builder.getI32Type();
  mlir::Value srcI64 = builder.create<mlir::arith::BitcastOp>(loc, i32, src);

  auto i64 = builder.getI64Type();
  srcI64 = builder.create<mlir::arith::ExtUIOp>(loc, i64, srcI64);

  auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, i64);
  auto absMask = builder.create<mlir::arith::ConstantIntOp>(
      loc, static_cast<int64_t>(0x7FFFFFFFFFFFFFFFULL), i64);

  mlir::Value absVal =
      builder.create<mlir::arith::AndIOp>(loc, srcI64, absMask);
  mlir::Value isZero = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, absVal, zero);

  auto signShift = builder.create<mlir::arith::ConstantIntOp>(loc, 31, i64);
  auto expShift = builder.create<mlir::arith::ConstantIntOp>(loc, 23, i64);
  auto expMask = builder.create<mlir::arith::ConstantIntOp>(loc, 0xFF, i64);
  auto manMask = builder.create<mlir::arith::ConstantIntOp>(loc, 0x7FFFFF, i64);
  auto b = builder.create<mlir::arith::ConstantIntOp>(loc, 1023 - 127, i64);
  auto _29 = builder.create<mlir::arith::ConstantIntOp>(loc, 29, i64);
  auto _52 = builder.create<mlir::arith::ConstantIntOp>(loc, 52, i64);
  auto _63 = builder.create<mlir::arith::ConstantIntOp>(loc, 63, i64);

  mlir::Value sign =
      builder.create<mlir::arith::ShRUIOp>(loc, srcI64, signShift);
  mlir::Value exponent =
      builder.create<mlir::arith::ShRUIOp>(loc, srcI64, expShift);
  exponent = builder.create<mlir::arith::AndIOp>(loc, exponent, expMask);
  mlir::Value mantissa =
      builder.create<mlir::arith::AndIOp>(loc, srcI64, manMask);

  mantissa = builder.create<mlir::arith::ShLIOp>(loc, mantissa, _29);
  exponent = builder.create<mlir::arith::AddIOp>(loc, exponent, b);

  exponent = builder.create<mlir::arith::ShLIOp>(loc, exponent, _52);
  sign = builder.create<mlir::arith::ShLIOp>(loc, sign, _63);

  mlir::Value res = mantissa;
  res = builder.create<mlir::arith::OrIOp>(loc, res, exponent);
  res = builder.create<mlir::arith::OrIOp>(loc, res, sign);

  res = builder.create<mlir::arith::SelectOp>(loc, isZero, srcI64, res);

  res = builder.create<imex::util::BitcastOp>(loc, resType, res);
  return res;
}

class ConvertF64LoadOp
    : public mlir::OpConversionPattern<mlir::memref::LoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::memref::LoadOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter && "Invalid type converter");

    auto origResType = op.getType();
    if (!origResType.isF64())
      return mlir::success();

    auto resType = converter->convertType(origResType);
    if (!resType)
      return mlir::success();

    auto loc = op.getLoc();
    mlir::Value result = rewriter.create<mlir::memref::LoadOp>(
        loc, adaptor.getMemref(), adaptor.getIndices());
    result = f64Tof32(rewriter, loc, result);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class ConvertF64StoreOp
    : public mlir::OpConversionPattern<mlir::memref::StoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op,
                  mlir::memref::StoreOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter && "Invalid type converter");

    auto origType = op.getValue().getType();
    if (!origType.isF64())
      return mlir::failure();

    auto memref = adaptor.getMemref();
    auto memrefType = memref.getType().dyn_cast<mlir::MemRefType>();
    if (!memrefType)
      return mlir::failure();

    auto loc = op.getLoc();
    mlir::Value f64val = f32Tof64(rewriter, loc, adaptor.getValue(),
                                  memrefType.getElementType());
    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, f64val, memref,
                                                       adaptor.getIndices());
    return mlir::success();
  }
};

struct TruncateF64ForGPUPass
    : public mlir::PassWrapper<TruncateF64ForGPUPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TruncateF64ForGPUPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::ConversionTarget target(*ctx);
    mlir::TypeConverter converter;

    // Convert unknown types to itself
    converter.addConversion([](mlir::Type type) { return type; });

    converter.addConversion([](mlir::Float64Type type) {
      return mlir::Float32Type::get(type.getContext());
    });

    converter.addConversion(
        [](mlir::MemRefType type) -> llvm::Optional<mlir::Type> {
          if (!type.getElementType().isF64())
            return llvm::None;

          int64_t shape[] = {2};
          auto elemType = mlir::IntegerType::get(type.getContext(), 32);
          auto newType = mlir::VectorType::get(shape, elemType);
          return type.clone(newType);
        });

    auto addCast = [](mlir::OpBuilder &builder, mlir::Type dstType,
                      mlir::ValueRange inputs,
                      mlir::Location loc) -> llvm::Optional<mlir::Value> {
      if (inputs.size() != 1)
        return llvm::None;

      auto src = inputs.front();
      auto srcType = src.getType();
      if (srcType.isF32() && dstType.isF64())
        return builder.create<mlir::arith::ExtFOp>(loc, dstType, src)
            .getResult();

      if (srcType.isF64() && dstType.isF32())
        return builder.create<mlir::arith::TruncFOp>(loc, dstType, src)
            .getResult();

      if (srcType.isa<mlir::MemRefType>() && dstType.isa<mlir::MemRefType>())
        return builder.create<imex::util::MemrefBitcastOp>(loc, dstType, src)
            .getResult();

      return llvm::None;
    };
    converter.addArgumentMaterialization(addCast);
    converter.addSourceMaterialization(addCast);
    converter.addTargetMaterialization(addCast);

    mlir::RewritePatternSet patterns(ctx);

    imex::populateArithConversionRewritesAndTarget(converter, patterns, target);
    imex::populateMathConversionRewritesAndTarget(converter, patterns, target);
    imex::populateControlFlowTypeConversionRewritesAndTarget(converter,
                                                             patterns, target);
    imex::populateTupleTypeConversionRewritesAndTarget(converter, patterns,
                                                       target);

    // TODO: remove
    mlir::populateFunctionOpInterfaceTypeConversionPattern<
        mlir::gpu::GPUFuncOp>(patterns, converter);
    target.addDynamicallyLegalOp<mlir::gpu::GPUFuncOp>(
        [&](mlir::gpu::GPUFuncOp op) -> llvm::Optional<bool> {
          if (converter.isSignatureLegal(op.getFunctionType()) &&
              converter.isLegal(&op.getBody()))
            return true;

          return llvm::None;
        });

    patterns.insert<ConvertF64LoadOp, ConvertF64StoreOp>(converter, ctx);
    target.addDynamicallyLegalOp<mlir::memref::LoadOp, mlir::memref::StoreOp>(
        [&converter](mlir::Operation *op) -> llvm::Optional<bool> {
          if (converter.isLegal(op))
            return true;

          return llvm::None;
        });

    mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    auto module = getOperation();

    llvm::SmallVector<mlir::Value> newArgs;
    mlir::OpBuilder builder(ctx);
    for (auto gpuModule : module.getOps<mlir::gpu::GPUModuleOp>()) {
      auto targetEnv = mlir::spirv::lookupTargetEnv(gpuModule);
      if (!targetEnv) {
        gpuModule->emitError("TargetEnv not found");
        return signalPassFailure();
      }

      auto caps = targetEnv.getCapabilities();
      if (llvm::is_contained(caps, mlir::spirv::Capability::Float64))
        continue;

      for (auto gpuFunc : gpuModule.getOps<mlir::gpu::GPUFuncOp>()) {
        auto origSig = gpuFunc.getFunctionType();
        if (mlir::failed(
                mlir::applyPartialConversion(gpuFunc, target, frozenPatterns)))
          return signalPassFailure();

        auto newSig = gpuFunc.getFunctionType();
        if (origSig == newSig)
          continue;

        auto funcUses = mlir::SymbolTable::getSymbolUses(gpuFunc, module);
        if (!funcUses)
          continue;

        for (auto use : llvm::make_early_inc_range(*funcUses)) {
          auto user = use.getUser();
          if (mlir::isa<gpu_runtime::GPUSuggestBlockSizeOp>(user))
            continue;

          auto launch = mlir::dyn_cast<mlir::gpu::LaunchFuncOp>(user);
          if (!launch) {
            user->emitError("Unknown gpu func user");
            return signalPassFailure();
          }

          builder.setInsertionPoint(launch);

          newArgs.clear();
          newArgs.reserve(launch.getNumKernelOperands());
          for (auto [origArg, newType] :
               llvm::zip(launch.getKernelOperands(), newSig.getInputs())) {
            auto origType = origArg.getType();
            if (newType == origType) {
              newArgs.emplace_back(origArg);
            } else if (origType.isF64() && newType.isF32()) {
              auto loc = launch.getLoc();
              mlir::Value newVal =
                  builder.create<mlir::arith::TruncFOp>(loc, newType, origArg);
              newArgs.emplace_back(newVal);
            } else if (origType.isa<mlir::MemRefType>() &&
                       newType.isa<mlir::MemRefType>()) {
              auto loc = launch.getLoc();
              mlir::Value newVal = builder.create<imex::util::MemrefBitcastOp>(
                  loc, newType, origArg);
              newArgs.emplace_back(newVal);
            } else {
              launch->emitError("Incompatible types: ")
                  << origType << " and " << newType;
              return signalPassFailure();
            }
          }

          launch.getKernelOperandsMutable().assign(newArgs);
        }
      }
    }
  }
};
} // namespace

// Expose the passes to the outside world
std::unique_ptr<mlir::Pass> gpu_runtime::createAbiAttrsPass() {
  return std::make_unique<AbiAttrsPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createSetSPIRVCapabilitiesPass(
    std::function<mlir::spirv::TargetEnvAttr(mlir::gpu::GPUModuleOp)> mapper) {
  return std::make_unique<SetSPIRVCapabilitiesPass>(std::move(mapper));
}

std::unique_ptr<mlir::Pass> gpu_runtime::createGPUToSpirvPass() {
  return std::make_unique<GPUToSpirvPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createInsertGPUAllocsPass() {
  return std::make_unique<InsertGPUAllocs>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createConvertGPUDeallocsPass() {
  return std::make_unique<ConvertGPUDeallocsPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createUnstrideMemrefsPass() {
  return std::make_unique<UnstrideMemrefsPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createSerializeSPIRVPass() {
  return std::make_unique<SerializeSPIRVPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createGPUExPass() {
  return std::make_unique<GPUExPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createParallelLoopGPUMappingPass() {
  return std::make_unique<ParallelLoopGPUMappingPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createTileParallelLoopsForGPUPass() {
  return std::make_unique<TileParallelLoopsForGPUPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createTruncateF64ForGPUPass() {
  return std::make_unique<TruncateF64ForGPUPass>();
}