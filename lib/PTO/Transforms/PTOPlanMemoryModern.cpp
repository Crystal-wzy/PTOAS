// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOPlanMemoryModern.cpp - modern static local memory planner -------===//

#include "AllocToPointerCast.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace mlir::pto;

namespace mlir::pto {
LogicalResult runModernPlanMemory(func::FuncOp func, llvm::StringRef memMode,
                                  bool orderBySize);
} // namespace mlir::pto

namespace {

struct MemSpec {
  uint64_t capacityBytes = 0;
  uint64_t alignmentBytes = 1;
};

struct RootInfo {
  Value root;
  Operation *defOp = nullptr;
  AddressSpace space = AddressSpace::Zero;
  uint64_t slotBytes = 0;
  uint64_t totalBytes = 0;
  uint64_t alignmentBytes = 1;
  uint64_t slotCount = 1;
  unsigned allocIndex = 0;
  unsigned freeIndex = 0;
  unsigned stableOrder = 0;
  SmallVector<uint64_t> offsets;
};

using RootList = SmallVector<Value, 4>;

static uint64_t alignUp(uint64_t value, uint64_t align) {
  if (align == 0)
    return value;
  return ((value + align - 1) / align) * align;
}

static std::optional<AddressSpace> getBufferAddressSpace(Type type) {
  if (auto tileType = dyn_cast<TileBufType>(type)) {
    if (auto attr =
            dyn_cast_or_null<AddressSpaceAttr>(tileType.getMemorySpace()))
      return attr.getAddressSpace();
    return std::nullopt;
  }

  if (auto memRefType = dyn_cast<BaseMemRefType>(type)) {
    if (auto attr =
            dyn_cast_or_null<AddressSpaceAttr>(memRefType.getMemorySpace()))
      return attr.getAddressSpace();
  }
  return std::nullopt;
}

static bool isPlannableLocalSpace(std::optional<AddressSpace> space) {
  return space && *space != AddressSpace::GM && *space != AddressSpace::Zero;
}

static MemSpec getMemSpec(PTOArch arch, AddressSpace space) {
  switch (space) {
  case AddressSpace::VEC:
    return {arch == PTOArch::A5 ? 253952ull : 196608ull, 256};
  case AddressSpace::MAT:
    return {524288ull, 256};
  case AddressSpace::LEFT:
  case AddressSpace::RIGHT:
    return {65536ull, 4096};
  case AddressSpace::ACC:
    return {arch == PTOArch::A5 ? 262144ull : 131072ull, 4096};
  case AddressSpace::BIAS:
  case AddressSpace::SCALING:
    return {524288ull, 256};
  case AddressSpace::GM:
  case AddressSpace::Zero:
    break;
  }
  return {};
}

static FailureOr<uint64_t> computeStaticBufferBytes(Value value) {
  ArrayRef<int64_t> shape;
  Type elementType;

  if (auto tileType = dyn_cast<TileBufType>(value.getType())) {
    shape = tileType.getShape();
    elementType = tileType.getElementType();
  } else if (auto memRefType = dyn_cast<BaseMemRefType>(value.getType())) {
    shape = memRefType.getShape();
    elementType = memRefType.getElementType();
  } else {
    return failure();
  }

  uint64_t elemBytes = getPTOStorageElemByteSize(elementType);
  if (elemBytes == 0)
    return failure();

  uint64_t numel = 1;
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic)
      return failure();
    numel *= static_cast<uint64_t>(dim);
  }
  return numel * elemBytes;
}

static void appendUniqueRoot(RootList &roots, Value root) {
  if (llvm::is_contained(roots, root))
    return;
  roots.push_back(root);
}

static RootList unionRoots(const RootList &lhs, const RootList &rhs) {
  RootList result = lhs;
  for (Value root : rhs)
    appendUniqueRoot(result, root);
  return result;
}

struct PlannerAnalysis {
  func::FuncOp func;
  DenseMap<Value, RootList> valueToRoots;
  SmallVector<Operation *> linearOps;
  DenseMap<Operation *, unsigned> opToIndex;
  SmallVector<RootInfo> roots;
  DenseMap<Value, unsigned> rootIndexByValue;
  DenseMap<Value, RootList> semanticConflictMap;
  bool failed = false;

  explicit PlannerAnalysis(func::FuncOp func) : func(func) {}

  RootList getRoots(Value value) const {
    auto it = valueToRoots.find(value);
    if (it == valueToRoots.end())
      return {};
    return it->second;
  }

  void setRoots(Value value, const RootList &roots) {
    if (roots.empty())
      return;
    valueToRoots[value] = roots;
  }

  void addRoot(Value value, Operation *defOp) {
    if (rootIndexByValue.count(value))
      return;

    auto space = getBufferAddressSpace(value.getType());
    if (!isPlannableLocalSpace(space))
      return;

    auto bytesOr = computeStaticBufferBytes(value);
    if (mlir::failed(bytesOr)) {
      defOp->emitError("requires a static local buffer shape and known element "
                       "byte size for memory planning");
      failed = true;
      return;
    }

    uint64_t slotCount = 1;
    if (auto attr =
            defOp->getAttrOfType<IntegerAttr>(pto::kPtoMultiBufferAttrName))
      slotCount = attr.getValue().getZExtValue();
    MemSpec spec = getMemSpec(getTargetArch(func), *space);
    uint64_t slotBytes = alignUp(*bytesOr, spec.alignmentBytes);

    RootInfo info;
    info.root = value;
    info.defOp = defOp;
    info.space = *space;
    info.slotBytes = slotBytes;
    info.totalBytes = slotBytes * slotCount;
    info.alignmentBytes = spec.alignmentBytes;
    info.slotCount = slotCount;
    info.stableOrder = roots.size();
    roots.push_back(info);
    rootIndexByValue[value] = roots.size() - 1;
    valueToRoots[value] = RootList{value};
  }

  void markUse(Value value, unsigned index) {
    for (Value root : getRoots(value)) {
      auto found = rootIndexByValue.find(root);
      if (found == rootIndexByValue.end())
        continue;
      RootInfo &info = roots[found->second];
      if (index < info.allocIndex)
        info.allocIndex = index;
      info.freeIndex = std::max(info.freeIndex, index);
    }
  }

  void addSemanticConflict(Value a, Value b) {
    if (a == b)
      return;
    if (!rootIndexByValue.count(a) || !rootIndexByValue.count(b))
      return;
    appendUniqueRoot(semanticConflictMap[a], b);
    appendUniqueRoot(semanticConflictMap[b], a);
  }

  bool hasSemanticConflict(Value a, Value b) const {
    if (a == b)
      return false;
    auto it = semanticConflictMap.find(a);
    if (it == semanticConflictMap.end())
      return false;
    return llvm::is_contained(it->second, b);
  }

  void recordLocalOpConflicts(Operation *op) {
    RootList operandRoots;
    RootList resultRoots;
    for (Value operand : op->getOperands())
      operandRoots = unionRoots(operandRoots, getRoots(operand));
    for (Value result : op->getResults())
      resultRoots = unionRoots(resultRoots, getRoots(result));

    if (operandRoots.empty() || resultRoots.empty())
      return;
    for (Value lhs : operandRoots) {
      for (Value rhs : resultRoots)
        addSemanticConflict(lhs, rhs);
    }
  }

  void seedForIterArgAliases(scf::ForOp forOp) {
    if (forOp.getRegion().empty())
      return;
    Block &body = forOp.getRegion().front();
    for (auto [iterArg, initArg] :
         llvm::zip(body.getArguments().drop_front(1), forOp.getInitArgs())) {
      setRoots(iterArg, getRoots(initArg));
    }
  }

  void walkRegion(Region &region) {
    for (Block &block : region) {
      for (Operation &opRef : block) {
        Operation *op = &opRef;
        unsigned index = linearOps.size();
        linearOps.push_back(op);
        opToIndex[op] = index;

        if (auto allocTile = dyn_cast<pto::AllocTileOp>(op)) {
          if (!allocTile.getAddr()) {
            addRoot(allocTile.getResult(), op);
            if (failed)
              return;
            auto found = rootIndexByValue.find(allocTile.getResult());
            if (found != rootIndexByValue.end()) {
              roots[found->second].allocIndex = index;
              roots[found->second].freeIndex = index;
            }
          }
        } else if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
          addRoot(alloc.getResult(), op);
          if (failed)
            return;
          auto found = rootIndexByValue.find(alloc.getResult());
          if (found != rootIndexByValue.end()) {
            roots[found->second].allocIndex = index;
            roots[found->second].freeIndex = index;
          }
        }

        if (auto bind = dyn_cast<pto::BindTileOp>(op)) {
          setRoots(bind.getResult(), getRoots(bind.getSource()));
        } else if (auto slotMarker = dyn_cast<pto::SlotMarkerOp>(op)) {
          setRoots(slotMarker.getResult(), getRoots(slotMarker.getSource()));
        } else if (auto select = dyn_cast<arith::SelectOp>(op)) {
          setRoots(select.getResult(),
                   unionRoots(getRoots(select.getTrueValue()),
                              getRoots(select.getFalseValue())));
        } else if (auto castOp = dyn_cast<memref::CastOp>(op)) {
          setRoots(castOp.getResult(), getRoots(castOp.getSource()));
        } else if (auto subview = dyn_cast<memref::SubViewOp>(op)) {
          setRoots(subview.getResult(), getRoots(subview.getSource()));
        } else if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(op)) {
          setRoots(reinterpret.getResult(), getRoots(reinterpret.getSource()));
        } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
          seedForIterArgAliases(forOp);
        }

        for (Value operand : op->getOperands())
          markUse(operand, index);

        for (Region &nested : op->getRegions()) {
          walkRegion(nested);
          if (failed)
            return;
        }

        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          if (ifOp.getNumResults() != 0) {
            auto thenYield =
                cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
            auto elseYield =
                cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
            for (auto [result, thenVal, elseVal] :
                 llvm::zip(ifOp.getResults(), thenYield.getResults(),
                           elseYield.getResults())) {
              setRoots(result,
                       unionRoots(getRoots(thenVal), getRoots(elseVal)));
            }
          }
        } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
          auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
          for (auto [result, initArg, yielded] :
               llvm::zip(forOp.getResults(), forOp.getInitArgs(),
                         yieldOp.getResults())) {
            setRoots(result, unionRoots(getRoots(initArg), getRoots(yielded)));
          }
        }

        recordLocalOpConflicts(op);
      }
    }
  }
};

static bool lifetimesOverlap(const RootInfo &lhs, const RootInfo &rhs) {
  return !(lhs.freeIndex < rhs.allocIndex || rhs.freeIndex < lhs.allocIndex);
}

static bool intervalsOverlap(uint64_t lhsOffset, uint64_t lhsSize,
                             uint64_t rhsOffset, uint64_t rhsSize) {
  return lhsOffset < rhsOffset + rhsSize && rhsOffset < lhsOffset + lhsSize;
}

static SmallVector<uint64_t> buildSlotOffsets(uint64_t base, uint64_t slotBytes,
                                              uint64_t slotCount) {
  SmallVector<uint64_t> offsets;
  offsets.reserve(slotCount);
  for (uint64_t slot = 0; slot < slotCount; ++slot)
    offsets.push_back(base + slot * slotBytes);
  return offsets;
}

static FailureOr<uint64_t> planReserveBufferBase(
    pto::ReserveBufferOp reserveOp, const MemSpec &spec,
    SmallVectorImpl<std::pair<uint64_t, uint64_t>> &occupied) {
  uint64_t sizeBytes =
      alignUp(static_cast<uint64_t>(reserveOp.getSize()), spec.alignmentBytes);
  llvm::sort(occupied, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  SmallVector<std::pair<uint64_t, uint64_t>> merged;
  for (const auto &interval : occupied) {
    if (merged.empty() || interval.first > merged.back().second) {
      merged.push_back(interval);
      continue;
    }
    merged.back().second = std::max(merged.back().second, interval.second);
  }

  uint64_t cursor = 0;
  for (const auto &interval : merged) {
    cursor = alignUp(cursor, spec.alignmentBytes);
    if (cursor + sizeBytes <= interval.first)
      return cursor;
    cursor = std::max(cursor, interval.second);
  }
  cursor = alignUp(cursor, spec.alignmentBytes);
  if (cursor + sizeBytes > spec.capacityBytes)
    return failure();
  occupied.push_back({cursor, cursor + sizeBytes});
  return cursor;
}

static LogicalResult
validateManualReserveBufferBase(pto::ReserveBufferOp reserveOp,
                                const MemSpec &spec) {
  auto baseAttr = reserveOp.getBaseAttr();
  if (!baseAttr)
    return reserveOp.emitError("expects 'base' when 'auto' is false");

  int64_t signedBase = baseAttr.getInt();
  if (signedBase < 0)
    return reserveOp.emitError(
        "expects 'base' to be non-negative when present");

  uint64_t base = static_cast<uint64_t>(signedBase);
  if (base % spec.alignmentBytes != 0) {
    return reserveOp.emitError("expects 'base' to be aligned to ")
           << spec.alignmentBytes << " bytes for "
           << stringifyEnum(reserveOp.getLocation().getAddressSpace());
  }

  uint64_t size = static_cast<uint64_t>(reserveOp.getSize());
  if (base > spec.capacityBytes || size > spec.capacityBytes - base) {
    return reserveOp.emitError("reserved range exceeds ")
           << stringifyEnum(reserveOp.getLocation().getAddressSpace())
           << " capacity: base " << base << " + size " << size << " > "
           << spec.capacityBytes << " bytes";
  }

  return success();
}

static LogicalResult
validateReserveBufferForPlanMemory(pto::ReserveBufferOp reserveOp,
                                   PTOArch arch) {
  MemSpec spec = getMemSpec(arch, reserveOp.getLocation().getAddressSpace());

  if (reserveOp.getAutoAlloc()) {
    if (reserveOp.getBaseAttr()) {
      return reserveOp.emitError(
          "unexpected pre-populated 'base' on auto reserve_buffer before "
          "pto-plan-memory; omit 'base' and let pto-plan-memory assign it");
    }
    return success();
  }

  if (mlir::failed(validateManualReserveBufferBase(reserveOp, spec)))
    return failure();

  return reserveOp.emitError(
      "pto.reserve_buffer with explicit 'base' (auto = false) is not "
      "supported in PlanMemory; use --pto-level=level3 or set auto = true");
}

class AllocTileOpAddPlannedAddressPattern
    : public OpRewritePattern<pto::AllocTileOp> {
public:
  explicit AllocTileOpAddPlannedAddressPattern(
      MLIRContext *context,
      DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets)
      : OpRewritePattern<pto::AllocTileOp>(context),
        buffer2Offsets(std::move(buffer2Offsets)) {}

  LogicalResult matchAndRewrite(pto::AllocTileOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getAddr())
      return failure();

    auto tileType = dyn_cast<TileBufType>(op.getResult().getType());
    if (!tileType)
      return failure();

    auto it = buffer2Offsets.find(op.getResult());
    if (it == buffer2Offsets.end() || it->second.empty())
      return failure();

    if (it->second.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "single alloc_tile root expects exactly one planned address");
    }

    Value addr = rewriter.create<arith::ConstantIntOp>(op.getLoc(),
                                                       it->second.front(), 64);
    auto planned = rewriter.create<pto::AllocTileOp>(
        op.getLoc(), tileType, addr,
        op.getValidRow() ? op.getValidRow() : Value(),
        op.getValidCol() ? op.getValidCol() : Value());
    for (NamedAttribute attr : op->getAttrs()) {
      if (attr.getName().getValue() == "operandSegmentSizes")
        continue;
      planned->setAttr(attr.getName(), attr.getValue());
    }

    rewriter.replaceOp(op, planned.getResult());
    return success();
  }

private:
  DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets;
};

static LogicalResult materializePlannedOffsets(
    func::FuncOp func, DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets) {
  RewritePatternSet patterns(func.getContext());
  patterns.add<MemrefAllocaOpToPointerCastOpPattern>(patterns.getContext(),
                                                     buffer2Offsets);
  patterns.add<AllocTileOpAddPlannedAddressPattern>(patterns.getContext(),
                                                    buffer2Offsets);
  if (mlir::failed(applyPatternsGreedily(func, std::move(patterns))))
    return failure();
  return success();
}

} // namespace

LogicalResult mlir::pto::runModernPlanMemory(func::FuncOp func,
                                             llvm::StringRef memMode,
                                             bool orderBySize) {
    if (!memMode.equals_insensitive("local")) {
      func.emitError("unsupported mem-mode '")
          << memMode << "'; only 'local' is currently implemented";
      return failure();
    }

    PlannerAnalysis analysis(func);
    analysis.walkRegion(func.getBody());
    if (analysis.failed) {
      return failure();
    }

    DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets;
    llvm::MapVector<AddressSpace, SmallVector<RootInfo *>> rootsBySpace;
    for (RootInfo &info : analysis.roots)
      rootsBySpace[info.space].push_back(&info);

    for (auto &entry : rootsBySpace) {
      AddressSpace space = entry.first;
      SmallVector<RootInfo *> &roots = entry.second;
      MemSpec spec = getMemSpec(getTargetArch(func), space);

      llvm::stable_sort(roots, [&](const RootInfo *lhs, const RootInfo *rhs) {
        if (orderBySize && lhs->totalBytes != rhs->totalBytes)
          return lhs->totalBytes > rhs->totalBytes;
        if (lhs->allocIndex != rhs->allocIndex)
          return lhs->allocIndex < rhs->allocIndex;
        return lhs->stableOrder < rhs->stableOrder;
      });

      SmallVector<RootInfo *> planned;
      uint64_t scopeRequiredBytes = 0;
      for (RootInfo *info : roots) {
        llvm::SmallSet<uint64_t, 16> candidateSet;
        candidateSet.insert(0);
        for (RootInfo *other : planned) {
          candidateSet.insert(other->offsets.front());
          candidateSet.insert(
              alignUp(other->offsets.front() + other->totalBytes,
                      info->alignmentBytes));
        }

        SmallVector<uint64_t> candidates(candidateSet.begin(),
                                         candidateSet.end());
        llvm::sort(candidates);

        std::optional<uint64_t> chosen;
        for (uint64_t candidate : candidates) {
          candidate = alignUp(candidate, info->alignmentBytes);
          if (candidate + info->totalBytes > spec.capacityBytes)
            continue;

          bool conflict = false;
          for (RootInfo *other : planned) {
            if (!intervalsOverlap(candidate, info->totalBytes,
                                  other->offsets.front(), other->totalBytes))
              continue;
            if (lifetimesOverlap(*info, *other) ||
                analysis.hasSemanticConflict(info->root, other->root)) {
              conflict = true;
              break;
            }
          }
          if (!conflict) {
            chosen = candidate;
            break;
          }
        }

        if (!chosen) {
          uint64_t fallback = 0;
          for (RootInfo *other : planned)
            fallback =
                std::max(fallback, other->offsets.front() + other->totalBytes);
          fallback = alignUp(fallback, info->alignmentBytes);
          chosen = fallback;
        }

        info->offsets =
            buildSlotOffsets(*chosen, info->slotBytes, info->slotCount);
        scopeRequiredBytes =
            std::max(scopeRequiredBytes, *chosen + info->totalBytes);
        planned.push_back(info);
      }

      if (scopeRequiredBytes > spec.capacityBytes) {
        func.emitError() << stringifyEnum(space) << " overflow, requires "
                         << (scopeRequiredBytes * 8) << " bits while "
                         << (spec.capacityBytes * 8) << " bits available";
        return failure();
      }

      for (RootInfo *info : roots)
        buffer2Offsets[info->root] = info->offsets;
    }

    DenseMap<AddressSpace, SmallVector<std::pair<uint64_t, uint64_t>>>
        occupiedBySpace;
    for (const RootInfo &info : analysis.roots) {
      if (info.offsets.empty())
        continue;
      occupiedBySpace[info.space].push_back(
          {info.offsets.front(), info.offsets.front() + info.totalBytes});
    }

    MLIRContext *ctx = func.getContext();
    PTOArch arch = getTargetArch(func);
    func.walk([&](pto::ReserveBufferOp reserveOp) -> WalkResult {
      if (mlir::failed(validateReserveBufferForPlanMemory(reserveOp, arch))) {
        analysis.failed = true;
        return WalkResult::interrupt();
      }

      AddressSpace space = reserveOp.getLocation().getAddressSpace();
      MemSpec spec = getMemSpec(arch, space);
      auto baseOr =
          planReserveBufferBase(reserveOp, spec, occupiedBySpace[space]);
      if (mlir::failed(baseOr)) {
        reserveOp.emitError("failed to reserve a local memory hole for "
                            "reserve_buffer");
        analysis.failed = true;
        return WalkResult::interrupt();
      }

      reserveOp->setAttr("base",
                         IntegerAttr::get(IntegerType::get(ctx, 32),
                                          static_cast<int32_t>(*baseOr)));
      return WalkResult::advance();
    });

    if (analysis.failed ||
        mlir::failed(materializePlannedOffsets(func, buffer2Offsets))) {
      return failure();
    }

    bool hasUnplannedAllocTile = false;
    func.walk([&](pto::AllocTileOp op) {
      if (op.getAddr())
        return;
      op.emitError(
          "PTOPlanMemory failed to assign an address to pto.alloc_tile");
      hasUnplannedAllocTile = true;
    });
    if (hasUnplannedAllocTile)
      return failure();
    return success();
}

namespace {
struct PlanMemoryModernPass
    : public PassWrapper<PlanMemoryModernPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanMemoryModernPass)

  PlanMemoryModernPass() = default;
  explicit PlanMemoryModernPass(const PlanMemoryOptions &options)
      : memMode(options.memMode), orderBySize(options.orderBySize) {}

  StringRef getArgument() const final { return "pto-plan-memory"; }
  StringRef getDescription() const final {
    return "Plan local memory using the modern PTO planner";
  }
  StringRef getName() const final { return "PlanMemoryModern"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::pto::PTODialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    for (func::FuncOp funcOp : moduleOp.getOps<func::FuncOp>()) {
      if (failed(mlir::pto::runModernPlanMemory(funcOp, memMode,
                                                orderBySize))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  std::string memMode = "local";
  bool orderBySize = false;
};
} // namespace

std::unique_ptr<Pass>
mlir::pto::createPlanMemoryModernPass(const PlanMemoryOptions &options) {
  return std::make_unique<PlanMemoryModernPass>(options);
}
