// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOResolveBufferSelect.cpp -----------------------------------------===//
//
// Lowering for multi-buffer slot selection.
//
// Consumes `pto.slot_marker %src[%k] : memref<...>` ops written by
// PTOViewToMemref while lowering `pto.multi_tile_get`. By the time this pass
// runs, PTOPlanMemory has already converted the underlying `memref.alloc` to
// a multi-address `pto.pointer_cast(addr0, ..., addrN-1)`. This pass picks
// the right per-slot address(es) for each slot_marker use:
//
//   * Constant slot k: emit a single-address `pto.pointer_cast(addrK)` at
//     the use site and replace the slot_marker.
//   * Dynamic slot %k: emit N single-address per-slot pointer_casts and
//     pick one via an N-way `arith.select` chain. The user's SSA selects
//     the slot -- this pass does NOT synthesize `iv mod N`.
//
// The original multi-address `pto.pointer_cast` is left in IR as the
// "alloc anchor" so future sync extensions can still see the multi-buffer
// geometry (e.g. for `set_flag_dyn` / `wait_flag_dyn` derivation).
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTORESOLVEBUFFERSELECT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

#define DEBUG_TYPE "pto-resolve-buffer-select"

using namespace mlir;

namespace {

static Value ensureI64(Value value, IRRewriter &rewriter, Location loc) {
  if (!value)
    return {};
  if (value.getType().isInteger(64))
    return value;
  if (value.getType().isIndex())
    return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), value);
  if (isa<IntegerType>(value.getType()))
    return rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), value);
  return {};
}

static bool getTilePointerStrides(pto::TileBufType type, int64_t &rowStride,
                                  int64_t &colStride) {
  auto shape = type.getShape();
  if (shape.size() != 2 || llvm::is_contained(shape, ShapedType::kDynamic))
    return false;

  auto config = type.getConfigAttr();
  int32_t bl = static_cast<int32_t>(config.getBLayout().getValue());
  int32_t sl = static_cast<int32_t>(config.getSLayout().getValue());
  if (sl == 0) {
    bool rowPlusOne =
        type.getCompactModeI32() ==
        static_cast<int32_t>(pto::CompactMode::RowPlusOne);
    rowStride = bl == 1 ? 1 : shape[1] + (rowPlusOne ? 1 : 0);
    colStride = bl == 1 ? shape[0] + (rowPlusOne ? 1 : 0) : 1;
    return true;
  }

  unsigned elemBytes = pto::getPTOStorageElemByteSize(type.getElementType());
  if (elemBytes == 0)
    return false;
  int64_t innerRows = 1;
  int64_t innerCols = 1;
  int32_t fractal = config.getSFractalSize().getInt();
  if (fractal == 1024) {
    innerRows = 16;
    innerCols = 16;
  } else if (fractal == 32) {
    innerRows = 16;
    innerCols = 2;
  } else if (fractal == 512 && sl == 1) {
    innerRows = 16;
    innerCols = 32 / elemBytes;
  } else if (fractal == 512 && sl == 2) {
    innerRows = 32 / elemBytes;
    innerCols = 16;
  } else {
    return false;
  }

  if (bl == 1) {
    if (sl != 1)
      return false;
    rowStride = innerCols;
    colStride = shape[0];
  } else {
    rowStride = shape[1];
    colStride = innerRows;
  }
  return true;
}

static Value computeTileAddress(Value value, IRRewriter &rewriter,
                                Location loc) {
  if (auto alloc = value.getDefiningOp<pto::AllocTileOp>())
    return ensureI64(alloc.getAddr(), rewriter, loc);
  if (auto subview = value.getDefiningOp<pto::SubViewOp>()) {
    Value base = computeTileAddress(subview.getSource(), rewriter, loc);
    auto sourceType = subview.getSource().getType();
    int64_t rowStride = 0;
    int64_t colStride = 0;
    if (!base || !getTilePointerStrides(sourceType, rowStride, colStride) ||
        subview.getOffsets().size() != 2)
      return {};
    Value row = ensureI64(subview.getOffsets()[0], rewriter, loc);
    Value col = ensureI64(subview.getOffsets()[1], rewriter, loc);
    if (!row || !col)
      return {};
    Value rowScale = rewriter.create<arith::ConstantIntOp>(loc, rowStride, 64);
    Value colScale = rewriter.create<arith::ConstantIntOp>(loc, colStride, 64);
    row = rewriter.create<arith::MulIOp>(loc, row, rowScale);
    col = rewriter.create<arith::MulIOp>(loc, col, colScale);
    Value elements = rewriter.create<arith::AddIOp>(loc, row, col);
    int64_t elemBytes = static_cast<int64_t>(
        pto::getPTOStorageElemByteSize(sourceType.getElementType()));
    if (elemBytes == 0)
      return {};
    Value byteScale = rewriter.create<arith::ConstantIntOp>(loc, elemBytes, 64);
    Value bytes = rewriter.create<arith::MulIOp>(loc, elements, byteScale);
    return rewriter.create<arith::AddIOp>(loc, base, bytes);
  }
  return {};
}

static pto::TileBufType getSubviewPhysicalType(pto::SubViewOp op) {
  pto::TileBufType sourceType = op.getSource().getType();
  pto::TileBufType resultType = op.getResult().getType();
  return pto::TileBufType::get(
      op.getContext(), sourceType.getShape(), resultType.getElementType(),
      resultType.getMemorySpace(), resultType.getValidShape(),
      resultType.getConfigAttr());
}

static Value getSubviewValidOperand(pto::SubViewOp op,
                                    pto::TileBufType physicalType,
                                    unsigned dim, IRRewriter &rewriter) {
  Value operand = dim == 0 ? op.getValidRow() : op.getValidCol();
  ArrayRef<int64_t> validShape = physicalType.getValidShape();
  if (validShape.size() <= dim || validShape[dim] >= 0)
    return {};
  if (operand)
    return operand;
  ArrayRef<int64_t> shape = physicalType.getShape();
  if (shape.size() > dim && shape[dim] != ShapedType::kDynamic)
    return rewriter.create<arith::ConstantIndexOp>(op.getLoc(), shape[dim]);
  return {};
}

static LogicalResult resolveTileNativeSubviews(ModuleOp module,
                                               MLIRContext *ctx) {
  SmallVector<pto::SubViewOp, 16> subviews;
  module.walk([&](pto::SubViewOp op) { subviews.push_back(op); });
  for (pto::SubViewOp op : subviews) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Value addr = computeTileAddress(op.getResult(), rewriter, op.getLoc());
    // A tile function argument is a symbolic runtime-bound handle. Keep its
    // subview tile-native; only planned local roots can be normalized to an
    // addressed alloc_tile here.
    if (!addr)
      continue;
    pto::TileBufType physicalType = getSubviewPhysicalType(op);
    auto alloc = rewriter.create<pto::AllocTileOp>(
        op.getLoc(), physicalType, addr,
        getSubviewValidOperand(op, physicalType, 0, rewriter),
        getSubviewValidOperand(op, physicalType, 1, rewriter));
    alloc->setAttr("pto.view_semantics", rewriter.getStringAttr("subview"));
    rewriter.replaceOp(op, alloc.getResult());
  }
  return success();
}

static FailureOr<uint64_t> getStaticSlotBytes(pto::TileBufType slotType) {
  uint64_t elemBytes = pto::getPTOStorageElemByteSize(slotType.getElementType());
  if (elemBytes == 0)
    return failure();
  uint64_t bytes = elemBytes;
  for (int64_t dim : slotType.getShape()) {
    if (dim == ShapedType::kDynamic)
      return failure();
    bytes *= static_cast<uint64_t>(dim);
  }
  return bytes;
}

static LogicalResult getMultiTileAddresses(pto::AllocMultiTileOp alloc,
                                           IRRewriter &rewriter,
                                           SmallVectorImpl<Value> &addrs) {
  uint32_t count = alloc.getResult().getType().getCount();
  if (auto planned = alloc->getAttrOfType<DenseI64ArrayAttr>(
          pto::kPtoMultiBufferAddrsAttrName)) {
    if (planned.size() != count)
      return alloc.emitError("planned address count does not match slot count");
    for (int64_t address : planned.asArrayRef())
      addrs.push_back(rewriter.create<arith::ConstantIntOp>(
          alloc.getLoc(), address, 64));
    return success();
  }

  Value base = alloc.getAddr();
  if (!base)
    return alloc.emitError(
        "has neither a level3 base address nor planner-assigned slot addresses");
  auto slotBytes = getStaticSlotBytes(alloc.getResult().getType().getSlotType());
  if (failed(slotBytes))
    return alloc.emitError(
        "requires a static slot shape and known element byte size");

  addrs.push_back(base);
  for (uint32_t slot = 1; slot < count; ++slot) {
    Value offset = rewriter.create<arith::ConstantIntOp>(
        alloc.getLoc(), static_cast<int64_t>(slot * *slotBytes), 64);
    addrs.push_back(
        rewriter.create<arith::AddIOp>(alloc.getLoc(), base, offset));
  }
  return success();
}

static LogicalResult resolveTileNativeMultiGets(ModuleOp module,
                                                MLIRContext *ctx) {
  SmallVector<pto::MultiTileGetOp, 8> gets;
  module.walk([&](pto::MultiTileGetOp op) { gets.push_back(op); });

  for (pto::MultiTileGetOp op : gets) {
    auto alloc = op.getSource().getDefiningOp<pto::AllocMultiTileOp>();
    if (!alloc)
      return op.emitError(
          "currently requires a direct pto.alloc_multi_tile source");

    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    SmallVector<Value, 8> addrs;
    if (failed(getMultiTileAddresses(alloc, rewriter, addrs)))
      return failure();

    Value selectedAddr;
    IntegerAttr constSlotAttr;
    if (matchPattern(op.getSlot(), m_Constant(&constSlotAttr))) {
      int64_t slot = constSlotAttr.getValue().getSExtValue();
      if (slot < 0 || slot >= static_cast<int64_t>(addrs.size()))
        return op.emitError("constant slot is outside planned address range");
      selectedAddr = addrs[static_cast<size_t>(slot)];
    } else {
      selectedAddr = addrs.front();
      for (uint32_t slot = 1; slot < addrs.size(); ++slot) {
        Value slotValue = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), slot);
        Value matches = rewriter.create<arith::CmpIOp>(
            op.getLoc(), arith::CmpIPredicate::eq, op.getSlot(), slotValue);
        selectedAddr = rewriter.create<arith::SelectOp>(
            op.getLoc(), matches, addrs[slot], selectedAddr);
      }
    }

    auto slotHandle = rewriter.create<pto::AllocTileOp>(
        op.getLoc(), op.getResult().getType(), selectedAddr,
        alloc.getValidRow() ? alloc.getValidRow() : Value(),
        alloc.getValidCol() ? alloc.getValidCol() : Value());
    rewriter.replaceOp(op, slotHandle.getResult());
  }

  SmallVector<pto::AllocMultiTileOp, 8> allocs;
  module.walk([&](pto::AllocMultiTileOp op) { allocs.push_back(op); });
  for (pto::AllocMultiTileOp alloc : allocs) {
    if (!alloc.getResult().use_empty())
      return alloc.emitError(
          "has unsupported uses after resolving pto.multi_tile_get");
    alloc.erase();
  }
  return success();
}

/// Walk back through pure metadata ops (`pto.bind_tile`, `pto.slot_marker`)
/// to find the root multi-address `pto.pointer_cast` that this view ties
/// to. Returns nullptr if the chain does not terminate on a
/// `pto.pointer_cast` -- in which case this slot_marker is not a multi-
/// buffer reference and should not be touched.
static pto::PointerCastOp lookupRootPointerCast(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto pc = dyn_cast<pto::PointerCastOp>(def))
      return pc;
    if (auto bind = dyn_cast<pto::BindTileOp>(def)) {
      v = bind.getSource();
      continue;
    }
    if (auto sm = dyn_cast<pto::SlotMarkerOp>(def)) {
      // Nested slot_marker should not happen (verifier disallows nested
      // multi_tile_get), but follow the chain defensively.
      v = sm.getSource();
      continue;
    }
    return {};
  }
  return {};
}

/// Lookup tile-buf config from the existing PointerCastOp's optional attr.
/// Returns nullptr if not set.
static Attribute getCastConfigAttr(pto::PointerCastOp root) {
  auto cfg = root.getConfig();
  if (cfg.has_value())
    return *cfg;
  return Attribute();
}

/// Create a fresh single-address pointer_cast that aliases slot `slotIdx`
/// of `root`. The result type matches `targetType`. `vRow` / `vCol` and
/// `config` are forwarded from the root.
static Value emitSlotPointerCast(IRRewriter &rewriter, Location loc,
                                 pto::PointerCastOp root, uint32_t slotIdx,
                                 Type targetType) {
  auto rootAddrs = root.getAddrs();
  assert(slotIdx < rootAddrs.size() && "slot index out of range");
  Value vRow = root.getValidRow();
  Value vCol = root.getValidCol();
  Attribute cfg = getCastConfigAttr(root);
  auto pc = rewriter.create<pto::PointerCastOp>(
      loc, targetType, ValueRange{rootAddrs[slotIdx]},
      vRow ? vRow : Value(), vCol ? vCol : Value(), cfg);
  return pc.getResult();
}

struct PTOResolveBufferSelectPass
    : public mlir::pto::impl::PTOResolveBufferSelectBase<
          PTOResolveBufferSelectPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOResolveBufferSelectPass)

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    if (failed(resolveTileNativeMultiGets(mod, ctx))) {
      signalPassFailure();
      return;
    }
    if (failed(resolveTileNativeSubviews(mod, ctx))) {
      signalPassFailure();
      return;
    }

    SmallVector<pto::SlotMarkerOp, 8> markers;
    mod.walk([&](pto::SlotMarkerOp op) { markers.push_back(op); });
    if (markers.empty())
      return;

    for (auto op : markers) {
      IRRewriter rewriter(ctx);
      rewriter.setInsertionPoint(op);
      Location loc = op.getLoc();

      // Find the root multi-address pto.pointer_cast that this slot_marker
      // refers to. If the chain does not land on one, the marker is not a
      // multi-buffer reference; downgrade silently by forwarding source.
      pto::PointerCastOp root = lookupRootPointerCast(op.getSource());
      if (!root) {
        rewriter.replaceOp(op, op.getSource());
        continue;
      }

      auto rootAddrs = root.getAddrs();
      uint32_t n = static_cast<uint32_t>(rootAddrs.size());
      if (n < 2) {
        // Single-address root: treat slot_marker as identity.
        rewriter.replaceOp(op, op.getSource());
        continue;
      }
      if (n > mlir::pto::kPtoMultiBufferMaxNum) {
        op.emitError() << "underlying pointer_cast has " << n
                       << " addresses, exceeds max "
                       << mlir::pto::kPtoMultiBufferMaxNum;
        signalPassFailure();
        return;
      }

      Type targetType = op.getResult().getType();

      // Constant slot: emit a single-address pointer_cast for that slot.
      IntegerAttr constSlotAttr;
      if (matchPattern(op.getSlot(), m_Constant(&constSlotAttr))) {
        int64_t slotI = constSlotAttr.getValue().getSExtValue();
        if (slotI < 0 || slotI >= static_cast<int64_t>(n)) {
          op.emitError() << "constant slot " << slotI
                         << " is out of range for "
                         << n << " physical buffers";
          signalPassFailure();
          return;
        }
        Value picked = emitSlotPointerCast(rewriter, loc, root,
                                           static_cast<uint32_t>(slotI),
                                           targetType);
        rewriter.replaceOp(op, picked);
        continue;
      }

      // Dynamic slot: emit per-slot single-addr casts + N-way arith.select.
      // The select chain uses the user-supplied SSA verbatim -- ptoas does
      // NOT replace it with `iv mod N`.
      SmallVector<Value, 8> slotMems;
      slotMems.reserve(n);
      for (uint32_t i = 0; i < n; ++i)
        slotMems.push_back(
            emitSlotPointerCast(rewriter, loc, root, i, targetType));

      Value selected = slotMems[0];
      Value slot = op.getSlot();
      for (uint32_t i = 1; i < n; ++i) {
        Value iIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);
        Value isThis = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, slot, iIdx);
        selected = rewriter.create<arith::SelectOp>(loc, isThis, slotMems[i],
                                                    selected);
      }
      rewriter.replaceOp(op, selected);
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOResolveBufferSelectPass() {
  return std::make_unique<PTOResolveBufferSelectPass>();
}
