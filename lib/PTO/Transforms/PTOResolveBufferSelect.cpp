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
