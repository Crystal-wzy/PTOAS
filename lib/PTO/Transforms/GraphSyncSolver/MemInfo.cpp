// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===------------- MemInfo.cpp ---- Graph Sync Solver ---------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/GraphSyncSolver/MemInfo.h"
#include "PTO/IR/PTO.h"
#include "../Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

using namespace mlir;
using namespace pto::syncsolver;

namespace mlir::pto::syncsolver {

static std::optional<int64_t> getBufferBitSize(Value value) {
  auto shaped = dyn_cast<ShapedType>(value.getType());
  if (!shaped || !shaped.hasStaticShape()) {
    return ShapedType::kDynamic;
  }
  Type elementType = shaped.getElementType();
  auto bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth == 0) {
    return ShapedType::kDynamic;
  }
  return shaped.getNumElements() * bitWidth;
}

llvm::SmallVector<int64_t> getAddresses(const llvm::SmallVector<Value> &addrs) {
  llvm::SmallVector<int64_t> offsets;
  for (auto addr : addrs) {
    auto constOp =
        llvm::dyn_cast_if_present<arith::ConstantOp>(addr.getDefiningOp());
    if (!constOp) {
      offsets.push_back(ShapedType::kDynamic);
      continue;
    }
    auto baseAddr =
        static_cast<int64_t>(cast<IntegerAttr>(constOp.getValue()).getInt());
    int64_t baseAddrInBits = baseAddr * pto::kBitsToByte;
    offsets.push_back(baseAddrInBits);
  }
  return offsets;
}

PointerLikeInfo getPointerLikeInfo(pto::PointerCastOp pointerCastOp) {
  PointerLikeInfo pointerLikeInfo(pointerCastOp);
  pointerLikeInfo.addresses = getAddresses(pointerCastOp.getAddrs());
  pointerLikeInfo.allocateSize = getBufferBitSize(pointerCastOp.getResult());
  if (!pointerLikeInfo.allocateSize.has_value()) {
    pointerCastOp.emitError("unknown buffer size");
    llvm_unreachable("unknown buffer size");
  }
  if (auto spaceAttr = GetBufferSpaceAttr(pointerCastOp.getResult())) {
    pointerLikeInfo.addressSpace = spaceAttr->getAddressSpace();
  }
  if (auto parentLoop = pointerCastOp->getParentOfType<LoopLikeOpInterface>()) {
    pointerLikeInfo.parentLoop = parentLoop;
  }
  return pointerLikeInfo;
}

// Walk back through metadata-only view ops (`pto.bind_tile`) to the
// nearest `pto.pointer_cast`. Used to anchor slot_marker MemInfo on its
// underlying multi-address alloc cast.
static pto::PointerCastOp findUnderlyingPointerCast(Value v) {
  int hops = 0;
  while (v && hops++ < 32) {
    Operation *op = v.getDefiningOp();
    if (!op)
      return {};
    if (auto pc = llvm::dyn_cast<pto::PointerCastOp>(op))
      return pc;
    if (auto bind = llvm::dyn_cast<pto::BindTileOp>(op)) {
      v = bind.getSource();
      continue;
    }
    return {};
  }
  return {};
}

// Build a MemInfo for a `pto.slot_marker` use. For a constant slot K the
// MemInfo carries just slot K's physical address so two const-slot
// accesses on different slots come back as non-conflicting via the
// existing `PointerLikeInfo::checkConflict` byte-range overlap. For a
// dynamic slot the MemInfo carries all N physical addresses; downstream
// `checkMultiBufferEventIdInfo` then deduces N event ids using the
// `(i % N) == (j % N)` slot-skipping rule, which is exactly the
// multi-buffer prefetch pattern.
//
// Note on `parentLoop`: `getPointerLikeInfo` records the parent loop of
// the cast op, which is typically outside the multi-buffer scf.for (the
// alloc/cast lives at function scope). The multi-buffer geometry, though,
// is keyed by the loop that *uses* the slot. We override `parentLoop`
// with the slot_marker's enclosing LoopLikeOpInterface so
// `getMultiBufferLoop` finds the right anchor.
static MemInfo getMemInfoForSlotMarker(pto::SlotMarkerOp slotMarker) {
  pto::PointerCastOp castOp = findUnderlyingPointerCast(slotMarker.getSource());
  if (!castOp) {
    return MemInfo(slotMarker.getResult(),
                   isWorkSpaceFuncArgument(slotMarker.getResult()));
  }

  PointerLikeInfo info = getPointerLikeInfo(castOp);

  IntegerAttr constSlotAttr;
  if (matchPattern(slotMarker.getSlot(), m_Constant(&constSlotAttr)) &&
      info.addresses.size() > 1) {
    int64_t slotIdx = constSlotAttr.getValue().getSExtValue();
    if (slotIdx >= 0 && slotIdx < static_cast<int64_t>(info.addresses.size())) {
      int64_t picked = info.addresses[static_cast<size_t>(slotIdx)];
      info.addresses.clear();
      info.addresses.push_back(picked);
    }
  }

  if (auto useLoop =
          slotMarker->template getParentOfType<LoopLikeOpInterface>()) {
    info.parentLoop = useLoop;
  }

  return MemInfo(slotMarker.getResult(), info);
}

MemInfo getMemInfo(Value val) {
  if (auto *defOp = val.getDefiningOp()) {
    if (auto pointerCastOp = llvm::dyn_cast<pto::PointerCastOp>(defOp)) {
      return MemInfo(val, getPointerLikeInfo(pointerCastOp));
    }
    if (auto slotMarker = llvm::dyn_cast<pto::SlotMarkerOp>(defOp)) {
      return getMemInfoForSlotMarker(slotMarker);
    }
  }
  return MemInfo(val, isWorkSpaceFuncArgument(val));
}

MemInfo getMemInfo(const llvm::SmallVector<int64_t> &addrs) {
  MemInfo memInfo;
  memInfo.pointerLikeInfo = PointerLikeInfo();
  memInfo.pointerLikeInfo->addresses = addrs;
  memInfo.pointerLikeInfo->allocateSize = 1;
  memInfo.pointerLikeInfo->addressSpace = pto::AddressSpace::Zero;
  return memInfo;
}

bool PointerLikeInfo::checkConflict(const PointerLikeInfo &pointerLikeInfo1,
                                    const PointerLikeInfo &pointerLikeInfo2,
                                    std::optional<int64_t> lcmLen,
                                    std::optional<int64_t> eventIdNum) {
  if (!pointerLikeInfo1.addressSpace.has_value() ||
      !pointerLikeInfo2.addressSpace.has_value()) {
    return false;
  }
  if (pointerLikeInfo1.addressSpace.value() !=
      pointerLikeInfo2.addressSpace.value()) {
    return false;
  }

  auto &offsets1 = pointerLikeInfo1.addresses;
  auto &offsets2 = pointerLikeInfo2.addresses;
  auto sz1 = static_cast<int64_t>(offsets1.size());
  auto sz2 = static_cast<int64_t>(offsets2.size());

  int64_t len1 = sz1;
  int64_t len2 = sz2;
  if (lcmLen.has_value()) {
    len1 = lcmLen.value();
    len2 = lcmLen.value();
  }

  for (int64_t i = 0; i < len1; i++) {
    for (int64_t j = 0; j < len2; j++) {
      if (eventIdNum.has_value()) {
        if ((i % eventIdNum.value()) == (j % eventIdNum.value())) {
          continue;
        }
      }

      auto offset1 = offsets1[i % sz1];
      auto offset2 = offsets2[j % sz2];
      if (offset1 == ShapedType::kDynamic || offset2 == ShapedType::kDynamic) {
        return true;
      }

      assert(pointerLikeInfo1.allocateSize.has_value());
      assert(pointerLikeInfo2.allocateSize.has_value());
      auto allocSz1 = pointerLikeInfo1.allocateSize.value();
      auto allocSz2 = pointerLikeInfo2.allocateSize.value();

      if ((allocSz1 != ShapedType::kDynamic) &&
          (offset1 + allocSz1 < offset2 + 1)) {
        continue;
      }
      if ((allocSz2 != ShapedType::kDynamic) &&
          (offset2 + allocSz2 < offset1 + 1)) {
        continue;
      }
      return true;
    }
  }
  return false;
}

bool MemInfo::checkConflict(const MemInfo &memInfo1, const MemInfo &memInfo2,
                            std::optional<int64_t> lcmLen,
                            std::optional<int64_t> eventIdNum) {
  if (memInfo1.pointerLikeInfo.has_value() &&
      memInfo2.pointerLikeInfo.has_value()) {
    return PointerLikeInfo::checkConflict(memInfo1.pointerLikeInfo.value(),
                                          memInfo2.pointerLikeInfo.value(),
                                          lcmLen, eventIdNum);
  }
  return memInfo1.value == memInfo2.value;
}

bool isWorkSpaceFuncArgument(Value value) {
  auto blockArg = dyn_cast_if_present<BlockArgument>(value);
  if (!blockArg) {
    return false;
  }
  auto *block = blockArg.getOwner();
  if (!block) {
    return false;
  }
  auto *region = block->getParent();
  if (!region) {
    return false;
  }
  auto *parentOp = region->getParentOp();
  if (!parentOp) {
    return false;
  }
  auto funcOp = dyn_cast<func::FuncOp>(parentOp);
  if (!funcOp) {
    return false;
  }
  return false;
}

} // namespace mlir::pto::syncsolver
