// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- AllocToPointerCast.cpp - convert memref.AllocOp to pto.pointercastOp.//
//===----------------------------------------------------------------------===//

#include "AllocToPointerCast.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace mlir::pto;

namespace {} // namespace

namespace {
constexpr uint64_t kDefaultAllocAlignmentBytes = 4096;
constexpr size_t kDynamicValidShapeRank = 2;

static uint64_t alignUpToDefault(uint64_t value) {
  return ((value + kDefaultAllocAlignmentBytes - 1) /
          kDefaultAllocAlignmentBytes) *
         kDefaultAllocAlignmentBytes;
}

static TileBufConfigAttr inferBindTileConfig(Value root, Operation *diagOp) {
  TileBufConfigAttr configAttr;
  for (Operation *user : root.getUsers()) {
    auto bind = dyn_cast<pto::BindTileOp>(user);
    if (!bind || bind.getSource() != root)
      continue;
    if (!configAttr) {
      configAttr = bind.getConfigAttr();
      continue;
    }
    if (configAttr != bind.getConfigAttr()) {
      diagOp->emitWarning(
          "alloc has multiple bind_tile users with different configs; "
          "using the first one");
      break;
    }
  }
  return configAttr;
}

static SmallVector<uint64_t> getAllocatedOffsets(
    Value root, Operation *allocLikeOp, BaseMemRefType memRefType,
    const DenseMap<Value, SmallVector<uint64_t>> &buffer2Offsets,
    uint64_t &fallbackNextOffset) {
  auto iter = buffer2Offsets.find(root);
  SmallVector<uint64_t> offsets;
  if (iter != buffer2Offsets.end())
    offsets = iter->second;

  if (!offsets.empty())
    return offsets;

  // Estimate buffer size (best-effort). Most PTO tile buffers are 32x32 and
  // naturally align to 4096 bytes.
  uint64_t bytes = kDefaultAllocAlignmentBytes;
  if (auto memrefTy = dyn_cast<MemRefType>(memRefType)) {
    uint64_t elemBytes = getPTOStorageElemByteSize(memrefTy.getElementType());
    if (elemBytes != 0) {
      uint64_t numel = 1;
      bool allStatic = true;
      for (int64_t d : memrefTy.getShape()) {
        if (d == ShapedType::kDynamic) {
          allStatic = false;
          break;
        }
        numel *= static_cast<uint64_t>(d);
      }
      if (allStatic && numel != 0)
        bytes = numel * elemBytes;
    }
  }

  uint64_t slotCount = 1;
  if (auto attr =
          allocLikeOp->getAttrOfType<IntegerAttr>(pto::kPtoMultiBufferAttrName)) {
    slotCount = attr.getValue().getZExtValue();
  }

  uint64_t stride = alignUpToDefault(bytes);
  uint64_t off = fallbackNextOffset;
  for (uint64_t i = 0; i < slotCount; ++i)
    offsets.push_back(off + i * stride);

  fallbackNextOffset +=
      std::max<uint64_t>(stride * slotCount, kDefaultAllocAlignmentBytes);
  return offsets;
}

static std::pair<Value, Value> getDynamicValidShapeValues(memref::AllocOp op) {
  Value vRow;
  Value vCol;
  auto dynSizes = op.getDynamicSizes();
  if (dynSizes.size() >= kDynamicValidShapeRank) {
    vRow = dynSizes[0];
    vCol = dynSizes[1];
  } else if (dynSizes.size() == 1) {
    vCol = dynSizes[0];
  }
  return {vRow, vCol};
}
} // namespace

LogicalResult MemrefAllocaOpToPointerCastOpPattern::matchAndRewrite(
    memref::AllocOp op, PatternRewriter &rewriter) const {
  const auto &currentMemRefType = cast<BaseMemRefType>(op.getType());
  TileBufConfigAttr configAttr = inferBindTileConfig(op.getResult(), op);
  SmallVector<uint64_t> offsets =
      getAllocatedOffsets(op.getResult(), op, currentMemRefType,
                          buffer2Offsets, fallbackNextOffset);
  SmallVector<Value> addrs;
  addrs.reserve(offsets.size());
  for (uint64_t offset : offsets) {
    auto constantIntOffsetOp =
        rewriter.create<arith::ConstantIntOp>(op->getLoc(), offset, 64);
    addrs.push_back(constantIntOffsetOp);
  }

  auto [vRow, vCol] = getDynamicValidShapeValues(op);
  auto ptoPointerCastOp = rewriter.create<pto::PointerCastOp>(
      op.getLoc(), currentMemRefType, ValueRange(addrs), vRow ? vRow : Value(),
      vCol ? vCol : Value(), configAttr);

  rewriter.replaceOp(op, ptoPointerCastOp->getResults());
  return success();
}
