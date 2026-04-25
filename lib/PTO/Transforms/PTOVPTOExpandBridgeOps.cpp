// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <algorithm>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVPTOEXPANDBRIDGEOPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static pto::AddressSpaceAttr getPointerMemorySpace(Attribute memorySpace,
                                                   MLIRContext *ctx) {
  if (auto addrSpace = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace))
    return addrSpace;
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memorySpace))
    return pto::AddressSpaceAttr::get(
        ctx, static_cast<pto::AddressSpace>(intAttr.getInt()));
  return pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::GM);
}

static Value materializeBufferPointer(Value value, PatternRewriter &rewriter,
                                      Location loc) {
  if (!value)
    return {};

  if (isa<pto::PtrType>(value.getType()))
    return value;

  auto memrefType = dyn_cast<MemRefType>(value.getType());
  if (!memrefType)
    return {};

  auto ptrType =
      pto::PtrType::get(rewriter.getContext(), memrefType.getElementType(),
                        getPointerMemorySpace(memrefType.getMemorySpace(),
                                              rewriter.getContext()));
  return rewriter.create<pto::CastPtrOp>(loc, ptrType, value).getResult();
}

static Value offsetBufferPointer(Value basePtr, Type elementType,
                                 Value elementOffset,
                                 PatternRewriter &rewriter, Location loc) {
  if (!basePtr)
    return {};

  Value offsetIndex = elementOffset;
  if (!offsetIndex.getType().isIndex())
    offsetIndex = rewriter.create<arith::IndexCastUIOp>(loc,
                                                        rewriter.getIndexType(),
                                                        elementOffset);
  return rewriter.create<pto::AddPtrOp>(loc, basePtr.getType(), basePtr,
                                        offsetIndex);
}

static bool isKnownOne(Value value) {
  APInt intValue;
  return value && matchPattern(value, m_ConstantInt(&intValue)) &&
         intValue.isOne();
}

static bool shouldRestoreDmaLoopSize(Value loop1Count, Value loop2Count) {
  if (!loop1Count)
    return false;
  return !isKnownOne(loop1Count) || !isKnownOne(loop2Count);
}

static SmallVector<pto::DmaLoopConfig> collectLoopConfigs(ValueRange counts,
                                                          ValueRange srcStrides,
                                                          ValueRange dstStrides) {
  SmallVector<pto::DmaLoopConfig> loops;
  loops.reserve(counts.size());
  for (auto [count, srcStride, dstStride] :
       llvm::zip(counts, srcStrides, dstStrides))
    loops.push_back({count, srcStride, dstStride});
  return loops;
}

static Value offsetPointerByBytes(Value basePtr, Value byteOffset,
                                  PatternRewriter &rewriter, Location loc) {
  if (!basePtr)
    return {};

  Value basePtrValue = materializeBufferPointer(basePtr, rewriter, loc);
  auto ptrType = dyn_cast_or_null<pto::PtrType>(basePtrValue.getType());
  if (!ptrType)
    return {};

  APInt constOffset;
  if (matchPattern(byteOffset, m_ConstantInt(&constOffset)) && constOffset.isZero())
    return basePtrValue;

  auto bytePtrType =
      pto::PtrType::get(rewriter.getContext(), rewriter.getI8Type(),
                        ptrType.getMemorySpace());
  Value bytePtr =
      rewriter.create<pto::CastPtrOp>(loc, bytePtrType, basePtrValue);
  Value offsetIndex = byteOffset;
  if (!offsetIndex.getType().isIndex())
    offsetIndex =
        rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getIndexType(),
                                              offsetIndex);
  Value advanced =
      rewriter.create<pto::AddPtrOp>(loc, bytePtrType, bytePtr, offsetIndex);
  return rewriter.create<pto::CastPtrOp>(loc, ptrType, advanced);
}

static Value buildAccumulatedByteOffset(Location loc, Value baseOffset,
                                        Value indexI64, Value stride,
                                        PatternRewriter &rewriter) {
  Value delta = rewriter.create<arith::MulIOp>(loc, indexI64, stride);
  return rewriter.create<arith::AddIOp>(loc, baseOffset, delta);
}

static Value packLoopPair(Location loc, Value low, Value high,
                          PatternRewriter &rewriter) {
  Value shift = rewriter.create<arith::ConstantIntOp>(loc, 40, 64);
  Value highShifted = rewriter.create<arith::ShLIOp>(loc, high, shift);
  return rewriter.create<arith::OrIOp>(loc, highShifted, low);
}

static Value packLoopSize(Location loc, Value loop2, Value loop1,
                          PatternRewriter &rewriter) {
  Value shift = rewriter.create<arith::ConstantIntOp>(loc, 21, 64);
  Value loop2Shifted = rewriter.create<arith::ShLIOp>(loc, loop2, shift);
  return rewriter.create<arith::OrIOp>(loc, loop2Shifted, loop1);
}

static Value packMte2NzPara(Location loc, Value groupCount, Value dstLoop2Stride,
                            Value dstLoop3Stride, Value dstLoop4Stride,
                            PatternRewriter &rewriter) {
  Value shift16 = rewriter.create<arith::ConstantIntOp>(loc, 16, 64);
  Value shift32 = rewriter.create<arith::ConstantIntOp>(loc, 32, 64);
  Value shift48 = rewriter.create<arith::ConstantIntOp>(loc, 48, 64);
  Value loop2Bits =
      rewriter.create<arith::ShLIOp>(loc, dstLoop2Stride, shift16);
  Value loop3Bits =
      rewriter.create<arith::ShLIOp>(loc, dstLoop3Stride, shift32);
  Value loop4Bits =
      rewriter.create<arith::ShLIOp>(loc, dstLoop4Stride, shift48);
  Value low = rewriter.create<arith::OrIOp>(loc, groupCount, loop2Bits);
  Value high = rewriter.create<arith::OrIOp>(loc, loop3Bits, loop4Bits);
  return rewriter.create<arith::OrIOp>(loc, low, high);
}

static Value packCopyMatrixCcToGmXm(Location loc, Value sid, Value nSize,
                                    Value mSize, Value dstStride,
                                    PatternRewriter &rewriter) {
  Value nShift4 = rewriter.create<arith::ConstantIntOp>(loc, 4, 64);
  Value mShift16 = rewriter.create<arith::ConstantIntOp>(loc, 16, 64);
  Value dstShift32 = rewriter.create<arith::ConstantIntOp>(loc, 32, 64);
  Value nBits = rewriter.create<arith::ShLIOp>(loc, nSize, nShift4);
  Value mBits = rewriter.create<arith::ShLIOp>(loc, mSize, mShift16);
  Value dstStrideBits = rewriter.create<arith::ShLIOp>(loc, dstStride, dstShift32);
  Value sidMask = rewriter.create<arith::ConstantIntOp>(loc, 0xf, 64);
  Value sidBits = rewriter.create<arith::AndIOp>(loc, sid, sidMask);
  Value xmLow = rewriter.create<arith::OrIOp>(loc, sidBits, nBits);
  xmLow = rewriter.create<arith::OrIOp>(loc, xmLow, mBits);
  return rewriter.create<arith::OrIOp>(loc, xmLow, dstStrideBits);
}

static Value packCopyMatrixCcToGmXt(Location loc, Value srcStride,
                                    Value unitFlagCtrl, Value quantPre,
                                    Value reluPreMode, Value l2CacheCtrl,
                                    Value nz2ndEn, Value channelSplitEn,
                                    Value nz2dnEn,
                                    PatternRewriter &rewriter) {
  Value l2CacheShift16 = rewriter.create<arith::ConstantIntOp>(loc, 16, 64);
  Value unitFlagShift32 = rewriter.create<arith::ConstantIntOp>(loc, 32, 64);
  Value quantBlockBitShift29 =
      rewriter.create<arith::ConstantIntOp>(loc, 29, 64);
  Value quantFieldShift34 = rewriter.create<arith::ConstantIntOp>(loc, 34, 64);
  Value reluShift39 = rewriter.create<arith::ConstantIntOp>(loc, 39, 64);
  Value channelSplitShift42 =
      rewriter.create<arith::ConstantIntOp>(loc, 42, 64);
  Value nz2ndShift43 = rewriter.create<arith::ConstantIntOp>(loc, 43, 64);
  Value nz2dnShift62 = rewriter.create<arith::ConstantIntOp>(loc, 62, 64);

  Value quantShift5 = rewriter.create<arith::ConstantIntOp>(loc, 5, 64);
  Value quantLowMask = rewriter.create<arith::ConstantIntOp>(loc, 0x1f, 64);
  Value quantBitMask = rewriter.create<arith::ConstantIntOp>(loc, 0x1, 64);
  Value l2CacheMask = rewriter.create<arith::ConstantIntOp>(loc, 0xf, 64);
  Value unitFlagMask = rewriter.create<arith::ConstantIntOp>(loc, 0x3, 64);
  Value reluMask = rewriter.create<arith::ConstantIntOp>(loc, 0x7, 64);

  Value l2CacheBits = rewriter.create<arith::AndIOp>(loc, l2CacheCtrl, l2CacheMask);
  l2CacheBits =
      rewriter.create<arith::ShLIOp>(loc, l2CacheBits, l2CacheShift16);

  Value unitFlagBits = rewriter.create<arith::AndIOp>(loc, unitFlagCtrl, unitFlagMask);
  unitFlagBits =
      rewriter.create<arith::ShLIOp>(loc, unitFlagBits, unitFlagShift32);

  Value quantBlockBit = rewriter.create<arith::ShRUIOp>(loc, quantPre, quantShift5);
  quantBlockBit =
      rewriter.create<arith::AndIOp>(loc, quantBlockBit, quantBitMask);
  quantBlockBit = rewriter.create<arith::ShLIOp>(loc, quantBlockBit,
                                                 quantBlockBitShift29);

  Value quantField = rewriter.create<arith::AndIOp>(loc, quantPre, quantLowMask);
  quantField =
      rewriter.create<arith::ShLIOp>(loc, quantField, quantFieldShift34);

  Value reluBits = rewriter.create<arith::AndIOp>(loc, reluPreMode, reluMask);
  reluBits = rewriter.create<arith::ShLIOp>(loc, reluBits, reluShift39);

  Value channelSplitBits =
      rewriter.create<arith::AndIOp>(loc, channelSplitEn, quantBitMask);
  channelSplitBits = rewriter.create<arith::ShLIOp>(loc, channelSplitBits,
                                                    channelSplitShift42);

  Value nz2ndBits = rewriter.create<arith::AndIOp>(loc, nz2ndEn, quantBitMask);
  nz2ndBits =
      rewriter.create<arith::ShLIOp>(loc, nz2ndBits, nz2ndShift43);

  Value nz2dnBits = rewriter.create<arith::AndIOp>(loc, nz2dnEn, quantBitMask);
  nz2dnBits =
      rewriter.create<arith::ShLIOp>(loc, nz2dnBits, nz2dnShift62);

  Value xt = rewriter.create<arith::OrIOp>(loc, srcStride, l2CacheBits);
  xt = rewriter.create<arith::OrIOp>(loc, xt, unitFlagBits);
  xt = rewriter.create<arith::OrIOp>(loc, xt, quantBlockBit);
  xt = rewriter.create<arith::OrIOp>(loc, xt, quantField);
  xt = rewriter.create<arith::OrIOp>(loc, xt, reluBits);
  xt = rewriter.create<arith::OrIOp>(loc, xt, channelSplitBits);
  xt = rewriter.create<arith::OrIOp>(loc, xt, nz2ndBits);
  return rewriter.create<arith::OrIOp>(loc, xt, nz2dnBits);
}

static Value packCopyMatrixCcToUbConfig1(Location loc, Value srcStride,
                                         Value dualDstMode, Value subBlockId,
                                         Value unitFlagCtrl, Value quantPre,
                                         Value reluPreMode, Value nz2ndEn,
                                         Value channelSplitEn, Value nz2dnEn,
                                         PatternRewriter &rewriter) {
  Value dualDstShift16 = rewriter.create<arith::ConstantIntOp>(loc, 16, 64);
  Value subBlockShift18 = rewriter.create<arith::ConstantIntOp>(loc, 18, 64);
  Value unitFlagShift32 = rewriter.create<arith::ConstantIntOp>(loc, 32, 64);
  Value quantBlockBitShift29 =
      rewriter.create<arith::ConstantIntOp>(loc, 29, 64);
  Value quantFieldShift34 = rewriter.create<arith::ConstantIntOp>(loc, 34, 64);
  Value reluShift39 = rewriter.create<arith::ConstantIntOp>(loc, 39, 64);
  Value channelSplitShift42 =
      rewriter.create<arith::ConstantIntOp>(loc, 42, 64);
  Value nz2ndShift43 = rewriter.create<arith::ConstantIntOp>(loc, 43, 64);
  Value nz2dnShift62 = rewriter.create<arith::ConstantIntOp>(loc, 62, 64);

  Value dualDstMask = rewriter.create<arith::ConstantIntOp>(loc, 0x3, 64);
  Value subBlockMask = rewriter.create<arith::ConstantIntOp>(loc, 0x1, 64);
  Value quantShift5 = rewriter.create<arith::ConstantIntOp>(loc, 5, 64);
  Value quantLowMask = rewriter.create<arith::ConstantIntOp>(loc, 0x1f, 64);
  Value quantBitMask = rewriter.create<arith::ConstantIntOp>(loc, 0x1, 64);
  Value unitFlagMask = rewriter.create<arith::ConstantIntOp>(loc, 0x3, 64);
  Value reluMask = rewriter.create<arith::ConstantIntOp>(loc, 0x7, 64);

  Value dualDstBits = rewriter.create<arith::AndIOp>(loc, dualDstMode, dualDstMask);
  dualDstBits =
      rewriter.create<arith::ShLIOp>(loc, dualDstBits, dualDstShift16);

  Value subBlockBits = rewriter.create<arith::AndIOp>(loc, subBlockId, subBlockMask);
  subBlockBits =
      rewriter.create<arith::ShLIOp>(loc, subBlockBits, subBlockShift18);

  Value unitFlagBits = rewriter.create<arith::AndIOp>(loc, unitFlagCtrl, unitFlagMask);
  unitFlagBits =
      rewriter.create<arith::ShLIOp>(loc, unitFlagBits, unitFlagShift32);

  Value quantBlockBit = rewriter.create<arith::ShRUIOp>(loc, quantPre, quantShift5);
  quantBlockBit =
      rewriter.create<arith::AndIOp>(loc, quantBlockBit, quantBitMask);
  quantBlockBit = rewriter.create<arith::ShLIOp>(loc, quantBlockBit,
                                                 quantBlockBitShift29);

  Value quantField = rewriter.create<arith::AndIOp>(loc, quantPre, quantLowMask);
  quantField =
      rewriter.create<arith::ShLIOp>(loc, quantField, quantFieldShift34);

  Value reluBits = rewriter.create<arith::AndIOp>(loc, reluPreMode, reluMask);
  reluBits = rewriter.create<arith::ShLIOp>(loc, reluBits, reluShift39);

  Value channelSplitBits =
      rewriter.create<arith::AndIOp>(loc, channelSplitEn, quantBitMask);
  channelSplitBits = rewriter.create<arith::ShLIOp>(loc, channelSplitBits,
                                                    channelSplitShift42);

  Value nz2ndBits = rewriter.create<arith::AndIOp>(loc, nz2ndEn, quantBitMask);
  nz2ndBits =
      rewriter.create<arith::ShLIOp>(loc, nz2ndBits, nz2ndShift43);

  Value nz2dnBits = rewriter.create<arith::AndIOp>(loc, nz2dnEn, quantBitMask);
  nz2dnBits =
      rewriter.create<arith::ShLIOp>(loc, nz2dnBits, nz2dnShift62);

  Value config1 = rewriter.create<arith::OrIOp>(loc, srcStride, dualDstBits);
  config1 = rewriter.create<arith::OrIOp>(loc, config1, subBlockBits);
  config1 = rewriter.create<arith::OrIOp>(loc, config1, unitFlagBits);
  config1 = rewriter.create<arith::OrIOp>(loc, config1, quantBlockBit);
  config1 = rewriter.create<arith::OrIOp>(loc, config1, quantField);
  config1 = rewriter.create<arith::OrIOp>(loc, config1, reluBits);
  config1 = rewriter.create<arith::OrIOp>(loc, config1, channelSplitBits);
  config1 = rewriter.create<arith::OrIOp>(loc, config1, nz2ndBits);
  return rewriter.create<arith::OrIOp>(loc, config1, nz2dnBits);
}

static Value packLoop3Config(Location loc, Value count, Value srcStride,
                             Value dstStride, PatternRewriter &rewriter) {
  Value srcShift16 = rewriter.create<arith::ConstantIntOp>(loc, 16, 64);
  Value dstShift32 = rewriter.create<arith::ConstantIntOp>(loc, 32, 64);
  Value srcBits = rewriter.create<arith::ShLIOp>(loc, srcStride, srcShift16);
  Value dstBits = rewriter.create<arith::ShLIOp>(loc, dstStride, dstShift32);
  Value low = rewriter.create<arith::OrIOp>(loc, count, srcBits);
  return rewriter.create<arith::OrIOp>(loc, low, dstBits);
}

static Value packChannelConfig(Location loc, Value loop0SrcStride,
                               PatternRewriter &rewriter) {
  Value shift48 = rewriter.create<arith::ConstantIntOp>(loc, 48, 64);
  return rewriter.create<arith::ShLIOp>(loc, loop0SrcStride, shift48);
}

struct LoadCbufToCbControl {
  Value mStart;
  Value kStart;
  Value mStep;
  Value kStep;
  Value srcStride;
  Value dstStride;
};

static FailureOr<LoadCbufToCbControl>
deriveLoadCbufToCbControl(Location loc, Value k, Value n, Type elementType,
                          bool transpose, PatternRewriter &rewriter) {
  unsigned elemBitWidth = elementType.getIntOrFloatBitWidth();
  if (elemBitWidth == 0 || (elemBitWidth % 8) != 0)
    return failure();
  uint64_t elemBytes = elemBitWidth / 8;

  auto constant = [&](uint64_t value) -> Value {
    return rewriter.create<arith::ConstantIntOp>(loc, value, 64);
  };
  auto ceilDivConst = [&](Value value, uint64_t divisor) -> Value {
    Value bias = constant(divisor - 1);
    Value sum = rewriter.create<arith::AddIOp>(loc, value, bias);
    return rewriter.create<arith::DivUIOp>(loc, sum, constant(divisor));
  };

  Value zero = constant(0);
  if (!transpose) {
    Value mStep = ceilDivConst(n, 16);
    Value kBytes = rewriter.create<arith::MulIOp>(loc, k, constant(elemBytes));
    Value kStep = ceilDivConst(kBytes, 32);
    Value stride = ceilDivConst(n, 16);
    return LoadCbufToCbControl{zero, zero, mStep, kStep, stride, stride};
  }

  uint64_t c0Size = std::max<uint64_t>(16, 32 / elemBytes);
  Value kAlign = ceilDivConst(k, c0Size);
  kAlign = rewriter.create<arith::MulIOp>(loc, kAlign, constant(c0Size));
  Value nAlign = ceilDivConst(n, c0Size);
  nAlign = rewriter.create<arith::MulIOp>(loc, nAlign, constant(c0Size));
  Value mStep = ceilDivConst(kAlign, 16);
  Value nBytes = rewriter.create<arith::MulIOp>(loc, nAlign, constant(elemBytes));
  Value kStep = ceilDivConst(nBytes, 32);
  Value srcStride = ceilDivConst(kAlign, 16);
  Value dstStride = ceilDivConst(nAlign, 16);
  return LoadCbufToCbControl{zero, zero, mStep, kStep, srcStride, dstStride};
}

static FailureOr<LoadCbufToCbControl>
deriveLoadCbufToCaControl(Location loc, Value m, Value k, Type elementType,
                          bool transpose, PatternRewriter &rewriter) {
  unsigned elemBitWidth = elementType.getIntOrFloatBitWidth();
  if (elemBitWidth == 0 || (elemBitWidth % 8) != 0)
    return failure();
  uint64_t elemBytes = elemBitWidth / 8;

  auto constant = [&](uint64_t value) -> Value {
    return rewriter.create<arith::ConstantIntOp>(loc, value, 64);
  };
  auto ceilDivConst = [&](Value value, uint64_t divisor) -> Value {
    Value bias = constant(divisor - 1);
    Value sum = rewriter.create<arith::AddIOp>(loc, value, bias);
    return rewriter.create<arith::DivUIOp>(loc, sum, constant(divisor));
  };

  Value zero = constant(0);
  if (!transpose) {
    Value mStep = ceilDivConst(m, 16);
    Value kBytes = rewriter.create<arith::MulIOp>(loc, k, constant(elemBytes));
    Value kStep = ceilDivConst(kBytes, 32);
    Value stride = ceilDivConst(m, 16);
    return LoadCbufToCbControl{zero, zero, mStep, kStep, stride, stride};
  }

  uint64_t c0Size = std::max<uint64_t>(16, 32 / elemBytes);
  Value mAlign = ceilDivConst(m, c0Size);
  mAlign = rewriter.create<arith::MulIOp>(loc, mAlign, constant(c0Size));
  Value kAlign = ceilDivConst(k, c0Size);
  kAlign = rewriter.create<arith::MulIOp>(loc, kAlign, constant(c0Size));
  Value mStep = ceilDivConst(kAlign, 16);
  Value mBytes = rewriter.create<arith::MulIOp>(loc, mAlign, constant(elemBytes));
  Value kStep = ceilDivConst(mBytes, 32);
  Value srcStride = ceilDivConst(kAlign, 16);
  Value dstStride = ceilDivConst(mAlign, 16);
  return LoadCbufToCbControl{zero, zero, mStep, kStep, srcStride, dstStride};
}

static Value extractConfigLow40(Location loc, Value packed,
                                PatternRewriter &rewriter) {
  Value lowMask =
      rewriter.create<arith::ConstantIntOp>(loc, 0xffffffffffULL, 64);
  return rewriter.create<arith::AndIOp>(loc, packed, lowMask);
}

static Value extractConfigHigh24(Location loc, Value packed,
                                 PatternRewriter &rewriter) {
  Value shift40 = rewriter.create<arith::ConstantIntOp>(loc, 40, 64);
  return rewriter.create<arith::ShRUIOp>(loc, packed, shift40);
}

template <typename BodyBuilder>
static void buildSoftwareLoopNest(PatternRewriter &rewriter, Location loc,
                                  ArrayRef<pto::DmaLoopConfig> loops,
                                  Value srcOffset, Value dstOffset,
                                  BodyBuilder &&buildLeaf) {
  if (loops.empty()) {
    buildLeaf(srcOffset, dstOffset);
    return;
  }

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value count = rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getIndexType(),
                                                      loops.front().count);
  scf::ForOp forOp = rewriter.create<scf::ForOp>(loc, c0, count, c1);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());
    Value ivI64 =
        rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getI64Type(),
                                              forOp.getInductionVar());
    Value nextSrcOffset = buildAccumulatedByteOffset(
        loc, srcOffset, ivI64, loops.front().srcStride, rewriter);
    Value nextDstOffset = buildAccumulatedByteOffset(
        loc, dstOffset, ivI64, loops.front().dstStride, rewriter);
    buildSoftwareLoopNest(rewriter, loc, loops.drop_front(), nextSrcOffset,
                          nextDstOffset, buildLeaf);
  }
}

struct ExpandUvldPattern : public OpRewritePattern<pto::UvldOp> {
  using OpRewritePattern<pto::UvldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::UvldOp op,
                                PatternRewriter &rewriter) const override {
    auto vecType = dyn_cast<pto::VRegType>(op.getResult().getType());
    if (!vecType)
      return failure();

    Value basePtr = materializeBufferPointer(op.getSource(), rewriter, op.getLoc());
    if (!basePtr)
      return op.emitOpError(
          "requires a recoverable pointer base for uvld expansion");

    Value loadPtr = offsetBufferPointer(basePtr, vecType.getElementType(),
                                       op.getOffset(), rewriter, op.getLoc());
    auto alignType = pto::AlignType::get(rewriter.getContext());
    Value align =
        rewriter.create<pto::VldasOp>(op.getLoc(), alignType, loadPtr);
    auto load = rewriter.create<pto::VldusOp>(
        op.getLoc(), TypeRange{vecType, alignType},
        ValueRange{loadPtr, align});
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

struct ExpandDmaLoadPattern : public OpRewritePattern<pto::DmaLoadOp> {
  using OpRewritePattern<pto::DmaLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::DmaLoadOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
    SmallVector<pto::DmaLoopConfig> loops =
        collectLoopConfigs(op.getLoopCounts(), op.getLoopSrcStrides(),
                           op.getLoopDstStrides());
    ArrayRef<pto::DmaLoopConfig> hwLoops = ArrayRef<pto::DmaLoopConfig>(loops).take_front(2);
    ArrayRef<pto::DmaLoopConfig> swLoops = ArrayRef<pto::DmaLoopConfig>(loops).drop_front(hwLoops.size());

    Value loop1Count;
    Value loop2Size = one;
    if (hwLoops.size() == 2) {
      rewriter.create<pto::SetLoop2StrideOutToUbOp>(
          loc, hwLoops[0].srcStride, hwLoops[0].dstStride);
      loop2Size = hwLoops[0].count;
      loop1Count = hwLoops[1].count;
      rewriter.create<pto::SetLoop1StrideOutToUbOp>(
          loc, hwLoops[1].srcStride, hwLoops[1].dstStride);
      rewriter.create<pto::SetLoopSizeOutToUbOp>(loc, loop2Size, loop1Count);
    } else if (hwLoops.size() == 1) {
      loop1Count = hwLoops[0].count;
      rewriter.create<pto::SetLoop1StrideOutToUbOp>(
          loc, hwLoops[0].srcStride, hwLoops[0].dstStride);
      rewriter.create<pto::SetLoopSizeOutToUbOp>(loc, loop2Size, loop1Count);
    }

    Value leftPadding = op.getLeftPaddingCount();
    if (!leftPadding)
      leftPadding = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value rightPadding = op.getRightPaddingCount();
    if (!rightPadding)
      rightPadding = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value dataSelect = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI1Type(),
        rewriter.getBoolAttr(static_cast<bool>(op.getPadValue())));

    if (Value padValue = op.getPadValue())
      rewriter.create<pto::SetMovPadValOp>(loc, padValue);

    buildSoftwareLoopNest(
        rewriter, loc, swLoops, zero, zero,
        [&](Value srcOffset, Value dstOffset) {
          Value source = offsetPointerByBytes(op.getSource(), srcOffset, rewriter, loc);
          Value destination =
              offsetPointerByBytes(op.getDestination(), dstOffset, rewriter, loc);
          rewriter.create<pto::CopyGmToUbufOp>(
              loc, source, destination, zero, op.getNBurst(), op.getLenBurst(),
              leftPadding, rightPadding, dataSelect, op.getL2CacheCtl(),
              op.getNburstSrcStride(), op.getNburstDstStride());
        });
    if (shouldRestoreDmaLoopSize(loop1Count, loop2Size))
      rewriter.create<pto::SetLoopSizeOutToUbOp>(loc, one, one);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandDmaStorePattern : public OpRewritePattern<pto::DmaStoreOp> {
  using OpRewritePattern<pto::DmaStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::DmaStoreOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
    SmallVector<pto::DmaLoopConfig> loops =
        collectLoopConfigs(op.getLoopCounts(), op.getLoopSrcStrides(),
                           op.getLoopDstStrides());
    ArrayRef<pto::DmaLoopConfig> hwLoops =
        ArrayRef<pto::DmaLoopConfig>(loops).take_front(2);
    ArrayRef<pto::DmaLoopConfig> swLoops =
        ArrayRef<pto::DmaLoopConfig>(loops).drop_front(hwLoops.size());

    Value loop1Count;
    Value loop2Size = one;
    if (hwLoops.size() == 2) {
      rewriter.create<pto::SetLoop2StrideUbToOutOp>(
          loc, hwLoops[0].srcStride, hwLoops[0].dstStride);
      loop2Size = hwLoops[0].count;
      loop1Count = hwLoops[1].count;
      rewriter.create<pto::SetLoop1StrideUbToOutOp>(
          loc, hwLoops[1].srcStride, hwLoops[1].dstStride);
      rewriter.create<pto::SetLoopSizeUbToOutOp>(loc, loop2Size, loop1Count);
    } else if (hwLoops.size() == 1) {
      loop1Count = hwLoops[0].count;
      rewriter.create<pto::SetLoop1StrideUbToOutOp>(
          loc, hwLoops[0].srcStride, hwLoops[0].dstStride);
      rewriter.create<pto::SetLoopSizeUbToOutOp>(loc, loop2Size, loop1Count);
    }

    buildSoftwareLoopNest(
        rewriter, loc, swLoops, zero, zero,
        [&](Value srcOffset, Value dstOffset) {
          Value source = offsetPointerByBytes(op.getSource(), srcOffset, rewriter, loc);
          Value destination =
              offsetPointerByBytes(op.getDestination(), dstOffset, rewriter, loc);
          rewriter.create<pto::CopyUbufToGmOp>(
              loc, source, destination, zero, op.getNBurst(), op.getLenBurst(),
              zero, op.getNburstDstStride(), op.getNburstSrcStride());
        });
    if (shouldRestoreDmaLoopSize(loop1Count, loop2Size))
      rewriter.create<pto::SetLoopSizeUbToOutOp>(loc, one, one);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandDmaCopyPattern : public OpRewritePattern<pto::DmaCopyOp> {
  using OpRewritePattern<pto::DmaCopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::DmaCopyOp op,
                                PatternRewriter &rewriter) const override {
    auto destinationType =
        dyn_cast<pto::PtrType>(materializeBufferPointer(op.getDestination(),
                                                        rewriter, op.getLoc())
                                   .getType());
    if (!destinationType)
      return rewriter.notifyMatchFailure(op, "expected pointer-like destination");
    Value zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
    switch (destinationType.getMemorySpace().getAddressSpace()) {
    case pto::AddressSpace::VEC:
      rewriter.replaceOpWithNewOp<pto::CopyUbufToUbufOp>(
          op, op.getSource(), op.getDestination(), zero, op.getNBurst(),
          op.getLenBurst(), op.getSrcStride(), op.getDstStride());
      return success();
    case pto::AddressSpace::MAT:
      rewriter.replaceOpWithNewOp<pto::CopyUbufToCbufOp>(
          op, op.getSource(), op.getDestination(), zero, op.getNBurst(),
          op.getLenBurst(), op.getSrcStride(), op.getDstStride());
      return success();
    default:
      return rewriter.notifyMatchFailure(
          op, "expected dma_copy destination in UB or MAT address space");
    }
  }
};

struct ExpandCubeLoadPattern : public OpRewritePattern<pto::CubeLoadOp> {
  using OpRewritePattern<pto::CubeLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::CubeLoadOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
    SmallVector<pto::DmaLoopConfig> loops =
        collectLoopConfigs(op.getLoopCounts(), op.getLoopSrcStrides(),
                           op.getLoopDstStrides());
    ArrayRef<pto::DmaLoopConfig> hwLoops =
        ArrayRef<pto::DmaLoopConfig>(loops).take_front(2);
    ArrayRef<pto::DmaLoopConfig> swLoops =
        ArrayRef<pto::DmaLoopConfig>(loops).drop_front(hwLoops.size());

    Value loop1Count;
    Value loop2Count = one;
    if (hwLoops.size() == 2) {
      rewriter.create<pto::SetLoop2StrideOutToL1Op>(
          loc,
          packLoopPair(loc, hwLoops[0].srcStride, hwLoops[0].dstStride,
                       rewriter));
      loop2Count = hwLoops[0].count;
      loop1Count = hwLoops[1].count;
      rewriter.create<pto::SetLoop1StrideOutToL1Op>(
          loc,
          packLoopPair(loc, hwLoops[1].srcStride, hwLoops[1].dstStride,
                       rewriter));
      rewriter.create<pto::SetLoopSizeOutToL1Op>(
          loc, packLoopSize(loc, loop2Count, loop1Count, rewriter));
    } else if (hwLoops.size() == 1) {
      loop1Count = hwLoops[0].count;
      rewriter.create<pto::SetLoop1StrideOutToL1Op>(
          loc,
          packLoopPair(loc, hwLoops[0].srcStride, hwLoops[0].dstStride,
                       rewriter));
      rewriter.create<pto::SetLoopSizeOutToL1Op>(
          loc, packLoopSize(loc, loop2Count, loop1Count, rewriter));
    }

    SmallVector<pto::DmaLoopConfig> swLoopNestOrder(swLoops.rbegin(),
                                                    swLoops.rend());
    buildSoftwareLoopNest(
        rewriter, loc, swLoopNestOrder, zero, zero,
        [&](Value srcOffset, Value dstOffset) {
          Value source =
              offsetPointerByBytes(op.getSource(), srcOffset, rewriter, loc);
          Value destination = offsetPointerByBytes(op.getDestination(), dstOffset,
                                                   rewriter, loc);
          rewriter.create<pto::CopyGmToCbufOp>(
              loc, source, destination, op.getNBurst(), op.getLenBurst(),
              op.getNburstSrcStride(), op.getNburstDstStride());
        });
    if (loop1Count && (!isKnownOne(loop1Count) || !isKnownOne(loop2Count)))
      rewriter.create<pto::SetLoopSizeOutToL1Op>(
          loc, packLoopSize(loc, one, one, rewriter));
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandCubeStorePattern : public OpRewritePattern<pto::CubeStoreOp> {
  using OpRewritePattern<pto::CubeStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::CubeStoreOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    SmallVector<pto::DmaLoopConfig> loops =
        collectLoopConfigs(op.getLoopCounts(), op.getLoopSrcStrides(),
                           op.getLoopDstStrides());
    SmallVector<pto::DmaLoopConfig> swLoopNestOrder(loops.rbegin(),
                                                    loops.rend());
    buildSoftwareLoopNest(
        rewriter, loc, swLoopNestOrder, zero, zero,
        [&](Value srcOffset, Value dstOffset) {
          Value source =
              offsetPointerByBytes(op.getSource(), srcOffset, rewriter, loc);
          Value destination =
              offsetPointerByBytes(op.getDestination(), dstOffset, rewriter, loc);
          rewriter.create<pto::CopyCbufToUbufOp>(
              loc, source, destination, zero, op.getNBurst(), op.getLenBurst(),
              op.getNburstSrcStride(), op.getNburstDstStride());
        });
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandBiasLoadPattern : public OpRewritePattern<pto::BiasLoadOp> {
  using OpRewritePattern<pto::BiasLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::BiasLoadOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto sourceType = dyn_cast<pto::PtrType>(
        materializeBufferPointer(op.getSource(), rewriter, loc).getType());
    if (!sourceType)
      return rewriter.notifyMatchFailure(op, "expected pointer-like source");

    Value convControl = rewriter.create<arith::ConstantIntOp>(
        loc, sourceType.getElementType().isF16() ? 1 : 0, 1);
    rewriter.replaceOpWithNewOp<pto::CopyCbufToBtOp>(
        op, op.getSource(), op.getDestination(), convControl, op.getNBurst(),
        op.getLenBurst(), op.getNburstSrcGap(), op.getNburstDstGap());
    return success();
  }
};

struct ExpandCubeLoadFracPattern : public OpRewritePattern<pto::CubeLoadFracOp> {
  using OpRewritePattern<pto::CubeLoadFracOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::CubeLoadFracOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value mte2NzPara = packMte2NzPara(
        loc, op.getGroupCount(), op.getDstLoop2Stride(), op.getDstLoop3Stride(),
        op.getDstLoop4Stride(), rewriter);
    rewriter.create<pto::SetMte2NzParaOp>(loc, mte2NzPara);

    Value srcOuterStride = op.getSrcOuterStride() ? op.getSrcOuterStride() : zero;
    Value source = materializeBufferPointer(op.getSource(), rewriter, loc);
    Value destination =
        materializeBufferPointer(op.getDestination(), rewriter, loc);
    switch (op.getMode()) {
    case pto::CubeLoadFracMode::Nd2nz:
      rewriter.create<pto::CopyGmToCbufMultiNd2NzOp>(
          loc, source, destination, zero, op.getSrcInnerStride(),
          op.getL2CacheCtrl(), op.getNValue(), op.getDValue(), srcOuterStride,
          op.getSmallc0En());
      break;
    case pto::CubeLoadFracMode::Dn2nz:
      rewriter.create<pto::CopyGmToCbufMultiDn2NzOp>(
          loc, source, destination, zero, op.getSrcInnerStride(),
          op.getL2CacheCtrl(), op.getNValue(), op.getDValue(), srcOuterStride,
          op.getSmallc0En());
      break;
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandLeftLoadPattern : public OpRewritePattern<pto::LeftLoadOp> {
  using OpRewritePattern<pto::LeftLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::LeftLoadOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto sourceType = dyn_cast<pto::PtrType>(op.getSource().getType());
    if (!sourceType)
      return rewriter.notifyMatchFailure(op, "expected typed L1 source");
    FailureOr<LoadCbufToCbControl> control = deriveLoadCbufToCaControl(
        loc, op.getM(), op.getK(), sourceType.getElementType(),
        op.getTranspose(), rewriter);
    if (failed(control))
      return rewriter.notifyMatchFailure(op,
                                         "failed to derive load_cbuf_to_ca control");
    auto load = rewriter.create<pto::LoadCbufToCaOp>(
        loc, op.getSource(), op.getDestination(), control->mStart,
        control->kStart, control->mStep, control->kStep, control->srcStride,
        control->dstStride);
    load->setAttr("transpose", rewriter.getBoolAttr(op.getTranspose()));
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandRightLoadPattern : public OpRewritePattern<pto::RightLoadOp> {
  using OpRewritePattern<pto::RightLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::RightLoadOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto sourceType = dyn_cast<pto::PtrType>(op.getSource().getType());
    if (!sourceType)
      return rewriter.notifyMatchFailure(op, "expected typed L1 source");
    FailureOr<LoadCbufToCbControl> control = deriveLoadCbufToCbControl(
        loc, op.getK(), op.getN(), sourceType.getElementType(),
        op.getTranspose(), rewriter);
    if (failed(control))
      return rewriter.notifyMatchFailure(op,
                                         "failed to derive load_cbuf_to_cb control");
    auto load = rewriter.create<pto::LoadCbufToCbOp>(
        loc, op.getSource(), op.getDestination(), control->mStart,
        control->kStart, control->mStep, control->kStep, control->srcStride,
        control->dstStride);
    load->setAttr("transpose", rewriter.getBoolAttr(op.getTranspose()));
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandAccStorePattern : public OpRewritePattern<pto::AccStoreOp> {
  using OpRewritePattern<pto::AccStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::AccStoreOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
    pto::DmaLoopConfig hwLoop{one, zero, zero};
    if (Value loop3Count = op.getLoop3Count()) {
      hwLoop = {loop3Count, op.getLoop3SrcStride(), op.getLoop3DstStride()};
    }

    Value channelLoop0Stride = zero;
    Value nz2ndEn = zero;
    Value channelSplitEn = zero;
    Value nz2dnEn = zero;
    switch (op.getMode()) {
    case pto::AccStoreMode::Nz2nd:
      nz2ndEn = one;
      break;
    case pto::AccStoreMode::Nz2dn:
      nz2dnEn = one;
      channelLoop0Stride = op.getLoop0SrcStride() ? op.getLoop0SrcStride() : one;
      break;
    case pto::AccStoreMode::Nz2nz:
      channelSplitEn = op.getSplit() ? op.getSplit() : zero;
      break;
    }

    Value loop3Config = packLoop3Config(loc, hwLoop.count, hwLoop.srcStride,
                                        hwLoop.dstStride, rewriter);
    Value channelConfig =
        packChannelConfig(loc, channelLoop0Stride, rewriter);
    rewriter.create<pto::SetLoop3ParaOp>(
        loc, extractConfigLow40(loc, loop3Config, rewriter),
        extractConfigHigh24(loc, loop3Config, rewriter));
    rewriter.create<pto::SetChannelParaOp>(
        loc, extractConfigLow40(loc, channelConfig, rewriter),
        extractConfigHigh24(loc, channelConfig, rewriter));
    Value xm =
        packCopyMatrixCcToGmXm(loc, zero, op.getN(), op.getM(),
                               op.getDstStride(), rewriter);
    Value xt = packCopyMatrixCcToGmXt(
        loc, op.getSrcStride(), op.getUnitFlagCtrl(), op.getQuantPre(),
        op.getReluPreMode(), zero, nz2ndEn, channelSplitEn, nz2dnEn,
        rewriter);
    rewriter.create<pto::CopyMatrixCcToCbufOp>(loc, op.getSource(),
                                               op.getDestination(), xm, xt);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandAccStoreGmPattern : public OpRewritePattern<pto::AccStoreGmOp> {
  using OpRewritePattern<pto::AccStoreGmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::AccStoreGmOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
    pto::DmaLoopConfig hwLoop{one, zero, zero};
    if (Value loop3Count = op.getLoop3Count()) {
      hwLoop = {loop3Count, op.getLoop3SrcStride(), op.getLoop3DstStride()};
    }

    Value channelLoop0Stride = zero;
    Value nz2ndEn = zero;
    Value channelSplitEn = zero;
    Value nz2dnEn = zero;
    switch (op.getMode()) {
    case pto::AccStoreMode::Nz2nd:
      nz2ndEn = one;
      break;
    case pto::AccStoreMode::Nz2dn:
      nz2dnEn = one;
      channelLoop0Stride =
          op.getLoop0SrcStride() ? op.getLoop0SrcStride() : one;
      break;
    case pto::AccStoreMode::Nz2nz:
      channelSplitEn = op.getSplit() ? op.getSplit() : zero;
      break;
    }

    Value loop3Config = packLoop3Config(loc, hwLoop.count, hwLoop.srcStride,
                                        hwLoop.dstStride, rewriter);
    Value channelConfig = packChannelConfig(loc, channelLoop0Stride, rewriter);
    rewriter.create<pto::SetLoop3ParaOp>(
        loc, extractConfigLow40(loc, loop3Config, rewriter),
        extractConfigHigh24(loc, loop3Config, rewriter));
    rewriter.create<pto::SetChannelParaOp>(
        loc, extractConfigLow40(loc, channelConfig, rewriter),
        extractConfigHigh24(loc, channelConfig, rewriter));
    Value xm = packCopyMatrixCcToGmXm(loc, op.getSid(), op.getN(), op.getM(),
                                      op.getDstStride(), rewriter);
    Value xt = packCopyMatrixCcToGmXt(
        loc, op.getSrcStride(), op.getUnitFlagCtrl(), op.getQuantPre(),
        op.getReluPreMode(), op.getL2CacheCtrl(), nz2ndEn, channelSplitEn,
        nz2dnEn, rewriter);
    rewriter.create<pto::CopyMatrixCcToGmOp>(loc, op.getSource(),
                                             op.getDestination(), xm, xt);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandAccStoreUbPattern : public OpRewritePattern<pto::AccStoreUbOp> {
  using OpRewritePattern<pto::AccStoreUbOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::AccStoreUbOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
    pto::DmaLoopConfig hwLoop{one, zero, zero};
    if (Value loop3Count = op.getLoop3Count()) {
      hwLoop = {loop3Count, op.getLoop3SrcStride(), op.getLoop3DstStride()};
    }

    Value channelLoop0Stride = zero;
    Value nz2ndEn = zero;
    Value channelSplitEn = zero;
    Value nz2dnEn = zero;
    switch (op.getMode()) {
    case pto::AccStoreMode::Nz2nd:
      nz2ndEn = one;
      break;
    case pto::AccStoreMode::Nz2dn:
      nz2dnEn = one;
      channelLoop0Stride = op.getLoop0SrcStride() ? op.getLoop0SrcStride() : one;
      break;
    case pto::AccStoreMode::Nz2nz:
      channelSplitEn = op.getSplit() ? op.getSplit() : zero;
      break;
    }

    Value loop3Config = packLoop3Config(loc, hwLoop.count, hwLoop.srcStride,
                                        hwLoop.dstStride, rewriter);
    Value channelConfig =
        packChannelConfig(loc, channelLoop0Stride, rewriter);
    rewriter.create<pto::SetLoop3ParaOp>(
        loc, extractConfigLow40(loc, loop3Config, rewriter),
        extractConfigHigh24(loc, loop3Config, rewriter));
    rewriter.create<pto::SetChannelParaOp>(
        loc, extractConfigLow40(loc, channelConfig, rewriter),
        extractConfigHigh24(loc, channelConfig, rewriter));

    Value config0 = packCopyMatrixCcToGmXm(loc, zero, op.getN(), op.getM(),
                                           op.getDstStride(), rewriter);
    Value config1 = packCopyMatrixCcToUbConfig1(
        loc, op.getSrcStride(), op.getDualDstMode(), op.getSubBlockid(),
        op.getUnitFlagCtrl(), op.getQuantPre(), op.getReluPreMode(), nz2ndEn,
        channelSplitEn, nz2dnEn, rewriter);
    rewriter.create<pto::CopyMatrixCcToUbOp>(loc, op.getSource(),
                                             op.getDestination(), config0,
                                             config1);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOVPTOExpandBridgeOpsPass
    : public pto::impl::PTOVPTOExpandBridgeOpsBase<PTOVPTOExpandBridgeOpsPass> {
  using pto::impl::PTOVPTOExpandBridgeOpsBase<
      PTOVPTOExpandBridgeOpsPass>::PTOVPTOExpandBridgeOpsBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, pto::PTODialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    RewritePatternSet patterns(&getContext());
    patterns.add<ExpandUvldPattern, ExpandDmaLoadPattern, ExpandDmaStorePattern,
                 ExpandDmaCopyPattern, ExpandCubeLoadPattern,
                 ExpandCubeStorePattern, ExpandBiasLoadPattern,
                 ExpandCubeLoadFracPattern, ExpandLeftLoadPattern,
                 ExpandRightLoadPattern, ExpandAccStorePattern,
                 ExpandAccStoreGmPattern,
                 ExpandAccStoreUbPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOVPTOExpandBridgeOpsPass() {
  return std::make_unique<PTOVPTOExpandBridgeOpsPass>();
}
