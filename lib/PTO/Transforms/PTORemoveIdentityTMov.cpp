// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTORemoveIdentityTMov.cpp -----------------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOREMOVEIDENTITYTMOV
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

struct AddressToken {
  Value value;
  std::optional<int64_t> constant;
};

static std::optional<int64_t> tryEvalI64Constant(Value value) {
  if (!value)
    return std::nullopt;

  APInt apInt;
  if (matchPattern(value, m_ConstantInt(&apInt)))
    return apInt.getSExtValue();

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  if (auto castOp = dyn_cast<arith::IndexCastOp>(defOp))
    return tryEvalI64Constant(castOp.getIn());
  if (auto castOp = dyn_cast<arith::ExtSIOp>(defOp))
    return tryEvalI64Constant(castOp.getIn());
  if (auto castOp = dyn_cast<arith::ExtUIOp>(defOp))
    return tryEvalI64Constant(castOp.getIn());
  if (auto castOp = dyn_cast<arith::TruncIOp>(defOp))
    return tryEvalI64Constant(castOp.getIn());

  return std::nullopt;
}

static Value peelMetadata(Value value) {
  while (true) {
    if (auto bind = value.getDefiningOp<BindTileOp>()) {
      value = bind.getSource();
      continue;
    }
    if (auto cast = value.getDefiningOp<memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    return value;
  }
}

static std::optional<AddressToken> getExplicitAddressToken(Value value) {
  value = peelMetadata(value);

  if (auto tassign = value.getDefiningOp<TAssignOp>())
    return AddressToken{tassign.getAddr(),
                        tryEvalI64Constant(tassign.getAddr())};

  auto pointerCast = value.getDefiningOp<PointerCastOp>();
  if (!pointerCast || pointerCast.getAddrs().size() != 1)
    return std::nullopt;

  Value addr = pointerCast.getAddrs().front();
  return AddressToken{addr, tryEvalI64Constant(addr)};
}

static bool sameAddressToken(const AddressToken &lhs, const AddressToken &rhs) {
  if (lhs.value == rhs.value)
    return true;
  if (lhs.constant && rhs.constant)
    return *lhs.constant == *rhs.constant;
  return false;
}

static const BaseMemInfo *
getSingleMemInfo(const Buffer2MemInfoMap &buffer2MemInfoMap, Value value) {
  auto it = buffer2MemInfoMap.find(value);
  if (it == buffer2MemInfoMap.end() || it->second.size() != 1)
    return nullptr;
  return it->second.front().get();
}

static std::optional<int64_t>
tryGetConcreteRootAddress(const BaseMemInfo *info) {
  if (!info)
    return std::nullopt;

  if (auto direct = tryEvalI64Constant(info->rootBuffer))
    return direct;

  Operation *defOp = info->rootBuffer.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  if (auto alloc = dyn_cast<pto::AllocTileOp>(defOp))
    return tryEvalI64Constant(alloc.getAddr());

  if (auto pointerCast = dyn_cast<pto::PointerCastOp>(defOp)) {
    if (pointerCast.getAddrs().size() == 1)
      return tryEvalI64Constant(pointerCast.getAddrs().front());
  }

  return std::nullopt;
}

static bool hasDynamicStaticList(ArrayRef<int64_t> values) {
  return llvm::any_of(
      values, [](int64_t value) { return value == ShapedType::kDynamic; });
}

static bool isStaticallyAddressableValue(Value value) {
  int depth = 0;
  constexpr int kMaxDepth = 32;
  while (value && depth++ < kMaxDepth) {
    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      return false;

    if (auto subView = dyn_cast<memref::SubViewOp>(defOp)) {
      if (hasDynamicStaticList(subView.getStaticOffsets()) ||
          hasDynamicStaticList(subView.getStaticSizes()) ||
          hasDynamicStaticList(subView.getStaticStrides()))
        return false;
      value = subView.getSource();
      continue;
    }

    if (isa<memref::ReinterpretCastOp>(defOp))
      return false;

    if (auto cast = dyn_cast<memref::CastOp>(defOp)) {
      value = cast.getSource();
      continue;
    }
    if (auto collapse = dyn_cast<memref::CollapseShapeOp>(defOp)) {
      value = collapse.getSrc();
      continue;
    }
    if (auto expand = dyn_cast<memref::ExpandShapeOp>(defOp)) {
      value = expand.getSrc();
      continue;
    }
    if (auto view = dyn_cast<memref::ViewOp>(defOp)) {
      if (view.getByteShift())
        return false;
      value = view.getSource();
      continue;
    }

    return true;
  }

  return false;
}

static bool hasExactSameAddressRange(const BaseMemInfo *srcInfo,
                                     const BaseMemInfo *dstInfo) {
  if (!srcInfo || !dstInfo)
    return false;
  if (srcInfo->scope != dstInfo->scope)
    return false;
  if (srcInfo->allocateSize == 0 || dstInfo->allocateSize == 0)
    return false;
  if (srcInfo->allocateSize != dstInfo->allocateSize)
    return false;
  if (srcInfo->baseAddresses.empty() || dstInfo->baseAddresses.empty())
    return false;
  if (srcInfo->baseAddresses != dstInfo->baseAddresses)
    return false;
  return true;
}

static bool hasPlainTMovSemantics(TMovOp op) {
  return !op.getFp() && !op.getPreQuantScalar() && !op.getAccToVecModeAttr() &&
         op.getReluPreMode() == ReluPreMode::NoRelu;
}

static bool hasCompatibleIdentityTypes(TMovOp op) {
  if (op.getSrc().getType() != op.getDst().getType())
    return false;
  for (OpResult result : op->getResults()) {
    if (result.getType() != op.getDst().getType())
      return false;
  }
  return true;
}

static bool isIdentityTMovByExplicitAddress(TMovOp op) {
  if (op.getSrc() == op.getDst())
    return true;

  std::optional<AddressToken> src = getExplicitAddressToken(op.getSrc());
  std::optional<AddressToken> dst = getExplicitAddressToken(op.getDst());
  if (!src || !dst)
    return false;
  return sameAddressToken(*src, *dst);
}

static bool
isIdentityTMovByMemInfo(TMovOp op, const Buffer2MemInfoMap &buffer2MemInfoMap) {
  Value src = op.getSrc();
  Value dst = op.getDst();

  if (!isStaticallyAddressableValue(src) || !isStaticallyAddressableValue(dst))
    return false;

  const BaseMemInfo *srcInfo = getSingleMemInfo(buffer2MemInfoMap, src);
  const BaseMemInfo *dstInfo = getSingleMemInfo(buffer2MemInfoMap, dst);
  if (!hasExactSameAddressRange(srcInfo, dstInfo))
    return false;

  if (srcInfo->rootBuffer == dstInfo->rootBuffer)
    return true;

  auto srcRootAddr = tryGetConcreteRootAddress(srcInfo);
  auto dstRootAddr = tryGetConcreteRootAddress(dstInfo);
  return srcRootAddr && dstRootAddr && *srcRootAddr == *dstRootAddr;
}

struct PTORemoveIdentityTMovPass
    : public mlir::pto::impl::PTORemoveIdentityTMovBase<
          PTORemoveIdentityTMovPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<TMovOp, 16> identityMoves;
    SmallVector<TMovOp, 16> memInfoCandidates;

    func.walk([&](TMovOp op) {
      if (!hasPlainTMovSemantics(op) || !hasCompatibleIdentityTypes(op))
        return;
      if (isIdentityTMovByExplicitAddress(op)) {
        identityMoves.push_back(op);
        return;
      }
      memInfoCandidates.push_back(op);
    });

    if (!memInfoCandidates.empty()) {
      MemoryDependentAnalyzer memAnalyzer;
      SyncIRs syncIR;
      Buffer2MemInfoMap buffer2MemInfoMap;
      PTOIRTranslator translator(syncIR, memAnalyzer, buffer2MemInfoMap, func,
                                 SyncAnalysisMode::NORMALSYNC);
      translator.Build();

      for (TMovOp op : memInfoCandidates) {
        if (isIdentityTMovByMemInfo(op, buffer2MemInfoMap))
          identityMoves.push_back(op);
      }
    }

    for (TMovOp op : identityMoves) {
      for (OpResult result : op->getResults())
        result.replaceAllUsesWith(op.getDst());
      op.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTORemoveIdentityTMovPass() {
  return std::make_unique<PTORemoveIdentityTMovPass>();
}
