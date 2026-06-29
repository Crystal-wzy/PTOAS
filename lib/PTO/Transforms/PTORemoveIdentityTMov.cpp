// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- PTORemoveIdentityTMov.cpp -----------------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
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
  std::optional<llvm::APInt> constant;
};

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
    return AddressToken{tassign.getAddr(), std::nullopt};

  auto pointerCast = value.getDefiningOp<PointerCastOp>();
  if (!pointerCast || pointerCast.getAddrs().size() != 1)
    return std::nullopt;

  Value addr = pointerCast.getAddrs().front();
  llvm::APInt constant;
  if (matchPattern(addr, m_ConstantInt(&constant)))
    return AddressToken{addr, constant};
  return AddressToken{addr, std::nullopt};
}

static bool sameAddressToken(const AddressToken &lhs, const AddressToken &rhs) {
  if (lhs.value == rhs.value)
    return true;
  if (lhs.constant && rhs.constant)
    return lhs.constant->getSExtValue() == rhs.constant->getSExtValue();
  return false;
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

static bool isIdentityTMov(TMovOp op) {
  if (!hasPlainTMovSemantics(op) || !hasCompatibleIdentityTypes(op))
    return false;

  if (op.getSrc() == op.getDst())
    return true;

  std::optional<AddressToken> src = getExplicitAddressToken(op.getSrc());
  std::optional<AddressToken> dst = getExplicitAddressToken(op.getDst());
  if (!src || !dst)
    return false;
  return sameAddressToken(*src, *dst);
}

struct PTORemoveIdentityTMovPass
    : public mlir::pto::impl::PTORemoveIdentityTMovBase<
          PTORemoveIdentityTMovPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<TMovOp, 16> identityMoves;

    func.walk([&](TMovOp op) {
      if (isIdentityTMov(op))
        identityMoves.push_back(op);
    });

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
