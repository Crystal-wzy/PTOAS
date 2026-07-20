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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOSOFTPOSTUPDATE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

// Candidate op info collected during traversal.
struct PostUpdateCandidate {
  Operation *op;
  Value base;
  Value offset;
  scf::ForOp forOp; // null if not inside scf.for
};

// Return the set of operation names eligible for post-update conversion.
static const DenseSet<StringRef> &getPostUpdateSet() {
  static const DenseSet<StringRef> set = {
      "pto.vlds",
      "pto.vsts",
      "pto.vsstb",
  };
  return set;
}

// Check if an operation is a post-update candidate.
static bool isPostUpdateCandidate(Operation *op) {
  return getPostUpdateSet().contains(op->getName().getStringRef());
}

// Extract base and offset operands from a candidate op.
// Returns false if the op structure is not recognized.
static bool extractBaseAndOffset(Operation *op, Value &base, Value &offset) {
  if (auto vlds = dyn_cast<pto::VldsOp>(op)) {
    base = vlds.getSource();
    offset = vlds.getOffset();
    return true;
  }
  if (auto vsts = dyn_cast<pto::VstsOp>(op)) {
    base = vsts.getDestination();
    offset = vsts.getOffset();
    return true;
  }
  if (auto vsstb = dyn_cast<pto::VsstbOp>(op)) {
    base = vsstb.getDestination();
    // vsstb uses block_stride/repeat_stride, not a single offset.
    // For delta analysis, treat the base as the varying part.
    // offset is not directly applicable; set to null.
    offset = Value();
    return true;
  }
  return false;
}

// Check if a value has an updated_base result (already post-update).
static bool isAlreadyPostUpdate(Operation *op) {
  if (auto vlds = dyn_cast<pto::VldsOp>(op))
    return static_cast<bool>(vlds.getUpdatedBase());
  if (auto vsts = dyn_cast<pto::VstsOp>(op))
    return static_cast<bool>(vsts.getUpdatedBase());
  if (auto vsstb = dyn_cast<pto::VsstbOp>(op))
    return static_cast<bool>(vsstb.getUpdatedBase());
  return false;
}

// Check if op is directly inside the scf.for body (not nested in scf.if etc).
static bool isDirectlyInForBody(Operation *op, scf::ForOp forOp) {
  return op->getParentOp() == forOp.getOperation();
}

//===----------------------------------------------------------------------===//
// Delta Analysis
//===----------------------------------------------------------------------===//

// Compute the per-iteration delta of value `v` within `forOp`.
// Returns the delta as a loop-invariant Value, or nullptr if unknown.
static Value computeDelta(Value v, scf::ForOp forOp, OpBuilder &builder) {
  // IV: delta = step
  if (v == forOp.getInductionVar())
    return forOp.getStep();

  // Constant or loop-invariant: delta = 0
  if (forOp.isDefinedOutsideOfLoop(v)) {
    builder.setInsertionPoint(forOp);
    return builder.create<arith::ConstantIndexOp>(forOp.getLoc(), 0);
  }

  // Block argument from iter_args: check yield = arg + c
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    if (blockArg.getOwner() == forOp.getBody() &&
        blockArg.getArgNumber() > 0) {
      unsigned idx = blockArg.getArgNumber() - 1;
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      Value yieldVal = yieldOp.getOperand(idx);
      if (auto addOp = yieldVal.getDefiningOp<arith::AddIOp>()) {
        Value other;
        if (addOp.getLhs() == blockArg)
          other = addOp.getRhs();
        else if (addOp.getRhs() == blockArg)
          other = addOp.getLhs();
        if (other && forOp.isDefinedOutsideOfLoop(other))
          return other;
      }
      return nullptr;
    }
  }

  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return nullptr;

  // arith.addi(a, b): delta = delta(a) + delta(b)
  if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
    Value da = computeDelta(addOp.getLhs(), forOp, builder);
    Value db = computeDelta(addOp.getRhs(), forOp, builder);
    if (!da || !db)
      return nullptr;
    // Optimize: if either is constant 0, return the other
    if (auto ca = getConstantIntValue(da); ca && *ca == 0)
      return db;
    if (auto cb = getConstantIntValue(db); cb && *cb == 0)
      return da;
    builder.setInsertionPoint(forOp);
    return builder.create<arith::AddIOp>(forOp.getLoc(), da, db);
  }

  // arith.subi(a, b): delta = delta(a) - delta(b)
  if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
    Value da = computeDelta(subOp.getLhs(), forOp, builder);
    Value db = computeDelta(subOp.getRhs(), forOp, builder);
    if (!da || !db)
      return nullptr;
    if (auto cb = getConstantIntValue(db); cb && *cb == 0)
      return da;
    builder.setInsertionPoint(forOp);
    return builder.create<arith::SubIOp>(forOp.getLoc(), da, db);
  }

  // arith.muli(a, b) where one is loop-invariant:
  //   delta = invariant * delta(other)
  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    Value lhs = mulOp.getLhs(), rhs = mulOp.getRhs();
    for (auto [invariant, variant] :
         {std::pair{rhs, lhs}, std::pair{lhs, rhs}}) {
      if (forOp.isDefinedOutsideOfLoop(invariant)) {
        Value dv = computeDelta(variant, forOp, builder);
        if (!dv)
          continue;
        if (auto cv = getConstantIntValue(dv); cv && *cv == 0) {
          builder.setInsertionPoint(forOp);
          return builder.create<arith::ConstantIndexOp>(forOp.getLoc(), 0);
        }
        if (auto cv = getConstantIntValue(dv); cv && *cv == 1)
          return invariant;
        builder.setInsertionPoint(forOp);
        return builder.create<arith::MulIOp>(forOp.getLoc(), invariant, dv);
      }
    }
    return nullptr;
  }

  // arith.index_castui / arith.index_cast: delta = delta(input)
  if (auto castOp = dyn_cast<arith::IndexCastUIOp>(defOp))
    return computeDelta(castOp.getIn(), forOp, builder);
  if (auto castOp = dyn_cast<arith::IndexCastOp>(defOp))
    return computeDelta(castOp.getIn(), forOp, builder);

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Rewrite: create new ForOp with additional iter_arg
//===----------------------------------------------------------------------===//

// Compute the value of `v` at the first iteration (IV = lower bound) by
// cloning the def-chain with IV replaced by the lower bound.  Returns nullptr
// if `v` cannot be materialized outside the loop.
static Value materializeAtLoopEntry(Value v, scf::ForOp forOp,
                                    OpBuilder &builder) {
  // IV → lower bound
  if (v == forOp.getInductionVar())
    return forOp.getLowerBound();

  // Already defined outside the loop — use directly.
  if (forOp.isDefinedOutsideOfLoop(v))
    return v;

  // iter_arg → its init value
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    if (blockArg.getOwner() == forOp.getBody() &&
        blockArg.getArgNumber() > 0) {
      unsigned idx = blockArg.getArgNumber() - 1;
      return forOp.getInitArgs()[idx];
    }
  }

  Operation *defOp = v.getDefiningOp();
  if (!defOp || !forOp->isAncestor(defOp))
    return nullptr;

  // Clone the defining op with operands materialized at loop entry.
  SmallVector<Value> newOperands;
  for (Value operand : defOp->getOperands()) {
    Value materialized = materializeAtLoopEntry(operand, forOp, builder);
    if (!materialized)
      return nullptr;
    newOperands.push_back(materialized);
  }
  builder.setInsertionPoint(forOp);
  Operation *cloned = builder.clone(*defOp);
  for (auto [i, operand] : llvm::enumerate(newOperands))
    cloned->setOperand(i, operand);
  return cloned->getResult(0);
}

// Compute the initial pointer for a post-update rewrite: base + offset_at_iter0.
// For post-update, the op loads/stores from the pointer directly (offset is
// repurposed as the stride), so the initial pointer must equal the first
// iteration's effective address.
static Value computeInitialPtr(Value base, Value offset, scf::ForOp forOp,
                               OpBuilder &builder) {
  if (!offset)
    return base;

  Value offsetAtEntry = materializeAtLoopEntry(offset, forOp, builder);
  if (!offsetAtEntry)
    return nullptr;

  // If initial offset is known to be zero, just use base.
  if (auto constVal = getConstantIntValue(offsetAtEntry);
      constVal && *constVal == 0)
    return base;

  builder.setInsertionPoint(forOp);
  return builder.create<pto::AddPtrOp>(forOp.getLoc(), base, offsetAtEntry);
}

// Information about a post-update transformation to apply.
struct PostUpdateRewrite {
  Operation *op;
  Value base;
  Value offset;
  Value stride;  // loop-invariant stride value
  Value initPtr; // base + offset_at_iter0
};

// A unique key for grouping rewrites that can share an iter_arg.
// Same base + same stride (by Value identity) = same group.
using IterArgGroupKey = std::pair<Value, Value>;

static IterArgGroupKey getGroupKey(const PostUpdateRewrite &rw) {
  return {rw.base, rw.stride};
}

// Apply post-update rewrites to a single scf.for.
// Returns the new ForOp if any rewrites were applied, null otherwise.
static scf::ForOp applyPostUpdateRewrites(
    scf::ForOp forOp, ArrayRef<PostUpdateRewrite> rewrites, OpBuilder &builder) {
  if (rewrites.empty())
    return nullptr;

  // Group rewrites by (base, stride). Ops in the same group share one iter_arg
  // and all use the pre-update pointer. Only one updated_base per group is
  // yielded. This avoids redundant iter_args for same-address ops (e.g. vlds
  // + vsts both accessing %base[%iv]).
  DenseMap<IterArgGroupKey, unsigned> groupToIdx; // group key -> iter_arg index
  SmallVector<unsigned> rwGroupIdx(rewrites.size()); // rewrite -> group index
  SmallVector<Value> groupInitPtrs; // initial pointer per group (base + offset_at_iter0)

  for (auto [i, rw] : llvm::enumerate(rewrites)) {
    auto key = getGroupKey(rw);
    auto [it, inserted] = groupToIdx.try_emplace(key, groupInitPtrs.size());
    if (inserted)
      groupInitPtrs.push_back(rw.initPtr);
    rwGroupIdx[i] = it->second;
  }

  unsigned numGroups = groupInitPtrs.size();

  // Build new init args: original + one new pointer per group.
  SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(),
                                forOp.getInitArgs().end());
  for (Value ptr : groupInitPtrs)
    newInitArgs.push_back(ptr);

  unsigned origIterArgCount = forOp.getInitArgs().size();

  // Create new ForOp.
  builder.setInsertionPoint(forOp);
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInitArgs);
  newForOp->setAttrs(forOp->getAttrs());

  // Map old block args to new: IV + original iter_args.
  IRMapping mapping;
  Block *oldBody = forOp.getBody();
  Block *newBody = newForOp.getBody();
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  for (unsigned i = 0; i < origIterArgCount; ++i)
    mapping.map(oldBody->getArgument(i + 1), newBody->getArgument(i + 1));

  // Clone the body, tracking old->new op correspondence.
  DenseMap<Operation *, Operation *> opMapping;
  builder.setInsertionPointToStart(newBody);
  for (auto &op : oldBody->without_terminator()) {
    Operation *cloned = builder.clone(op, mapping);
    opMapping[&op] = cloned;
  }

  // Apply rewrites. All ops in a group use the same pre-update pointer (block
  // arg). Track the last updated_base per group for yielding.
  SmallVector<Value> groupYieldPtrs(numGroups);
  for (unsigned g = 0; g < numGroups; ++g)
    groupYieldPtrs[g] = newBody->getArgument(origIterArgCount + 1 + g);

  for (auto [rwIdx, rw] : llvm::enumerate(rewrites)) {
    auto it = opMapping.find(rw.op);
    if (it == opMapping.end())
      continue;
    Operation *clonedOp = it->second;
    unsigned gIdx = rwGroupIdx[rwIdx];
    Value ptr = newBody->getArgument(origIterArgCount + 1 + gIdx);
    Value stride = mapping.lookupOrDefault(rw.stride);

    builder.setInsertionPoint(clonedOp);

    if (auto vlds = dyn_cast<pto::VldsOp>(clonedOp)) {
      auto newVlds = builder.create<pto::VldsOp>(
          vlds.getLoc(),
          /*result=*/vlds.getResult().getType(),
          /*updated_base=*/ptr.getType(),
          /*source=*/ptr,
          /*offset=*/stride,
          /*dist=*/vlds.getDistAttr());
      vlds.getResult().replaceAllUsesWith(newVlds.getResult());
      groupYieldPtrs[gIdx] = newVlds.getUpdatedBase();
      clonedOp->erase();
    } else if (auto vsts = dyn_cast<pto::VstsOp>(clonedOp)) {
      Value value = vsts.getValue();
      Value mask = vsts.getMask();
      auto newVsts = builder.create<pto::VstsOp>(
          vsts.getLoc(),
          /*updated_base=*/ptr.getType(),
          /*value=*/value,
          /*destination=*/ptr,
          /*offset=*/stride,
          /*dist=*/vsts.getDistAttr(),
          /*mask=*/mask);
      groupYieldPtrs[gIdx] = newVsts.getUpdatedBase();
      clonedOp->erase();
    }
    // TODO: handle vsstb
  }

  // Build yield: original yields + one pointer per group.
  auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
  SmallVector<Value> newYields;
  for (Value v : oldYield.getOperands())
    newYields.push_back(mapping.lookupOrDefault(v));
  for (Value ptr : groupYieldPtrs)
    newYields.push_back(ptr);

  builder.setInsertionPointToEnd(newBody);
  builder.create<scf::YieldOp>(oldYield.getLoc(), newYields);

  // Replace original ForOp results (only the original ones).
  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));

  forOp.erase();
  return newForOp;
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct VPTOSoftPostUpdatePass
    : public pto::impl::VPTOSoftPostUpdateBase<VPTOSoftPostUpdatePass> {
  using pto::impl::VPTOSoftPostUpdateBase<
      VPTOSoftPostUpdatePass>::VPTOSoftPostUpdateBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());

    module.walk([&](pto::VecScopeOp vecscope) {
      processVecScope(vecscope, builder);
    });
  }

private:
  void processVecScope(pto::VecScopeOp vecscope, OpBuilder &builder) {
    // Collect scf.for ops inside this vecscope (inner-to-outer order).
    SmallVector<scf::ForOp> forOps;
    vecscope.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

    // Process inner-to-outer (walk gives us pre-order, reverse for post-order).
    std::reverse(forOps.begin(), forOps.end());

    for (scf::ForOp forOp : forOps)
      processForOp(forOp, builder);
  }

  void processForOp(scf::ForOp forOp, OpBuilder &builder) {
    SmallVector<PostUpdateRewrite> rewrites;

    for (Operation &op : *forOp.getBody()) {
      if (!isPostUpdateCandidate(&op))
        continue;
      if (isAlreadyPostUpdate(&op))
        continue;
      if (!isDirectlyInForBody(&op, forOp))
        continue;

      Value base, offset;
      if (!extractBaseAndOffset(&op, base, offset))
        continue;

      // Skip vsstb for now (needs special stride handling).
      if (isa<pto::VsstbOp>(&op))
        continue;

      // Delta analysis.
      Value deltaBase = computeDelta(base, forOp, builder);
      Value deltaOffset = offset ? computeDelta(offset, forOp, builder)
                                 : nullptr;

      if (!deltaBase && !deltaOffset)
        continue;

      // Compute stride = delta(base) + delta(offset).
      Value stride;
      if (deltaBase && deltaOffset) {
        auto cb = getConstantIntValue(deltaBase);
        auto co = getConstantIntValue(deltaOffset);
        if (cb && *cb == 0)
          stride = deltaOffset;
        else if (co && *co == 0)
          stride = deltaBase;
        else {
          builder.setInsertionPoint(forOp);
          stride = builder.create<arith::AddIOp>(forOp.getLoc(),
                                                  deltaBase, deltaOffset);
        }
      } else if (deltaBase) {
        stride = deltaBase;
      } else {
        stride = deltaOffset;
      }

      if (!stride)
        continue;

      // Stride must be loop-invariant.
      if (!forOp.isDefinedOutsideOfLoop(stride))
        continue;

      // Compute initial pointer: base + offset_at_iter0.
      Value initPtr = computeInitialPtr(base, offset, forOp, builder);
      if (!initPtr)
        continue;

      rewrites.push_back({&op, base, offset, stride, initPtr});
    }

    if (!rewrites.empty())
      applyPostUpdateRewrites(forOp, rewrites, builder);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOSoftPostUpdatePass() {
  return std::make_unique<VPTOSoftPostUpdatePass>();
}
