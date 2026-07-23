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

// Per-op-type descriptor: how to extract address operands and check post-update.
// base/strideOperand indices are operand positions; updatedBaseResultIdx is the
// result index for updated_base (or -1 if no such result exists).
struct PostUpdateOpInfo {
  int baseOperandIdx;
  int strideOperandIdx;
  int64_t weight;              // 1 for standard, 32 for block-stride
  unsigned minResultsForPost;  // numResults > this means already post-update
};

using PostUpdateTable = llvm::StringMap<PostUpdateOpInfo>;

static const PostUpdateTable &getPostUpdateTable() {
  static const PostUpdateTable table = [] {
    PostUpdateTable t;
    //                       base  strideOp  weight  minResultsForPost
    t["pto.vlds"]         = { 0,    1,        1,      1 };
    t["pto.vsts"]         = { 1,    2,        1,      0 };
    t["pto.vsstb"]        = { 1,    3,        32,     0 };
    return t;
  }();
  return table;
}

static const PostUpdateOpInfo *getPostUpdateInfo(Operation *op) {
  auto it = getPostUpdateTable().find(op->getName().getStringRef());
  if (it == getPostUpdateTable().end())
    return nullptr;
  return &it->second;
}

// Extract base and stride operand from a candidate op using table info.
static void extractBaseAndStrideOperand(Operation *op,
                                        const PostUpdateOpInfo &info,
                                        Value &base, Value &strideOperand) {
  base = op->getOperand(info.baseOperandIdx);
  strideOperand = op->getOperand(info.strideOperandIdx);
}

// Check if op already has an updated_base result.
static bool isAlreadyPostUpdate(Operation *op, const PostUpdateOpInfo &info) {
  return op->getNumResults() > info.minResultsForPost;
}

// Check if op is directly inside the scf.for body (not nested in scf.if etc).
static bool isDirectlyInForBody(Operation *op, scf::ForOp forOp) {
  return op->getParentOp() == forOp.getOperation();
}

//===----------------------------------------------------------------------===//
// Accumulator Analysis: Linear Decomposition
//===----------------------------------------------------------------------===//

// Result of decomposing a value into blockArg * coeff + increment.
struct LinearDecomp {
  int64_t coeff;
  Value increment; // nullptr means 0
};

static Value addIncrements(Value a, Value b, scf::ForOp forOp,
                           OpBuilder &builder) {
  if (!a) return b;
  if (!b) return a;
  if (auto ca = getConstantIntValue(a); ca && *ca == 0) return b;
  if (auto cb = getConstantIntValue(b); cb && *cb == 0) return a;
  builder.setInsertionPoint(forOp.getBody()->getTerminator());
  return builder.create<arith::AddIOp>(forOp.getLoc(), a, b);
}

static Value subIncrements(Value a, Value b, scf::ForOp forOp,
                           OpBuilder &builder) {
  if (!b) return a;
  if (auto cb = getConstantIntValue(b); cb && *cb == 0) return a;
  builder.setInsertionPoint(forOp.getBody()->getTerminator());
  if (!a) {
    if (b.getType().isIndex())
      a = builder.create<arith::ConstantIndexOp>(forOp.getLoc(), 0);
    else
      a = builder.create<arith::ConstantIntOp>(
          forOp.getLoc(), 0, b.getType().getIntOrFloatBitWidth());
  }
  return builder.create<arith::SubIOp>(forOp.getLoc(), a, b);
}

// Decompose `v` into blockArg * coeff + increment by recursing through
// addi/subi/muli/index_cast/addptr chains.
static std::optional<LinearDecomp>
decomposeLinear(Value v, BlockArgument blockArg, scf::ForOp forOp,
                OpBuilder &builder) {
  // v == blockArg → {1, nullptr}
  if (v == blockArg)
    return LinearDecomp{1, nullptr};

  // v is other block arg (IV, different iter_arg, func arg) → {0, v}
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return LinearDecomp{0, v};

  // v is loop-invariant or constant → {0, v}
  if (forOp.isDefinedOutsideOfLoop(v) ||
      defOp->hasTrait<OpTrait::ConstantLike>())
    return LinearDecomp{0, v};

  // v = addi(a, b) → {ca + cb, ia + ib}
  // v = subi(a, b) → {ca - cb, ia - ib}
  if (isa<arith::AddIOp, arith::SubIOp>(defOp)) {
    auto da = decomposeLinear(defOp->getOperand(0), blockArg, forOp, builder);
    auto db = decomposeLinear(defOp->getOperand(1), blockArg, forOp, builder);
    if (!da || !db)
      return std::nullopt;
    if (da->coeff == 0 && db->coeff == 0)
      return LinearDecomp{0, v};
    bool isSub = isa<arith::SubIOp>(defOp);
    return LinearDecomp{
        isSub ? da->coeff - db->coeff : da->coeff + db->coeff,
        isSub ? subIncrements(da->increment, db->increment, forOp, builder)
              : addIncrements(da->increment, db->increment, forOp, builder)};
  }

  // v = muli(a, b), one side blockArg-free with constant k → {c * k, i * k}
  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    auto da = decomposeLinear(mulOp.getLhs(), blockArg, forOp, builder);
    auto db = decomposeLinear(mulOp.getRhs(), blockArg, forOp, builder);
    if (!da || !db)
      return std::nullopt;
    if (da->coeff == 0 && db->coeff == 0)
      return LinearDecomp{0, v};
    if (da->coeff != 0 && db->coeff != 0)
      return std::nullopt;
    auto &withBA = (da->coeff != 0) ? *da : *db;
    Value multiplier = (da->coeff != 0) ? mulOp.getRhs() : mulOp.getLhs();
    auto constMul = getConstantIntValue(multiplier);
    if (!constMul)
      return std::nullopt;
    Value newInc = nullptr;
    if (withBA.increment && *constMul != 0) {
      if (*constMul == 1)
        newInc = withBA.increment;
      else {
        builder.setInsertionPoint(forOp.getBody()->getTerminator());
        newInc = builder.create<arith::MulIOp>(forOp.getLoc(),
                                                withBA.increment, multiplier);
      }
    }
    return LinearDecomp{withBA.coeff * *constMul, newInc};
  }

  // v = index_cast(a) → {ca, cast(ia)}
  if (isa<arith::IndexCastUIOp, arith::IndexCastOp>(defOp)) {
    auto d = decomposeLinear(defOp->getOperand(0), blockArg, forOp, builder);
    if (!d)
      return std::nullopt;
    if (d->coeff == 0)
      return LinearDecomp{0, v};
    if (d->increment) {
      builder.setInsertionPoint(forOp.getBody()->getTerminator());
      Operation *cast = builder.clone(*defOp);
      cast->setOperand(0, d->increment);
      d->increment = cast->getResult(0);
    }
    return *d;
  }

  // v = addptr(ptr, offset) → {c_ptr, i_ptr + offset}
  if (auto addPtrOp = dyn_cast<pto::AddPtrOp>(defOp)) {
    auto dp = decomposeLinear(addPtrOp.getPtr(), blockArg, forOp, builder);
    if (!dp)
      return std::nullopt;
    if (dp->coeff == 0)
      return LinearDecomp{0, v};
    return LinearDecomp{
        dp->coeff,
        addIncrements(dp->increment, addPtrOp.getOffset(), forOp, builder)};
  }

  // Unrecognized op → unknown
  return std::nullopt;
}

// Trace `v` back to an iter_arg BlockArgument of `forOp`, then decompose
// the yield expression to extract the per-iteration increment. Walks through
// index_cast (type-changing), addi/subi with loop-invariant offset, and
// addptr with loop-invariant offset. Type casts along the path are applied
// to the increment so its type matches v's context.
static std::optional<Value> getIterArgIncrement(Value v, scf::ForOp forOp,
                                                OpBuilder &builder) {
  SmallVector<Operation *> casts;
  Value current = v;

  while (true) {
    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      if (blockArg.getOwner() != forOp.getBody() ||
          blockArg.getArgNumber() == 0)
        return std::nullopt;

      unsigned idx = blockArg.getArgNumber() - 1;
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      auto decomp =
          decomposeLinear(yieldOp.getOperand(idx), blockArg, forOp, builder);
      if (!decomp || decomp->coeff != 1)
        return Value();

      Value inc = decomp->increment;
      if (!inc) {
        builder.setInsertionPoint(forOp);
        inc = builder.create<arith::ConstantIndexOp>(forOp.getLoc(), 0);
      }
      for (Operation *op : llvm::reverse(casts)) {
        builder.setInsertionPoint(forOp.getBody()->getTerminator());
        Operation *c = builder.clone(*op);
        c->setOperand(0, inc);
        inc = c->getResult(0);
      }
      return inc;
    }

    Operation *defOp = current.getDefiningOp();
    if (!defOp || forOp.isDefinedOutsideOfLoop(current) ||
        defOp->hasTrait<OpTrait::ConstantLike>())
      return std::nullopt;

    if (isa<arith::IndexCastUIOp, arith::IndexCastOp>(defOp)) {
      casts.push_back(defOp);
      current = defOp->getOperand(0);
      continue;
    }

    if (isa<arith::AddIOp>(defOp)) {
      if (forOp.isDefinedOutsideOfLoop(defOp->getOperand(1))) {
        current = defOp->getOperand(0);
        continue;
      }
      if (forOp.isDefinedOutsideOfLoop(defOp->getOperand(0))) {
        current = defOp->getOperand(1);
        continue;
      }
    }

    if (isa<arith::SubIOp>(defOp)) {
      if (forOp.isDefinedOutsideOfLoop(defOp->getOperand(1))) {
        current = defOp->getOperand(0);
        continue;
      }
    }

    if (auto addPtrOp = dyn_cast<pto::AddPtrOp>(defOp)) {
      if (forOp.isDefinedOutsideOfLoop(addPtrOp.getOffset())) {
        current = addPtrOp.getPtr();
        continue;
      }
    }

    return std::nullopt;
  }
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

// Get the per-iteration stride of `v`: tries accumulator analysis first
// (for iter_arg-derived values with possibly loop-varying increment),
// falls back to delta analysis (for IV-derived values, loop-invariant result).
// getIterArgIncrement returns:
//   nullopt        → not iter_arg-related, fall through to delta
//   Some(Value())  → iter_arg found but decomposition failed, give up
//   Some(non-null) → success
static Value getStride(Value v, scf::ForOp forOp, OpBuilder &builder) {
  if (auto result = getIterArgIncrement(v, forOp, builder))
    return *result;
  return computeDelta(v, forOp, builder);
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

// Compute the initial pointer: base_at_iter0 + weight * strideOperand_at_iter0.
// weight=1 for vlds/vsts (offset in elements), weight=32 for vsstb/vsldb.
static Value computeInitialPtr(Value base, Value strideOperand,
                               int64_t weight, scf::ForOp forOp,
                               OpBuilder &builder) {
  Value baseAtEntry = materializeAtLoopEntry(base, forOp, builder);
  if (!baseAtEntry)
    return nullptr;

  if (!strideOperand)
    return baseAtEntry;

  Value soAtEntry = materializeAtLoopEntry(strideOperand, forOp, builder);
  if (!soAtEntry)
    return nullptr;

  if (auto constVal = getConstantIntValue(soAtEntry);
      constVal && *constVal == 0)
    return baseAtEntry;

  builder.setInsertionPoint(forOp);
  Value scaledOffset = soAtEntry;
  if (weight != 1) {
    Value soIndex = soAtEntry;
    if (soAtEntry.getType() != builder.getIndexType())
      soIndex = builder.create<arith::IndexCastUIOp>(
          forOp.getLoc(), builder.getIndexType(), soAtEntry);
    Value weightVal =
        builder.create<arith::ConstantIndexOp>(forOp.getLoc(), weight);
    scaledOffset =
        builder.create<arith::MulIOp>(forOp.getLoc(), soIndex, weightVal);
  }
  return builder.create<pto::AddPtrOp>(forOp.getLoc(), baseAtEntry,
                                       scaledOffset);
}

// Combine per-operand strides into the final stride_new for the post-update op.
// total = deltaBase + weight * deltaOffset; stride_new = total / weight.
// Returns nullptr if stride is zero or weight doesn't divide evenly.
// `strideOperandType` is needed for weight>1 to create the right integer type.
static Value computeFinalStride(Value deltaBase, Value deltaOffset,
                              int64_t weight, Type strideOperandType,
                              Operation *op, scf::ForOp forOp,
                              OpBuilder &builder) {
  bool anyLoopVarying = !forOp.isDefinedOutsideOfLoop(deltaBase) ||
                        !forOp.isDefinedOutsideOfLoop(deltaOffset);
  auto setCombineIP = [&]() {
    if (anyLoopVarying)
      builder.setInsertionPoint(op);
    else
      builder.setInsertionPoint(forOp);
  };

  Value weightedDeltaOffset = deltaOffset;
  if (weight != 1) {
    if (auto co = getConstantIntValue(deltaOffset); co && *co != 0) {
      setCombineIP();
      weightedDeltaOffset = builder.create<arith::ConstantIndexOp>(
          forOp.getLoc(), *co * weight);
    } else if (auto co = getConstantIntValue(deltaOffset); co && *co == 0) {
      // 0 * weight = 0
    } else {
      setCombineIP();
      Value weightVal =
          builder.create<arith::ConstantIndexOp>(forOp.getLoc(), weight);
      weightedDeltaOffset =
          builder.create<arith::MulIOp>(forOp.getLoc(), deltaOffset, weightVal);
    }
  }

  Value stride;
  auto cb = getConstantIntValue(deltaBase);
  auto co = getConstantIntValue(weightedDeltaOffset);
  if (cb && *cb == 0)
    stride = weightedDeltaOffset;
  else if (co && *co == 0)
    stride = deltaBase;
  else {
    setCombineIP();
    stride = builder.create<arith::AddIOp>(forOp.getLoc(), deltaBase,
                                            weightedDeltaOffset);
  }

  if (auto constTotal = getConstantIntValue(stride);
      constTotal && *constTotal == 0)
    return nullptr;

  if (weight != 1) {
    auto constTotal = getConstantIntValue(stride);
    if (!constTotal || *constTotal % weight != 0)
      return nullptr;
    setCombineIP();
    unsigned bitWidth = strideOperandType.getIntOrFloatBitWidth();
    return builder.create<arith::ConstantIntOp>(forOp.getLoc(),
                                                *constTotal / weight, bitWidth);
  }
  return stride;
}

// Information about a post-update transformation to apply.
struct PostUpdateRewrite {
  Operation *op;
  Value base;
  Value stride;  // stride value (stride_new for block-stride ops)
  Value initPtr; // base + weight * strideOperand_at_iter0
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
    scf::ForOp forOp, ArrayRef<PostUpdateRewrite> rewrites,
    OpBuilder &builder) {
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
    Value strideNew = mapping.lookupOrDefault(rw.stride);

    builder.setInsertionPoint(clonedOp);

    const PostUpdateOpInfo *info = getPostUpdateInfo(clonedOp);
    if (!info)
      continue;

    // Build the post-update op generically: replace base and strideOperand,
    // keep all other operands, append updated_base to result types.
    OperationState state(clonedOp->getLoc(), clonedOp->getName());
    for (auto [i, operand] : llvm::enumerate(clonedOp->getOperands())) {
      if (static_cast<int>(i) == info->baseOperandIdx)
        state.addOperands(ptr);
      else if (static_cast<int>(i) == info->strideOperandIdx)
        state.addOperands(strideNew);
      else
        state.addOperands(operand);
    }
    for (Type t : clonedOp->getResultTypes())
      state.addTypes(t);
    state.addTypes(ptr.getType()); // updated_base (appended last)
    state.addAttributes(clonedOp->getAttrs());

    Operation *newOp = builder.create(state);

    // Replace old results with new and update the mapping so that later
    // yield construction via mapping.lookupOrDefault sees the new results
    // instead of dangling pointers to the erased clonedOp.
    for (unsigned r = 0; r < clonedOp->getNumResults(); ++r) {
      clonedOp->getResult(r).replaceAllUsesWith(newOp->getResult(r));
      mapping.map(rw.op->getResult(r), newOp->getResult(r));
    }

    // updated_base is the last result.
    groupYieldPtrs[gIdx] = newOp->getResult(newOp->getNumResults() - 1);
    clonedOp->erase();
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
      const PostUpdateOpInfo *info = getPostUpdateInfo(&op);
      if (!info)
        continue;
      if (isAlreadyPostUpdate(&op, *info))
        continue;
      if (!isDirectlyInForBody(&op, forOp))
        continue;

      Value base, strideOperand;
      extractBaseAndStrideOperand(&op, *info, base, strideOperand);
      int64_t weight = info->weight;

      // Analyze each operand independently: accumulator (iter_arg) first,
      // delta (IV/affine) fallback. Both return the per-iteration stride.
      Value deltaBase = getStride(base, forOp, builder);
      Value deltaOffset = getStride(strideOperand, forOp, builder);

      if (!deltaBase || !deltaOffset)
        continue;

      Value strideNew = computeFinalStride(deltaBase, deltaOffset, weight,
                                          strideOperand.getType(), &op,
                                          forOp, builder);
      if (!strideNew)
        continue;

      Value initPtr =
          computeInitialPtr(base, strideOperand, weight, forOp, builder);
      if (!initPtr)
        continue;

      rewrites.push_back({&op, base, strideNew, initPtr});
    }

    if (!rewrites.empty())
      applyPostUpdateRewrites(forOp, rewrites, builder);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOSoftPostUpdatePass() {
  return std::make_unique<VPTOSoftPostUpdatePass>();
}
