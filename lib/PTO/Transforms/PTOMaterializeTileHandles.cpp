// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOMaterializeTileHandles.cpp -------------------------------------===//
//===----------------------------------------------------------------------===//
//
// Reintroduce explicit tile_buf handles after memory planning/sync have used
// memref IR. EmitC can then lower tile operations from tile-typed operands
// instead of rediscovering tile metadata from every memref use.

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <memory>
#include <optional>
#include <utility>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOMATERIALIZETILEHANDLES
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace mlir {
namespace pto {
namespace {

static constexpr llvm::StringLiteral kForceDynamicValidShapeAttrName =
    "__pto.force_dynamic_valid_shape";

static int64_t getIntegerAttrSignedValue(IntegerAttr attr) {
  const APInt &value = attr.getValue();
  return value.getBitWidth() == 0 ? 0 : value.getSExtValue();
}

struct TileHandleMetadata {
  Value source;
  Value validRow;
  Value validCol;
  TileBufConfigAttr config;
  bool explicitConfig = false;
  SmallVector<NamedAttribute, 2> attrs;
};

static bool isLocalTileMemRef(Type type) {
  auto memTy = dyn_cast<MemRefType>(type);
  if (!memTy || memTy.getRank() != 2)
    return false;

  auto asAttr = dyn_cast_or_null<AddressSpaceAttr>(memTy.getMemorySpace());
  if (!asAttr)
    return false;

  switch (asAttr.getAddressSpace()) {
  case AddressSpace::GM:
  case AddressSpace::Zero:
    return false;
  case AddressSpace::VEC:
  case AddressSpace::MAT:
  case AddressSpace::LEFT:
  case AddressSpace::RIGHT:
  case AddressSpace::ACC:
  case AddressSpace::BIAS:
  case AddressSpace::SCALING:
    return true;
  }
  return false;
}

static bool shouldMaterializeOperand(Operation *owner) {
  if (isa<AllocTileOp, PointerCastOp>(owner))
    return false;

  StringRef name = owner->getName().getStringRef();
  if (name == "pto.set_validshape")
    return true;
  if (name == "pto.get_validshape")
    return true;
  if (!name.consume_front("pto."))
    return false;
  return name.starts_with("t");
}

static bool shouldMaterializeYieldOperand(Operation *owner) {
  return isa<scf::YieldOp, pto::YieldOp>(owner);
}

static bool hasStringAttr(ArrayRef<NamedAttribute> attrs, StringRef name,
                          StringRef value) {
  return llvm::any_of(attrs, [&](NamedAttribute attr) {
    if (attr.getName().getValue() != name)
      return false;
    auto strAttr = dyn_cast<StringAttr>(attr.getValue());
    return strAttr && strAttr.getValue() == value;
  });
}

static bool hasAttr(ArrayRef<NamedAttribute> attrs, StringRef name) {
  return llvm::any_of(attrs, [&](NamedAttribute attr) {
    return attr.getName().getValue() == name;
  });
}

static int64_t getConstantIndexOrDynamic(Value value) {
  if (!value)
    return ShapedType::kDynamic;
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  return ShapedType::kDynamic;
}

static void copyTileHandleAttrs(Operation *from,
                                SmallVectorImpl<NamedAttribute> &attrs) {
  StringRef names[] = {"pto.view_semantics", kForceDynamicValidShapeAttrName};
  for (StringRef name : names) {
    if (Attribute attr = from->getAttr(name))
      attrs.push_back(NamedAttribute(StringAttr::get(from->getContext(), name),
                                     attr));
  }
}

static std::optional<AddressSpace> getAddressSpace(Type type) {
  if (auto memTy = dyn_cast<MemRefType>(type)) {
    if (auto asAttr =
            dyn_cast_or_null<AddressSpaceAttr>(memTy.getMemorySpace()))
      return asAttr.getAddressSpace();
    return std::nullopt;
  }
  if (auto tileTy = dyn_cast<TileBufType>(type)) {
    if (auto asAttr =
            dyn_cast_or_null<AddressSpaceAttr>(tileTy.getMemorySpace()))
      return asAttr.getAddressSpace();
  }
  return std::nullopt;
}

static bool isA5Target(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return false;

  if (auto arch = module->getAttrOfType<StringAttr>("pto.target_arch")) {
    if (arch.getValue().equals_insensitive("a5"))
      return true;
  }
  if (auto spec = module->getAttrOfType<StringAttr>("pto.device-spec")) {
    StringRef value = spec.getValue();
    if (value.starts_with("Ascend950") || value.starts_with("Ascend910_95"))
      return true;
  }
  return false;
}

static TileBufConfigAttr makeTileConfig(MLIRContext *ctx, BLayout bl,
                                        SLayout sl) {
  Builder builder(ctx);
  return TileBufConfigAttr::get(
      ctx, BLayoutAttr::get(ctx, bl), SLayoutAttr::get(ctx, sl),
      builder.getI32IntegerAttr(512), PadValueAttr::get(ctx, PadValue::Null),
      CompactModeAttr::get(ctx, CompactMode::Null));
}

static void inferConfigForMaterializedUse(Operation *owner, unsigned operandNo,
                                          Type operandType,
                                          TileHandleMetadata &meta,
                                          MLIRContext *ctx) {
  if (meta.explicitConfig)
    return;

  auto colRow = [&]() {
    return makeTileConfig(ctx, BLayout::ColMajor, SLayout::RowMajor);
  };
  auto rowCol = [&]() {
    return makeTileConfig(ctx, BLayout::RowMajor, SLayout::ColMajor);
  };

  if (isa<TMatmulOp>(owner)) {
    if (!isA5Target(owner))
      return;
    if (operandNo == 0 || operandNo == 2)
      meta.config = colRow();
    else if (operandNo == 1)
      meta.config = rowCol();
    return;
  }

  if (isa<TMatmulAccOp>(owner)) {
    if (!isA5Target(owner))
      return;
    if (operandNo == 0 || operandNo == 1 || operandNo == 3)
      meta.config = colRow();
    else if (operandNo == 2)
      meta.config = rowCol();
    return;
  }

  if (isa<TInsertOp>(owner)) {
    if (operandNo != 0 && operandNo != 3)
      return;
    auto as = getAddressSpace(operandType);
    if (!as)
      return;
    if (*as == AddressSpace::ACC || *as == AddressSpace::MAT)
      meta.config = colRow();
  }
}

static TileHandleMetadata getTileHandleMetadata(Value value,
                                                MLIRContext *ctx) {
  TileHandleMetadata meta;
  meta.source = value;
  meta.config = TileBufConfigAttr::getDefault(ctx);

  if (auto cast = value.getDefiningOp<PointerCastOp>()) {
    meta.validRow = cast.getValidRow();
    meta.validCol = cast.getValidCol();
    if (auto config = cast.getConfig()) {
      meta.config = *config;
      meta.explicitConfig = true;
    }
    copyTileHandleAttrs(cast, meta.attrs);
    return meta;
  }

  return meta;
}

static bool getTilePointerStrides(TileBufConfigAttr configAttr, Type elemTy,
                                  int64_t rows, int64_t cols,
                                  int64_t &rowStride, int64_t &colStride) {
  if (rows == ShapedType::kDynamic || cols == ShapedType::kDynamic)
    return false;

  int32_t blVal = 0;
  if (auto blAttr = dyn_cast<BLayoutAttr>(configAttr.getBLayout()))
    blVal = static_cast<int32_t>(blAttr.getValue());
  else if (auto intAttr = dyn_cast<IntegerAttr>(configAttr.getBLayout()))
    blVal = static_cast<int32_t>(getIntegerAttrSignedValue(intAttr));

  int32_t slVal = 0;
  if (auto slAttr = dyn_cast<SLayoutAttr>(configAttr.getSLayout()))
    slVal = static_cast<int32_t>(slAttr.getValue());
  else if (auto intAttr = dyn_cast<IntegerAttr>(configAttr.getSLayout()))
    slVal = static_cast<int32_t>(getIntegerAttrSignedValue(intAttr));

  bool boxed = slVal != 0;
  int64_t innerRows = 1;
  int64_t innerCols = 1;
  if (boxed) {
    int32_t fractal = 512;
    if (auto frAttr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
      fractal = static_cast<int32_t>(getIntegerAttrSignedValue(frAttr));

    unsigned elemBytes = pto::getPTOStorageElemByteSize(elemTy);
    if (elemBytes == 0)
      return false;

    switch (fractal) {
    case 1024:
      innerRows = 16;
      innerCols = 16;
      break;
    case 32:
      innerRows = 16;
      innerCols = 2;
      break;
    case 512:
      if (slVal == 1) {
        innerRows = 16;
        innerCols = 32 / elemBytes;
      } else if (slVal == 2) {
        innerRows = 32 / elemBytes;
        innerCols = 16;
      } else {
        return false;
      }
      break;
    default:
      return false;
    }
    if (innerRows <= 0 || innerCols <= 0)
      return false;
  }

  if (!boxed) {
    if (blVal == 1) {
      rowStride = 1;
      colStride = rows;
    } else {
      rowStride = cols;
      colStride = 1;
    }
    return true;
  }

  if (blVal == 1) {
    if (slVal != 1)
      return false;
    rowStride = innerCols;
    colStride = rows;
    return true;
  }

  rowStride = cols;
  colStride = innerRows;
  return true;
}

static SmallVector<int64_t, 2>
getMaterializedTileShape(MemRefType memTy, const TileHandleMetadata &meta) {
  SmallVector<int64_t, 2> shape(memTy.getShape().begin(),
                                memTy.getShape().end());
  if (!hasStringAttr(meta.attrs, "pto.view_semantics", "subview"))
    return shape;

  auto sourceMrTy = dyn_cast_or_null<MemRefType>(meta.source.getType());
  if (!sourceMrTy || sourceMrTy.getRank() < 2 ||
      !meta.source.getDefiningOp<memref::SubViewOp>())
    return shape;

  int64_t subRows = sourceMrTy.getDimSize(0);
  int64_t subCols = sourceMrTy.getDimSize(1);
  if (pto::isPTOFloat4PackedType(sourceMrTy.getElementType()) ||
      subRows == ShapedType::kDynamic || subCols == ShapedType::kDynamic)
    return shape;

  SmallVector<int64_t> inheritedStrides;
  int64_t inheritedOffset = ShapedType::kDynamic;
  if (failed(mlir::pto::getPTOMemRefStridesAndOffset(
          sourceMrTy, inheritedStrides, inheritedOffset)) ||
      inheritedStrides.size() < 2)
    return shape;

  int64_t childRowStride = 0;
  int64_t childColStride = 0;
  if (!getTilePointerStrides(meta.config, sourceMrTy.getElementType(), subRows,
                             subCols, childRowStride, childColStride))
    return shape;

  if (inheritedStrides[0] == childRowStride &&
      inheritedStrides[1] == childColStride) {
    shape[0] = subRows;
    shape[1] = subCols;
  }

  return shape;
}

static TileBufType buildTileTypeFromMemRef(MemRefType memTy,
                                           const TileHandleMetadata &meta,
                                           MLIRContext *ctx) {
  SmallVector<int64_t, 2> shape = getMaterializedTileShape(memTy, meta);
  SmallVector<int64_t, 2> validShape(shape.begin(), shape.end());
  bool forceDynamic = hasAttr(meta.attrs, kForceDynamicValidShapeAttrName);
  if (forceDynamic) {
    validShape[0] = ShapedType::kDynamic;
    validShape[1] = ShapedType::kDynamic;
  } else {
    if (meta.validRow)
      validShape[0] = getConstantIndexOrDynamic(meta.validRow);
    if (meta.validCol)
      validShape[1] = getConstantIndexOrDynamic(meta.validCol);
  }

  return TileBufType::get(ctx, shape, memTy.getElementType(),
                          memTy.getMemorySpace(), validShape, meta.config);
}

static bool isMaterializedTileAnchor(Operation *op) {
  return isa<PointerCastOp>(op);
}

static Value makeI64Constant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantIntOp>(loc, value, 64);
}

static Value ensureI64(Value value, OpBuilder &builder, Location loc) {
  if (!value)
    return Value();

  auto i64Ty = builder.getI64Type();
  if (value.getType() == i64Ty)
    return value;
  if (isa<IndexType>(value.getType()))
    return builder.create<arith::IndexCastOp>(loc, i64Ty, value);
  if (auto intTy = dyn_cast<IntegerType>(value.getType())) {
    if (intTy.getWidth() == 64)
      return value;
    if (intTy.getWidth() < 64)
      return builder.create<arith::ExtSIOp>(loc, i64Ty, value);
    return builder.create<arith::TruncIOp>(loc, i64Ty, value);
  }
  return Value();
}

static Value materializeOffset(OpFoldResult ofr, OpBuilder &builder,
                               Location loc) {
  if (isa<Attribute>(ofr)) {
    Attribute attr = cast<Attribute>(ofr);
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return makeI64Constant(builder, loc, getIntegerAttrSignedValue(intAttr));
    return Value();
  }
  return ensureI64(cast<Value>(ofr), builder, loc);
}

static Value addI64(Value lhs, Value rhs, OpBuilder &builder, Location loc) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  return builder.create<arith::AddIOp>(loc, lhs, rhs);
}

static Value mulI64(Value lhs, int64_t rhs, OpBuilder &builder, Location loc) {
  if (!lhs)
    return Value();
  if (rhs == 0)
    return makeI64Constant(builder, loc, 0);
  if (rhs == 1)
    return lhs;
  return builder.create<arith::MulIOp>(loc, lhs,
                                       makeI64Constant(builder, loc, rhs));
}

static Value computeExplicitAddress(Value value, OpBuilder &builder,
                                    Location loc);

static Value computeSubviewAddress(memref::SubViewOp subview,
                                   OpBuilder &builder, Location loc) {
  Value base = computeExplicitAddress(subview.getSource(), builder, loc);
  if (!base)
    return Value();

  auto sourceTy = dyn_cast<MemRefType>(subview.getSource().getType());
  if (!sourceTy)
    return Value();
  unsigned elemBytes = getPTOStorageElemByteSize(sourceTy.getElementType());
  if (elemBytes == 0)
    return Value();

  SmallVector<int64_t> sourceStrides;
  int64_t sourceOffset = ShapedType::kDynamic;
  if (failed(mlir::pto::getPTOMemRefStridesAndOffset(
          sourceTy, sourceStrides, sourceOffset)))
    return Value();

  auto mixedOffsets = subview.getMixedOffsets();
  if (sourceStrides.size() < mixedOffsets.size())
    return Value();

  Value linearOffset;
  for (auto [offsetOfr, stride] :
       llvm::zip_equal(mixedOffsets, ArrayRef<int64_t>(sourceStrides).take_front(
                                         mixedOffsets.size()))) {
    if (stride == ShapedType::kDynamic)
      return Value();
    Value offset = materializeOffset(offsetOfr, builder, loc);
    if (!offset)
      return Value();
    linearOffset = addI64(linearOffset, mulI64(offset, stride, builder, loc),
                          builder, loc);
  }

  if (!linearOffset)
    return base;
  linearOffset = mulI64(linearOffset, elemBytes, builder, loc);
  return builder.create<arith::AddIOp>(loc, base, linearOffset);
}

static Value computePTOSubViewAddress(pto::SubViewOp subview,
                                      OpBuilder &builder, Location loc) {
  Value base = computeExplicitAddress(subview.getSource(), builder, loc);
  if (!base)
    return Value();

  auto sourceTy = dyn_cast<TileBufType>(subview.getSource().getType());
  if (!sourceTy || sourceTy.getShape().size() < 2)
    return Value();

  unsigned elemBytes = getPTOStorageElemByteSize(sourceTy.getElementType());
  if (elemBytes == 0)
    return Value();

  int64_t rowStride = 0;
  int64_t colStride = 0;
  if (!getTilePointerStrides(sourceTy.getConfigAttr(), sourceTy.getElementType(),
                             sourceTy.getShape()[0], sourceTy.getShape()[1],
                             rowStride, colStride))
    return Value();

  SmallVector<int64_t, 2> strides = {rowStride, colStride};
  Value linearOffset;
  for (auto [offset, stride] :
       llvm::zip(subview.getOffsets(), ArrayRef<int64_t>(strides))) {
    Value offsetI64 = ensureI64(offset, builder, loc);
    if (!offsetI64)
      return Value();
    linearOffset =
        addI64(linearOffset, mulI64(offsetI64, stride, builder, loc), builder,
               loc);
  }

  if (!linearOffset)
    return base;
  linearOffset = mulI64(linearOffset, elemBytes, builder, loc);
  return builder.create<arith::AddIOp>(loc, base, linearOffset);
}

static Value computeExplicitAddress(Value value, OpBuilder &builder,
                                    Location loc) {
  if (auto alloc = value.getDefiningOp<AllocTileOp>())
    return ensureI64(alloc.getAddr(), builder, loc);

  if (auto cast = value.getDefiningOp<PointerCastOp>()) {
    if (cast.getAddrs().empty())
      return Value();
    return ensureI64(cast.getAddrs().front(), builder, loc);
  }

  if (auto subview = value.getDefiningOp<memref::SubViewOp>())
    return computeSubviewAddress(subview, builder, loc);

  if (auto subview = value.getDefiningOp<pto::SubViewOp>())
    return computePTOSubViewAddress(subview, builder, loc);

  if (auto select = value.getDefiningOp<arith::SelectOp>()) {
    Value trueAddr = computeExplicitAddress(select.getTrueValue(), builder, loc);
    Value falseAddr =
        computeExplicitAddress(select.getFalseValue(), builder, loc);
    if (!trueAddr || !falseAddr)
      return Value();
    return builder.create<arith::SelectOp>(loc, select.getCondition(),
                                           trueAddr, falseAddr);
  }

  if (auto cast = value.getDefiningOp<memref::CastOp>())
    return computeExplicitAddress(cast.getSource(), builder, loc);

  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getNumOperands() == 1)
      return computeExplicitAddress(cast.getOperand(0), builder, loc);
  }

  return Value();
}

static bool isControlFlowAddressProducer(Operation *op) {
  if (!op)
    return false;

  StringRef name = op->getName().getStringRef();
  return name == "scf.if" || name == "scf.for" || name == "scf.while" ||
         name == "scf.execute_region" || name == "scf.index_switch";
}

static Value peelAddressSource(Value value) {
  while (true) {
    if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
      value = subview.getSource();
      continue;
    }

    if (auto cast = value.getDefiningOp<memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }

    return value;
  }
}

static bool isFunctionEntryBlockArgument(BlockArgument arg) {
  Operation *parent = arg.getOwner()->getParentOp();
  auto func = dyn_cast_or_null<func::FuncOp>(parent);
  return func && arg.getOwner() == &func.getBody().front();
}

static bool isUnsupportedControlFlowAddress(Value value) {
  value = peelAddressSource(value);

  if (auto arg = dyn_cast<BlockArgument>(value))
    return !isFunctionEntryBlockArgument(arg);

  return isControlFlowAddressProducer(value.getDefiningOp());
}

static void emitMissingExplicitAddressError(Operation *owner, Value value) {
  value = peelAddressSource(value);
  auto diag = owner->emitOpError()
              << "cannot materialize tile handle for local memref because its "
                 "explicit byte address cannot be recovered";

  if (isa<BlockArgument>(value)) {
    diag << "; region block arguments and loop-carried memref values are "
            "unsupported here";
    return;
  }

  Operation *def = value.getDefiningOp();
  if (!def) {
    diag << "; value has no defining op";
    return;
  }

  if (isControlFlowAddressProducer(def)) {
    diag << "; control-flow result '" << def->getName()
         << "' cannot carry a local memref into tile materialization";
    return;
  }

  diag << "; unsupported defining op is '" << def->getName() << "'";
}

static Value getAllocValidOperand(TileBufType tileTy, Value operand,
                                  unsigned dim, OpBuilder &builder,
                                  Location loc) {
  auto validShape = tileTy.getValidShape();
  if (validShape.size() <= dim || validShape[dim] >= 0)
    return Value();
  if (operand)
    return operand;

  auto shape = tileTy.getShape();
  if (shape.size() > dim && shape[dim] != ShapedType::kDynamic)
    return builder.create<arith::ConstantIndexOp>(loc, shape[dim]);
  return Value();
}

static Attribute getAttr(ArrayRef<NamedAttribute> attrs, StringRef name) {
  for (NamedAttribute attr : attrs) {
    if (attr.getName().getValue() == name)
      return attr.getValue();
  }
  return {};
}

static void copyMaterializedTileAttrs(ArrayRef<NamedAttribute> attrs,
                                      Operation *to) {
  if (Attribute attr = getAttr(attrs, kForceDynamicValidShapeAttrName))
    to->setAttr(kForceDynamicValidShapeAttrName, attr);
}

static void updateResultTypesAfterMaterializingOperand(Operation *op,
                                                       unsigned operandNo,
                                                       Type tileTy) {
  if (auto tassign = dyn_cast<TAssignOp>(op)) {
    if (operandNo == 0)
      tassign.getResult().setType(tileTy);
  }
}

static bool isTileViewSemantics(StringAttr viewSemantics) {
  return viewSemantics && (viewSemantics.getValue() == "treshape" ||
                           viewSemantics.getValue() == "bitcast");
}

static Value materializeAnchorResult(Operation *anchor, Value anchoredValue,
                                     OpBuilder &builder, MLIRContext *ctx,
                                     DenseMap<Value, Value> &tileHandles,
                                     const DenseSet<Value> &mustMaterialize,
                                     bool &failedMaterialization) {
  auto memTy = dyn_cast<MemRefType>(anchoredValue.getType());
  if (!memTy || !isLocalTileMemRef(memTy))
    return Value();

  SmallVector<OpOperand *> usesToRewrite;
  for (OpOperand &use : anchoredValue.getUses()) {
    if (shouldMaterializeOperand(use.getOwner()) ||
        shouldMaterializeYieldOperand(use.getOwner()))
      usesToRewrite.push_back(&use);
  }

  TileHandleMetadata meta = getTileHandleMetadata(anchoredValue, ctx);
  auto viewSemantics = dyn_cast_or_null<StringAttr>(
      getAttr(meta.attrs, "pto.view_semantics"));
  bool isTileView = isTileViewSemantics(viewSemantics);
  if (usesToRewrite.empty() && !isTileView &&
      !mustMaterialize.contains(anchoredValue))
    return Value();

  if (Value existing = tileHandles.lookup(anchoredValue)) {
    for (OpOperand *use : usesToRewrite) {
      Operation *owner = use->getOwner();
      unsigned operandNo = use->getOperandNumber();
      use->set(existing);
      updateResultTypesAfterMaterializingOperand(owner, operandNo,
                                                 existing.getType());
    }
    return existing;
  }

  for (OpOperand *use : usesToRewrite)
    inferConfigForMaterializedUse(use->getOwner(), use->getOperandNumber(),
                                  anchoredValue.getType(), meta, ctx);
  auto tileTy = buildTileTypeFromMemRef(memTy, meta, ctx);

  builder.setInsertionPointAfter(anchor);
  Value materialized;
  Value sourceTile = meta.source ? tileHandles.lookup(meta.source) : Value();
  if (sourceTile && viewSemantics &&
      viewSemantics.getValue() == "treshape") {
    materialized =
        builder.create<TReshapeOp>(anchor->getLoc(), tileTy, sourceTile)
            .getResult();
  } else if (sourceTile && viewSemantics &&
             viewSemantics.getValue() == "bitcast") {
    materialized =
        builder.create<BitcastOp>(anchor->getLoc(), tileTy, sourceTile)
            .getResult();
  } else {
    Value addr = computeExplicitAddress(anchoredValue, builder, anchor->getLoc());
    if (!addr && isUnsupportedControlFlowAddress(anchoredValue)) {
      emitMissingExplicitAddressError(anchor, anchoredValue);
      failedMaterialization = true;
      return Value();
    }
    auto alloc = builder.create<AllocTileOp>(
        anchor->getLoc(), tileTy, addr ? addr : Value(),
        getAllocValidOperand(tileTy, meta.validRow, 0, builder,
                             anchor->getLoc()),
        getAllocValidOperand(tileTy, meta.validCol, 1, builder,
                             anchor->getLoc()));
    copyMaterializedTileAttrs(meta.attrs, alloc);
    materialized = alloc.getResult();
  }

  for (OpOperand *use : usesToRewrite) {
    Operation *owner = use->getOwner();
    unsigned operandNo = use->getOperandNumber();
    use->set(materialized);
    updateResultTypesAfterMaterializingOperand(owner, operandNo, tileTy);
  }

  tileHandles[anchoredValue] = materialized;
  return materialized;
}

struct PTOMaterializeTileHandlesPass
    : public impl::PTOMaterializeTileHandlesBase<
          PTOMaterializeTileHandlesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    bool hasUnresolvedMultiTile = false;
    module.walk([&](Operation *op) {
      if (!isa<AllocMultiTileOp, MultiTileGetOp>(op))
        return;
      op->emitError(
          "must be resolved by pto-resolve-buffer-select before "
          "pto-materialize-tile-handles");
      hasUnresolvedMultiTile = true;
    });
    if (hasUnresolvedMultiTile) {
      signalPassFailure();
      return;
    }

    OpBuilder builder(ctx);
    DenseMap<Value, Value> tileHandles;
    DenseSet<Value> mustMaterialize;
    bool failedMaterialization = false;

    SmallVector<Operation *, 32> anchors;
    module.walk([&](Operation *op) {
      if (!isMaterializedTileAnchor(op))
        return;
      anchors.push_back(op);
    });
    for (Operation *anchor : anchors) {
      if (anchor->getNumResults() != 1)
        continue;
      Value anchoredValue = anchor->getResult(0);
      if (!isLocalTileMemRef(anchoredValue.getType()))
        continue;
      TileHandleMetadata meta = getTileHandleMetadata(anchoredValue, ctx);
      auto viewSemantics = dyn_cast_or_null<StringAttr>(
          getAttr(meta.attrs, "pto.view_semantics"));
      if (isTileViewSemantics(viewSemantics) && meta.source)
        mustMaterialize.insert(meta.source);
    }

    for (Operation *anchor : anchors) {
      if (anchor->getNumResults() != 1)
        continue;
      materializeAnchorResult(anchor, anchor->getResult(0), builder, ctx,
                              tileHandles, mustMaterialize,
                              failedMaterialization);
    }

    if (failedMaterialization) {
      signalPassFailure();
      return;
    }

    SmallVector<std::pair<Operation *, unsigned>, 32> operandsToRewrite;
    module.walk([&](Operation *op) {
      if (!shouldMaterializeOperand(op))
        return;
      for (OpOperand &operand : op->getOpOperands()) {
        if (isLocalTileMemRef(operand.get().getType()))
          operandsToRewrite.push_back({op, operand.getOperandNumber()});
      }
    });

    for (auto [op, operandNo] : operandsToRewrite) {
      Value oldValue = op->getOperand(operandNo);
      if (!isa<MemRefType>(oldValue.getType()))
        continue;
      if (op->getName().getStringRef() == "pto.tassign" && operandNo == 0)
        continue;

      if (Value existing = tileHandles.lookup(oldValue)) {
        op->setOperand(operandNo, existing);
        updateResultTypesAfterMaterializingOperand(op, operandNo,
                                                   existing.getType());
        continue;
      }

      auto memTy = cast<MemRefType>(oldValue.getType());
      TileHandleMetadata meta = getTileHandleMetadata(oldValue, ctx);
      inferConfigForMaterializedUse(op, operandNo, oldValue.getType(), meta,
                                    ctx);
      auto tileTy = buildTileTypeFromMemRef(memTy, meta, ctx);

      builder.setInsertionPoint(op);
      Value materialized;
      Value addr = computeExplicitAddress(oldValue, builder, op->getLoc());
      if (!addr && isUnsupportedControlFlowAddress(oldValue)) {
        emitMissingExplicitAddressError(op, oldValue);
        failedMaterialization = true;
        continue;
      }
      auto alloc = builder.create<AllocTileOp>(
          op->getLoc(), tileTy, addr ? addr : Value(),
          getAllocValidOperand(tileTy, meta.validRow, 0, builder,
                               op->getLoc()),
          getAllocValidOperand(tileTy, meta.validCol, 1, builder,
                               op->getLoc()));
      copyMaterializedTileAttrs(meta.attrs, alloc);
      materialized = alloc.getResult();
      tileHandles[oldValue] = materialized;
      op->setOperand(operandNo, materialized);
      updateResultTypesAfterMaterializingOperand(op, operandNo, tileTy);
    }

    if (failedMaterialization) {
      signalPassFailure();
      return;
    }

  }
};

} // namespace

std::unique_ptr<Pass> createPTOMaterializeTileHandlesPass() {
  return std::make_unique<PTOMaterializeTileHandlesPass>();
}

} // namespace pto
} // namespace mlir
