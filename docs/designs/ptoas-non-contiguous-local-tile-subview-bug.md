# PTOAS Non-Contiguous Local Tile Subview Bug

## 问题描述

`PTOMaterializeTileHandles` 在处理非连续 local tile subview 时，会把 `memref.subview<child shape, inherited stride>` 物化成 `tile_buf<parent physical shape, valid=child shape>`。

这种物化方式保留了父 tile stride 语义，但会让后续 A2/A3 `pto.tmov` 看到 src/dst physical shape 不一致，从而触发 verifier 报错：

```text
'pto.tmov' op expects A2/A3 non-mat tmov to use matching src/dst shapes
```

因此 latest main 对非连续 local tile subview 不能正确 lowering。

## 最小复现

```mlir
module attributes {pto.target_arch = "a2a3"} {
  func.func @non_contig_subview_tmov(%arg0: !pto.ptr<f32>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0_i64 = arith.constant 0 : i64
    %c0_index = arith.constant 0 : index
    %c1_index = arith.constant 1 : index
    %c2_index = arith.constant 2 : index
    %c4_index = arith.constant 4 : index
    %c8_index = arith.constant 8 : index
    %c16_index = arith.constant 16 : index
    %cst0 = arith.constant 0.000000e+00 : f32
    %cst1 = arith.constant 1.000000e+00 : f32

    %out_view = pto.make_tensor_view %arg0,
      shape = [%c4_index, %c16_index],
      strides = [%c16_index, %c1_index]
      {layout = #pto.layout<nd>} : !pto.tensor_view<?x?xf32>

    %parent = pto.alloc_tile valid_row = %c4_index valid_col = %c16_index
      : !pto.tile_buf<loc=vec, dtype=f32, rows=4, cols=16,
                      v_row=?, v_col=?, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>
    pto.texpands ins(%cst0 : f32) outs(%parent
      : !pto.tile_buf<loc=vec, dtype=f32, rows=4, cols=16,
                      v_row=?, v_col=?, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)

    %src = pto.alloc_tile valid_row = %c2_index valid_col = %c8_index
      : !pto.tile_buf<loc=vec, dtype=f32, rows=2, cols=8,
                      v_row=?, v_col=?, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>
    pto.texpands ins(%cst1 : f32) outs(%src
      : !pto.tile_buf<loc=vec, dtype=f32, rows=2, cols=8,
                      v_row=?, v_col=?, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)

    %sub = pto.subview %parent[%c0_index, %c2_index]
      sizes [2, 8] valid [%c2_index, %c8_index]
      : !pto.tile_buf<loc=vec, dtype=f32, rows=4, cols=16,
                      v_row=?, v_col=?, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>
     -> !pto.tile_buf<loc=vec, dtype=f32, rows=2, cols=8,
                      v_row=2, v_col=8, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>

    pto.tmov ins(%src
      : !pto.tile_buf<loc=vec, dtype=f32, rows=2, cols=8,
                      v_row=?, v_col=?, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
      outs(%sub
      : !pto.tile_buf<loc=vec, dtype=f32, rows=2, cols=8,
                      v_row=2, v_col=8, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)

    %out_pview = pto.partition_view %out_view,
      offsets = [%c0_index, %c0_index],
      sizes = [%c4_index, %c16_index]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<4x16xf32>

    pto.tstore ins(%parent
      : !pto.tile_buf<loc=vec, dtype=f32, rows=4, cols=16,
                      v_row=?, v_col=?, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
      outs(%out_pview : !pto.partition_tensor_view<4x16xf32>)
      {layout = #pto.layout<nd>}

    return
  }
}
```

## 复现命令

```bash
ptoas non_contig_subview_tmov.pto \
  --enable-insert-sync \
  --pto-level=level2 \
  --pto-arch a3 \
  -mlir-print-ir-after-all
```

## 实际结果

编译失败：

```text
loc("non_contig_subview_tmov.pto":18:5): error:
'pto.tmov' op expects A2/A3 non-mat tmov to use matching src/dst shapes
Error: Pass execution failed.
```

`PTOMaterializeTileHandles` 失败前的 IR 中可以看到：

```mlir
%subview = memref.subview %parent[0, 2] [2, 8] [1, 1]
  : memref<4x16xf32, strided<[16, 1], offset: ?>, #pto.address_space<vec>>
 to memref<2x8xf32, strided<[16, 1], offset: ?>, #pto.address_space<vec>>

%addr = ...
%dst = pto.alloc_tile addr = %addr
  : !pto.tile_buf<vec, 4x16xf32, valid=2x8>

pto.tmov ins(%src : !pto.tile_buf<vec, 2x8xf32, valid=?x?>)
  outs(%dst : !pto.tile_buf<vec, 4x16xf32, valid=2x8>)
```

于是 A2/A3 `tmov` verifier 比较 physical shape：

```text
srcShape = 2x8
dstShape = 4x16
```

并报错。

## 原因分析

`PTOMaterializeTileHandles` 从 `BindTileOp` / `PointerCastOp` 这类 anchor 开始，把 local memref handle 重新 materialize 成 `pto.alloc_tile`。关键逻辑在 `getMaterializedTileShape()`：

```cpp
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

  SmallVector<int64_t> inheritedStrides;
  int64_t inheritedOffset = ShapedType::kDynamic;
  getPTOMemRefStridesAndOffset(sourceMrTy, inheritedStrides, inheritedOffset);

  int64_t childRowStride = 0;
  int64_t childColStride = 0;
  getTilePointerStrides(meta.config, sourceMrTy.getElementType(),
                        subRows, subCols,
                        childRowStride, childColStride);

  if (inheritedStrides[0] == childRowStride &&
      inheritedStrides[1] == childColStride) {
    shape[0] = subRows;
    shape[1] = subCols;
  }

  return shape;
}
```

也就是说：

- 连续 subview：如果 inherited stride 等于 child compact tile stride，就可以安全物化为 child shape。
- 非连续 subview：如果 inherited stride 不等于 child compact tile stride，就保留原来的 parent/bind result shape。

在本复现中：

```text
subview shape = 2x8
inherited stride = [16, 1]
child compact stride = [8, 1]
```

因为 `[16, 1] != [8, 1]`，`PTOMaterializeTileHandles` 保留 parent shape `4x16`，再用 `bind_tile` 的 valid row/col 构造出 `tile_buf<4x16, valid=2x8>`。

该策略试图保留非连续 subview 的父 stride 语义，但普通 `tile_buf<parent shape, valid=child shape>` 又不能直接作为 A2/A3 `tmov` 的 dst 使用，因为 `tmov` verifier 要求 non-MAT src/dst physical shape 一致。

## 期望结果

PTOAS 应该能够正确处理非连续 local tile subview，或者在暂不支持时给出明确诊断。

可能的修复方向：

- 对连续 subview：继续物化为 compact child-shape tile_buf。
- 对非连续 subview：不能简单物化为普通 compact child tile，也不能直接交给 `tmov` 看到 parent-shape tile。
- 需要为 tile-native subview 保留 stride/offset 语义，或在 lowering 中针对非连续 subview 生成正确的分行/分块 copy。
- 如果当前后端暂不支持，应在更早阶段报明确错误，例如 `non-contiguous local tile subview is not supported`。

## 影响范围

这个问题影响 local tile 上的非连续 `pto.subview`，尤其是：

```text
parent row stride != child compact row stride
```

例如：

```text
parent: 4x16
subview: [0, 2] sizes [2, 8]
```

该 subview 的逻辑 shape 是 `2x8`，但 inherited row stride 是 `16`，不是 compact `8`。
