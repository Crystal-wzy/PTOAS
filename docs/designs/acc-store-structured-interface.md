# `acc_store` 统一接口设计

## 1. 目标

统一原先两套桥接接口：

- `ACC -> GM`
- `ACC -> CBUF`

收敛为单一 `pto.acc_store`：

```mlir
pto.acc_store %src, %dst, %m, %n, %src_stride, %dst_stride,
              %unit_flag_ctrl, %quant_pre, %relu_pre_mode,
              nz2nd|nz2dn(%loop0_src_stride)?|nz2nz(%split)?
              loop3(%count, %src_stride, %dst_stride)?
```

其中：

- `%src` 必须是 `!pto.ptr<..., acc>`
- `%dst` 允许是 `!pto.ptr<..., gm>` 或 `!pto.ptr<..., mat>`
- 根据 `%dst` 的地址空间，expand 时选择不同的终端 intrinsic

## 2. 统一依据

根据 `pto-isa` 和 `disa-cube.json`，下面这组 FIX 控制语义是共通的：

- `m / n`
- `src_stride / dst_stride`
- `unit_flag_ctrl`
- `quant_pre`
- `relu_pre_mode`
- `nz2nd / nz2dn / nz2nz(split)`
- `loop3(count, src_stride, dst_stride)`
- `nz2dn` 下的 `loop0_src_stride`

也就是说，`ACC -> GM` 和 `ACC -> CBUF` 的差别不在于这组控制字段本身，  
而在于最后一条搬移 intrinsic 的目标地址空间不同。

## 3. 统一后的 lowering

统一 `pto.acc_store` 的 bridge expand 逻辑：

- 总是先 lower 为
  - `pto.set_loop3_para`
  - `pto.set_channel_para`
- 然后根据 `%dst` 地址空间分流

### 3.1 `dst : !pto.ptr<..., gm>`

lower 为：

```mlir
pto.copy_matrix_cc_to_gm %src, %dst, %xm, %xt
```

### 3.2 `dst : !pto.ptr<..., mat>`

lower 为：

```mlir
pto.copy_matrix_cc_to_cbuf %src, %dst, %xm, %xt
```

这里 `copy_matrix_cc_to_cbuf` 的两个 config 也直接复用同一组 FIX 配置值。

## 4. 统一后的 verifier

`pto.acc_store` verifier 约束为：

- source 必须是 ACC
- destination 必须是 GM 或 MAT
- `nz2nd`
  - 不允许 `split`
  - 不允许 `loop0_src_stride`
- `nz2dn`
  - 不允许 `split`
- `nz2nz`
  - 不允许 `loop0_src_stride`
  - 不允许 `loop3`

## 5. 结果

统一后仓内只保留一套对外接口：

- `pto.acc_store`

不再保留旧的拆分接口与命名。

这样做的好处是：

- 对外只保留一套 FIX accumulator-store 语义
- `gm` / `mat` 只是 destination address space 的分流
- parser / printer / verifier / builder / docs / tests 都不再需要维护两套名称
