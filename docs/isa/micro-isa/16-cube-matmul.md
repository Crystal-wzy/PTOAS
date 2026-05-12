# 16. Cube Matrix Multiply (MAT)

> **Category:** Cube unit ops — staged load/store and matrix multiply
> **Raw-op reference:** See `16-cube-matmul-raw.md` for low-level bridge/raw ops

---

## Wrapper-Layer Compute Ops

### `pto.mad`

- **syntax:**
```mlir
pto.mad %lhs, %rhs, %dst, %m, %n, %k
  : !pto.ptr<A, left>, !pto.ptr<B, right>, !pto.ptr<C, acc>, i64, i64, i64
```
- **semantics:** Zero-init cube matmul, `dst = lhs * rhs`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` | ptr | L0A input (`left`) |
| `%rhs` | ptr | L0B input (`right`) |
| `%dst` | ptr | L0C accumulator (`acc`) |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%k` | i64 | K size |
| `unit_flag_ctrl` | i32 attr | Accumulator control flag |
| `disable_gemv` | bool attr | GEMV-disable control bit |

**Constraints:**

- Address spaces must be `left`, `right`, `acc`.
- `unit_flag_ctrl` currently uses `0/2/3` values in existing tests.

**Example:**

```mlir
pto.mad %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c16_i64
  : !pto.ptr<f16, left>, !pto.ptr<f16, right>, !pto.ptr<f32, acc>, i64, i64, i64
```

---

### `pto.mad_acc`

- **syntax:**
```mlir
pto.mad_acc %lhs, %rhs, %dst, %m, %n, %k
  : !pto.ptr<A, left>, !pto.ptr<B, right>, !pto.ptr<C, acc>, i64, i64, i64
```
- **semantics:** Accumulating cube matmul, `dst += lhs * rhs`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` | ptr | L0A input (`left`) |
| `%rhs` | ptr | L0B input (`right`) |
| `%dst` | ptr | L0C accumulator (`acc`) |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%k` | i64 | K size |
| `unit_flag_ctrl` | i32 attr | Accumulator control flag |
| `disable_gemv` | bool attr | GEMV-disable control bit |

**Constraints:**

- Same address space/type family requirements as `pto.mad`.

**Example:**

```mlir
pto.mad_acc %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c16_i64 {unit_flag_ctrl = 2 : i32}
  : !pto.ptr<f16, left>, !pto.ptr<f16, right>, !pto.ptr<f32, acc>, i64, i64, i64
```

---

### `pto.mad_bias`

- **syntax:**
```mlir
pto.mad_bias %lhs, %rhs, %dst, %bias, %m, %n, %k
  : !pto.ptr<A, left>, !pto.ptr<B, right>, !pto.ptr<C, acc>, !pto.ptr<C, bias>, i64, i64, i64
```
- **semantics:** Bias-init cube matmul, `dst = lhs * rhs + bias`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` / `%rhs` / `%dst` / `%m` / `%n` / `%k` | - | Same meaning as `pto.mad` |
| `%bias` | ptr | Bias-table pointer (`!pto.ptr<C, bias>`) |
| `unit_flag_ctrl` | i32 attr | Accumulator control flag |
| `disable_gemv` | bool attr | GEMV-disable control bit |

**Constraints:**

- `%bias` must be in `bias` address space.

**Example:**

```mlir
pto.mad_bias %l0a, %l0b, %l0c, %bt, %c16_i64, %c16_i64, %c16_i64
  : !pto.ptr<f16, left>, !pto.ptr<f16, right>, !pto.ptr<f32, acc>, !pto.ptr<f32, bias>, i64, i64, i64
```

---

### `pto.mad_mx`

- **syntax:**
```mlir
pto.mad_mx %lhs, %rhs, %dst, %m, %n, %k
  : !pto.ptr<A, left>, !pto.ptr<B, right>, !pto.ptr<C, acc>, i64, i64, i64
```
- **semantics:** Zero-init MX cube matmul.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` | ptr | MX L0A input (`left`) |
| `%rhs` | ptr | MX L0B input (`right`) |
| `%dst` | ptr | L0C accumulator (`acc`) |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%k` | i64 | K size, typically matching MX tile granularity |
| `unit_flag_ctrl` | i32 attr | Accumulator control flag |
| `disable_gemv` | bool attr | GEMV-disable control bit |

**Constraints:**

- MX-capable dtype combinations must be respected by backend lowering.

**Example:**

```mlir
pto.mad_mx %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, left>, !pto.ptr<f8E4M3FN, right>, !pto.ptr<f32, acc>, i64, i64, i64
```

---

### `pto.mad_mx_acc`

- **syntax:**
```mlir
pto.mad_mx_acc %lhs, %rhs, %dst, %m, %n, %k
  : !pto.ptr<A, left>, !pto.ptr<B, right>, !pto.ptr<C, acc>, i64, i64, i64
```
- **semantics:** Accumulating MX cube matmul.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` | ptr | MX L0A input (`left`) |
| `%rhs` | ptr | MX L0B input (`right`) |
| `%dst` | ptr | L0C accumulator (`acc`) |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%k` | i64 | K size, typically matching MX tile granularity |
| `unit_flag_ctrl` | i32 attr | Accumulator control flag |
| `disable_gemv` | bool attr | GEMV-disable control bit |

**Constraints:** same as `pto.mad_mx`.

**Example:**

```mlir
pto.mad_mx_acc %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, left>, !pto.ptr<f8E4M3FN, right>, !pto.ptr<f32, acc>, i64, i64, i64
```

---

### `pto.mad_mx_bias`

- **syntax:**
```mlir
pto.mad_mx_bias %lhs, %rhs, %dst, %bias, %m, %n, %k
  : !pto.ptr<A, left>, !pto.ptr<B, right>, !pto.ptr<C, acc>, !pto.ptr<C, bias>, i64, i64, i64
```
- **semantics:** Bias-init MX cube matmul.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` | ptr | MX L0A input (`left`) |
| `%rhs` | ptr | MX L0B input (`right`) |
| `%dst` | ptr | L0C accumulator (`acc`) |
| `%bias` | ptr | Bias-table pointer (`bias`) |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%k` | i64 | K size, typically matching MX tile granularity |
| `unit_flag_ctrl` | i32 attr | Accumulator control flag |
| `disable_gemv` | bool attr | GEMV-disable control bit |

**Constraints:** same as `pto.mad_mx` plus bias address-space requirement.

**Example:**

```mlir
pto.mad_mx_bias %l0a, %l0b, %l0c, %bt, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, left>, !pto.ptr<f8E4M3FN, right>, !pto.ptr<f32, acc>, !pto.ptr<f32, bias>, i64, i64, i64
```

---

## Cube Bridge Wrapper Ops

### `pto.cube_load`

- **syntax:**
```mlir
pto.cube_load %src, %dst, %len_burst
  nburst(%count, %src_stride, %dst_stride)
  [loop(%count_i, %src_stride_i, %dst_stride_i)]*
  : !pto.ptr<T, gm>, !pto.ptr<T, mat>, i64, i64, i64, i64
```
- **semantics:** Structured GM-to-L1 (`cbuf`) wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | GM source pointer |
| `%dst` | ptr | L1 destination pointer (`mat`) |
| `%len_burst` | i64 | Burst length |
| `nburst(%count, %src_stride, %dst_stride)` | i64 triple | Inner DMA burst count and strides |
| `loop(%count_i, %src_stride_i, %dst_stride_i)` | i64 triple | Optional outer loop triplet, repeatable |

**Constraints:**

- Wrapper lowers to loop/stride setup plus `pto.copy_gm_to_cbuf`.

**Example:**

```mlir
pto.cube_load %a_gm, %l1_a, %c16_i64
  nburst(%c1_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, mat>, i64, i64, i64, i64
```

---

### `pto.cube_store`

- **syntax:**
```mlir
pto.cube_store %src, %dst, %len_burst
  nburst(%count, %src_stride, %dst_stride)
  [loop(%count_i, %src_stride_i, %dst_stride_i)]*
  : !pto.ptr<T, mat>, !pto.ptr<T, ub>, i64, i64, i64, i64
```
- **semantics:** Structured L1 (`cbuf`) to UB wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer (`mat`) |
| `%dst` | ptr | UB destination pointer |
| `%len_burst` | i64 | Burst length |
| `nburst(%count, %src_stride, %dst_stride)` | i64 triple | Inner DMA burst count and strides |
| `loop(%count_i, %src_stride_i, %dst_stride_i)` | i64 triple | Optional outer loop triplet, repeatable |

**Constraints:**

- Wrapper lowers to `pto.copy_cbuf_to_ubuf` and optional outer loops.

**Example:**

```mlir
pto.cube_store %l1_src, %ub_dst, %c16_i64
  nburst(%c1_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f16, mat>, !pto.ptr<f16, ub>, i64, i64, i64, i64
```

---

### `pto.cube_load_frac`

- **syntax:**
```mlir
pto.cube_load_frac %src, %dst, nd2nz|dn2nz, shape(%n_value, %d_value), src_layout(%src_inner_stride[, %src_outer_stride]), dst_group(%group_count, %dst_loop2_stride, %dst_loop3_stride, %dst_loop4_stride), ctrl(%l2_cache_ctrl, %smallc0_en)
  : !pto.ptr<T, gm>, !pto.ptr<T, mat>, ...
```
- **semantics:** Structured fractal-load wrapper for `nd2nz` / `dn2nz`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | GM source pointer |
| `%dst` | ptr | L1 destination pointer (`mat`) |
| `nd2nz` / `dn2nz` | enum token | Fractal load mode |
| `shape(%n_value, %d_value)` | i64 pair | Logical N and D shape |
| `src_layout(%src_inner_stride[, %src_outer_stride])` | i64 / i64 pair | Source layout stride fields |
| `dst_group(%group_count, %dst_loop2_stride, %dst_loop3_stride, %dst_loop4_stride)` | i64 tuple | Destination group count and nested destination strides |
| `ctrl(%l2_cache_ctrl, %smallc0_en)` | i64, i1 | L2 cache control and small-C0 enable |

**Constraints:**

- Lowers to `set_mte2_nz_para` plus `copy_gm_to_cbuf_multi_*`.

**Example:**

```mlir
pto.cube_load_frac %src, %dst, nd2nz,
  shape(%n, %d),
  src_layout(%sis),
  dst_group(%g, %l2s, %l3s, %l4s),
  ctrl(%l2, %small)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, mat>, nd2nz, shape i64, i64, src_layout(i64), dst_group i64, i64, i64, i64, ctrl i64, i1
```

---

### `pto.bias_load`

- **syntax:**
```mlir
pto.bias_load %src, %dst, %len_burst
  nburst(%count, %src_gap, %dst_gap)
  : !pto.ptr<T, mat>, !pto.ptr<U, bias>, i64, i64, i64, i64
```
- **semantics:** Structured helper for L1 (`cbuf`) to bias-table load.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer (`mat`) |
| `%dst` | ptr | Bias destination pointer (`bias`) |
| `%len_burst` | i64 | Burst length |
| `%count` | i64 | Burst count |
| `%src_gap` | i64 | Source gap |
| `%dst_gap` | i64 | Destination gap |

**Constraints:**

- Supported type pairs: `f32->f32`, `i32->i32`, `f16->f32`, `bf16->f32`.

**Example:**

```mlir
pto.bias_load %l1_bias, %bt, %c16_i64 nburst(%c1_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f16, mat>, !pto.ptr<f32, bias>, i64, i64, i64, i64
```

---

### `pto.fp_load`

- **syntax:**
```mlir
pto.fp_load %src, %dst, %len_burst
  nburst(%count, %src_gap, %dst_gap)
  : !pto.ptr<T, mat>, !pto.ptr<U, scaling>, i64, i64, i64, i64
```
- **semantics:** Structured helper for L1 (`cbuf`) to Fixpipe Buffer (`scaling`) load.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer (`mat`) |
| `%dst` | ptr | Fixpipe-buffer destination pointer (`scaling`) |
| `%len_burst` | i64 | Burst length |
| `%count` | i64 | Burst count |
| `%src_gap` | i64 | Source gap |
| `%dst_gap` | i64 | Destination gap |

**Constraints:**

- Lowers to `pto.copy_cbuf_to_fbuf`.
- `%src` must be in `mat`, `%dst` must be in `scaling`.

**Example:**

```mlir
pto.fp_load %l1_fp, %fb_fp, %c2_i64 nburst(%c1_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f32, mat>, !pto.ptr<ui64, scaling>, i64, i64, i64, i64
```

---

### `pto.left_load`

- **syntax:**
```mlir
pto.left_load %src, %dst, %m, %k
  : !pto.ptr<T, mat>, !pto.ptr<T, left>, i64, i64
```
- **semantics:** Structured L1-to-L0A wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer (`mat`) |
| `%dst` | ptr | L0A destination pointer (`left`) |
| `%m` | i64 | M tile size |
| `%k` | i64 | K tile size |

**Constraints:**

- Lowers to `pto.load_cbuf_to_ca`.

**Example:**

```mlir
pto.left_load %l1_a, %l0a, %c16_i64, %c16_i64
  : !pto.ptr<f16, mat>, !pto.ptr<f16, left>, i64, i64
```

---

### `pto.right_load`

- **syntax:**
```mlir
pto.right_load %src, %dst, %k, %n
  : !pto.ptr<T, mat>, !pto.ptr<T, right>, i64, i64
```
- **semantics:** Structured L1-to-L0B wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer (`mat`) |
| `%dst` | ptr | L0B destination pointer (`right`) |
| `%k` | i64 | K tile size |
| `%n` | i64 | N tile size |

**Constraints:**

- Lowers to `pto.load_cbuf_to_cb`.

**Example:**

```mlir
pto.right_load %l1_b, %l0b, %c16_i64, %c16_i64
  : !pto.ptr<f16, mat>, !pto.ptr<f16, right>, i64, i64
```

---

### `pto.left_load_mx`

- **syntax:**
```mlir
pto.left_load_mx %src, %dst, %m, %k
  : !pto.ptr<T, mat>, !pto.ptr<T, left>, i64, i64
```
- **semantics:** MX-mode L1-to-L0A wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | MX-formatted L1 source pointer (`mat`) |
| `%dst` | ptr | MX L0A destination pointer (`left`) |
| `%m` | i64 | M tile size |
| `%k` | i64 | K tile size |

**Constraints:**

- Lowers to `pto.load_cbuf_to_ca_mx`.

**Example:**

```mlir
pto.left_load_mx %l1_a, %l0a, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, mat>, !pto.ptr<f8E4M3FN, left>, i64, i64
```

---

### `pto.right_load_mx`

- **syntax:**
```mlir
pto.right_load_mx %src, %dst, %k, %n
  : !pto.ptr<T, mat>, !pto.ptr<T, right>, i64, i64
```
- **semantics:** MX-mode L1-to-L0B wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | MX-formatted L1 source pointer (`mat`) |
| `%dst` | ptr | MX L0B destination pointer (`right`) |
| `%k` | i64 | K tile size |
| `%n` | i64 | N tile size |

**Constraints:**

- Lowers to `pto.load_cbuf_to_cb_mx`.

**Example:**

```mlir
pto.right_load_mx %l1_b, %l0b, %c64_i64, %c16_i64
  : !pto.ptr<f8E4M3FN, mat>, !pto.ptr<f8E4M3FN, right>, i64, i64
```

---

### `pto.acc_store`

- **syntax:**
```mlir
pto.acc_store %src, %dst, %m, %n, %src_stride, %dst_stride, %unit_flag_ctrl, %quant_pre, %relu_pre_mode, [fpc(%fpc_addr),]? nz2nd|nz2dn[, loop0_src_stride(%loop0_src_stride)]?|nz2nz[, split(%split)]? [loop3(%count, %src_stride3, %dst_stride3)]?
  : !pto.ptr<T, acc>, !pto.ptr<T, mat>, ..., [fpc(!pto.ptr<ui64, scaling>),]? ...
```
- **semantics:** Structured L0C (`acc`) to L1 (`cbuf`) wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L0C source pointer (`acc`) |
| `%dst` | ptr | L1 destination pointer (`mat`) |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%src_stride` | i64 | Source stride field used in fixpipe config |
| `%dst_stride` | i64 | Destination stride field used in fixpipe config |
| `%unit_flag_ctrl` | i64 | Unit-flag control bits |
| `%quant_pre` | i64 | Quantization pre-mode bits |
| `%relu_pre_mode` | i64 | ReLU pre-mode bits |
| `fpc(%fpc_addr)` | optional `!pto.ptr<ui64, scaling>` | Optional Fixpipe parameter block pointer. Lowering casts the pointer to an integer address before emitting `set_fpc`. |
| `nz2nd` / `nz2dn` / `nz2nz` | enum token | Layout/store mode selector |
| `loop0_src_stride(%loop0_src_stride)` | optional i64 | Extra mode parameter used by `nz2dn` |
| `split(%split)` | optional i64 | Extra mode parameter used by `nz2nz` |
| `loop3(%count, %src_stride3, %dst_stride3)` | optional i64 triple | Optional loop3 hardware control tuple |

**Constraints:**

- `nz2nz` mode does not accept `loop3(...)`.

**Example:**

```mlir
pto.acc_store %l0c, %l1_out, %c16_i64, %c16_i64, %c16_i64, %c16_i64, %c0_i64, %c33_i64, %c0_i64, fpc(%fb_fp), nz2nd
  : !pto.ptr<f32, acc>, !pto.ptr<f32, mat>, i64, i64, i64, i64, i64, i64, i64, fpc(!pto.ptr<ui64, scaling>), nz2nd
```

---

### `pto.acc_store_gm`

- **syntax:**
```mlir
pto.acc_store_gm %src, %dst, %m, %n, %src_stride, %dst_stride, %unit_flag_ctrl, %quant_pre, %relu_pre_mode, %sid, %l2_cache_ctrl, [fpc(%fpc_addr),]? nz2nd|nz2dn[, loop0_src_stride(%loop0_src_stride)]?|nz2nz[, split(%split)]? [loop3(%count, %src_stride3, %dst_stride3)]?
  : !pto.ptr<T, acc>, !pto.ptr<T, gm>, ..., [fpc(!pto.ptr<ui64, scaling>),]? ...
```
- **semantics:** Structured L0C (`acc`) to GM wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L0C source pointer (`acc`) |
| `%dst` | ptr | GM destination pointer |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%src_stride` | i64 | Source stride field used in fixpipe config |
| `%dst_stride` | i64 | Destination stride field used in fixpipe config |
| `%unit_flag_ctrl` | i64 | Unit-flag control bits |
| `%quant_pre` | i64 | Quantization pre-mode bits |
| `%relu_pre_mode` | i64 | ReLU pre-mode bits |
| `%sid` | i64 | GM store SID control |
| `%l2_cache_ctrl` | i64 | GM-side L2 cache control |
| `fpc(%fpc_addr)` | optional `!pto.ptr<ui64, scaling>` | Optional Fixpipe parameter block pointer. Lowering casts the pointer to an integer address before emitting `set_fpc`. |
| `nz2nd` / `nz2dn` / `nz2nz` | enum token | Layout/store mode selector |
| `loop0_src_stride(%loop0_src_stride)` | optional i64 | Extra mode parameter used by `nz2dn` |
| `split(%split)` | optional i64 | Extra mode parameter used by `nz2nz` |
| `loop3(%count, %src_stride3, %dst_stride3)` | optional i64 triple | Optional loop3 hardware control tuple |

**Constraints:**

- GM output path controls (`sid`, `l2_cache_ctrl`) must be provided.

**Example:**

```mlir
pto.acc_store_gm %l0c, %c_gm, %c16_i64, %c16_i64, %c16_i64, %c16_i64, %c0_i64, %c33_i64, %c0_i64, %c0_i64, %c0_i64, fpc(%fb_fp), nz2nd
  : !pto.ptr<f32, acc>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64, i64, i64, i64, fpc(!pto.ptr<ui64, scaling>), nz2nd
```

---

### `pto.acc_store_ub`

- **syntax:**
```mlir
pto.acc_store_ub %src, %dst, %m, %n, %src_stride, %dst_stride, %unit_flag_ctrl, %quant_pre, %relu_pre_mode, [fpc(%fpc_addr),]? %dual_dst_mode, %sub_blockid, nz2nd|nz2dn[, loop0_src_stride(%loop0_src_stride)]?|nz2nz[, split(%split)]? [loop3(%count, %src_stride3, %dst_stride3)]?
  : !pto.ptr<T, acc>, !pto.ptr<T, ub>, ..., [fpc(!pto.ptr<ui64, scaling>),]? ...
```
- **semantics:** Structured L0C (`acc`) to UB wrapper.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L0C source pointer (`acc`) |
| `%dst` | ptr | UB destination pointer |
| `%m` | i64 | M size |
| `%n` | i64 | N size |
| `%src_stride` | i64 | Source stride field used in fixpipe config |
| `%dst_stride` | i64 | Destination stride field used in fixpipe config |
| `%unit_flag_ctrl` | i64 | Unit-flag control bits |
| `%quant_pre` | i64 | Quantization pre-mode bits |
| `%relu_pre_mode` | i64 | ReLU pre-mode bits |
| `fpc(%fpc_addr)` | optional `!pto.ptr<ui64, scaling>` | Optional Fixpipe parameter block pointer. Lowering casts the pointer to an integer address before emitting `set_fpc`. |
| `%dual_dst_mode` | i64 | Dual-destination mode control |
| `%sub_blockid` | i64 | Sub-block id control |
| `nz2nd` / `nz2dn` / `nz2nz` | enum token | Layout/store mode selector |
| `loop0_src_stride(%loop0_src_stride)` | optional i64 | Extra mode parameter used by `nz2dn` |
| `split(%split)` | optional i64 | Extra mode parameter used by `nz2nz` |
| `loop3(%count, %src_stride3, %dst_stride3)` | optional i64 triple | Optional loop3 hardware control tuple |

**Constraints:**

- `nz2nz` mode does not accept `loop3(...)`.

**Example:**

```mlir
pto.acc_store_ub %l0c, %ub_out, %c16_i64, %c16_i64, %c16_i64, %c16_i64, %c0_i64, %c33_i64, %c0_i64, fpc(%fb_fp), %c0_i64, %c0_i64, nz2nd
  : !pto.ptr<f32, acc>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i64, i64, fpc(!pto.ptr<ui64, scaling>), i64, i64, nz2nd
```

---

## Current PTOAS Coverage

- VPTO->LLVM (`--vpto-emit-hivm-llvm`) lowers this chapter's ops to
  `llvm.hivm.*` intrinsics with cube-related address spaces.
