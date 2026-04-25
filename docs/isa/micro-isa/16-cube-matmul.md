# 16. Cube Matrix Multiply (MAT)

> **Category:** Cube unit — GM/L1 staging, L0A/L0B loads, L0C accumulate, and matrix-side side-buffer moves

---

## Core Data Path Ops

### `pto.copy_gm_to_cbuf`

- **syntax:**
```mlir
pto.copy_gm_to_cbuf %src, %dst, %n_burst, %len_burst, %src_stride, %dst_stride
  : !pto.ptr<T, gm>, !pto.ptr<T, mat>, i64, i64, i64, i64
```
- **semantics:** Copy matrix tile data from GM to L1 (`cbuf`).

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | GM source pointer (`!pto.ptr<T, gm>`) |
| `%dst` | ptr | L1 destination pointer (`!pto.ptr<T, mat>`) |
| `%n_burst` | i64 | Burst count |
| `%len_burst` | i64 | Bytes per burst row |
| `%src_stride` | i64 | Source row stride |
| `%dst_stride` | i64 | Destination row stride |

**Constraints:**

- Source/destination element types must match.
- Address spaces must be `gm -> mat`.

**Example:**

```mlir
pto.copy_gm_to_cbuf %a_gm, %l1_a, %c1_i64, %c16_i64, %c0_i64, %c0_i64
  : !pto.ptr<f16, gm>, !pto.ptr<f16, mat>, i64, i64, i64, i64
```

---

### `pto.load_cbuf_to_ca`

- **syntax:**
```mlir
pto.load_cbuf_to_ca %src, %dst, %m_start, %k_start, %m_step, %k_step, %src_stride, %dst_stride
  : !pto.ptr<T, mat>, !pto.ptr<T, left>, i64, i64, i64, i64, i64, i64
```
- **semantics:** Load L1 (`cbuf`) tile to L0A.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer (`!pto.ptr<T, mat>`) |
| `%dst` | ptr | L0A destination pointer (`!pto.ptr<T, left>`) |
| `%m_start` | i64 | M start index |
| `%k_start` | i64 | K start index |
| `%m_step` | i64 | M step |
| `%k_step` | i64 | K step |
| `%src_stride` | i64 | Source stride control |
| `%dst_stride` | i64 | Destination stride control |

**Constraints:**

- Address spaces must be `mat -> left`.
- Optional `transpose` attribute controls transpose mode.

**Example:**

```mlir
pto.load_cbuf_to_ca %l1_a, %l0a, %c0_i64, %c0_i64, %c1_i64, %c1_i64, %c1_i64, %c1_i64
  : !pto.ptr<f16, mat>, !pto.ptr<f16, left>, i64, i64, i64, i64, i64, i64
```

---

### `pto.load_cbuf_to_cb`

- **syntax:**
```mlir
pto.load_cbuf_to_cb %src, %dst, %m_start, %k_start, %m_step, %k_step, %src_stride, %dst_stride
  : !pto.ptr<T, mat>, !pto.ptr<T, right>, i64, i64, i64, i64, i64, i64
```
- **semantics:** Load L1 (`cbuf`) tile to L0B.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer (`!pto.ptr<T, mat>`) |
| `%dst` | ptr | L0B destination pointer (`!pto.ptr<T, right>`) |
| `%m_start` | i64 | M start index |
| `%k_start` | i64 | K start index |
| `%m_step` | i64 | M step |
| `%k_step` | i64 | K step |
| `%src_stride` | i64 | Source stride control |
| `%dst_stride` | i64 | Destination stride control |

**Constraints:**

- Address spaces must be `mat -> right`.
- Optional `transpose` attribute controls transpose mode.

**Example:**

```mlir
pto.load_cbuf_to_cb %l1_b, %l0b, %c0_i64, %c0_i64, %c1_i64, %c1_i64, %c1_i64, %c1_i64
  : !pto.ptr<f16, mat>, !pto.ptr<f16, right>, i64, i64, i64, i64, i64, i64
```

---

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

**Parameter Table:** same as `pto.mad`.

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

**Parameter Table:** same as `pto.mad`.

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

**Parameter Table:** same as `pto.mad`.

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

**Parameter Table:** same as `pto.mad_bias`.

**Constraints:** same as `pto.mad_mx` plus bias address-space requirement.

**Example:**

```mlir
pto.mad_mx_bias %l0a, %l0b, %l0c, %bt, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, left>, !pto.ptr<f8E4M3FN, right>, !pto.ptr<f32, acc>, !pto.ptr<f32, bias>, i64, i64, i64
```

---

### `pto.copy_matrix_cc_to_gm`

- **syntax:**
```mlir
pto.copy_matrix_cc_to_gm %src, %dst, %xm, %xt
  : !pto.ptr<T, acc>, !pto.ptr<T, gm>, i64, i64
```
- **semantics:** Write L0C (`acc`) tile back to GM.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L0C source pointer (`acc`) |
| `%dst` | ptr | GM destination pointer |
| `%xm` | i64 | Matrix writeback config field |
| `%xt` | i64 | Matrix writeback config field |

**Constraints:**

- Address spaces must be `acc -> gm`.

**Example:**

```mlir
pto.copy_matrix_cc_to_gm %l0c, %c_gm, %xm, %xt
  : !pto.ptr<f32, acc>, !pto.ptr<f32, gm>, i64, i64
```

---

### `pto.copy_matrix_cc_to_cbuf`

- **syntax:**
```mlir
pto.copy_matrix_cc_to_cbuf %src, %dst, %config0, %config1
  : !pto.ptr<T, acc>, !pto.ptr<T, mat>, i64, i64
```
- **semantics:** Move L0C (`acc`) tile to L1 (`cbuf`).

**Parameter Table:** `%src`, `%dst`, `%config0`, `%config1`.

**Constraints:**

- Address spaces must be `acc -> mat`.

**Example:**

```mlir
pto.copy_matrix_cc_to_cbuf %l0c, %l1_out, %cfg0, %cfg1
  : !pto.ptr<f32, acc>, !pto.ptr<f32, mat>, i64, i64
```

---

### `pto.copy_matrix_cc_to_ub`

- **syntax:**
```mlir
pto.copy_matrix_cc_to_ub %src, %dst, %config0, %config1
  : !pto.ptr<T, acc>, !pto.ptr<T, ub>, i64, i64
```
- **semantics:** Move L0C (`acc`) tile to UB.

**Parameter Table:** `%src`, `%dst`, `%config0`, `%config1`.

**Constraints:**

- Address spaces must be `acc -> ub`.

**Example:**

```mlir
pto.copy_matrix_cc_to_ub %l0c, %ub_out, %cfg0, %cfg1
  : !pto.ptr<f32, acc>, !pto.ptr<f32, ub>, i64, i64
```

---

### `pto.copy_cbuf_to_bt`

- **syntax:**
```mlir
pto.copy_cbuf_to_bt %src, %dst, %len_burst, %n_burst, %src_gap, %dst_gap
  : !pto.ptr<T, mat>, !pto.ptr<U, bias>, i64, i64, i64, i64
```
- **semantics:** Move L1 (`cbuf`) data to BT buffer.

**Parameter Table:** `%src`, `%dst`, `%len_burst`, `%n_burst`, `%src_gap`, `%dst_gap`.

**Constraints:**

- Destination must be bias/BT address space.

**Example:**

```mlir
pto.copy_cbuf_to_bt %l1_bias, %bt, %c16_i64, %c1_i64, %c0_i64, %c0_i64
  : !pto.ptr<f16, mat>, !pto.ptr<f32, bias>, i64, i64, i64, i64
```

---

### `pto.copy_cbuf_to_fbuf`

- **syntax:**
```mlir
pto.copy_cbuf_to_fbuf %src, %dst, %n_burst, %len_burst, %src_gap, %dst_gap
  : !pto.ptr<T, mat>, !pto.ptr<T, ub>, i64, i64, i64, i64
```
- **semantics:** Move L1 (`cbuf`) data to FB-related destination path.

**Parameter Table:** `%src`, `%dst`, `%n_burst`, `%len_burst`, `%src_gap`, `%dst_gap`.

**Constraints:**

- Source must be `mat` address space.

**Example:**

```mlir
pto.copy_cbuf_to_fbuf %l1_src, %ub_dst, %c1_i64, %c16_i64, %c0_i64, %c0_i64
  : !pto.ptr<f16, mat>, !pto.ptr<f16, ub>, i64, i64, i64, i64
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

### `pto.copy_gm_to_cbuf_multi_nd2nz`

- **syntax:**
```mlir
pto.copy_gm_to_cbuf_multi_nd2nz %src, %dst, %sid, %loop1_src_stride, %l2_cache_ctrl, %n_value, %d_value, %loop4_src_stride, %smallc0_en
  : !pto.ptr<T, gm>, !pto.ptr<T, mat>, i64, i64, i64, i64, i64, i64, i1
```
- **semantics:** Multi-fractal `ND2NZ` staging from GM to L1 (`cbuf`).

**Parameter Table:** `%src`, `%dst`, `%sid`, `%loop1_src_stride`, `%l2_cache_ctrl`, `%n_value`, `%d_value`, `%loop4_src_stride`, `%smallc0_en`.

**Constraints:**

- `smallc0_en` is valid only when `d_value <= 4`.

**Example:**

```mlir
pto.copy_gm_to_cbuf_multi_nd2nz %src, %dst, %sid, %l1s, %l2, %n, %d, %l4s, %small
  : !pto.ptr<f16, gm>, !pto.ptr<f16, mat>, i64, i64, i64, i64, i64, i64, i1
```

---

### `pto.copy_gm_to_cbuf_multi_dn2nz`

- **syntax:**
```mlir
pto.copy_gm_to_cbuf_multi_dn2nz %src, %dst, %sid, %loop1_src_stride, %l2_cache_ctrl, %n_value, %d_value, %loop4_src_stride, %smallc0_en
  : !pto.ptr<T, gm>, !pto.ptr<T, mat>, i64, i64, i64, i64, i64, i64, i1
```
- **semantics:** Multi-fractal `DN2NZ` staging from GM to L1 (`cbuf`).

**Parameter Table:** same as `pto.copy_gm_to_cbuf_multi_nd2nz`.

**Constraints:** same as `pto.copy_gm_to_cbuf_multi_nd2nz`.

**Example:**

```mlir
pto.copy_gm_to_cbuf_multi_dn2nz %src, %dst, %sid, %l1s, %l2, %n, %d, %l4s, %small
  : !pto.ptr<f16, gm>, !pto.ptr<f16, mat>, i64, i64, i64, i64, i64, i64, i1
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

**Parameter Table:** `%src`, `%dst`, `%len_burst`, `nburst(...)`, optional `loop(...)`.

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

**Parameter Table:** `%src`, `%dst`, `%len_burst`, `nburst(...)`, optional `loop(...)`.

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

**Parameter Table:** source/destination pointers, shape fields, source layout fields, destination group fields, control fields.

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

### `pto.left_load`

- **syntax:**
```mlir
pto.left_load %src, %dst, %m, %k
  : !pto.ptr<T, mat>, !pto.ptr<T, left>, i64, i64
```
- **semantics:** Structured L1-to-L0A wrapper.

**Parameter Table:** `%src`, `%dst`, `%m`, `%k`.

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

**Parameter Table:** `%src`, `%dst`, `%k`, `%n`.

**Constraints:**

- Lowers to `pto.load_cbuf_to_cb`.

**Example:**

```mlir
pto.right_load %l1_b, %l0b, %c16_i64, %c16_i64
  : !pto.ptr<f16, mat>, !pto.ptr<f16, right>, i64, i64
```

---

### `pto.acc_store`

- **syntax:**
```mlir
pto.acc_store %src, %dst, %m, %n, %src_stride, %dst_stride, %unit_flag_ctrl, %quant_pre, %relu_pre_mode, nz2nd|nz2dn(%loop0_src_stride)?|nz2nz(%split)? [loop3(%count, %src_stride3, %dst_stride3)]?
  : !pto.ptr<T, acc>, !pto.ptr<T, mat>, ...
```
- **semantics:** Structured L0C (`acc`) to L1 (`cbuf`) wrapper.

**Parameter Table:** `%src`, `%dst`, shape/stride fields, pre/post fields, layout mode (`nz2nd` / `nz2dn` / `nz2nz`), optional `loop3`.

**Constraints:**

- `nz2nz` mode does not accept `loop3(...)`.

**Example:**

```mlir
pto.acc_store %l0c, %l1_out, %c16_i64, %c16_i64, %c16_i64, %c16_i64, %c0_i64, %c0_i64, %c0_i64, nz2nd
  : !pto.ptr<f32, acc>, !pto.ptr<f32, mat>, i64, i64, i64, i64, i64, i64, i64, nz2nd
```

---

### `pto.acc_store_gm`

- **syntax:**
```mlir
pto.acc_store_gm %src, %dst, %m, %n, %src_stride, %dst_stride, %unit_flag_ctrl, %quant_pre, %relu_pre_mode, %sid, %l2_cache_ctrl, nz2nd|nz2dn(%loop0_src_stride)?|nz2nz(%split)? [loop3(%count, %src_stride3, %dst_stride3)]?
  : !pto.ptr<T, acc>, !pto.ptr<T, gm>, ...
```
- **semantics:** Structured L0C (`acc`) to GM wrapper.

**Parameter Table:** same fields as `pto.acc_store` plus `%sid` and `%l2_cache_ctrl`.

**Constraints:**

- GM output path controls (`sid`, `l2_cache_ctrl`) must be provided.

**Example:**

```mlir
pto.acc_store_gm %l0c, %c_gm, %c16_i64, %c16_i64, %c16_i64, %c16_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, nz2nd
  : !pto.ptr<f32, acc>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64, i64, i64, i64, nz2nd
```

---

### `pto.acc_store_ub`

- **syntax:**
```mlir
pto.acc_store_ub %src, %dst, %m, %n, %src_stride, %dst_stride, %unit_flag_ctrl, %quant_pre, %relu_pre_mode, %dual_dst_mode, %sub_blockid, nz2nd|nz2dn(%loop0_src_stride)?|nz2nz(%channel_split_en)? [loop3(%count, %src_stride3, %dst_stride3)]?
  : !pto.ptr<T, acc>, !pto.ptr<T, ub>, ...
```
- **semantics:** Structured L0C (`acc`) to UB wrapper.

**Parameter Table:** same fields as `pto.acc_store` plus `%dual_dst_mode`, `%sub_blockid`.

**Constraints:**

- `nz2nz` mode does not accept `loop3(...)`.

**Example:**

```mlir
pto.acc_store_ub %l0c, %ub_out, %c16_i64, %c16_i64, %c16_i64, %c16_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, nz2nd
  : !pto.ptr<f32, acc>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i64, i64, i64, i64, nz2nd
```

---

## Current PTOAS Coverage

- VPTO->LLVM (`--vpto-emit-hivm-llvm`) lowers this chapter's ops to
  `llvm.hivm.*` intrinsics with cube-related address spaces.
- Basic coverage is under `test/basic/vpto_mad_*.pto` and
  `test/basic/vpto_cube_dma_matmul_*.pto`.
- Micro-op coverage for `mad` / `mad_bias` / `mad_mx` families is under
  `test/vpto/cases/micro-op/cube-matmul/`.
