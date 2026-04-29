# 16. Cube Matrix Multiply (MAT)

> **Category:** Cube unit — GM/L1 staging, L0A/L0B loads, L0C accumulate, and matrix-side side-buffer moves
> **Pipelines:** MTE2 (GM→L1 / cbuf), Cube (L0A/L0B→L0C), MTE3/FIX (L0C→{GM,L1,UB}, L1→{BT,FB})

This group documents **buffer-pointer** PTO ops used to express a minimal **cube matmul data path** on A5: data is moved from GM into L1-aligned buffers (`cbuf`), loaded into L0A/L0B, multiplied into L0C (`cc`), then written back or redistributed to GM/L1/UB and related matrix-side buffers. These ops are distinct from the vector `!pto.vreg<…>` surface in groups 3–13.

Typical usage keeps the body inside `pto.vecscope { … }` (or another enclosing region required by the PTO verifier) so cube-side effects remain ordered with respect to other PTO work.

---

## `pto.copy_gm_to_cbuf`

- **syntax:** `pto.copy_gm_to_cbuf %src, %dst, %n_burst, %len_burst, %src_stride, %dst_stride : !pto.ptr<…, gm>, !pto.ptr<…, ub>, i64, i64, i64, i64`
- **semantics:** GM→L1 (`cbuf`) aligned copy. `%src` is GM; `%dst` is UB-backed L1 (`cbuf`) staging.

Operands `%n_burst`, `%len_burst`, `%src_stride`, and `%dst_stride` configure the transfer shape; they are lowered to packed `i64` configuration tokens for the target `llvm.hivm` MOV family intrinsic.

---

## `pto.load_cbuf_to_ca`

- **syntax:** `pto.load_cbuf_to_ca %src, %dst, %m, %k : !pto.ptr<…, ub>, !pto.ptr<…, ub>, i64, i64`
- **semantics:** L1 (`cbuf`) → L0A load. `%src` is `cbuf`; `%dst` is UB-backed L0A staging.

---

## `pto.load_cbuf_to_cb`

- **syntax:** `pto.load_cbuf_to_cb %src, %dst, %k, %n : !pto.ptr<…, ub>, !pto.ptr<…, ub>, i64, i64`
- **semantics:** L1 (`cbuf`) → L0B load. `%src` is `cbuf`; `%dst` is UB-backed L0B staging.

---

## `pto.mad`

- **syntax:** `pto.mad %lhs, %rhs, %dst, %m, %n, %k : !pto.ptr<…, ub>, !pto.ptr<…, ub>, !pto.ptr<…, ub>, i64, i64, i64`
- **semantics:** Cube **multiply** on L0A (`%lhs`) and L0B (`%rhs`) into L0C (`%dst`). All three pointers must be UB-backed **buffer** pointers (`!pto.ptr<…, ub>`) classified as left / right / accumulator roles in lowering.

Supported element-type combinations follow the HIVM intrinsic selection in the compiler (for example `f16` × `f16` → `f32` accumulation, and MX-dtyped paths where applicable).

---

## `pto.copy_matrix_cc_to_gm`

- **syntax:** `pto.copy_matrix_cc_to_gm %src, %dst, %m, %n : !pto.ptr<…, ub>, !pto.ptr<…, gm>, i64, i64`
- **semantics:** L0C (`cc`) → GM matrix writeback. `%src` is UB-backed `cc`; `%dst` is GM.

---

## `pto.copy_gm_to_cbuf_multi_nd2nz` / `pto.copy_gm_to_cbuf_multi_dn2nz`

- **syntax:** `pto.copy_gm_to_cbuf_multi_* %src, %dst, ... : !pto.ptr<…, gm>, !pto.ptr<…, ub>, ...`
- **semantics:** GM→L1 (`cbuf`) multi-fractal staging paths for cube data layout conversion variants (`ND2NZ` / `DN2NZ`), lowered to `llvm.hivm.MOV.OUT.TO.L1.MULTI.*` families.

---

## `pto.copy_matrix_cc_to_cbuf` / `pto.copy_matrix_cc_to_ub`

- **syntax:** `pto.copy_matrix_cc_to_* %src, %dst, %config0, %config1 : !pto.ptr<…, ub>, !pto.ptr<…, ub>, i64, i64`
- **semantics:** L0C (`cc`) redistribution to L1 (`cbuf`) or UB destinations. These are post-matmul movement ops typically used before follow-up fusion or output formatting.

---

## `pto.copy_cbuf_to_bt` / `pto.copy_cbuf_to_fbuf`

- **syntax:** `pto.copy_cbuf_to_bt ...` and `pto.copy_cbuf_to_fbuf ...`
- **semantics:** L1 (`cbuf`) to bias/scaling-side buffers for matrix post-processing setup. Lowered to `llvm.hivm.MOV.L1.TO.BT.f16` and `llvm.hivm.MOV.L1.TO.FB.V2`.

---

## Verified A5 Op Set (Current Batch)

The following PTO ops have been verified in the current A5 VPTO→LLVM/HIVM and Bisheng cube-flow validation batch:

- `pto.copy_cbuf_to_bt`
- `pto.copy_cbuf_to_fbuf`
- `pto.copy_gm_to_cbuf_multi_dn2nz`
- `pto.copy_gm_to_cbuf_multi_nd2nz`
- `pto.copy_matrix_cc_to_cbuf`
- `pto.copy_matrix_cc_to_ub`
- `pto.load_cbuf_to_ca_mx`
- `pto.load_cbuf_to_ca_s4`
- `pto.load_cbuf_to_cb_mx`
- `pto.load_cbuf_to_cb_s4`
- `pto.set_atomic_s32`
- `pto.set_atomic_s8`
- `pto.set_channel_para`
- `pto.set_fpc`
- `pto.set_loop1_stride_outtol1`
- `pto.set_loop2_stride_outtol1`
- `pto.set_loop3_para`
- `pto.set_loop_size_outtol1`
- `pto.set_mte2_nz_para`
- `pto.set_pad_val_outtol1`
- `pto.set_quant_pre`

---

## Current PTOAS Coverage

- VPTO → LLVM (`--vpto-emit-hivm-llvm`) lowers the ops in this group to target-specific `llvm.hivm.*` intrinsics with explicit address spaces for GM / cbuf / L0A / L0B / L0C and matrix-side side buffers.
- FileCheck coverage lives under `test/basic/vpto_mad_*.pto` and `test/basic/vpto_cube_dma_matmul_*.pto`.
- TileLang ST builds a cube-linked host testcase via `pto_tilelang_cube_st(tmatmul)` in `test/tilelang_st/npu/a5/src/st/testcase/tmatmul/`.
