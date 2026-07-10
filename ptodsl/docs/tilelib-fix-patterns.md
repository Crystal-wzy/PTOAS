# PTODSL TileLib Fix Patterns

This note is an implementation playbook for the `mani/ptodsl` migration work.
It is not a user manual. The goal is to record the common failure shapes we
have already seen, how to recognize them quickly, and what kinds of fixes have
actually worked.

## Why keep this

Many ST failures look similar at first:

- `ExpandTileOp requires at least one template candidate`
- `NoMatchingTemplate`
- `custom constraints are not satisfied`
- build passes, but compare fails

In practice, these come from a small number of recurring root causes. Writing
them down helps avoid re-debugging the same pattern from scratch.

## Fast classification

When a testcase fails, classify it first:

| Failure shape | Usual meaning | First place to look |
|---|---|---|
| `dtype signature ... is not supported` | template exists, but the registered dtype matrix is too narrow | tile template `dtypes=` list and any dtype-specific body logic |
| `custom constraints are not satisfied` | template exists, but legality is too narrow for ST operand layout / memory-space / valid-shape form | template constraint helpers |
| `requires at least one template candidate` / `no candidate survives` | candidate was never inserted, was dropped by a pass, or selection key changed shape | daemon metadata path, `InsertTemplateAttributes`, `PTOViewToMemref`, `ExpandTileOp` |
| build succeeds, compare fails | semantics differ from TileLangDSL or ST golden | template body logic, store offsets, padding, reduction loop shape, precision path |
| smoke passes, non-smoke fails | usually a missing dtype/version/path not exercised by smoke, or a large-shape / high-precision semantic gap | non-smoke-only ST cases and legacy template body |
| PTODSL tracing error | template mixes Python compile-time control with runtime SSA values | PTODSL-authored control flow, scalar coercion, loop structure |

## Common problem patterns and fixes

### 1. Missing dtype/signature coverage

**Symptom**

- `NoMatchingTemplate`
- `dtype signature ('f16', 'f32', 'f16') is not supported`

**What it usually means**

The template is fundamentally right, but PTODSL registered fewer legal
signatures than TileLangDSL/ST actually uses.

**What fixed it**

- extend the template `dtypes=` list
- if needed, add a small dtype-specific path inside the body

**Recent example**

- `tlrelu`
  - smoke passed because it only used `f32`
  - non-smoke used `src=f16, slope=f32, dst=f16`
  - fix: add `("f16", "f32", "f16")` and coerce the runtime slope to `f16`

### 2. Constraint too narrow for the real ST operand form

**Symptom**

- `custom constraints are not satisfied`
- template exists, but ST still cannot select it

**What it usually means**

The template is present, but the accepted memory space, layout, valid shape, or
operand form is narrower than the real ST emission.

**What fixed it**

- compare the ST `.pto` operand form with the PTODSL constraint helper
- relax the legality only to the real used shape/layout combinations
- do not immediately rewrite the body; first make sure selection is correct

**Recent examples**

- `tcmps`
  - initial constraints were too narrow
  - ST used packed predicate destination shapes that did not satisfy the old
    assumptions
- likely next candidates for this pattern:
  - `tsel`
  - `textract`
  - `txors`

### 3. Context attributes reached selection, but not rendering

**Symptom**

- candidate selection succeeds
- expanded template body behaves as if attr-driven mode stayed at the default
- wrong compare mode, round mode, precision mode, etc.

**What it usually means**

The daemon saw `context_attrs` during metadata selection, but the actual
render/specialization step dropped them.

**What fixed it**

- forward `context_attrs` into the template `specialize(...)` call during
  render, not only during metadata selection

**Recent example**

- `tcmp` / `tcmps`
  - `cmp_mode` was effectively rendered as default behavior
  - fix: preserve `context_attrs` all the way into template specialization

### 4. Op attrs lost during `PTOViewToMemref`

**Symptom**

- metadata insertion seems fine upstream
- later `ExpandTileOp` sees no candidates
- op recreation in a transform silently drops attrs

**What it usually means**

One of the view-to-memref rewrites recreated the TileOp but did not preserve
attrs like `candidates` or other mode fields.

**What fixed it**

- replace manual op recreation with the local helper that clones attrs
- audit all rewritten operands of that op at the same time

**Recent example**

- `TCmpSOp` in `PTOViewToMemref.cpp`
  - manual recreation dropped attrs
  - fix: use the cloned-attrs replacement helper
- `TMatmul*` / `TGemv*` variants in `PTOViewToMemref.cpp`
  - plain `tmatmul` preserved attrs, but `.acc`, `.bias`, `.mx`, and GEMV
    variants were recreated without `candidates`
  - fix: use the cloned-attrs replacement helper for the variant rewrites too

### 5. Callable form mismatch with TileLang/ST

**Symptom**

- template exists
- selection fails or body assumptions do not match actual operand list
- often happens on ops with optional tmp/scalar/extra-vector operands

**What it usually means**

PTODSL encoded a simplified form, but ST emits a different operand order or a
different auxiliary operand shape/dtype.

**What fixed it**

- inspect the ST `.pto` op directly
- compare against the legacy TileLangDSL template parameter order
- fix the PTODSL signature and only then adjust body logic

**Recent examples**

- `tcmps`
- `tmrgsort`
- `tsort32`

### 6. Runtime scalar vs compile-time literal confusion

**Symptom**

- PTODSL tracing error when trying to cast or branch on a scalar kernel
  argument
- constructors like `pto.f16(...)` work for Python literals but fail for runtime
  scalar values

**What it usually means**

The template is treating a runtime scalar SSA value like a compile-time Python
literal.

**What fixed it**

- use PTODSL scalar coercion utilities for runtime values
- do not use constant constructors for runtime scalar adaptation

**Recent example**

- `tlrelu`
  - `pto.f16(slope)` was wrong when `slope` was a runtime kernel argument
  - fix: use runtime scalar coercion to `f16`

### 7. Python control flow mixed with PTODSL runtime values

**Symptom**

- tracing misuse errors like:
  - runtime value used as Python loop bound
  - runtime value used in native `if`

**What it usually means**

The template relies on Python control flow for something that became a device
side value during tracing.

**What fixed it**

- keep branching on compile-time quantities only
- use PTODSL-authored control flow for runtime-dependent decisions
- simplify the loop structure so the split between compile-time and runtime is
  explicit

**Recent example**

- `tsort32`
  - large non-smoke path hit runtime/control-flow trouble in earlier attempts

### 8. Build problem turns into semantic problem after template coverage lands

**Symptom**

- older snapshot showed `NoMatchingTemplate`
- fresh rerun now builds and runs
- compare still fails on a narrow case family

**What it usually means**

This is progress. The issue moved from coverage to behavior.

**What to do**

- do not keep documenting it as a build blocker
- move it to the semantic-parity list
- isolate the exact failing ST case family

**Recent examples**

- `tdivs`
  - no longer a template-selection failure
  - now fails a high-precision subnormal scalar-src compare case
- `tsort32`
  - no longer a no-template failure
  - now fails large unaligned non-smoke semantics

### 9. PTODSL helper cache ignores view metadata

**Symptom**

- build succeeds
- a row-reduction / row-arg case computes the right per-row values, but compare
  sees zeros or stale data in later rows
- failures may appear only in full non-smoke runs, while an isolated one-function
  compile looks correct

**What it usually means**

PTODSL templates can bake `ViewSpec` shape/stride metadata into helper bodies.
If `ExpandTileOp` reuses a helper specialization using only the tile type and
view dtype/layout, an earlier compact destination view can poison a later
strided destination view with the same tile type.

TileLangDSL is less exposed to this particular cache bug because its helper
keeps a `partition_tensor_view` argument and asks for the tensor-view stride in
IR. PTODSL often receives/renders a memref-shaped helper and materializes the
view stride as constants.

**Debug trail from `trowargmin`**

1. First isolate the failing case with `run_st.py -c`:

   ```bash
   PTOAS_TILE_LIB_BACKEND=ptodsl \
   python3 test/tilelang_st/script/run_st.py \
     -r sim -v a5 \
     -p build-llvm21/tools/ptoas/ptoas \
     -t trowargmin \
     -c uint32_float_3x8_3x3480_3x3473 \
     &> mani_log/manual_20260709/trowargmin_one.log
   ```

2. Inspect `golden.bin` and `output.bin`, not only the compare message.

   The failing case had:

   ```text
   golden = [1088, 661, 176]
   output first 24 = [1088, 661, 176, 0, 0, ...]
   output as 3x8 first column = [1088, 0, 0]
   ```

   That proved `trowargmin` found the right row answers, but writeback packed
   them into GM offsets `0, 1, 2` instead of first-column offsets `0, 8, 16`.

3. Compare TileLangDSL and PTODSL emitted VPTO for the same case.

   Useful commands:

   ```bash
   build-llvm21/tools/ptoas/ptoas \
     --pto-arch=a5 --pto-backend=vpto --emit-vpto \
     --tile-lib-backend=tilelang --enable-insert-sync \
     test/tilelang_st/npu/a5/src/st/testcase/trowargmin/trowargmin.pto \
     -o /tmp/trowargmin_tilelang.vpto

   build-llvm21/tools/ptoas/ptoas \
     --pto-arch=a5 --pto-backend=vpto --emit-vpto \
     --tile-lib-backend=ptodsl --enable-insert-sync \
     test/tilelang_st/npu/a5/src/st/testcase/trowargmin/trowargmin.pto \
     -o /tmp/trowargmin_ptodsl.vpto
   ```

   The row-arg body was not the real problem: both backends stored the row result
   in UB at `row * 8`. The difference was the final store helper:

   - TileLangDSL used a 32-byte GM row stride.
   - PTODSL reused a helper with a 4-byte GM row stride.

4. Dump after `ExpandTileOp` to find where the divergence entered.

   ```bash
   build-llvm21/tools/ptoas/ptoas \
     --pto-arch=a5 --pto-backend=vpto --emit-vpto \
     --tile-lib-backend=ptodsl --enable-insert-sync \
     --mlir-print-ir-after=pto-expand-tile-op \
     --mlir-print-ir-tree-dir=/tmp/trowargmin_after_expand_ptodsl \
     test/tilelang_st/npu/a5/src/st/testcase/trowargmin/trowargmin.pto \
     -o /tmp/trowargmin_ptodsl.vpto
   ```

   Avoid relying on `--mlir-print-ir-module-scope` in this build; it may require
   a threading-disable flag that is not exposed by this `ptoas`.

   The after-expand IR already showed the bad PTODSL helper:

   ```mlir
   pto.mte_ub_gm ... nburst(%c3_i64, %c32_i64, %c4_i64)
   ```

   The matching TileLangDSL helper kept `!pto.partition_tensor_view` and used
   `pto.get_tensor_view_stride` for the destination row stride.

5. Look earlier in the full generated testcase for another op that shares the
   same destination tile type.

   In `trowargmin.pto`, an earlier case used `tile_ui32_3_8_v3_v1` with a
   physically compact destination view `3x1`. Later,
   `uint32_float_3x8_3x3480_3x3473` used the same tile type but a physical
   destination row width of `8`. Because the PTODSL helper cache key/name did
   not include view shape/strides, the later case reused the compact helper.

**What fixed it**

- make PTODSL `ExpandTileOp` helper specialization include view shape, strides,
  memory space, and layout in:
  - `OperandTypeInfo::operator==`
  - `SpecKeyInfo::getHashValue`
  - `buildUniqueFunctionBaseName`
- add a focused lit regression with two `tstore`s that have the same tile type
  but different destination strides:
  - `test/lit/vpto/expand_tile_op_ptodsl_view_stride_cache.pto`

**Validated by**

- `ninja -C build-llvm21 tools/ptoas/ptoas`
- `llvm-lit -v build-llvm21/test/lit --filter expand_tile_op_ptodsl_view_stride_cache`
- compiler-only full `trowargmin.pto` emit now shows the strided case using
  32-byte GM and UB row strides
- full non-smoke `trowargmin` ST passed after the fix

**Likely affected ops**

- `trowargmin`
- `trowargmax`
- any PTODSL template that bakes `ViewSpec` strides into a helper and can see
  multiple physical destination views with the same tile type

## Reusable debugging workflow

### A. For `NoMatchingTemplate` / `custom constraints are not satisfied`

1. Read the failing `.pto` call form in the ST testcase.
2. Read the TileLangDSL template for the same op.
3. Read the PTODSL template decorator:
   - `dtypes=`
   - constraint helpers
   - parameter order
4. Decide whether the miss is:
   - missing dtype
   - too-narrow constraint
   - wrong callable form
5. Add a small focused catalog/daemon test if the fix is local.

### B. For `no candidate survives`

1. Confirm whether PTODSL has a template for that op at all.
2. Check whether daemon metadata returns candidates.
3. If metadata is correct, inspect attr preservation across:
   - `InsertTemplateAttributes`
   - `PTOViewToMemref`
   - `ExpandTileOp`
4. Diff a working nearby op if possible.

### C. For build-pass but compare-fail

1. Identify the smallest failing ST case.
2. Inspect the binary outputs, not just the compare log:
   - read `golden.bin`
   - read `output.bin`
   - reshape output to the physical destination shape used by ST
   - check whether values are wrong, missing, or stored at the wrong stride
3. Diff PTODSL body vs TileLangDSL body.
3. Look for:
   - pack/store offsets
   - pad constants
   - reduction loop depth
   - tail handling
   - precision widening/casting
4. If the body looks right, compare the final store/load movement IR:
   - emitted VPTO for `tilelang`
   - emitted VPTO for `ptodsl`
   - `--mlir-print-ir-after=pto-expand-tile-op` when the final VPTO only shows
     the symptom, not the entry point
5. Check for helper cache collisions:
   - same generated helper name
   - same tile type
   - different view shape or stride metadata
6. Re-run only the affected testcase after each change.

## Patterns that are usually low effort

These tend to be good grouped fixes:

- dtype matrix expansion
- constraint widening to real ST forms
- runtime scalar coercion fixes
- attr forwarding / attr preservation fixes

## Patterns that are usually not low effort

These tend to sprawl:

- true backend candidate-propagation failures in cube paths
- high-precision semantic parity
- wrong-output bugs in reduction/arg-reduction writeback
- large-shape tail handling
- random / sort semantic parity

## Keep this note current

When an ST failure is fixed, add a short line here if it introduced a new
repeatable debugging lesson. The goal is not to list every tileop, but to keep
the small set of recurring fix patterns visible.
