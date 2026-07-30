# Soft `SYNCALL` ABI Alignment with PTO-ISA f24

- Status: Draft
- Tracking issue: [PTOAS #1061](https://github.com/hw-native-sys/PTOAS/issues/1061)
- PTO-ISA change:
  [`f24f7b736b689cc107b9eb2d362be6a7718fcc99`](https://github.com/hw-native-sys/pto-isa/commit/f24f7b736b689cc107b9eb2d362be6a7718fcc99)

## 1. Summary

PTO-ISA f24 replaced the core-type-specific soft `SYNCALL` overloads with one
public ABI:

```cpp
SYNCALL<SyncAllMode::Soft, CoreType>(gmWorkspace, usedCores);
```

The UB and L1 scratch arguments no longer exist. PTOAS still models those
arguments in `pto.syncall`, verifies them, materializes local Tiles for them,
and emits three- or four-argument C++ calls.

This design changes the canonical PTO IR contract to:

- hard mode: no operands;
- soft mode: required `gm_workspace` and optional `used_cores`;
- all soft core types: exactly two EmitC arguments, materializing
  `int32_t{0}` when `used_cores` is omitted.

The recommended rollout is a direct switch to the new ABI. A one-release
compatibility option is described in [Section 8](#8-compatibility-and-rollout)
if an atomic producer upgrade is not possible.

## 2. Problem statement

The current PTOAS soft forms depend on `core_type`:

| Core type | Current PTO operands | Current C++ arguments |
| --- | --- | --- |
| `aiv_only` | GM + UB + optional used cores | GM + UB + used cores |
| `aic_only` | GM + L1 + optional used cores | GM + L1 + used cores |
| `mix` | GM + UB + L1 + optional used cores | GM + UB + L1 + used cores |

For example, current PTOAS emits:

```cpp
Tile<TileType::Vec, int32_t, 1, 64> ubWorkspace;
TASSIGN(ubWorkspace, ubAddress);
Tile<TileType::Mat, int32_t, 1, 64> l1Workspace;
TASSIGN(l1Workspace, l1Address);
SYNCALL<SyncAllMode::Soft, SyncCoreType::Mix>(
    gmWorkspace, ubWorkspace, l1Workspace, usedCores);
```

That call does not match PTO-ISA f24 or its descendants. It also reserves and
initializes local storage that the new implementation never consumes.

The new PTO-ISA implementation uses a shared GM atomic counter. It requires one
exclusive 64-byte cache line, corresponding to at least 16 `int32_t` elements,
and the workspace must be zero-initialized before its first use.

## 3. Goals and non-goals

### 3.1 Goals

- Align textual PTO IR, ODS-generated APIs, verification, and EmitC with
  PTO-ISA f24.
- Keep hard `SYNCALL` behavior unchanged.
- Accept `memref`, `tensor_view`, and `partition_tensor_view` GM workspaces.
- Reject statically known GM workspaces smaller than 16 `int32_t` elements.
- Generate no soft-`SYNCALL` UB/L1 Tile, `TASSIGN`, or local scratch.
- Update Python examples, the PTO IR manual, and focused regression tests in
  the same change.
- Compile generated A5 C++ against PTO-ISA f24 or an explicit descendant SHA.

### 3.2 Non-goals

- Changing the PTO-ISA synchronization algorithm.
- Proving cache-line exclusivity, alignment, or zero-initialization in the
  PTO IR verifier. These remain caller/runtime obligations.
- Changing the meaning of `used_cores = 0`.
- Changing hard-mode FFTS behavior.
- Adding a new CLI option or changing the PTOAS pass pipeline.

## 4. Canonical PTO IR contract

### 4.1 ODS operands

`SyncAllOp` keeps `AttrSizedOperandSegments` because both operands are
mode-dependent:

```tablegen
let arguments = (ins
  Optional<PTODpsType>:$gm_workspace,
  Optional<I32>:$used_cores,
  PTO_SyncAllModeAttr:$mode,
  PTO_SyncCoreTypeAttr:$core_type
);
```

`gm_workspace` is optional in ODS so the same op can represent zero-operand
hard mode. The verifier makes it required for soft mode.

The segment layouts become:

| Form | `operandSegmentSizes` |
| --- | --- |
| hard | `[0, 0]` |
| soft, inferred core count | `[1, 0]` |
| soft, explicit core count | `[1, 1]` |

`core_type` no longer changes the operand layout.

### 4.2 Soft mode with explicit `used_cores`

```mlir
module {
  func.func @soft_aiv(
      %gm: memref<16xi32, #pto.address_space<gm>>,
      %used: i32) {
    pto.syncall(
      %gm, %used : memref<16xi32, #pto.address_space<gm>>, i32
    ) mode = #pto.sync_all_mode<soft>,
      core_type = #pto.sync_core_type<aiv_only>
    return
  }
}
```

Expected C++:

```cpp
SYNCALL<SyncAllMode::Soft, SyncCoreType::AIVOnly>(
    gmWorkspace, usedCores);
```

The AIC-only and Mix forms use the same operands:

```mlir
pto.syncall(%gm, %used : memref<16xi32, #pto.address_space<gm>>, i32)
  mode = #pto.sync_all_mode<soft>,
  core_type = #pto.sync_core_type<aic_only>

pto.syncall(%gm, %used : memref<16xi32, #pto.address_space<gm>>, i32)
  mode = #pto.sync_all_mode<soft>,
  core_type = #pto.sync_core_type<mix>
```

### 4.3 Soft mode with inferred core count

```mlir
pto.syncall(%gm : !pto.partition_tensor_view<16xi32>)
  mode = #pto.sync_all_mode<soft>,
  core_type = #pto.sync_core_type<mix>
```

PTOAS must still emit two C++ arguments:

```cpp
SYNCALL<SyncAllMode::Soft, SyncCoreType::Mix>(
    gmWorkspace, int32_t{0});
```

The generated call does not rely on the C++ default argument. Keeping an
explicit, typed zero makes the PTO IR omission semantics visible in EmitC and
stable under overload changes.

### 4.4 Hard mode

Hard mode remains a zero-operand op:

```mlir
pto.syncall()
  mode = #pto.sync_all_mode<hard>,
  core_type = #pto.sync_core_type<mix>
```

Expected C++:

```cpp
SYNCALL<SyncCoreType::Mix>();
```

## 5. Parsing, printing, and verification

### 5.1 Custom parser

The parser applies mode-dependent arity rules:

1. Parse the operand and type lists, then parse `mode` and `core_type`.
2. For hard mode, require zero operands and add segment sizes `[0, 0]`.
3. For soft mode, require one or two operands.
4. Resolve operand 0 as `gm_workspace`.
5. Resolve operand 1, when present, as `used_cores`.
6. Add segment sizes `[1, 0]` or `[1, 1]`.

The parser does not branch on `core_type`.

Representative diagnostics:

```text
custom op 'pto.syncall' expects hard syncall to have no operands
```

```text
custom op 'pto.syncall' expects soft syncall to have gm_workspace
and optional used_cores
```

### 5.2 Printer

The printer emits operands in the fixed order:

1. `gm_workspace`, if present;
2. `used_cores`, if present.

It elides `operandSegmentSizes`, `mode`, and `core_type` from the optional
attribute dictionary as it does today. Parse/print round trips therefore
produce only the canonical new form.

### 5.3 Verifier

Hard mode succeeds only when both optional operands are absent.

Soft mode verifies:

- `gm_workspace` is present;
- the workspace is a ranked GM `memref`, `!pto.tensor_view`, or
  `!pto.partition_tensor_view`;
- the element type is `i32`;
- the rank is at least one;
- every static dimension is positive;
- when every dimension is static, the element-count product is at least 16;
- `used_cores`, when present, is `i32`.

The capacity calculation must be overflow-safe. Because only the threshold
`16` matters, the implementation may stop multiplying as soon as the running
capacity reaches 16.

Examples:

```mlir
// Accepted: exactly 16 elements.
memref<16xi32, #pto.address_space<gm>>

// Accepted: multidimensional static capacity of 16.
memref<4x4xi32, #pto.address_space<gm>>

// Rejected: statically known capacity is too small.
memref<15xi32, #pto.address_space<gm>>

// Accepted by static verification: the runtime must provide at least 16.
memref<?xi32, #pto.address_space<gm>>
```

Suggested diagnostic:

```text
'pto.syncall' op expects soft syncall gm_workspace to contain at least
16 i32 elements (64 bytes), but static capacity is 15
```

The verifier cannot prove that the buffer starts on a 64-byte boundary, owns
the entire cache line without aliasing, or is zero-initialized. Those
requirements must be documented in `docs/PTO_IR_manual.md` and enforced by the
producer/runtime.

## 6. EmitC lowering

The hard-mode branch and `coreTypeTok()` remain unchanged.

For soft mode:

1. Convert `gm_workspace` to the existing GlobalTensor representation.
2. Use the converted `used_cores` value when present.
3. Otherwise create an EmitC value whose rendered literal is
   `int32_t{0}`.
4. Emit one `SYNCALL<SyncAllMode::Soft, CoreType>` call with exactly
   `{gmWorkspace, usedCores}`.

Conceptually:

```cpp
FailureOr<Value> gmWorkspace = buildGmWorkspace();
Value usedCores = adaptor.getUsedCores()
    ? peelUnrealized(adaptor.getUsedCores())
    : makeTypedInt32Zero();

rewriter.create<emitc::CallOpaqueOp>(
    op.getLoc(), TypeRange{}, callee,
    ArrayAttr{}, ArrayAttr{},
    ValueRange{*gmWorkspace, usedCores});
```

The following old-ABI code is removed:

- the `core_type` switch that selects UB and/or L1;
- `buildSyncAllWorkspaceTileValue()`;
- soft-`SYNCALL` Tile construction;
- soft-`SYNCALL` `TASSIGN` generation.

## 7. Python API, documentation, and examples

The ODS change regenerates a Python API without `ub_workspace` and
`l1_workspace`.

Canonical Python construction:

```python
pto.syncall(
    _mode("soft"),
    _core_type("aiv_only"),
    gm_workspace=gm_workspace,
    used_cores=used_cores,
)
```

Inferred core count:

```python
pto.syncall(
    _mode("soft"),
    _core_type("mix"),
    gm_workspace=gm_workspace,
)
```

The `test/samples/SyncAll` sample must stop creating an `AllocTileOp` solely
for `SYNCALL`.

`docs/PTO_IR_manual.md` must describe:

- the two canonical soft operands;
- the 16-element static capacity rule;
- the exclusive 64-byte cache-line requirement;
- zero-initialization before first use;
- the meaning of an omitted or zero `used_cores`;
- unchanged hard-mode behavior.

## 8. Compatibility and rollout

### 8.1 Recommended: direct ABI switch

The recommended implementation removes UB/L1 from ODS, parser, verifier,
bindings, and EmitC in one change.

Advantages:

- one canonical IR form;
- no dead local scratch operands;
- generated bindings cannot accidentally create the removed ABI;
- no hidden canonicalization dependency;
- acceptance criteria are straightforward to test.

Cost:

- cached old `.pto` files and Python producers must be updated together with
  the PTOAS wheel.

This is appropriate when PTOAS and PyPTO can move to the new wheel/IR contract
atomically.

### 8.2 Optional: one-release compatibility window

If old producers must keep working for one transition release:

1. Temporarily keep the legacy optional UB/L1 fields in ODS.
2. Extend the custom parser to distinguish new `(gm, i32)` from legacy
   `(gm, local_workspace [, i32])` forms by operand type and count.
3. Emit a deprecation warning when a legacy local operand is present.
4. Run an early canonicalization before memory planning that rebuilds the op
   without UB/L1 operands.
5. Always emit the new two-argument C++ call.
6. Remove the compatibility fields and canonicalization in the next release.

The early canonicalization is mandatory. Merely ignoring UB/L1 in
`PTOSyncAllToEmitC` can leave their allocations live long enough to generate
local scratch and `TASSIGN`, violating the new contract.

This option is intentionally not the default because it expands the public
surface and test matrix for a short-lived migration path.

## 9. Test plan

### 9.1 Lit coverage

| Case | Expected result |
| --- | --- |
| soft AIV-only with used cores | exact two-argument AIV call |
| soft AIC-only with used cores | exact two-argument AIC call |
| soft Mix with used cores | exact two-argument Mix call |
| soft mode without used cores | second argument is `int32_t{0}` |
| hard AIV/AIC/Mix | unchanged zero-argument calls |
| static GM capacity 16 | accepted |
| static GM capacity `4x4` | accepted |
| static GM capacity 15 | rejected with actionable error |
| dynamic GM capacity | accepted by static verifier |
| non-i32 GM workspace | rejected |
| non-GM memref workspace | rejected |
| non-i32 used cores | rejected |

The positive EmitC test must include:

```text
CHECK-NOT: Tile<TileType::Vec
CHECK-NOT: Tile<TileType::Mat
CHECK-NOT: TASSIGN
```

It must check the complete call shape rather than only the callee prefix:

```text
CHECK: SYNCALL<SyncAllMode::Soft, SyncCoreType::AIVOnly>(
CHECK: SYNCALL<SyncAllMode::Soft, SyncCoreType::AICOnly>(
CHECK: SYNCALL<SyncAllMode::Soft, SyncCoreType::Mix>(
```

Captured FileCheck variables should be used to ensure each call has only the GM
and used-core operands.

### 9.2 Python sample coverage

Run the existing `syncall_binding.py` sample flow and verify:

- Python construction succeeds with the regenerated binding;
- printed PTO IR uses only GM and optional used cores;
- PTOAS emits the new C++ call;
- no local `AllocTileOp` is created for synchronization.

### 9.3 PTO-ISA compile validation

Use PTO-ISA commit f24 or an explicit descendant:

```bash
git -C "${PTO_ISA_ROOT}" checkout \
  f24f7b736b689cc107b9eb2d362be6a7718fcc99

ptoas --pto-arch=a5 test/lit/pto/syncall_emitc.pto \
  -o build/issue_1061_syncall.cpp
```

Compile the generated kernel C++ with the repository's existing A5/bisheng
flags and `${PTO_ISA_ROOT}/include`. The validation must cover AIV-only,
AIC-only, and Mix template instantiations.

The PR should record the exact PTO-ISA SHA and compile command. A targeted
compile-only validation is preferred over changing the global PTO-ISA pin for
unrelated NPU tests.

## 10. Planned file changes

| Layer | File | Planned change |
| --- | --- | --- |
| ODS | `include/PTO/IR/PTOOps.td` | remove UB/L1 operands and update description |
| Parser/printer/verifier | `lib/PTO/IR/PTO.cpp` | implement new arity and capacity rules |
| EmitC | `lib/PTO/Transforms/PTOToEmitC.cpp` | emit only GM + used cores; remove helper |
| Positive tests | `test/lit/pto/syncall_emitc.pto` | cover all core modes and exact call shape |
| Negative tests | `test/lit/pto/syncall_invalid_*.pto` | cover missing/invalid/small GM |
| Python sample | `test/samples/SyncAll/syncall_binding.py` | use new generated binding |
| PTO sample | `test/samples/SyncAll/syncall_binding.pto` | use canonical new IR |
| Documentation | `docs/PTO_IR_manual.md` | document new ABI and runtime obligations |

No CLI or pass-pipeline change is expected.

## 11. Acceptance criteria

The implementation is complete when:

- AIV-only, AIC-only, and Mix soft calls contain exactly two C++ arguments.
- Soft lowering produces no UB/L1 Tile, `TASSIGN`, or local scratch.
- Omitted `used_cores` produces an explicit typed zero.
- Statically known GM capacity below 16 `int32_t` elements is rejected.
- Hard `SYNCALL` output is unchanged.
- Python construction and printed PTO IR use the new contract.
- Focused lit tests pass.
- Generated A5 C++ compiles against the recorded PTO-ISA f24-or-newer SHA.

## 12. Open review decisions

Before implementation, reviewers should confirm:

1. Whether PTOAS and PyPTO can switch atomically. If yes, use the direct ABI
   switch; otherwise approve the one-release compatibility window.
2. Whether the emitted omitted-core literal must be textually
   `int32_t{0}` or only an EmitC value typed as `int32_t` with value zero.
3. Which PTO-ISA descendant SHA should be used for the PR's compile validation
   if f24 itself is not the current integration pin.
