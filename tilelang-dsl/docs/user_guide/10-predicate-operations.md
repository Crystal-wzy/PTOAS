### Predicate Operations

Operations for creating and manipulating typed masks.

**Recommended API**: For most use cases, prefer the unified `pto.make_mask()` function which automatically selects the appropriate mask granularity based on element type and supports both tail processing (remaining element count) and pattern-based mask generation. This eliminates the need to manually choose between `plt_b8`/`plt_b16`/`plt_b32` (tail processing) and `pset_b8`/`pset_b16`/`pset_b32` (pattern generation) operations.

**Pattern alias**: For brevity in examples, the documentation uses `PAT` as an alias for `pto.MaskPattern` (e.g., `PAT.ALL` instead of `pto.MaskPattern.ALL`). In practice, you can create this alias with `from pto import MaskPattern as PAT` or `PAT = pto.MaskPattern`.

**Predicate Part Enum**: `pto.ppack` and `pto.punpack` require the `PredicatePart` enum. Use `PredicatePart.LOWER` or `PredicatePart.HIGHER`; these lower to the VPTO canonical `PART` tokens `"LOWER"` and `"HIGHER"`.

**Predicate Dist Enum**: The `PredicateDist` enum provides type-safe distribution mode selection for predicate memory families. Load families (`plds`, `pld`, `pldi`) use `NORM`, `US`, and `DS`. Store families (`psts`, `pst`, `psti`) use `NORM` and `PK`.

**Pattern coverage**: The VPTO canonical predicate-generation families use `PAT_*` tokens such as `PAT_ALL`, `PAT_ALLF`, `PAT_H`, `PAT_Q`, `PAT_VL*`, `PAT_M3`, and `PAT_M4`. The Python DSL surface may expose only a subset through `pto.MaskPattern`; check the enum for currently available values.

#### `pto.pset_b8(pattern: pto.MaskPattern) -> pto.mask_b8`

**Description**: Creates an 8-bit granularity mask from a pattern.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern` | `pto.MaskPattern` | Mask pattern enum (for example `pto.MaskPattern.ALL`, `pto.MaskPattern.ALLF`, or `pto.MaskPattern.VL32`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b8` | 8-bit granularity mask |

**Constraints**:
- Used with `i8` vector operations

**Example**:
```python
mask8 = pto.pset_b8(PAT.ALL)
```

#### `pto.pset_b16(pattern: pto.MaskPattern) -> pto.mask_b16`

**Description**: Creates a 16-bit granularity mask from a pattern.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern` | `pto.MaskPattern` | Mask pattern enum (for example `pto.MaskPattern.ALL`, `pto.MaskPattern.ALLF`, or `pto.MaskPattern.VL32`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b16` | 16-bit granularity mask |

**Constraints**:
- Used with `f16`/`bf16`/`i16` vector operations

**Example**:
```python
mask16 = pto.pset_b16(PAT.ALL)
```

#### `pto.pset_b32(pattern: pto.MaskPattern) -> pto.mask_b32`

**Description**: Creates a 32-bit granularity mask from a pattern.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern` | `pto.MaskPattern` | Mask pattern enum (for example `pto.MaskPattern.ALL`, `pto.MaskPattern.ALLF`, or `pto.MaskPattern.VL32`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b32` | 32-bit granularity mask |

**Constraints**:
- Used with `f32`/`i32` vector operations

**Example**:
```python
mask32 = pto.pset_b32(PAT.ALL)
```

#### `pto.pge_b8(pattern: pto.MaskPattern) -> pto.mask_b8`

**Description**: Generate tail mask — first N lanes active based on pattern. Creates an 8-bit granularity mask where the first N lanes are active according to the specified pattern.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern` | `pto.MaskPattern` | Tail mask pattern enum lowered to a VPTO `PAT_*` token (for example `pto.MaskPattern.VL16` or `pto.MaskPattern.VL32`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b8` | 8-bit granularity tail mask |

**Constraints**:
- Used with `i8` vector operations
- Pattern must be a valid tail mask pattern (typically `PAT_VL*` variants)

**Example**:
```python
# Tail mask pattern lowered as `PAT_VL16`
tail_mask = pto.pge_b8(PAT.VL16)
```

#### `pto.pge_b16(pattern: pto.MaskPattern) -> pto.mask_b16`

**Description**: Generate tail mask — first N lanes active based on pattern. Creates a 16-bit granularity mask where the first N lanes are active according to the specified pattern.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern` | `pto.MaskPattern` | Tail mask pattern enum lowered to a VPTO `PAT_*` token (for example `pto.MaskPattern.VL16` or `pto.MaskPattern.VL32`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b16` | 16-bit granularity tail mask |

**Constraints**:
- Used with `f16`/`bf16`/`i16` vector operations
- Pattern must be a valid tail mask pattern (typically `PAT_VL*` variants)

**Example**:
```python
# Tail mask for first 16 lanes
tail_mask = pto.pge_b16(PAT.VL16)
```

#### `pto.pge_b32(pattern: pto.MaskPattern) -> pto.mask_b32`

**Description**: Generate tail mask — first N lanes active based on pattern. Creates a 32-bit granularity mask where the first N lanes are active according to the specified pattern.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern` | `pto.MaskPattern` | Tail mask pattern enum lowered to a VPTO `PAT_*` token (for example `pto.MaskPattern.VL16` or `pto.MaskPattern.VL32`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b32` | 32-bit granularity tail mask |

**Constraints**:
- Used with `f32`/`i32` vector operations
- Pattern must be a valid tail mask pattern (typically `PAT_VL*` variants)

**Example**:
```python
# Tail mask for first 32 lanes
tail_mask = pto.pge_b32(PAT.VL32)
```

#### `pto.plt_b8(scalar: pto.i32) -> (pto.mask_b8, pto.i32)`

**Description**: Generate predicate state together with updated scalar state (tail processing). Creates an 8-bit granularity mask and returns updated scalar value for state progression.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `pto.i32` | Input scalar value (typically remaining element count) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b8` | 8-bit granularity mask |
| `scalar_out` | `pto.i32` | Updated scalar state |

**Constraints**:
- Used with `i8` vector operations for tail processing
- The scalar input is typically a remaining element count that decrements across successive calls

**Example**:
```python
remaining: pto.i32 = 64
mask, remaining = pto.plt_b8(remaining)  # generates mask for next chunk, updates remaining count
```

#### `pto.plt_b16(scalar: pto.i32) -> (pto.mask_b16, pto.i32)`

**Description**: Generate predicate state together with updated scalar state (tail processing). Creates a 16-bit granularity mask and returns updated scalar value for state progression.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `pto.i32` | Input scalar value (typically remaining element count) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b16` | 16-bit granularity mask |
| `scalar_out` | `pto.i32` | Updated scalar state |

**Constraints**:
- Used with `f16`/`bf16`/`i16` vector operations for tail processing
- The scalar input is typically a remaining element count that decrements across successive calls

**Example**:
```python
remaining: pto.i32 = 64
mask, remaining = pto.plt_b16(remaining)  # generates mask for next chunk, updates remaining count
```

#### `pto.plt_b32(scalar: pto.i32) -> (pto.mask_b32, pto.i32)`

**Description**: Generate predicate state together with updated scalar state (tail processing). Creates a 32-bit granularity mask and returns updated scalar value for state progression.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `pto.i32` | Input scalar value (typically remaining element count) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `pto.mask_b32` | 32-bit granularity mask |
| `scalar_out` | `pto.i32` | Updated scalar state |

**Constraints**:
- Used with `f32`/`i32` vector operations for tail processing
- The scalar input is typically a remaining element count that decrements across successive calls

**Example**:
```python
remaining: pto.i32 = 64
mask, remaining = pto.plt_b32(remaining)  # generates mask for next chunk, updates remaining count
```

#### `pto.make_mask(element_type: Type, value: pto.i32 | pto.MaskPattern) -> MaskType | (MaskType, pto.i32)`

**Description**: Creates a mask with appropriate bitwidth (8, 16, or 32) based on element type, automatically inferring whether to perform tail processing or pattern-based mask generation based on the `value` parameter type. This convenience function eliminates the need to manually choose between `plt_b8`/`plt_b16`/`plt_b32` and `pset_b8`/`pset_b16`/`pset_b32` operations.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `element_type` | `Type` | Element type (e.g., `pto.f32`, `pto.f16`, `pto.i8`) |
| `value` | `pto.i32` \| `pto.MaskPattern` | Either: <br>- Remaining element count (as `pto.i32`) for tail processing <br>- Mask pattern enum value for fixed mask generation (for example `pto.MaskPattern.ALL` or `pto.MaskPattern.VL32`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `MaskType` | Generated mask with appropriate granularity |
| `remaining` | `pto.i32` | Updated remaining element count (only returned when `value` is a `pto.i32` for tail processing) |

**Constraints**:
- The `element_type` must be one of: `f32`, `f16`, `bf16`, or an 8/16/32-bit integer family member (`i*`, `si*`, `ui*`)
- The returned mask granularity matches the element type: 32-bit for `f32`/`i32`/`si32`/`ui32`, 16-bit for `f16`/`bf16`/`i16`/`si16`/`ui16`, and 8-bit for `i8`/`si8`/`ui8`
- The function infers the operation mode from the `value` parameter type at compile time:
  - `pto.i32` value → tail processing mode (returns `(mask, updated_remaining)`)
  - `pto.MaskPattern` enum value → pattern mode (returns `mask` only)

**Implementation Note**: This function is a DSL macro that performs type-based dispatch at compile time:
- When `value` is a `pto.i32` expression: expands to corresponding `plt_b` instruction (`plt_b32`, `plt_b16`, or `plt_b8`)
- When `value` is a `pto.MaskPattern` enum value: expands to corresponding `pset_b` instruction (`pset_b32`, `pset_b16`, or `pset_b8`)

**Example**:
```python
# Tail processing with f32 vectors: value is pto.i32 → expands to plt_b32
mask_f32, remaining_f32 = pto.make_mask(pto.f32, remaining_elements)

# Tail processing with f16 vectors: value is pto.i32 → expands to plt_b16  
mask_f16, remaining_f16 = pto.make_mask(pto.f16, remaining_elements)

# Tail processing with i8 vectors: value is pto.i32 → expands to plt_b8
mask_i8, remaining_i8 = pto.make_mask(pto.i8, remaining_elements)

# Pattern-based mask with f32 vectors: value is MaskPattern enum → expands to pset_b32
mask_all_f32 = pto.make_mask(pto.f32, PAT.ALL)

# Pattern-based mask with f16 vectors: value is MaskPattern enum → expands to pset_b16  
mask_even_f16 = pto.make_mask(pto.f16, PAT.EVEN)

# Pattern-based mask with i8 vectors: value is MaskPattern enum → expands to pset_b8
mask_all_i8 = pto.make_mask(pto.i8, PAT.ALL)

# Type annotations help clarify expected parameter types
remaining: pto.i32 = 1024
mask1, updated = pto.make_mask(pto.f32, remaining)     # tail processing
mask2 = pto.make_mask(pto.f32, PAT.ALL)              # pattern mode
```

#### `pto.ppack(mask: MaskType, part: PredicatePart) -> MaskType`

**Description**: Narrowing pack of a predicate register.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Input mask (`mask_b8`, `mask_b16`, or `mask_b32`) |
| `part` | `PredicatePart` | Part selector enum. Use `PredicatePart.LOWER` or `PredicatePart.HIGHER`. |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `packed` | `MaskType` | Packed mask |

**Example**:
```python
packed = pto.ppack(mask, pto.PredicatePart.LOWER)
```

#### `pto.punpack(mask: MaskType, part: PredicatePart) -> MaskType`

**Description**: Widening unpack of a predicate register.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Input mask |
| `part` | `PredicatePart` | Part selector enum. Use `PredicatePart.LOWER` or `PredicatePart.HIGHER`. |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `MaskType` | Unpacked mask |

**Example**:
```python
unpacked = pto.punpack(mask, pto.PredicatePart.HIGHER)
```

#### `pto.pnot(mask: MaskType, gate: MaskType) -> MaskType`

**Description**: Predicate negation under a same-granularity mask gate.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Input mask |
| `gate` | `MaskType` | Gating mask with the same granularity |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `negated` | `MaskType` | Negated mask |

#### `pto.psel(src0: MaskType, src1: MaskType, mask: MaskType) -> MaskType`

**Description**: Selects between two masks using a third mask as selector.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `MaskType` | First input mask |
| `src1` | `MaskType` | Second input mask |
| `mask` | `MaskType` | Selection mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `MaskType` | Selected mask |

#### `pto.plds(buf: ptr, offset: Index, dist: PredicateDist = PredicateDist.NORM) -> MaskType`  [Advanced Tier]

**Description**: Predicate load with scalar-index style offset form. This is the default DSL surface for loading predicate masks from UB memory.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `ptr` | Source pointer in UB memory space |
| `offset` | `Index` | Scalar/index-style offset |
| `dist` | `PredicateDist` | Distribution mode (default: `PredicateDist.NORM`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `MaskType` | Loaded predicate mask |

**Example**:
```python
mask = pto.plds(buf, offset, pto.PredicateDist.NORM)
```

#### `pto.pld(buf: ptr, offset: Index, dist: PredicateDist) -> MaskType`  [Advanced Tier]

**Description**: Predicate load with areg/index register style offset encoding.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `ptr` | Source pointer in UB memory space |
| `offset` | `Index` | Areg/index-style offset |
| `dist` | `PredicateDist` | Distribution mode |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `MaskType` | Loaded predicate mask |

**Example**:
```python
mask = pto.pld(buf, offset, pto.PredicateDist.NORM)
```

#### `pto.pldi(buf: ptr, imm_offset: pto.i32, dist: PredicateDist) -> MaskType`  [Advanced Tier]

**Description**: Predicate load with immediate-offset encoding form.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `ptr` | Source pointer in UB memory space |
| `imm_offset` | `pto.i32` | Immediate-offset operand |
| `dist` | `PredicateDist` | Distribution mode |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `MaskType` | Loaded predicate mask |

**Example**:
```python
mask = pto.pldi(buf, 0, pto.PredicateDist.NORM)
```

#### `pto.psts(mask: MaskType, buf: ptr, offset: Index, dist: PredicateDist = PredicateDist.NORM) -> None`  [Advanced Tier]

**Description**: Stores a predicate mask to UB memory using the VPTO dynamic-offset
`psts` form. This is the dynamic counterpart of `psti`: both encode the same
predicate payload semantics, while offset delivery differs (runtime `index` vs
constant immediate).

**Parameters (Advanced Tier: explicit pointer surface)**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Predicate mask to store |
| `buf` | `ptr` | Pointer to destination UB buffer |
| `offset` | `Index` | Runtime offset (`index`) |
| `dist` | `PredicateDist` | Distribution mode. Use `PredicateDist.NORM` or `PredicateDist.PK` (default: `PredicateDist.NORM`). |

**DIST semantics (VPTO-aligned)**:
- `NORM`: stores packed predicate payload into destination space of size `VL/8`.
- `PK`: stores packed predicate payload into destination space of size `VL/16`,
  keeping one bit out of every two bits.

**Returns**: None (side-effect operation)

**Example**:
```python
pto.psts(mask, buf, offset, pto.PredicateDist.NORM)
```

#### `pto.pst(mask: MaskType, buf: ptr, offset: Index, dist: PredicateDist = PredicateDist.NORM) -> None`  [Advanced Tier]

**Description**: Stores a predicate mask to UB memory using areg/index offset encoding.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Predicate mask to store |
| `buf` | `ptr` | Pointer to destination UB buffer |
| `offset` | `Index` | Areg/index-style offset |
| `dist` | `PredicateDist` | Distribution mode for predicate store. Use `PredicateDist.NORM` or `PredicateDist.PK`. Default is `PredicateDist.NORM`. |

**Returns**: None (side-effect operation)

**Example**:
```python
pto.pst(mask, buf, offset, pto.PredicateDist.NORM)
```

#### `pto.psti(mask: MaskType, buf: ptr, imm_offset: pto.i32, dist: PredicateDist = PredicateDist.NORM) -> None`  [Advanced Tier]

**Description**: Stores a predicate mask to UB memory using immediate-offset encoding.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Predicate mask to store |
| `buf` | `ptr` | Pointer to destination UB buffer |
| `imm_offset` | `pto.i32` | Immediate-offset operand |
| `dist` | `PredicateDist` | Distribution mode for predicate store. Use `PredicateDist.NORM` or `PredicateDist.PK`. Default is `PredicateDist.NORM`. |

**Returns**: None (side-effect operation)

**Example**:
```python
pto.psti(mask, buf, pto.i32(8), pto.PredicateDist.PK)
```

#### `pto.pstu(align_in: pto.align, mask: MaskType, buf: ptr) -> (pto.align, ptr)`  [Advanced Tier]

**Description**: Unaligned predicate store with align-state update.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `align_in` | `pto.align` | Input alignment state |
| `mask` | `MaskType` | Predicate mask to store |
| `buf` | `ptr` | Pointer to destination UB buffer |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `align_out` | `pto.align` | Updated alignment state |
| `base_out` | `ptr` | Updated destination pointer |

**Example**:
```python
align_out, base_out = pto.pstu(align_in, mask, buf)
```

#### `pto.pand(src0: MaskType, src1: MaskType, mask: MaskType) -> MaskType`

**Description**: Bitwise AND of two predicate masks under a gating mask.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `MaskType` | First input mask |
| `src1` | `MaskType` | Second input mask |
| `mask` | `MaskType` | Gating mask with the same granularity |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `MaskType` | Bitwise AND result |

**Example**:
```python
result = pto.pand(mask1, mask2, gate)
```

#### `pto.por(src0: MaskType, src1: MaskType, mask: MaskType) -> MaskType`

**Description**: Bitwise OR of two predicate masks under a gating mask.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `MaskType` | First input mask |
| `src1` | `MaskType` | Second input mask |
| `mask` | `MaskType` | Gating mask with the same granularity |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `MaskType` | Bitwise OR result |

**Example**:
```python
result = pto.por(mask1, mask2, gate)
```

#### `pto.pxor(src0: MaskType, src1: MaskType, mask: MaskType) -> MaskType`

**Description**: Bitwise XOR of two predicate masks under a gating mask.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `MaskType` | First input mask |
| `src1` | `MaskType` | Second input mask |
| `mask` | `MaskType` | Gating mask with the same granularity |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `MaskType` | Bitwise XOR result |

**Example**:
```python
result = pto.pxor(mask1, mask2, gate)
```

#### `pto.pdintlv_b8(src0: pto.mask_b8, src1: pto.mask_b8) -> (pto.mask_b8, pto.mask_b8)`

**Description**: Predicate deinterleave for 8-bit masks.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `pto.mask_b8` | First input mask |
| `src1` | `pto.mask_b8` | Second input mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `low` | `pto.mask_b8` | First result mask |
| `high` | `pto.mask_b8` | Second result mask |

**Example**:
```python
low8, high8 = pto.pdintlv_b8(mask_a, mask_b)
```

#### `pto.pintlv_b16(src0: pto.mask_b16, src1: pto.mask_b16) -> (pto.mask_b16, pto.mask_b16)`

**Description**: Predicate interleave for 16-bit masks.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `pto.mask_b16` | First input mask |
| `src1` | `pto.mask_b16` | Second input mask |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `low` | `pto.mask_b16` | First result mask |
| `high` | `pto.mask_b16` | Second result mask |

**Example**:
```python
low16, high16 = pto.pintlv_b16(mask_a, mask_b)
```

**Note**: Prefer `pto.make_mask()` for automatic bitwidth selection and unified tail/pattern mask generation.
