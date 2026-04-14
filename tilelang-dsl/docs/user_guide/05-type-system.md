## Type System

### Scalar Types

| DSL Type | Description | Bit Width |
|----------|-------------|-----------|
| `pto.i1` | Boolean | 1 |
| `pto.i8` | 8-bit signless integer | 8 |
| `pto.si8` | 8-bit signed integer | 8 |
| `pto.ui8` | 8-bit unsigned integer | 8 |
| `pto.i16` | 16-bit signless integer | 16 |
| `pto.si16` | 16-bit signed integer | 16 |
| `pto.ui16` | 16-bit unsigned integer | 16 |
| `pto.i32` | 32-bit signless integer | 32 |
| `pto.si32` | 32-bit signed integer | 32 |
| `pto.ui32` | 32-bit unsigned integer | 32 |
| `pto.i64` | 64-bit signless integer | 64 |
| `pto.si64` | 64-bit signed integer | 64 |
| `pto.ui64` | 64-bit unsigned integer | 64 |
| `pto.f16` | Half precision float | 16 |
| `pto.bf16` | Brain float 16 | 16 |
| `pto.f32` | Single precision float | 32 |

Python literals are automatically typed:
- `bool` → `pto.i1`
- `int` → Context-dependent (typically `pto.i32` or `pto.i64`)
- `float` → `pto.f32`

For explicit typing, use type constructors:
```python
x = pto.i32(1024)      # Explicit i32 constant
y: pto.i32 = 1024      # Type annotation
z = pto.ui16(7)        # Explicit unsigned 16-bit constant
```

Integer sign semantics are part of the DSL type surface. `pto.si16`,
`pto.ui16`, and `pto.i16` are distinct scalar dtypes and lower to `si16`,
`ui16`, and `i16` respectively in VPTO IR.

### Floating-Point Literal Forms

`pto.f16(...)`, `pto.bf16(...)`, and `pto.f32(...)` accept multiple literal forms.

```python
# Signed numeric literals
a = pto.f16(-1.5)
b = pto.bf16(+2.5)
c = pto.f32(-3.5)

# Special floating-point values
pos_inf = pto.f32("inf")
neg_inf = pto.f32("-inf")
qnan = pto.f32("nan")

# Bit-pattern form (hex string, interpreted by target dtype)
f16_neg_inf = pto.f16("0xFC00")
bf16_neg_inf = pto.bf16("0xFF80")
f32_neg_inf = pto.f32("0xFF800000")
```

Notes:
- Prefer dtype constructors for reduction seeds and boundary values (for example rowmax initialization).
- For float bit-pattern constants, pass a **string** hex literal to the matching dtype constructor.
- Avoid passing raw integer bit-patterns directly into vector broadcast/dup APIs when a floating vector is expected.
- `float(...)` function calls are not part of the TileLang DSL public call surface; use constructor forms above.

### Vector Register Type

Vector registers have fixed 256-byte width:

```python
v_f32 = pto.vreg(pto.f32)  # !pto.vreg<64xf32>
v_f16 = pto.vreg(pto.f16)  # !pto.vreg<128xf16>
v_i8 = pto.vreg(pto.i8)    # !pto.vreg<256xi8>
```

`pto.vreg(dtype)` only takes the element type. The frontend infers the element count automatically from the fixed 256-byte register width:

- `pto.f32` → `!pto.vreg<64xf32>`
- `pto.f16` → `!pto.vreg<128xf16>`
- `pto.bf16` → `!pto.vreg<128xbf16>`
- `pto.i32` → `!pto.vreg<64xi32>`
- `pto.si32` → `!pto.vreg<64xsi32>`
- `pto.ui32` → `!pto.vreg<64xui32>`
- `pto.i16` → `!pto.vreg<128xi16>`
- `pto.si16` → `!pto.vreg<128xsi16>`
- `pto.ui16` → `!pto.vreg<128xui16>`
- `pto.i8` → `!pto.vreg<256xi8>`
- `pto.si8` → `!pto.vreg<256xsi8>`
- `pto.ui8` → `!pto.vreg<256xui8>`

Constraint: `element_count × bitwidth(element_type) = 2048`

Use `pto.elements_per_vreg(dtype)` when you need the inferred element count explicitly:

```python
v_dtype = pto.vreg(pto.f32)
lanes0 = v_dtype.elements_per_vreg       # 64
lanes1 = pto.elements_per_vreg(pto.f32)  # 64
```

Current TileLang DSL v1 vector lowering supports the 8/16/32-bit integer
families (`i*`, `si*`, `ui*`) plus `f16`, `bf16`, and `f32` element types.

### Typed Masks

Masks are typed by their bit granularity:

| DSL Type | VPTO Type | Description |
|----------|-----------|-------------|
| `pto.mask_b8` | `!pto.mask<b8>` | 8-bit granularity mask |
| `pto.mask_b16` | `!pto.mask<b16>` | 16-bit granularity mask |
| `pto.mask_b32` | `!pto.mask<b32>` | 32-bit granularity mask |

```python
mask_ty = pto.mask_b32
mask: pto.mask_b32 = pto.make_mask(pto.f32, PAT.ALL)
```

Mask operations must match the vector element family:
- `f32`, `i32`, `si32`, and `ui32` vectors use `mask_b32`
- `f16`, `bf16`, `i16`, `si16`, and `ui16` vectors use `mask_b16`
- `i8`, `si8`, and `ui8` vectors use `mask_b8`

```python
# Correct: f32 vector with b32 mask
mask32 = pto.make_mask(pto.f32, PAT.ALL)
vec_f32 = pto.vlds(ptr, offset)
out = pto.vabs(vec_f32, mask32)

# Error: mismatched mask granularity
mask16 = pto.make_mask(pto.f16, PAT.ALL)
out = pto.vabs(vec_f32, mask16)  # Type error!
```

### Pointer Types [Advanced Tier]

Pointers combine element type and memory space:

```python
from pto import MemorySpace

ptr_gm = pto.ptr(pto.f32, MemorySpace.GM)    # GM pointer to f32
ptr_ub = pto.ptr(pto.f16, MemorySpace.UB)    # UB pointer to f16
```

The `MemorySpace` enum provides type-safe memory space specification:

| Enum Value | Description |
|------------|-------------|
| `MemorySpace.GM` | Global Memory (off-chip HBM/DDR) |
| `MemorySpace.UB` | Unified Buffer (on-chip SRAM, 256KB) |

This replaces string literals (`MemorySpace.GM`/`MemorySpace.UB`) with compile-time checked enums.

### Public Buffer Types

TileLang uses two public buffer-facing type names in kernel signatures:

| Public Type | Description |
|-------------|-------------|
| `pto.TensorView` | GM-facing tensor view descriptor used for DMA-oriented data access |
| `pto.Tile` | UB-facing tile buffer value used for tiled compute |

### TensorView Types

TensorView types represent multi-dimensional (up to 5D) views into tensors residing in Global Memory (GM). They are used as kernel parameters for describing GM data and support slicing operations to create logical partitions for DMA load/store operations.

#### TensorView Type Definition

TensorView types are parameterized by shape (a tuple of up to 5 dimensions) and element type:

```python
# Kernel parameter using TensorView
@pto.vkernel(target="a5", op="custom", dtypes=[(pto.AnyFloat, pto.AnyFloat, pto.AnyFloat)], priority=10)
def tiled_kernel(
    input_tensor: pto.TensorView,   # GM tensor view
    output_tensor: pto.TensorView,  # GM tensor view
    tile_buf: pto.Tile              # UB tile
):
    # Access tensor view properties
    shape = input_tensor.shape           # tuple of dimensions (dynamic or static, up to 5D)
    dtype = input_tensor.element_type    # e.g., pto.f32
    strides = input_tensor.strides       # stride in elements
```

Important notes:
- TensorView is a read-only descriptor for GM data, though DMA store operations can write through it.
- Shape can be static (compile-time constants) or dynamic (determined at runtime).
- Strides are expressed in elements, not bytes.
- Memory space is always GM (Global Memory).
- Maximum rank is 5. PTO ISA right-aligns lower-rank shapes to 5D.
- When higher dimensions are 1, a 5D TensorView can be abbreviated to lower-rank forms. For example, shape `(1, 1, 64, 32, 16)` can be written as `(64, 32, 16)`, and shape `(1, 1, 1, 32, 16)` can be written as `(32, 16)`.

#### TensorView Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int, ...]` | Tensor dimensions (supports up to 5 dimensions, right-aligned to 5D in PTO ISA) |
| `element_type` | `Type` | Element data type (for example `pto.f32`, `pto.f16`) |
| `strides` | `tuple[int, ...]` | Stride in elements for each dimension |
| `offset` | `pto.i64` | Byte offset from base pointer (internal) |

#### Padding Mode Enum

Padding mode controls how out-of-bounds accesses are handled during DMA load/store operations:

| Enum Value | Description |
|------------|-------------|
| `PadMode.PadNull` | No padding. Out-of-bounds access is invalid |
| `PadMode.PadFirstElem` | Pad using the first element of the source |
| `PadMode.PadValue` | Pad using a specified value and requires `pad_value` |

#### Slicing Syntax

TensorView supports Python slicing syntax to create logical partitions:

```python
# Create a partition from a tensor view
partition = tensor_view[dim0_start:dim0_end, dim1_start:dim1_end]

# Example: extract a 16x16 tile from a larger tensor
tile_view = large_tensor[0:16, 0:16]

# Dynamic offsets and sizes
dim0_start = tensor_view.shape[0] // 2
dynamic_partition = tensor_view[dim0_start:tensor_view.shape[0], 4:20]

# Static positive step on dimension 0
stepped_partition = tensor_view[0:32:2, 0:16]

# Right-aligned shorthand on a 5D descriptor
partition_3d = tensor_view[d2_start:d2_end, d3_start:d3_end, d4_start:d4_end]

# Full 5D spelling remains available when needed
partition_5d = tensor_view[
    d0_start:d0_end,
    d1_start:d1_end,
    d2_start:d2_end,
    d3_start:d3_end,
    d4_start:d4_end,
]
```

Constraints:
- Slicing returns a new TensorView representing the logical partition.
- The partition must be within the original tensor bounds.
- When fewer than 5 slice axes are written, they are right-aligned to the trailing physical axes of the 5D descriptor.
- `stop` must be explicit on all dimensions.
- `start` may be static or dynamic.
- `step` must be a static positive integer.
- Dimension 0 may use `step > 1`.
- Dimension 1 must keep `step == 1` in the current DMA-oriented implementation.

### Tile Types

Tile types represent data blocks in memory with layout and configuration information, corresponding to `!pto.tile_buf` in the VPTO IR. Tiles are commonly used as kernel parameters for tiled computations.

#### Tile Type Definition

`pto.Tile` is the public tile type used in kernel signatures and vectorized UB compute. User code does not construct tiles with standalone helper APIs in the stable guide surface.

Important notes on shape and valid shape:
- `shape` must be a compile-time constant. Tile dimensions are fixed at compilation time and cannot change at runtime.
- `valid_shape` can be either static or dynamic and must be less than or equal to `shape` in each dimension.
- When `valid_shape` is not specified, it defaults to the full `shape`.

#### Tile Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int, ...]` | Full tile dimensions. These are compile-time constants |
| `element_type` | `Type` | Element data type (for example `pto.f32`) |
| `memory_space` | `MemorySpace` | Memory space such as GM or UB |
| `valid_shape` | `tuple[int, ...]` | Actual data dimensions within the tile. Must be less than or equal to `shape` in each dimension |
| `config` | `TileConfig` | Layout and padding configuration |

#### Tile Pad Values

`TileConfig.pad_value` is modeled after the C++ `PadValue : uint64_t` design.

Standard pad values use small integer encodings:

| DSL Value | Encoded Value | Meaning |
|-----------|---------------|---------|
| `pto.PadValue.NULL` | `0` | No concrete fill value |
| `pto.PadValue.ZERO` | `1` | Zero fill |
| `pto.PadValue.MAX` | `2` | Maximum finite / integer max for the tile element dtype |
| `pto.PadValue.MIN` | `3` | Minimum finite / integer min for the tile element dtype |

Custom pad values use the `CustomBase = 0x100000000` convention and are authored with `pto.PadValue.custom_f32(...)`:

```python
pad0 = pto.PadValue.ZERO
pad1 = pto.PadValue.custom_f32(-1.0)
pad2 = pto.PadValue.custom_f32("0xBF800000")  # float32 bit pattern for -1.0f
```

Notes:
- `PadValue.value` on the host-side descriptor still exposes the encoded integer payload.
- `PadValue.text` exposes the standard textual spelling for built-ins such as `null` and `zero`.
- Custom pad values currently model an `f32` payload. In DSL v1, materializing a custom pad into a scalar is only supported for floating tile element dtypes.
- `PadValue.NULL` does not denote a usable scalar fill constant. Reading `tile.pad_value.value` or `tile.config.pad_value.value` when the enum is `NULL` is a frontend error.

#### Tile Shape Concepts

- `shape` is the static physical allocation size of the tile buffer.
- `valid_shape` is the logical data region and may be static or dynamic.
- `valid_shape[i] <= shape[i]` must hold for each dimension.
- Fixed-size tiles with smaller valid regions are useful for padding and partial-tile cases.

#### Basic Access Operations

```python
# Get tile properties
shape = tile.shape                    # (256, 128)
elem_type = tile.element_type         # pto.f32
mem_space = tile.memory_space         # MemorySpace.UB
valid_shape = tile.valid_shape        # (240, 120) or same as shape

# Get configuration properties
config = tile.config
b_layout = config.b_layout            # pto.BLayout.ROW_MAJOR
s_layout = config.s_layout            # pto.SLayout.NONE_BOX
s_fractal = config.s_fractal_size     # pto.i32(512)
pad_desc = tile.config.pad_value      # PadValue enum bound to the tile element dtype
pad_desc2 = tile.pad_value            # direct sugar for the same PadValue enum

# Dynamic properties
rank = tile.rank                      # 2
```

`tile.config.pad_value` and `tile.pad_value` are enum-typed inside kernel code. Use `.value` to materialize the configured pad descriptor against the tile element dtype:

- `tile.pad_value.value` with `PadValue.ZERO` becomes `0` / `0.0`
- `tile.pad_value.value` with `PadValue.MAX` becomes dtype-aware max
- `tile.pad_value.value` with `PadValue.MIN` becomes dtype-aware min
- `tile.pad_value.value` with `PadValue.custom_f32(...)` becomes the authored floating scalar
- `tile.pad_value.value` with `PadValue.NULL` raises a frontend error

Example: reading pad value from a `Tile`

```python
@pto.vkernel(op="fill_pad_demo", dtypes=[(pto.f16,)])
def kernel(dst: pto.Tile):
    mask, _ = pto.make_mask(pto.f16, 8)

    # Read the Tile-bound PadValue enum.
    pad0 = dst.pad_value

    # Equivalent form through TileConfig metadata.
    pad1 = dst.config.pad_value

    if pto.constexpr(pad0 != pto.PadValue.NULL):
        scalar0 = pad0.value
        scalar1 = pad1.value
        vec0 = pto.vdup(scalar0, mask)
        vec1 = pto.vdup(scalar1, mask)
        pto.vsts(vec0, dst[0, 0:], mask)
        pto.vsts(vec1, dst[1, 0:], mask)
```

If `dst` is specialized with `config=pto.TileConfig.from_mapping({"pad_value": pto.PadValue.ZERO})`,
both `pad0` and `pad1` are `PadValue.ZERO`, and `pad0.value` / `pad1.value` materialize to the scalar `0.0` for an `f16` tile.

#### Conversion Operations

Basic mode syntax uses tile element-indexing directly in vector operations:

```python
# 2D tile indexing
vec = pto.vlds(tile[row, col:])
pto.vsts(vec, tile[row, col:], mask)

# 1D tile indexing
vec = pto.vlds(tile[start:])
pto.vsts(vec, tile[start:], mask)
```

Advanced mode syntax converts tiles to typed pointers for byte-offset operations:

```python
# Convert tile to pointer
ptr = tile.as_ptr()                # Returns pto.ptr(pto.f32, MemorySpace.UB)

# Use pointer with byte offset
vec = pto.vlds(ptr, offset)
pto.vsts(vec, ptr, offset, mask)
```

#### Kernel Parameter Usage

```python
@pto.vkernel(target="a5", op="scale", dtypes=[(pto.AnyFloat, pto.AnyFloat)], priority=10)
def tiled_kernel(
    input_tile: pto.Tile,
    output_tile: pto.Tile,
    scale: pto.f32
):
    all_mask = pto.make_mask(pto.f32, PAT.ALL)
    for i in range(0, 256, 64):
        vec = pto.vlds(input_tile[i, 0:])
        scaled = pto.vmuls(vec, scale, all_mask)
        pto.vsts(scaled, output_tile[i, 0:], all_mask)
```

### Alignment Type

The `pto.align` type is used for alignment carrier operations and maps to `!pto.align`.
