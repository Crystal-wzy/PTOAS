# 10. Broadcast and Expansion Operations

> **Category:** Tile-local VEC broadcast and expansion compute
> **Pipeline:** PIPE_V

This chapter documents the TileLib broadcast, row-expansion, and column-expansion families. These ops populate destination tiles by broadcasting one logical scalar across a larger region — either from a standalone scalar operand, one source value per destination row, or one source value per destination column.

---

## 10.1 Scalar Broadcast: `pto.texpands`

- **syntax:**
```mlir
pto.texpands ins(%scalar : <scalar_type>)
             outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** `dst[i, j] = scalar` for every element inside `dst`'s valid region.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | signless integer / floating-point scalar | Scalar value broadcast into the destination tile. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- The TileLib template is VEC-oriented and fills `dst.valid_shape`.
- The scalar type must be compatible with `dst.dtype`.

**Example:**

```mlir
pto.texpands ins(%scalar : f32)
             outs(%dst : !pto.tile_buf<vec, 16x64xf32>)
```

---

## 10.2 Row-Wise Broadcast: `pto.trowexpand`

- **syntax:**
```mlir
pto.trowexpand ins(%src : !pto.tile_buf<...>)
               outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** `dst[row, col] = src[row, 0]`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile carrying one logical scalar per destination row. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- `src` and `dst` must have the same number of valid rows.
- `src` must encode exactly one logical source value per destination row.
- Templates target row-major VEC layouts.

**Example:**

```mlir
pto.trowexpand ins(%src : !pto.tile_buf<vec, 16x1xf32>)
               outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
```

---

## 10.3 Row-Wise Broadcast Arithmetic and Transform Families

The row-expansion family combines a full tile `%src0` with a per-row scalar carrier `%src1`:

| Op | Semantics |
|----|-----------|
| `pto.trowexpandadd` | `dst[row, col] = src0[row, col] + src1[row, 0]` |
| `pto.trowexpandsub` | `dst[row, col] = src0[row, col] - src1[row, 0]` |
| `pto.trowexpandmul` | `dst[row, col] = src0[row, col] * src1[row, 0]` |
| `pto.trowexpanddiv` | `dst[row, col] = src0[row, col] / src1[row, 0]` |
| `pto.trowexpandmax` | `dst[row, col] = max(src0[row, col], src1[row, 0])` |
| `pto.trowexpandmin` | `dst[row, col] = min(src0[row, col], src1[row, 0])` |
| `pto.trowexpandexpdif` | `dst[row, col] = exp(src0[row, col] - src1[row, 0])` |

### Common Syntax

```mlir
pto.<op> ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `pto.tile_buf` | Main source tile. |
| `src1` | `pto.tile_buf` | Tile carrying one logical scalar per destination row. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- `src0` and `dst` must be shape/valid-region compatible.
- `src1` must provide one logical scalar per destination row.
- Templates target row-major VEC layouts.
- `pto.trowexpanddiv` and `pto.trowexpandexpdif` are floating-point-only.

**Example:**

```mlir
pto.trowexpandadd ins(%src0, %src1 : !pto.tile_buf<vec, 16x128xf32>,
                                     !pto.tile_buf<vec, 16x1xf32, blayout=col_major>)
                  outs(%dst : !pto.tile_buf<vec, 16x128xf32>)
```

---

## 10.4 Column-Wise Broadcast: `pto.tcolexpand`

- **syntax:**
```mlir
pto.tcolexpand ins(%src : !pto.tile_buf<...>)
               outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** `dst[row, col] = src[0, col]`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile carrying one logical scalar per destination column. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- `src` and `dst` must have the same number of valid columns.
- `src` must encode exactly one logical source value per destination column.
- Templates target row-major VEC layouts.

**Example:**

```mlir
pto.tcolexpand ins(%src : !pto.tile_buf<vec, 1x16xf32>)
               outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
```

---

## 10.5 Column-Wise Broadcast Arithmetic and Transform Families

The column-expansion family combines a full tile `%src0` with a per-column scalar carrier `%src1`:

| Op | Semantics |
|----|-----------|
| `pto.tcolexpandadd` | `dst[row, col] = src0[row, col] + src1[0, col]` |
| `pto.tcolexpandsub` | `dst[row, col] = src0[row, col] - src1[0, col]` |
| `pto.tcolexpandmul` | `dst[row, col] = src0[row, col] * src1[0, col]` |
| `pto.tcolexpanddiv` | `dst[row, col] = src0[row, col] / src1[0, col]` |
| `pto.tcolexpandmax` | `dst[row, col] = max(src0[row, col], src1[0, col])` |
| `pto.tcolexpandmin` | `dst[row, col] = min(src0[row, col], src1[0, col])` |
| `pto.tcolexpandexpdif` | `dst[row, col] = exp(src0[row, col] - src1[0, col])` |

### Common Syntax

```mlir
pto.<op> ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `pto.tile_buf` | Main source tile. |
| `src1` | `pto.tile_buf` | Tile carrying one logical scalar per destination column. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- `src0` and `dst` must be shape/valid-region compatible.
- `src1` must provide one logical scalar per destination column.
- Templates target row-major VEC layouts.
- `pto.tcolexpanddiv` and `pto.tcolexpandexpdif` are floating-point-only.

**Example:**

```mlir
pto.tcolexpandadd ins(%src0, %src1 : !pto.tile_buf<vec, 16x128xf32>,
                                     !pto.tile_buf<vec, 1x128xf32>)
                  outs(%dst : !pto.tile_buf<vec, 16x128xf32>)
```
