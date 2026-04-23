# 2. DMA Copy Programming

> **Category:** DMA transfer configuration and execution
> **Pipelines:** MTE2 (GM→UB), MTE3 (UB→GM)

DMA transfers move data between Global Memory (GM) and Unified Buffer (UB). The MTE engines operate asynchronously from the Vector core, requiring explicit sync (see [Pipeline Sync](01-pipeline-sync.md)).

This document describes the public grouped DMA interfaces:

- `pto.dma_load`
- `pto.dma_store`
- `pto.dma_copy`

This chapter covers the public grouped DMA interfaces. The legacy raw copy
family remains documented separately; in particular, `pto.copy_ubuf_to_ubuf`
shares the same UB→UB copy contract as `pto.dma_copy` but remains a legacy
surface op.

---

## DMA Transfer Execution

### `pto.dma_load`

- **syntax:**
```mlir
pto.dma_load %gm_src, %ub_dst, %sid, %l2_cache_ctl, %len_burst
  nburst(%n_burst, %src_stride, %dst_stride)
  [loop1(%loop1_count, %loop1_src_stride, %loop1_dst_stride)]
  [loop2(%loop2_count, %loop2_src_stride, %loop2_dst_stride)]
  [pad(%pad_value[, %left_padding_count, %right_padding_count])]
  : !pto.ptr<T, gm>, !pto.ptr<T, ub>, i64, i64, i64,
    i64, i64, i64,
    [loop1 i64, i64, i64,]
    [loop2 i64, i64, i64,]
    [pad T[, i64, i64]]
```
- **semantics:** Grouped GM→UB DMA transfer. It carries the burst, optional HW loop, and optional padding configuration on the copy op itself.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%gm_src` | ptr | GM source pointer (`!pto.ptr<T, gm>`) |
| `%ub_dst` | ptr | UB destination pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%sid` | 32 bits | Stream ID |
| `%l2_cache_ctl` | 2 bits | L2 cache allocate control |
| `%len_burst` | 16 bits | Contiguous bytes transferred per burst row |
| `nburst(%n_burst, %src_stride, %dst_stride)` | 16 bits / 40 bits / 21 bits | Required innermost burst loop: count, GM source stride, UB destination stride |
| `loop1(%loop1_count, %loop1_src_stride, %loop1_dst_stride)` | 21 bits / 40 bits / 21 bits | Optional inner HW loop: count, GM source stride, UB destination stride |
| `loop2(%loop2_count, %loop2_src_stride, %loop2_dst_stride)` | 21 bits / 40 bits / 21 bits | Optional outer HW loop: count, GM source stride, UB destination stride |
| `pad(%pad_value[, %left_padding_count, %right_padding_count])` | scalar / 8 bits / 8 bits | Optional padding: fill value, optional left padding count, optional right padding count |

**Constraints:**

- `nburst(...)` is always required.
- `loop1(...)` and `loop2(...)` must each be provided as a complete group when present.
- `pad(...)` may contain only `%pad_value`; omitted left and right padding counts default to 0.
- If either left or right padding count is provided, both counts must be provided.
- `loop1(...)` may be used without `loop2(...)`; in that case `loop2_count` is treated as 1 when programming the loop-size register.
- `loop2(...)` requires `loop1(...)`; `loop2` without `loop1` is rejected by the verifier.
- `pad(...)` is independent of `loop1(...)` and `loop2(...)`.
- A DMA load may use `nburst(...) pad(...)` without any HW loop group.

**Example:**

```mlir
pto.dma_load %gm_in, %ub_out, %sid, %cache, %len_burst
  nburst(%rows, %gm_row_stride, %ub_row_stride)
  loop1(%tiles, %gm_tile_stride, %ub_tile_stride)
  pad(%pad)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64,
    i64, i64, i64, loop1 i64, i64, i64, pad f16
```

---

### `pto.dma_store`

- **syntax:**
```mlir
pto.dma_store %ub_src, %gm_dst, %sid, %reserved, %len_burst
  nburst(%n_burst, %src_stride, %dst_stride)
  [loop1(%loop1_count, %loop1_src_stride, %loop1_dst_stride)]
  [loop2(%loop2_count, %loop2_src_stride, %loop2_dst_stride)]
  : !pto.ptr<T, ub>, !pto.ptr<T, gm>, i64, i64, i64,
    i64, i64, i64,
    [loop1 i64, i64, i64,]
    [loop2 i64, i64, i64]
```
- **semantics:** Grouped UB→GM DMA transfer. It carries the burst and optional HW loop configuration on the copy op itself.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%ub_src` | ptr | UB source pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%gm_dst` | ptr | GM destination pointer (`!pto.ptr<T, gm>`) |
| `%sid` | 32 bits | Stream ID |
| `%reserved` | 8 bits | Reserved field, normally 0 |
| `%len_burst` | 16 bits | Contiguous bytes transferred per burst row |
| `nburst(%n_burst, %src_stride, %dst_stride)` | 16 bits / 21 bits / 40 bits | Required innermost burst loop: count, UB source stride, GM destination stride |
| `loop1(%loop1_count, %loop1_src_stride, %loop1_dst_stride)` | 21 bits / 21 bits / 40 bits | Optional inner HW loop: count, UB source stride, GM destination stride |
| `loop2(%loop2_count, %loop2_src_stride, %loop2_dst_stride)` | 21 bits / 21 bits / 40 bits | Optional outer HW loop: count, UB source stride, GM destination stride |

**Constraints:**

- `nburst(...)` is always required.
- `loop1(...)` and `loop2(...)` must each be provided as a complete group when present.
- `loop1(...)` may be used without `loop2(...)`; in that case `loop2_count` is treated as 1 when programming the loop-size register.
- `loop2(...)` requires `loop1(...)`; `loop2` without `loop1` is rejected by the verifier.

**Example:**

```mlir
pto.dma_store %ub_in, %gm_out, %sid, %zero, %len_burst
  nburst(%rows, %ub_row_stride, %gm_row_stride)
  loop1(%tiles, %ub_tile_stride, %gm_tile_stride)
  loop2(%batches, %ub_batch_stride, %gm_batch_stride)
  : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64,
    i64, i64, i64, loop1 i64, i64, i64, loop2 i64, i64, i64
```

---

### `pto.dma_copy`

- **syntax:**
```mlir
pto.dma_copy %ub_src, %ub_dst, %sid, %len_burst
  nburst(%n_burst, %src_gap, %dst_gap)
  : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64, i64, i64, i64, i64
```
- **semantics:** Grouped UB→UB raw copy..

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%ub_src` | ptr | UB source pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%ub_dst` | ptr | UB destination pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%sid` | 16 bits | Stream ID |
| `%len_burst` | 16 bits | Burst length in units of 32 bytes |
| `nburst(%n_burst, %src_gap, %dst_gap)` | 16 bits / 16 bits / 16 bits | Required UB→UB outer burst group: count, source gap, destination gap |

**Constraints:**

- UB source and destination addresses must be 32B-aligned.
- `%len_burst`, `%src_gap`, and `%dst_gap` are encoded in units of 32 bytes.

**Example:**

```mlir
pto.dma_copy %ub_src, %ub_dst, %sid, %len32b
  nburst(%rows, %src_gap, %dst_gap)
  : !pto.ptr<i16, ub>, !pto.ptr<i16, ub>, i64, i64, i64, i64, i64
```

---

## GM↔UB Burst / Stride / Pad Model

This section describes the grouped GM↔UB DMA interfaces in this document:
`pto.dma_load` and `pto.dma_store`.

For these grouped GM↔UB DMA ops, the innermost `nburst(...)` group is
**stride-based**: the source and destination stride operands are the
start-to-start byte distance from one burst row to the next row.

### Key Terms

```
burst    = lenBurst contiguous bytes transferred per row
stride   = distance (bytes) from start of row[r] to start of row[r+1]
pad      = ub_stride - lenBurst, padded to the 32B alignment boundary
```

### Alignment Constraints

- **UB addresses** (both source and destination) must be **32-byte aligned**.
- **GM→UB padding**: When `pad(...)` is present on `pto.dma_load`, each UB row is padded from `lenBurst` up to the **32B-aligned boundary** of `ub_stride` with `pad_val`. This ensures every UB row starts at a 32B-aligned offset.
- **UB→GM de-padding**: MTE3 reads `lenBurst` bytes from each 32B-aligned UB row (skipping any padding that was added during load), writing only valid data to GM. This effectively strips padding on store.

---

## UB→UB Burst / Gap Model

This section describes the grouped UB→UB DMA interface in this document:
`pto.dma_copy`.

For `pto.dma_copy`, each burst copies `len_burst * 32` bytes.

The next burst starts at:

```text
src_next = src_curr + (len_burst + src_gap) * 32 bytes
dst_next = dst_curr + (len_burst + dst_gap) * 32 bytes
```

So `src_gap` and `dst_gap` are additional gaps after the copied 32B blocks.
They are not start-to-start strides.

### 2D Diagram: GM→UB (`pto.dma_load`)

```
GM (source, `!pto.ptr<T, gm>`):

          |<--- src_stride (start-to-start) --->|
          |<- len_burst ->|                     |
Row 0:    [##DATA########]......................|
Row 1:    [##DATA########]......................|
Row 2:    [##DATA########]......................|
          ...
Row N-1:  [##DATA########]

UB (destination, `!pto.ptr<T, ub>`, 32B-aligned):

          |<---------- dst_stride (32B-aligned) ---------->|
          |<- len_burst ->|<- pad (to 32B boundary) ->|    |
Row 0:    [##DATA########][000000 PAD 000000000000000]
Row 1:    [##DATA########][000000 PAD 000000000000000]
Row 2:    [##DATA########][000000 PAD 000000000000000]
          ...
Row N-1:  [##DATA########][000000 PAD 000000000000000]

N = n_burst
stride = start of row[r] to start of row[r+1]
pad    = filled with pad_val to 32B boundary (`pad(...)` present)
[DATA] = valid data transferred by DMA
[PAD]  = pad_val fill (from `pad(...)`)
```

### 2D Diagram: UB→GM (`pto.dma_store`)

```
UB (source, `!pto.ptr<T, ub>`, 32B-aligned start addr):

          |<---------- src_stride (32B-aligned) --------->|
          |<- len_burst ->|<-- pad (ignored on read) -->| |
Row 0:    [##DATA########][000 pad 000000000000000000]
Row 1:    [##DATA########][000 pad 000000000000000000]
Row 2:    [##DATA########][000 pad 000000000000000000]
          ...
Row N-1:  [##DATA########][000 pad 000000000000000000]

GM (destination, `!pto.ptr<T, gm>`):

          |<--- dst_stride (start-to-start) --->|
          |<- len_burst ->|                     |
Row 0:    [##DATA########]......................|
Row 1:    [##DATA########]......................|
Row 2:    [##DATA########]......................|
          ...
Row N-1:  [##DATA########]

N = n_burst
MTE3 reads only len_burst bytes from each UB row (de-padding).
Only len_burst bytes are written to each GM row.
```

---

## Multi-Level Loop Semantics (C Code)

The full DMA transfer is a nested loop. `loop1(...)` / `loop2(...)` control the
outer levels, and `nburst(...)` controls the innermost burst level.

### GM→UB Full Loop

```c
// C equivalent of what the HW executes:
for (int j = 0; j < loop2_count; j++) {                // HW outer loop
    uint8_t *gm1 = gm_src + j * loop2_src_stride;
    uint8_t *ub1 = ub_dst + j * loop2_dst_stride;

    for (int k = 0; k < loop1_count; k++) {            // HW inner loop
        uint8_t *gm2 = gm1 + k * loop1_src_stride;
        uint8_t *ub2 = ub1 + k * loop1_dst_stride;

        for (int r = 0; r < n_burst; r++) {            // burst engine
            memcpy(ub2 + r * dst_stride,               //   UB dest row
                   gm2 + r * src_stride,               //   GM src row
                   len_burst);                          //   contiguous bytes
            if (pad_enabled)
                memset(ub2 + r * dst_stride + len_burst,
                       pad_val, dst_stride - len_burst);
        }
    }
}
```

### UB→GM Full Loop

```c
// C equivalent:
for (int j = 0; j < loop2_count; j++) {
    uint8_t *ub1 = ub_src + j * loop2_src_stride;
    uint8_t *gm1 = gm_dst + j * loop2_dst_stride;

    for (int k = 0; k < loop1_count; k++) {
        uint8_t *ub2 = ub1 + k * loop1_src_stride;
        uint8_t *gm2 = gm1 + k * loop1_dst_stride;

        for (int r = 0; r < n_burst; r++) {
            memcpy(gm2 + r * dst_stride,               //   GM dest row
                   ub2 + r * src_stride,               //   UB src row
                   len_burst);                          //   contiguous bytes
        }
    }
}
```

---

## Example 1: GM→UB — Load a 32×32 f32 Tile (Simple Case)

Load a 32×32 f32 tile from GM into UB. This matches the `abs_kernel_2d` test case.

```
GM layout (32 × 32 f32, contiguous):

    |<- len_burst = 128B (32 × 4) ->|
    |<- src_stride = 128B --------->|
    +--[#######TILE#######]--+  row 0
    +--[#######TILE#######]--+  row 1
    ...
    +--[#######TILE#######]--+  row 31

UB layout (32 × 32 f32, 32B-aligned, contiguous):

    |<- dst_stride = 128B (32B-aligned) ->|
    +--[#######TILE#######]--+  row 0
    +--[#######TILE#######]--+  row 1
    ...
    +--[#######TILE#######]--+  row 31

    len_burst   = 32 × 4 = 128 bytes
    src_stride  = 128 bytes (contiguous rows)
    dst_stride  = 128 bytes (already 32B-aligned, no padding)
```

```mlir
// Simple 2D load — only nburst(...) is needed
pto.dma_load %arg0, %ub_in, %c0_i64, %c0_i64, %c128_i64
  nburst(%c32_i64, %c128_i64, %c128_i64)
  : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64,
    i64, i64, i64
```

---

## Example 2: GM→UB — Load a 2D Tile from a Larger Matrix

Load a 64×128 tile (f16) from a 1024×512 matrix in GM into UB.

```
GM layout (1024 × 512 f16):

    col 0          col 128               col 512
    |              |                     |
    +--[###TILE###]+.....................+  row R
    +--[###TILE###]+.....................+  row R+1
    ...
    +--[###TILE###]+.....................+  row R+63

    |<--------- src_stride = 1024B ----------->|
    |<-len_burst=256B->|

    len_burst   = 128 × 2 = 256 bytes (128 f16 elements)
    src_stride  = 512 × 2 = 1024 bytes (start-to-start, full GM row)

UB layout (64 × 128 f16, 32B-aligned, contiguous):

    +--[###TILE###]--+  row 0  (256 bytes, 32B-aligned, no pad)
    +--[###TILE###]--+  row 1
    ...
    +--[###TILE###]--+  row 63

    dst_stride = 256 bytes (= len_burst, already 32B-aligned, no padding)
```

```mlir
pto.dma_load %gm_ptr, %ub_ptr, %c0_i64, %c0_i64, %c256_i64
  nburst(%c64_i64, %c1024_i64, %c256_i64)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64,
    i64, i64, i64
```

---

## Example 3: GM→UB — Load with Padding

Load 100 valid columns from GM into a 128-wide UB tile (f16). The remaining 28 columns are zero-padded.

```
GM (100 cols valid, contiguous):

    |<-len_burst=200B->|
    |<- src_stride=200B (start-to-start) ->|
    +--[####DATA####]-+  row 0
    +--[####DATA####]-+  row 1
    ...
    +--[####DATA####]-+  row 63

UB (128 cols wide, 32B-aligned, padded):

    |<--------- dst_stride = 256B (32B-aligned) --------->|
    |<-len_burst=200B->|<---- pad = 56B to 32B boundary ->|
    +--[####DATA####]-+[0000000 PAD 0000000000000000000000]+  row 0
    +--[####DATA####]-+[0000000 PAD 0000000000000000000000]+  row 1
    ...
    +--[####DATA####]-+[0000000 PAD 0000000000000000000000]+  row 63

    len_burst   = 100 × 2 = 200 bytes
    src_stride  = 200 bytes (start-to-start, contiguous in GM)
    dst_stride  = 128 × 2 = 256 bytes (32B-aligned tile width in UB)
    pad         = 256 - 200 = 56 bytes (padded to 32B boundary with pad_val)
```

```mlir
%pad = arith.constant 0 : i16
pto.dma_load %gm_ptr, %ub_ptr, %c0_i64, %c0_i64, %c200_i64
  nburst(%c64_i64, %c200_i64, %c256_i64)
  pad(%pad, %c0_i64, %c0_i64)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64,
    i64, i64, i64, pad i16, i64, i64
```

---

## Example 4: UB→GM — Store a 32×32 f32 Tile (Simple Case)

Store a 32×32 f32 tile from UB back to GM. This matches the `abs_kernel_2d` test case.

```
UB (source, 32B-aligned, 32 × 32 f32):

    |<- src_stride = 128B (32B-aligned) ->|
    |<- len_burst = 128B ->|
    +--[#######TILE#######]---+  row 0
    +--[#######TILE#######]---+  row 1
    ...
    +--[#######TILE#######]---+  row 31

    (no padding here — len_burst == src_stride)

GM (dest, 32 × 32 f32):

    |<- dst_stride = 128B ->|
    |<- len_burst = 128B -->|
    +--[#######TILE#######]---+  row 0
    +--[#######TILE#######]---+  row 1
    ...
    +--[#######TILE#######]---+  row 31
```

```mlir
pto.dma_store %ub_out, %arg1, %c0_i64, %c0_i64, %c128_i64
  nburst(%c32_i64, %c128_i64, %c128_i64)
  : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64,
    i64, i64, i64
```

---

## Example 5: UB→GM — Store a 2D Tile Back to a Larger Matrix

Store a 64×128 tile (f16) from UB back to a 1024×512 GM matrix at an offset.

```
UB (source, 32B-aligned, 64 × 128 f16):

    |<- src_stride = 256B (32B-aligned) ->|
    |<- len_burst = 256B ->|
    +--[#####TILE#####]---+  row 0
    +--[#####TILE#####]---+  row 1
    ...
    +--[#####TILE#####]---+  row 63

    (no padding here — len_burst == src_stride)

GM (dest, into 1024 × 512 matrix):

    |<----------- dst_stride = 1024B (start-to-start) --------->|
    |<- len_burst = 256B ->|                                    |
    col 0          col 128                              col 512
    +--[#####TILE#####]---+.............................+  row R
    +--[#####TILE#####]---+.............................+  row R+1
    ...
    +--[#####TILE#####]---+.............................+  row R+63

    MTE3 reads len_burst bytes from each 32B-aligned UB row,
    writes only len_burst bytes per GM row (stride controls row spacing).
```

```mlir
pto.dma_store %ub_ptr, %gm_ptr, %c0_i64, %c0_i64, %c256_i64
  nburst(%c64_i64, %c256_i64, %c1024_i64)
  : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64,
    i64, i64, i64
```

---

## Example 6: GM→UB with Multi-Level Loop (Batch of Tiles)

Load 4 batches of 8×128 tiles from a [4, 8, 128] f16 tensor using loop1.

```
GM [4, 8, 128] f16 (contiguous):        UB (4 tiles laid out sequentially):

    batch 0: 8 rows × 256 bytes          [batch 0: 8×128][batch 1: 8×128]
    batch 1: 8 rows × 256 bytes          [batch 2: 8×128][batch 3: 8×128]
    batch 2: 8 rows × 256 bytes
    batch 3: 8 rows × 256 bytes          loop1 src_stride = 2048 bytes (8 × 256)
                                          loop1 dst_stride = 2048 bytes (8 × 256)
    Each batch = 8 × 256 = 2048 bytes     loop1_count = 4 (iterate over batches)
```

```mlir
// loop1_count = 4 batches, loop2 omitted
pto.dma_load %gm_ptr, %ub_ptr, %c0_i64, %c0_i64, %c256_i64
  nburst(%c8_i64, %c256_i64, %c256_i64)
  loop1(%c4_i64, %c2048_i64, %c2048_i64)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64,
    i64, i64, i64, loop1 i64, i64, i64
```

Execution trace:

```
loop1 iter 0: gm_ptr + 0×2048 → ub_ptr + 0×2048, DMA 8 rows × 256B
loop1 iter 1: gm_ptr + 1×2048 → ub_ptr + 1×2048, DMA 8 rows × 256B
loop1 iter 2: gm_ptr + 2×2048 → ub_ptr + 2×2048, DMA 8 rows × 256B
loop1 iter 3: gm_ptr + 3×2048 → ub_ptr + 3×2048, DMA 8 rows × 256B
```
