"""TileLang DSL template for pto.trowargmax"""

import sys
from pathlib import Path
import tilelang_dsl as pto

@pto.vkernel(
    target="a5",
    op="pto.trowargmax",
    advanced=True,
)
def template_trowargmax(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    src_dtype = src.element_type
    idx_dtype = dst.element_type
    lanes = pto.get_lanes(src_dtype)
    valid_rows, valid_cols = src.valid_shape

    # Initialize with negative infinity for ROWARGMAX
    if pto.constexpr(src_dtype == pto.f32):
        init_val = pto.f32(0xFF800000)  # Negative infinity
        init_zero = pto.f32(0)
    elif pto.constexpr(src_dtype == pto.f16):
        init_val = pto.f16(0xFC00)  # Negative infinity
        init_zero = pto.f16(0)

    # Since index is valid in lane 0, we can use mask_1
    mask_1, _ = pto.make_mask(src_dtype, 1)
    mask_1_final, _ = pto.make_mask(idx_dtype, 1)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        
        v_val_acc = pto.vbr(init_val)
        v_idx_acc = pto.vbr(init_zero)
        v_zero = pto.vbr(init_zero)
        
        # Process all column chunks
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(src_dtype, remained)
            v_src = pto.vlds(src[row, col:])
            v_reduced = pto.vcmax(v_src, mask)
            
            v_val, v_idx = pto.vdintlv(v_reduced, v_zero)
            
            # Add absolute col offset to the chunk's local index
            if pto.constexpr(src_dtype == pto.f32):
                v_col = pto.f32(col)
            elif pto.constexpr(src_dtype == pto.f16):
                v_col = pto.f16(col)

            v_idx = pto.vadds(v_idx, v_col, mask_1)
            
            # Compare current chunk max with global max so far
            # vcmp returns a mask
            cmp_mask = pto.vcmp(v_val_acc, v_val, mask_1, "lt")
            
            # Update global max and global argmax depending on who is greater
            v_val_acc = pto.vsel(v_val, v_val_acc, cmp_mask)
            v_idx_acc = pto.vsel(v_idx, v_idx_acc, cmp_mask)

        # Store the extracted index into the dst tile
        if pto.constexpr(src_dtype != idx_dtype):
            # vcvt attrs are type-pair sensitive in VPTO verifier:
            # - f32 -> i32 requires rnd + sat
            # - f16 -> i32 requires part
            if pto.constexpr(src_dtype == pto.f32 and idx_dtype == pto.i32):
                v_idx_acc_casted = pto.vcvt(
                    v_idx_acc,
                    idx_dtype,
                    mask_1_final,
                    rnd=pto.VcvtRoundMode.R,
                    sat=pto.VcvtSatMode.SAT,
                )
            elif pto.constexpr(src_dtype == pto.f16 and idx_dtype == pto.i32):
                v_idx_acc_casted = pto.vcvt(
                    v_idx_acc,
                    idx_dtype,
                    mask_1_final,
                    part=pto.VcvtPartMode.ODD,
                )
            else:
                v_idx_acc_casted = pto.vcvt(v_idx_acc, idx_dtype, mask_1_final)
            pto.vsts(v_idx_acc_casted, dst[row, 0:], mask_1_final)
        else:
            pto.vsts(v_idx_acc, dst[row, 0:], mask_1_final)
    return
