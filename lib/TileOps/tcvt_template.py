"""TileLang DSL template for pto.tcvt."""

import tilelang_dsl as pto


def _supports_basic_rowwise_tcvt(
    src_shape=(),
    src_valid_shape=(),
    dst_shape=(),
    dst_valid_shape=(),
    src_config=None,
    dst_config=None,
):
    if tuple(src_shape) != tuple(dst_shape):
        return False
    if tuple(src_valid_shape) != tuple(dst_valid_shape):
        return False
    if len(src_shape) != 2 or len(dst_shape) != 2:
        return False
    if src_config is None or dst_config is None:
        return False
    if src_config.b_layout != pto.BLayout.ROW_MAJOR:
        return False
    if dst_config.b_layout != pto.BLayout.ROW_MAJOR:
        return False
    if src_config.s_layout != pto.SLayout.NONE_BOX:
        return False
    if dst_config.s_layout != pto.SLayout.NONE_BOX:
        return False
    return True

@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f32, pto.i32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f32_to_i32(src: pto.Tile, dst: pto.Tile):
    dst_dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dst_dtype)):
            mask, remained = pto.make_mask(dst_dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                mask,
                rnd=rnd,
                sat=pto.VcvtSatMode.SAT,
            )
            pto.vsts(converted, dst[row, col:], mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.f16, pto.f32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_f16_to_f32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(pto.f16, pto.PAT.ALL)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f32)):
            store_mask, remained = pto.make_mask(pto.f32, remained)
            vec = pto.vlds(src[row, col:], dist="UNPK_B16")
            converted = pto.vcvt(
                vec,
                pto.f32,
                full_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tcvt",
    dtypes=[
        (pto.i32, pto.f32),
    ],
    constraints=[_supports_basic_rowwise_tcvt],
)
def template_tcvt_i32_to_f32(src: pto.Tile, dst: pto.Tile):
    dst_dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    round_mode = pto.get_op_attr("round_mode", "RINT")
    rnd = pto.VcvtRoundMode.R
    if pto.constexpr(round_mode == "ROUND"):
        rnd = pto.VcvtRoundMode.A
    elif pto.constexpr(round_mode == "FLOOR"):
        rnd = pto.VcvtRoundMode.F
    elif pto.constexpr(round_mode == "CEIL"):
        rnd = pto.VcvtRoundMode.C
    elif pto.constexpr(round_mode == "TRUNC"):
        rnd = pto.VcvtRoundMode.Z
    elif pto.constexpr(round_mode == "ODD"):
        rnd = pto.VcvtRoundMode.O

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dst_dtype)):
            mask, remained = pto.make_mask(dst_dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                mask,
                rnd=rnd,
            )
            pto.vsts(converted, dst[row, col:], mask)
    return
