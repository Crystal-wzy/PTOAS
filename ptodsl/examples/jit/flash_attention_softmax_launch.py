# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Flash-attention softmax stage — end-to-end launch demo.

This example is the launchable counterpart to the compile-only
``flash_attention_sketch.py`` demo. It intentionally keeps only the online
softmax update stage from flash attention because the current PTODSL runtime
path is already strong enough for vector-heavy softmax, while the full
flash-attention stack still depends on simt/cube capabilities that are not yet
complete for an end-to-end runtime demo.

Each kernel instance updates one block of up to 8 rows:

    m_next = max(m_prev, row_max(scores))
    p      = exp(scores - m_next)
    l_next = l_prev * exp(m_prev - m_next) + row_sum(p)
    expmax = l_prev * exp(m_prev - m_next) / l_next
    out    = p / l_next

The demo offers two fixed-shape launchable kernels so the current launch ABI
does not need runtime scalar parameters:

- ``rows8_seq128``: full-width 128-column softmax
- ``rows17_seq96``: multi-block + tail-mask coverage
"""

import argparse
import time
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "ptodsl" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            break
    else:
        raise RuntimeError(
            "Unable to locate the PTODSL Python package root from flash_attention_softmax_launch.py"
        )

from ptodsl import pto, scalar

s = scalar

_DEVICE = "npu:0"
_ROWS_PER_BLOCK = 8
_PHYSICAL_COLS = 128


def _make_flash_attention_softmax_kernel(name: str, *, rows: int, seq: int):
    if rows <= 0:
        raise ValueError("rows must be positive")
    if not 0 < seq <= _PHYSICAL_COLS:
        raise ValueError(f"seq must be in [1, {_PHYSICAL_COLS}]")

    @pto.jit(
        name=name,
        kernel_kind="vector",
        target="a5",
        mode="explicit",
        insert_sync=False
    )
    def kernel(
        oldmax: pto.tensor_spec(rank=2, dtype=pto.f32),
        oldsum: pto.tensor_spec(rank=2, dtype=pto.f32),
        scores: pto.tensor_spec(rank=2, dtype=pto.f32),
        newmax: pto.tensor_spec(rank=2, dtype=pto.f32),
        newsum: pto.tensor_spec(rank=2, dtype=pto.f32),
        expmax: pto.tensor_spec(rank=2, dtype=pto.f32),
        out: pto.tensor_spec(rank=2, dtype=pto.f32),
    ):
        c0 = pto.const(0)
        c1 = pto.const(1)
        c8 = pto.const(_ROWS_PER_BLOCK)
        c64 = pto.const(64)
        c128 = pto.const(_PHYSICAL_COLS)
        c_rows = pto.const(rows)
        c_seq = pto.const(seq)
        c_rows_x_128 = pto.const(rows * _PHYSICAL_COLS)

        c0_i64 = pto.const(0, dtype=pto.int64)
        c128_i64 = pto.const(128, dtype=pto.int64)
        c256_i64 = pto.const(256, dtype=pto.int64)
        c8448_i64 = pto.const(8448, dtype=pto.int64)
        c16640_i64 = pto.const(16640, dtype=pto.int64)
        c16768_i64 = pto.const(16768, dtype=pto.int64)
        c16896_i64 = pto.const(16896, dtype=pto.int64)

        c0_i32 = pto.const(0, dtype=pto.int32)
        c1_i32 = pto.const(1, dtype=pto.int32)
        c8_i32 = pto.const(_ROWS_PER_BLOCK, dtype=pto.int32)
        c_seq_i32 = pto.const(seq, dtype=pto.int32)
        c_rows_i32 = pto.const(rows, dtype=pto.int32)

        block_i64 = pto.get_block_idx()
        block_idx = s.index_cast(block_i64)
        row_base = s.muli(block_idx, c8)
        row_base_i32 = s.index_cast(pto.int32, row_base)
        remaining_rows = s.subi(c_rows_i32, row_base_i32)
        has_rows = remaining_rows > c0_i32
        too_many_rows = remaining_rows > c8_i32
        row_count_i32 = s.select(too_many_rows, c8_i32, remaining_rows)
        row_count = s.index_cast(row_count_i32)

        with pto.if_(has_rows) as has_rows_br:
            with has_rows_br.then_:
                s1 = [c_rows, c_rows, c_rows, c1, c_rows]
                s128 = [c_rows_x_128, c_rows_x_128, c_rows_x_128, c128, c1]
                sh1 = [c1, c1, c1, c_rows, c1]
                sh128 = [c1, c1, c1, c_rows, c128]

                oldmax_view = pto.make_tensor_view(oldmax, shape=sh1, strides=s1)
                oldsum_view = pto.make_tensor_view(oldsum, shape=sh1, strides=s1)
                scores_view = pto.make_tensor_view(scores, shape=sh128, strides=s128)
                newmax_view = pto.make_tensor_view(newmax, shape=sh1, strides=s1)
                newsum_view = pto.make_tensor_view(newsum, shape=sh1, strides=s1)
                expmax_view = pto.make_tensor_view(expmax, shape=sh1, strides=s1)
                out_view = pto.make_tensor_view(out, shape=sh128, strides=s128)

                off = [c0, c0, c0, row_base, c0]
                z1 = [c1, c1, c1, row_count, c1]
                zs = [c1, c1, c1, row_count, c_seq]

                oldmax_part = pto.partition_view(oldmax_view, offsets=off, sizes=z1)
                oldsum_part = pto.partition_view(oldsum_view, offsets=off, sizes=z1)
                scores_part = pto.partition_view(scores_view, offsets=off, sizes=zs)
                newmax_part = pto.partition_view(newmax_view, offsets=off, sizes=z1)
                newsum_part = pto.partition_view(newsum_view, offsets=off, sizes=z1)
                expmax_part = pto.partition_view(expmax_view, offsets=off, sizes=z1)
                out_part = pto.partition_view(out_view, offsets=off, sizes=zs)

                tile_col = pto.tile_buf_type([8, 1], pto.float32, [-1, 1], blayout="ColMajor")
                tile_w = pto.tile_buf_type([8, 128], pto.float32, [-1, -1])

                oldmax_tile = pto.alloc_tile(tile_col, addr=c0_i64, valid_row=row_count)
                oldsum_tile = pto.alloc_tile(tile_col, addr=c128_i64, valid_row=row_count)
                scores_tile = pto.alloc_tile(tile_w, addr=c256_i64, valid_row=row_count, valid_col=c_seq)
                out_tile = pto.alloc_tile(tile_w, addr=c8448_i64, valid_row=row_count, valid_col=c_seq)
                newmax_tile = pto.alloc_tile(tile_col, addr=c16640_i64, valid_row=row_count)
                newsum_tile = pto.alloc_tile(tile_col, addr=c16768_i64, valid_row=row_count)
                expmax_tile = pto.alloc_tile(tile_col, addr=c16896_i64, valid_row=row_count)

                pto.tile.load(oldmax_part, oldmax_tile)
                pto.tile.load(oldsum_part, oldsum_tile)
                pto.tile.load(scores_part, scores_tile)

                pto.set_flag("MTE2", "V", event_id=0)
                pto.wait_flag("MTE2", "V", event_id=0)

                with pto.vecscope():
                    ptr_ub = pto.ptr(pto.float32, "ub")
                    vf32 = pto.vreg_type(64, pto.float32)

                    ub_om = pto.as_ptr(oldmax_tile, ptr_ub)
                    ub_os = pto.as_ptr(oldsum_tile, ptr_ub)
                    ub_scores = pto.as_ptr(scores_tile, ptr_ub)
                    ub_out = pto.as_ptr(out_tile, ptr_ub)
                    ub_nm = pto.as_ptr(newmax_tile, ptr_ub)
                    ub_ns = pto.as_ptr(newsum_tile, ptr_ub)
                    ub_em = pto.as_ptr(expmax_tile, ptr_ub)

                    active = pto.pset_b32(pto.MaskPattern.ALL)
                    one_mask, _ = pto.plt_b32(c1_i32)

                    with pto.for_(c0, row_count, step=c1) as row:
                        row_scores = s.muli(row, c128)
                        oldmax_bc = pto.vbrc_load(ub_om, row, vf32)
                        oldsum_bc = pto.vbrc_load(ub_os, row, vf32)

                        with pto.for_(c0, c128, step=c64, iter_args=(oldmax_bc, oldsum_bc)) as softmax_loop:
                            chunk = softmax_loop.iv
                            running_max, running_sum = softmax_loop.iter_args

                            chunk_i32 = s.index_cast(pto.int32, chunk)
                            remaining_cols = s.subi(c_seq_i32, chunk_i32)
                            has_chunk = remaining_cols > c0_i32

                            with pto.if_(has_chunk) as br:
                                with br.then_:
                                    chunk_mask, _ = pto.plt_b32(remaining_cols)
                                    chunk_base = s.addi(row_scores, chunk)
                                    vec = pto.vlds(ub_scores, chunk_base, vf32)
                                    chunk_max = pto.vcmax(vec, chunk_mask)
                                    chunk_max_bc = pto.vdup(chunk_max, active, position="LOWEST")
                                    merged_max = pto.vmax(running_max, chunk_max_bc, active)
                                    scaled_running = pto.vexpdif(running_max, merged_max, active)
                                    running_sum_scaled = pto.vmul(scaled_running, running_sum, active)
                                    chunk_exp = pto.vexpdif(vec, merged_max, chunk_mask)
                                    chunk_sum = pto.vcadd(chunk_exp, chunk_mask)
                                    chunk_sum_bc = pto.vdup(chunk_sum, active, position="LOWEST")
                                    merged_sum = pto.vadd(running_sum_scaled, chunk_sum_bc, active)
                                    br.assign(next_max=merged_max, next_sum=merged_sum)
                                with br.else_:
                                    br.assign(next_max=running_max, next_sum=running_sum)
                            pto.yield_(br.next_max, br.next_sum)

                        final_max, final_sum = softmax_loop.results

                        raw_em = pto.vexpdif(oldmax_bc, final_max, active)
                        scaled_oldsum = pto.vmul(raw_em, oldsum_bc, active)
                        expmax = pto.vdiv(scaled_oldsum, final_sum, active)

                        pto.vsts_1pt(final_max, ub_nm, row, one_mask)
                        pto.vsts_1pt(final_sum, ub_ns, row, one_mask)
                        pto.vsts_1pt(expmax, ub_em, row, one_mask)

                        with pto.for_(c0, c128, step=c64) as chunk2:
                            rem2 = s.subi(c_seq_i32, s.index_cast(pto.int32, chunk2))
                            has_chunk2 = rem2 > c0_i32
                            with pto.if_(has_chunk2) as br2:
                                with br2.then_:
                                    cmask2, _ = pto.plt_b32(rem2)
                                    cbase2 = s.addi(row_scores, chunk2)
                                    vec2 = pto.vlds(ub_scores, cbase2, vf32)
                                    exp2 = pto.vexpdif(vec2, final_max, cmask2)
                                    out2 = pto.vdiv(exp2, final_sum, cmask2)
                                    pto.vsts(out2, ub_out, cbase2, cmask2)

                pto.set_flag("V", "MTE3", event_id=0)
                pto.wait_flag("V", "MTE3", event_id=0)

                pto.tile.store(newmax_tile, newmax_part)
                pto.tile.store(newsum_tile, newsum_part)
                pto.tile.store(expmax_tile, expmax_part)
                pto.tile.store(out_tile, out_part)

        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


FLASH_SOFTMAX_ROWS8_SEQ128 = _make_flash_attention_softmax_kernel(
    "flash_attention_softmax_rows8_seq128",
    rows=8,
    seq=128,
)
FLASH_SOFTMAX_ROWS17_SEQ96 = _make_flash_attention_softmax_kernel(
    "flash_attention_softmax_rows17_seq96",
    rows=17,
    seq=96,
)

KERNELS = (
    FLASH_SOFTMAX_ROWS8_SEQ128,
    FLASH_SOFTMAX_ROWS17_SEQ96,
)

CASES = [
    {
        "name": "rows8_seq128",
        "kernel": FLASH_SOFTMAX_ROWS8_SEQ128,
        "rows": 8,
        "seq": 128,
    },
    {
        "name": "rows17_seq96",
        "kernel": FLASH_SOFTMAX_ROWS17_SEQ96,
        "rows": 17,
        "seq": 96,
    },
]


def emit_mlir():
    return pto.merge_jit_modules(*KERNELS)


def reference_online_softmax_update(oldmax: np.ndarray, oldsum: np.ndarray, scores: np.ndarray, seq: int):
    rows = oldmax.shape[0]
    newmax = np.empty_like(oldmax)
    newsum = np.empty_like(oldsum)
    expmax = np.empty_like(oldsum)
    out = np.full_like(scores, np.nan)

    for row in range(rows):
        m_prev = float(oldmax[row, 0])
        l_prev = float(oldsum[row, 0])
        row_scores = scores[row, :seq]
        m_next = max(m_prev, float(np.max(row_scores)))
        shifted = np.exp(row_scores - m_next)
        l_scaled = l_prev * np.exp(m_prev - m_next)
        l_next = l_scaled + float(np.sum(shifted))

        newmax[row, 0] = m_next
        newsum[row, 0] = l_next
        expmax[row, 0] = l_scaled / l_next
        out[row, :seq] = shifted / l_next

    return newmax, newsum, expmax, out


def init_runtime():
    import torch
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)
    return torch


def npu_stream(torch):
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def make_case_inputs(case: dict[str, object]):
    rows = int(case["rows"])
    seq = int(case["seq"])
    rng = np.random.RandomState(hash(case["name"]) & 0xFFFFFFFF)

    oldmax = rng.uniform(-2.0, 2.0, size=(rows, 1)).astype(np.float32)
    oldsum = rng.uniform(0.25, 3.0, size=(rows, 1)).astype(np.float32)
    scores = np.full((rows, _PHYSICAL_COLS), -1000.0, dtype=np.float32)
    scores[:, :seq] = rng.uniform(-4.0, 4.0, size=(rows, seq)).astype(np.float32)

    newmax = np.full((rows, 1), np.nan, dtype=np.float32)
    newsum = np.full((rows, 1), np.nan, dtype=np.float32)
    expmax = np.full((rows, 1), np.nan, dtype=np.float32)
    out = np.full((rows, _PHYSICAL_COLS), np.nan, dtype=np.float32)

    return oldmax, oldsum, scores, newmax, newsum, expmax, out


def run_case(case: dict[str, object], torch) -> None:
    rows = int(case["rows"])
    seq = int(case["seq"])
    grid = (rows + _ROWS_PER_BLOCK - 1) // _ROWS_PER_BLOCK
    oldmax, oldsum, scores, newmax, newsum, expmax, out = make_case_inputs(case)
    ref_newmax, ref_newsum, ref_expmax, ref_out = reference_online_softmax_update(
        oldmax,
        oldsum,
        scores,
        seq,
    )

    oldmax_t = torch.from_numpy(oldmax).to(_DEVICE)
    oldsum_t = torch.from_numpy(oldsum).to(_DEVICE)
    scores_t = torch.from_numpy(scores).to(_DEVICE)
    newmax_t = torch.from_numpy(newmax).to(_DEVICE)
    newsum_t = torch.from_numpy(newsum).to(_DEVICE)
    expmax_t = torch.from_numpy(expmax).to(_DEVICE)
    out_t = torch.from_numpy(out).to(_DEVICE)
    stream = npu_stream(torch)

    t0 = time.perf_counter()
    compiled = case["kernel"].compile()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled[grid, stream](
        oldmax_t,
        oldsum_t,
        scores_t,
        newmax_t,
        newsum_t,
        expmax_t,
        out_t,
    )
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    np.testing.assert_allclose(newmax_t.cpu().numpy(), ref_newmax, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(newsum_t.cpu().numpy(), ref_newsum, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(expmax_t.cpu().numpy(), ref_expmax, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out_t.cpu().numpy()[:, :seq], ref_out[:, :seq], rtol=1e-5, atol=1e-5)
    if seq < _PHYSICAL_COLS:
        assert np.isnan(out_t.cpu().numpy()[:, seq:]).all(), "tail columns should remain untouched"

    print(
        f"PASS {case['name']}  "
        f"compile={compile_s:.3f}s launch={launch_s:.3f}s"
    )


def test_flash_attention_softmax() -> None:
    torch = init_runtime()
    for case in CASES:
        run_case(case, torch)
    print("All cases passed.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-mlir",
        action="store_true",
        help="print the merged MLIR module and exit",
    )
    args = parser.parse_args(argv)

    if args.emit_mlir:
        print(emit_mlir())
        return 0

    test_flash_attention_softmax()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
