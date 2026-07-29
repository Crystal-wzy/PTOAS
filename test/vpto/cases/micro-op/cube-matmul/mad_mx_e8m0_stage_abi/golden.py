#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
from pathlib import Path

import numpy as np

M = 256
N = 256
K = 256
TILE_K_SUB = 128
SCALE_GROUP_K = 32
SCALE_X_BLOCK = 16
SCALE_SRC_STRIDE = 8
SCALE_PAIR_COUNT = K // (2 * SCALE_GROUP_K)


def e8m0_to_f32(bits: np.ndarray) -> np.ndarray:
    return np.exp2(bits.astype(np.int32) - 127).astype(np.float32)


def pack_scale_pairs(scale_bytes: np.ndarray) -> np.ndarray:
    """Pack [MN, K/32] E8M0 bytes into a stride-8 uint16 SF fractal."""
    assert scale_bytes.ndim == 2
    assert scale_bytes.shape[1] == K // SCALE_GROUP_K

    # The MX source coordinates address pair-packed uint16 values. Two adjacent
    # K/32 E8M0 scales form each uint16, and 16 logical M/N lanes form the
    # innermost physical fractal. Logical [MN, K/32] therefore occupies
    # [MN/16, padded K/64, 16], with the K/64 axis padded to SCALE_SRC_STRIDE.
    # E8M0 value 127 is identity, so it safely fills the padding. The resulting
    # source stride is eight uint16 values per 16-lane fractal.
    mn = scale_bytes.shape[0]
    pairs = np.ascontiguousarray(
        scale_bytes.reshape(mn, SCALE_PAIR_COUNT, 2)
    ).view(np.uint16).reshape(mn, SCALE_PAIR_COUNT)
    assert mn % SCALE_X_BLOCK == 0
    packed = np.full(
        (mn // SCALE_X_BLOCK, SCALE_SRC_STRIDE, SCALE_X_BLOCK),
        0x7F7F,
        dtype=np.uint16,
    )
    packed[:, :SCALE_PAIR_COUNT, :] = pairs.reshape(
        mn // SCALE_X_BLOCK, SCALE_X_BLOCK, SCALE_PAIR_COUNT
    ).transpose(0, 2, 1)
    return packed.reshape(-1)


def pack_a_scale(scale_bytes: np.ndarray) -> np.ndarray:
    """Pack A's per-row E8M0 scales into uint16 GM storage."""
    assert scale_bytes.shape == (M, K // SCALE_GROUP_K)
    return pack_scale_pairs(scale_bytes)


def pack_b_scale(scale_bytes: np.ndarray) -> np.ndarray:
    """Pack B's per-column E8M0 scales into uint16 GM storage."""
    assert scale_bytes.shape == (K // SCALE_GROUP_K, N)
    return pack_scale_pairs(scale_bytes.T)


def generate(output_dir: Path) -> None:
    # The data is exactly 1.0 in E4M3. Scale values vary by every logical M/N
    # lane and K/32 group, exposing wrong source stride, y-start, or stage-1
    # L0 scale association.
    a = np.full((M, K), 0x38, dtype=np.uint8)
    b = np.full((K, N), 0x38, dtype=np.uint8)
    group_ids = np.arange(K // SCALE_GROUP_K, dtype=np.uint8)
    m_ids = np.arange(M, dtype=np.uint8)[:, None]
    n_ids = np.arange(N, dtype=np.uint8)[None, :]
    a_scale = (126 + ((m_ids + group_ids) % 3)).astype(np.uint8)
    b_scale = (126 + ((group_ids[:, None] + 2 * n_ids) % 3)).astype(np.uint8)
    packed_a = pack_a_scale(a_scale)
    packed_b = pack_b_scale(b_scale)

    reference = np.zeros((M, N), dtype=np.float32)
    for group in range(K // SCALE_GROUP_K):
        reference += (
            SCALE_GROUP_K
            * e8m0_to_f32(a_scale[:, group : group + 1])
            * e8m0_to_f32(b_scale[group : group + 1, :])
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    a.reshape(-1).tofile(output_dir / "v1.bin")
    b.reshape(-1).tofile(output_dir / "v2.bin")
    np.zeros((M, N), dtype=np.float32).reshape(-1).tofile(output_dir / "v3.bin")
    packed_a.reshape(-1).tofile(output_dir / "v4.bin")
    packed_b.reshape(-1).tofile(output_dir / "v5.bin")
    reference.reshape(-1).tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
