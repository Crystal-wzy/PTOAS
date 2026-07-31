#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

from ptodsl import pto


@pto.jit(target="a5", backend="vpto", mode="explicit")
def legal_group_probe():
    src = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    dst = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    offset = pto.const(0, dtype=pto.index)
    stride = pto.const(4, dtype=pto.index)
    mask = pto.vmi.create_mask(16, size=64, group=4)
    value = pto.vmi.vload(src.as_ptr(), offset, size=64, group=4, stride=stride)
    slots = pto.vmi.vcmax(value, mask, group=4)
    dense = pto.vmi.vbrc(slots, size=64, group=4)
    pto.vmi.vstore(dense, dst.as_ptr(), offset, group=4, stride=stride)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def invalid_group_probe():
    _ = pto.vmi.create_mask(1, size=64, group=3)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def invalid_reduce_group_probe():
    value = pto.vmi.vbrc(pto.f32(0.0), size=64)
    mask = pto.vmi.create_mask(64, size=64)
    _ = pto.vmi.vcmax(value, mask, group=16)


def main() -> None:
    for lanes in (1, 2, 4, 8, 64, 128, 256):
        pto.vmi.vreg(lanes, pto.f32)
        pto.vmi.mask(lanes)
    for lanes in (3, 16, 32, 512):
        try:
            pto.vmi.vreg(lanes, pto.f32)
        except ValueError as exc:
            assert "1, 2, 4, 8, 64, 128, 256" in str(exc)
        else:
            raise AssertionError(f"PTODSL VMI vreg must reject lanes={lanes}")
        try:
            pto.vmi.mask(lanes)
        except ValueError as exc:
            assert "1, 2, 4, 8, 64, 128, 256" in str(exc)
        else:
            raise AssertionError(f"PTODSL VMI mask must reject lanes={lanes}")

    text = legal_group_probe.compile().mlir_text()
    for spelling in (
        "pto.vmi.create_group_mask",
        "pto.vmi.vload",
        "pto.vmi.vcmax",
        "pto.vmi.vbrc",
        "pto.vmi.vstore",
    ):
        assert spelling in text
    assert text.count("group = 4") >= 4

    try:
        invalid_group_probe.compile()
    except ValueError as exc:
        assert "size to be divisible by group" in str(exc)
    else:
        raise AssertionError("grouped mask must reject a group that does not divide size")

    try:
        invalid_reduce_group_probe.compile()
    except ValueError as exc:
        assert "group to be one of 1, 2, 4, 8" in str(exc)
    else:
        raise AssertionError("PTODSL grouped reduce must reject group=16")


if __name__ == "__main__":
    main()
