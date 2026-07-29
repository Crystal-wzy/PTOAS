#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


def main() -> None:
    golden = np.fromfile("golden_v3.bin", dtype=np.float32)
    output = np.fromfile("v3.bin", dtype=np.float32)
    if golden.shape != output.shape:
        raise SystemExit(f"[ERROR] shape mismatch: {golden.shape} vs {output.shape}")
    close = np.isclose(golden, output, atol=1e-3, rtol=1e-3)
    if not np.all(close):
        index = int(np.flatnonzero(~close)[0])
        raise SystemExit(
            f"[ERROR] mismatch at {index}: golden={golden[index]}, output={output[index]}"
        )
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
