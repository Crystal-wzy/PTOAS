#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

CASES = [
    {
        "name": "nd_f32_16x64",
        "dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-6,
    },
    {
        "name": "dn_f32_16x64",
        "dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-6,
    },
    {
        "name": "nz_f32_128x128",
        "dtype": np.float32,
        "shape": (128, 128),
        "valid_shape": (128, 128),
        "eps": 1e-6,
    },
]
