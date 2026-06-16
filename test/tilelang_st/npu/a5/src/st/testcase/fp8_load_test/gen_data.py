#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FP8 Load Test - Generate test data
"""

import os
import numpy as np
from ml_dtypes import float8_e4m3fn

def gen_fp8_data(shape, filename):
    """Generate random FP8 data and save to file"""
    m, n = shape
    # Generate random int8 values and cast to float8_e4m3fn
    data = np.random.randint(-128, 127, size=(m, n), dtype=np.int8)
    data_fp8 = data.view(float8_e4m3fn)
    
    # Save to file
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data_fp8.tofile(filename)
    print(f"[INFO] Generated {filename}: shape={shape}, dtype={data_fp8.dtype}")
    return data_fp8

def main():
    test_cases = [
        ("fp8_load_32x64", (32, 64)),
        ("fp8_load_64x64", (64, 64)),
        ("fp8_load_128x64", (128, 64)),
    ]
    
    for name, shape in test_cases:
        case_dir = name
        os.makedirs(case_dir, exist_ok=True)
        input_file = os.path.join(case_dir, "input.bin")
        gen_fp8_data(shape, input_file)
    
    print("[INFO] All test data generated successfully")

if __name__ == "__main__":
    main()
