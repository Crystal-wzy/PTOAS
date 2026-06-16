#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FP8 Load Test - Compare input and output data
"""

import os
import sys
import numpy as np
from ml_dtypes import float8_e4m3fn

def compare_data(case_name, shape):
    """Compare input and output data for a test case"""
    m, n = shape
    case_dir = case_name
    input_file = os.path.join(case_dir, "input.bin")
    output_file = os.path.join(case_dir, "output.bin")
    
    if not os.path.exists(input_file):
        print(f"[ERROR] {input_file} not found")
        return False
    
    if not os.path.exists(output_file):
        print(f"[ERROR] {output_file} not found")
        return False
    
    # Load data
    input_data = np.fromfile(input_file, dtype=float8_e4m3fn).reshape(m, n)
    output_data = np.fromfile(output_file, dtype=np.float32).reshape(m, n)
    
    # Convert input to float32 for comparison
    input_as_float32 = input_data.astype(np.float32)
    
    # Compare with tolerance (FP8 has limited precision)
    if np.allclose(input_as_float32, output_data, rtol=1e-2, atol=1e-2):
        print(f"[PASS] {case_name}: All {m*n} elements match (within tolerance)")
        return True
    else:
        # Find mismatches
        mismatch_mask = ~np.isclose(input_as_float32, output_data, rtol=1e-2, atol=1e-2)
        num_mismatches = np.sum(mismatch_mask)
        total_elements = m * n
        accuracy = (total_elements - num_mismatches) / total_elements * 100
        
        print(f"[FAIL] {case_name}: {num_mismatches}/{total_elements} mismatches ({accuracy:.2f}% accuracy)")
        
        # Show first few mismatches
        mismatch_indices = np.argwhere(mismatch_mask)
        for i, (row, col) in enumerate(mismatch_indices[:5]):
            print(f"  [{row},{col}]: input={input_as_float32[row,col]:.6f}, output={output_data[row,col]:.6f}")
        
        if num_mismatches > 5:
            print(f"  ... and {num_mismatches - 5} more mismatches")
        
        return False

def main():
    test_cases = [
        ("fp8_load_32x64", (32, 64)),
        ("fp8_load_64x64", (64, 64)),
        ("fp8_load_128x64", (128, 64)),
    ]
    
    all_passed = True
    for name, shape in test_cases:
        if not compare_data(name, shape):
            all_passed = False
    
    if all_passed:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[FAILURE] Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
