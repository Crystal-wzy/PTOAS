# e2e tests
End-to-end numerical-correctness tests for PTODSL vector ops.

## Quick start

```bash
# From repo root inside the hardware docker image:
cd /workspace && PYTHONPATH=/workspace/ptodsl python3 -m pytest ptodsl/tests/e2e/test_binary_elementwise.py -v

# Run a single test
python3 -m pytest ptodsl/tests/e2e/test_binary_elementwise.py -v -k "add-float32-1x64"

# Run all f32 tests
python3 -m pytest ptodsl/tests/e2e/test_binary_elementwise.py -v -k "float32"

# Run only div tests
python3 -m pytest ptodsl/tests/e2e/test_binary_elementwise.py -v -k "div"
```

## Hardware and backend selection
### A3 vpto backend
This is the default, or can be selected with `--backend vpto --vpto-target a3`.

### A3 emitc backend
This can be selected with  `--backend emitc --vpto-target a3`.

### A5 vpto backend
This can be selected with  `--backend vpto --vpto-target a5`.

## Test matrix

### Binary elementwise (tadd/tsub/tmul/tdiv)

| Category | Ops | Dtypes | Shapes | Total |
|----------|-----|--------|--------|-------|
| f32 | add, sub, mul, div | float32 | 12 | 48 |
| f16 | add, sub, mul, div | float16 | 6 | 24 |

### Shape coverage (exercises lowering code paths)

**f32** (`elementsPerRepeat=64`):

| Shape | Lowering Path |
|-------|---------------|
| (1, 32) | modeSmall single-row |
| (4, 32) | modeSmall multi-row |
| (11, 32) | modeSmall multi-row odd |
| (1, 64) | modeNorm1L single-repeat |
| (1, 128) | modeNorm1L multi-repeat aligned |
| (64, 64) | modeNorm1L square aligned |
| (16, 64) | modeNorm1L multi-row aligned |
| (16, 128) | modeNorm1L 16x128 |
| (4, 256) | modeNorm1L 4x256 |
| (1, 1024) | modeNorm1L 16-repeat |
| (16, 256) | modeNorm1L 16x256 |
| (32, 32) | modeSmall 32x32 square small |
| (1, 200) | modeNorm1L 1x200 tail |
| (32, 100) | modeCount1L nonVLAligned odd rows |
| (4, 200) | modeCount1L 4x200 |
| (32, 32) | modeSmall 32x32 square small |

**f16** (`elementsPerRepeat=128`):

| Shape | Lowering Path |
|-------|---------------|
| (1, 64) | modeSmall 1x64 |
| (4, 64) | modeSmall 4x64 |
| (1, 128) | modeNorm1L single-repeat |
| (16, 128) | modeNorm1L 16x128 |
| (64, 128) | modeNorm1L multi-row |
| (1, 512) | modeNorm1L 4-repeat |

## Adding new test categories

1. Create a new test file in `ptodsl/tests/e2e/` (e.g. `test_elementwise_unary.py`)
2. Add a kernel builder function in `common.py`
3. Parametrize over ops, dtypes, and shapes using the `make_binary_kernel` / `launch_and_check` pattern
4. Run with pytest

## Requirements

- NPU device with torch_npu installed
- ptoas and bisheng on PATH
- Properly built PTOAS with Python bindings (MLIR 19.1)
