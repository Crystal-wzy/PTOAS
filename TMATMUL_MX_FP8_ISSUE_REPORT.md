# tmatmul_mx FP8 精度问题报告

## 问题概述

**问题类型**: 数值精度错误  
**影响范围**: tmatmul_mx 算子在使用 FP8 数据类型时计算结果完全错误  
**严重程度**: 高 - 导致所有 FP8 tmatmul_mx 测试用例失败  

## 问题描述

### 现象

在使用 PTOAS 编译器编译 tmatmul_mx 算子时，**FP8 数据类型的计算结果与 golden 值完全不匹配**，而相同配置的 FP4 数据类型测试用例能够正常工作。

### 具体表现

| 数据类型 | 测试用例 | 正确率 | 状态 |
|---------|---------|--------|------|
| FP4 (e2m1) | fp4_e2m1_128x64x64 | ~95% | ✅ 正常 |
| FP4 (e1m2) | fp4_e1m2_128x64x64 | ~93% | ✅ 正常 |
| FP8 (e5m2) | fp8_e5m2_128x64x64 | 0% | ❌ 完全错误 |
| FP8 (e4m3) | fp8_e4m3_128x64x64 | 0% | ❌ 完全错误 |

### 错误示例

以 `fp8_e5m2_128x64x64` 为例：

```
Golden 值 (前 8 个元素):
[-1274. -1612. -1410.  1174.   -56.   -84.   276.  1854.]

实际输出 (前 8 个元素):
[ 1120.   396.   208.   544.  -896.  1920.  1984. -1024.]

差异: 完全不一致，数值符号和大小都错误
```

## 复现步骤

### 1. 编译 PTOAS

```bash
cd /home/wujiajun/llvm-workspace/00_pto/PTOAS/build
ninja ptoas
```

### 2. 运行测试

```bash
cd /home/wujiajun/llvm-workspace/00_pto/PTOAS
python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t tmatmul_mx
```

### 3. 观察结果

- FP4 测试用例：大部分通过（精度在可接受范围内）
- FP8 测试用例：全部失败，计算结果与 golden 值完全不匹配

## 已尝试的解决方案

### 方案 1: 修改 DMA Intrinsic 选择

**假设**: FP8 使用了错误的 DMA intrinsic

**实施**: 
- 修改 `lib/PTO/Transforms/VPTOLLVMEmitter.cpp` 和 `VPTOCANN900LLVMEmitter.cpp`
- 尝试让 FP8 使用与 FP4 相同的 "u8" intrinsic

**结果**: ❌ 失败  
**原因**: FP8 确实需要使用特定的 intrinsic（e4m3, e5m2），使用通用 intrinsic 会导致结果更差（准确率从 ~70% 降到 0.07%）

### 方案 2: 添加 TFILLPAD 操作

**假设**: 缺少 TFILLPAD 操作导致 padding 区域包含垃圾数据

**实施**:
- 检查 pto-isa 的实现，发现使用了 TFILLPAD
- 尝试在 PTOAS 中添加 tfillpad 操作

**结果**: ❌ 失败  
**原因**: PTOAS 的 tfillpad 是高级 tile 操作，而 tmatmul_mx.pto 使用 VPTO 低级风格，无法直接使用

### 方案 3: 分块计算

**假设**: MMAD.MX intrinsic 对 FP8 有 N 维度限制（N > 32 时只能计算前 32 列）

**实施**:
- 修改 `fp8_e5m2_128x64x64` kernel
- 将 N=64 分成两次 N=32 的计算
- 第一次计算列 0-31，第二次计算列 32-63

**结果**: ❌ 失败  
**原因**: 前 32 列的计算结果仍然完全错误（0% 正确率），说明问题不在 N 维度限制

### 方案 4: 预填充 Padding 区域

**假设**: Padding 区域未初始化为零导致计算错误

**实施**:
- 修改 `gen_data.py`，确保 padding 区域初始化为零

**结果**: ❌ 失败  
**原因**: 检查发现 padding 区域已经是零，问题不在这里

## 技术分析

### 对比 FP4 和 FP8 的实现

#### FP4 Kernel (工作正常)

```mlir
// 数据加载
pto.mte_gm_l1 %a_gm, %l1_a_data, %c1024_i64 nburst(%c8_i64, %c0_i64, %c0_i64)
  : !pto.ptr<!pto.f4E2M1x2, gm>, !pto.ptr<!pto.f4E2M1x2, l1>, i64, i64, i64, i64

// 移动到 L0
pto.mte_l1_l0a %l1_a_data, %l0a, %c128_i64, %c64_i64, %c0_i64, %c0_i64
  : !pto.ptr<!pto.f4E2M1x2, l1>, !pto.ptr<!pto.f4E2M1x2, l0a>, i64, i64, i64, i64

// 加载 scale
pto.mte_l1_l0a_mx %l1_a_scale, %l0a, %c128_i64, %c64_i64
  : !pto.ptr<!pto.f4E2M1x2, l1>, !pto.ptr<!pto.f4E2M1x2, l0a>, i64, i64

// 计算
pto.mad_mx %l0a, %l0b, %l0c, %c128_i64, %c64_i64, %c64_i64 unit_flag(check_only) disable_gemv sat
  : !pto.ptr<!pto.f4E2M1x2, l0a>, !pto.ptr<!pto.f4E2M1x2, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
```

#### FP8 Kernel (完全错误)

```mlir
// 数据加载
pto.mte_gm_l1 %a_gm, %l1_a_data, %c1024_i64 nburst(%c8_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f8E5M2, gm>, !pto.ptr<f8E5M2, l1>, i64, i64, i64, i64

// 移动到 L0
pto.mte_l1_l0a %l1_a_data, %l0a, %c128_i64, %c64_i64, %c0_i64, %c0_i64
  : !pto.ptr<f8E5M2, l1>, !pto.ptr<f8E5M2, l0a>, i64, i64, i64, i64

// 加载 scale
pto.mte_l1_l0a_mx %l1_a_scale, %l0a, %c128_i64, %c64_i64
  : !pto.ptr<f8E5M2, l1>, !pto.ptr<f8E5M2, l0a>, i64, i64

// 计算
pto.mad_mx %l0a, %l0b, %l0c, %c128_i64, %c64_i64, %c64_i64 unit_flag(check_only) disable_gemv sat
  : !pto.ptr<f8E5M2, l0a>, !pto.ptr<f8E5M2, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
```

**观察**: 两者的结构完全相同，只是数据类型不同（`!pto.f4E2M1x2` vs `f8E5M2`）

### 编译器生成的 Intrinsic

#### FP4

```llvm
call void @llvm.hivm.MOV.OUT.TO.L1.ALIGN.V2.u8.DV(...)
call void @llvm.hivm.MMAD.MX.e2m1x2e2m1x2(...)
```

#### FP8

```llvm
call void @llvm.hivm.MOV.OUT.TO.L1.ALIGN.V2.e5m2.DV(...)
call void @llvm.hivm.MMAD.MX.e5m2e5m2(...)
```

**观察**: FP8 使用了特定的 intrinsic，而不是通用的 u8 intrinsic

## 可能的原因

### 1. Scale 数据处理错误

**假设**: FP8 的 MX scale 加载或应用方式与 FP4 不同

**证据**:
- FP4 和 FP8 都使用 `mte_l1_l0a_mx` 和 `mte_l1_l0b_mx` 加载 scale
- 但 FP8 的 scale 格式可能不同（e8m0 vs u8）

**需要验证**:
- 检查 scale 数据的生成和加载是否正确
- 对比 pto-isa 中 FP4 和 FP8 的 scale 处理方式

### 2. FP8 数据布局问题

**假设**: FP8 数据在 L1/L0 中的布局与 FP4 不同

**证据**:
- FP4 使用打包格式（2 个 fp4 打包成 1 个 byte）
- FP8 使用标准格式（1 个 fp8 = 1 个 byte）
- 可能导致 DMA 传输或 L0 加载时的对齐问题

**需要验证**:
- 检查 L1 和 L0 中的实际数据内容
- 对比 pto-isa 的数据布局

### 3. mad_mx 参数错误

**假设**: m, n, k 参数对于 FP8 不适用

**证据**:
- FP4 和 FP8 使用相同的参数（m=128, n=64, k=64）
- 但 FP8 可能需要不同的参数解释

**需要验证**:
- 查阅硬件文档，确认 mad_mx 对于 FP8 的参数要求
- 尝试不同的参数组合

### 4. 硬件 Bug

**假设**: MMAD.MX intrinsic 对于 FP8 存在硬件 bug

**证据**:
- 所有尝试的软件层面修复都失败
- FP4 工作正常，说明硬件基本功能正常
- FP8 完全错误，可能是特定于 FP8 的硬件问题

**需要验证**:
- 联系硬件团队确认
- 检查是否有已知的 FP8 相关 errata

## 影响范围

### 受影响的测试用例

- `fp8_e5m2_128x64x64`
- `fp8_e4m3_128x64x64`
- `fp8_e4m3_127x72x64`
- `fp8_e4m3_e5m2_128x110x63`
- 所有使用 FP8 数据类型的 tmatmul_mx 测试用例

### 不受影响的测试用例

- 所有 FP4 数据类型的 tmatmul_mx 测试用例
- 其他非 MX 的 tmatmul 测试用例

## 建议的下一步行动

### 短期（1-2 周）

1. **详细检查 Scale 数据**
   - 对比 FP4 和 FP8 的 scale 生成和加载方式
   - 检查 scale 格式是否正确（e8m0 vs u8）
   - 验证 scale 在 L0 中的实际内容

2. **查阅硬件文档**
   - 确认 MMAD.MX 对于 FP8 的具体要求
   - 检查是否有特殊的对齐或格式要求
   - 查找已知的 FP8 相关问题

3. **联系硬件团队**
   - 报告问题现象
   - 询问是否有已知的 FP8 bug
   - 请求硬件层面的诊断支持

### 中期（1-2 个月）

1. **实现 Workaround**
   - 如果无法修复，考虑避免使用 FP8 same-type 组合
   - 或者将 FP8 转换为 FP4 进行计算
   - 评估 workaround 对性能的影响

2. **扩展测试覆盖**
   - 添加更多 FP8 测试用例
   - 测试不同的 M, N, K 组合
   - 测试 mixed-type FP8（e4m3 x e5m2）

### 长期（3-6 个月）

1. **优化 FP8 支持**
   - 如果硬件问题得到修复，优化编译器生成代码
   - 添加自动检测和 fallback 机制
   - 提供最佳实践指南

2. **文档化限制**
   - 在用户文档中说明 FP8 的限制
   - 提供替代方案建议
   - 更新 API 文档

## 相关文件

### 测试文件

- `/home/wujiajun/llvm-workspace/00_pto/PTOAS/test/tilelang_st/npu/a5/src/st/testcase/tmatmul_mx/tmatmul_mx.pto`
- `/home/wujiajun/llvm-workspace/00_pto/PTOAS/test/tilelang_st/npu/a5/src/st/testcase/tmatmul_mx/gen_data.py`
- `/home/wujiajun/llvm-workspace/00_pto/PTOAS/test/tilelang_st/npu/a5/src/st/testcase/tmatmul_mx/compare.py`

### 编译器文件

- `/home/wujiajun/llvm-workspace/00_pto/PTOAS/lib/PTO/Transforms/VPTOLLVMEmitter.cpp`
- `/home/wujiajun/llvm-workspace/00_pto/PTOAS/lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp`
- `/home/wujiajun/llvm-workspace/00_pto/PTOAS/lib/PTO/IR/VPTO.cpp`

### 调试文档

- `/home/wujiajun/llvm-workspace/00_pto/PTOAS/TMATMUL_MX_DEBUG.md`

## 联系信息

**问题报告人**: [您的姓名]  
**报告日期**: 2026-06-16  
**相关团队**: PTOAS 编译器团队、硬件团队  

---

**备注**: 本文档将持续更新，记录问题的调查进展和解决方案。
