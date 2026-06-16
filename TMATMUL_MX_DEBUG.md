# tmatmul_mx 测试问题调试记录

**创建时间**: 2026-06-16  
**最后更新**: 2026-06-16  
**状态**: 进行中 - 需要继续调试 DMA 数据加载问题

---

## 1. 问题概述

tmatmul_mx 测试用例（19 个）全部能够成功执行（无崩溃、无 DMA 断言错误），但所有比较都失败。

**测试结果**:
- ✅ 编译通过
- ✅ 数据生成通过
- ✅ 所有 19 个 case 执行成功
- ❌ 所有 19 个 case 比较失败

---

## 2. 已完成的工作

### 2.1 编译器修改（3 个文件）

#### 文件 1: `lib/PTO/IR/VPTO.cpp`
- **位置**: 第 786 行
- **修改**: `isMxElementType` 函数
- **原因**: 原实现只接受 `f8E4M3FN`，需要扩展支持其他 MX 类型
- **修改内容**:
```cpp
// 修改前
static bool isMxElementType(Type type) { return isa<Float8E4M3FNType>(type); }

// 修改后
static bool isMxElementType(Type type) {
  return isa<Float8E4M3FNType, Float8E5M2Type>(type) ||
         isa<pto::F4E1M2x2Type, pto::F4E2M1x2Type>(type);
}
```
- **错误消息更新**: 第 790 行，更新为更准确的类型列表

#### 文件 2: `lib/PTO/Transforms/VPTOLLVMEmitter.cpp`
- **修改 1** (第 290 行): `isMxElementType` 添加 fp4 类型检查
```cpp
static bool isMxElementType(Type ty) {
  if (auto floatType = dyn_cast<FloatType>(ty))
    return floatType.getWidth() == 8;
  if (isa<pto::F4E1M2x2Type, pto::F4E2M1x2Type>(ty))
    return true;
  std::string typeText;
  llvm::raw_string_ostream os(typeText);
  ty.print(os);
  os.flush();
  return StringRef(typeText).starts_with("f8");
}
```

- **修改 2** (第 791 行): `getCopyElementFragment` 为 fp4 返回 "u8"
```cpp
if (StringRef(lower).contains("e1m2x2") || StringRef(lower).contains("e2m1x2"))
  return "u8";
```

- **修改 3** (第 601 行): `getL0LoadElementFragment` 为 fp4 返回 "s8"
```cpp
if (StringRef(lower).contains("e1m2x2") || StringRef(lower).contains("e2m1x2"))
  return "s8";
```

#### 文件 3: `lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp`
- 与 VPTOLLVMEmitter.cpp 相同的三处修改

### 2.2 PTO 文件修复 (`tmatmul_mx.pto`)

#### 修复 1: 补充缺失的 SSA 常量
多个函数缺少 `%c5_i64`, `%c2_i64`, `%c4_i64` 等常量定义，导致 "use of undeclared SSA value name" 错误。

#### 修复 2: mte_l1_bt 参数
- **问题**: 模拟器断言 `(isa_para->burst % 2 == 0x0)` 失败
- **原因**: DMA burst 计数必须为偶数
- **修改**: 将 `bias_fp8_e4m3_200x192x95` 的 `mte_l1_bt` len_burst 从 95 改为 96

#### 修复 3: mte_l1_l0a/l0b 参数
- **问题**: "expected ','" 语法错误
- **原因**: 缺少 stride 参数
- **修改**: 所有 `mte_l1_l0a` 和 `mte_l1_l0b` 添加 `%c0_i64, %c0_i64` stride 参数

### 2.3 测试文件修复

#### 文件 1: `gen_data.py`
- **修改 1**: 所有数据 pad 到 aligned 大小（m_padded, n_padded, k_aligned）
- **修改 2**: bias 数据 pad 到偶数大小（`ceil_align(n, 2)`）
- **修改 3**: golden 结果也 pad 到 aligned 大小

#### 文件 2: `main.cpp`
- **修改**: TestCase 结构体添加 `m_padded` 和 `n_padded` 字段
- **修改**: kCases 数组更新所有 19 个 case 的参数
- **修改**: RunCase 函数使用 `m_padded * n_padded` 作为输出大小

#### 文件 3: `compare.py`
- **修改**: 从 aligned 大小 slice 到 valid 区域再比较
```python
golden = np.fromfile(...).reshape(m_padded, n_padded)[:m, :n]
output = np.fromfile(...).reshape(m_padded, n_padded)[:m, :n]
```

#### 文件 4: `cases.py`
- **修改**: 所有 19 个 case 添加 `m_padded` 和 `n_padded` 字段

---

## 3. 当前问题：DMA 数据加载格式不正确

### 3.1 问题现象

通过详细分析第一个 case `fp8_e5m2_128x64x64` 的输出：

```python
golden[0,:5] = [1606. -164.  333. -752.  448.]
output[0,:5] = [1606. -164.  333. -752.  448.]  # 完全正确

golden[32,:5] = [2068.  952. 1296. -568.  438.]
output[32,:5] = [0. 0. 0. 0. 0.]  # 全部为零

golden[127,:5] = [  -40.  -440.   -63. -1344.  -978.]
output[127,:5] = [0. 0. 0. 0. 0.]  # 全部为零
```

**统计**:
- 前 32 行的前 32 列完全正确
- 第 32 行之后的所有数据为零
- 第 32 列之后的所有数据为零
- 总共 128×64 = 8192 个元素，只有 32×32 = 1024 个正确

### 3.2 根因分析

#### 对比 tmatmul（通过）和 tmatmul_mx（失败）

**tmatmul (f16, 通过)**:
```mlir
pto.mte_gm_l1_frac %a_gm, %l1_a, nd2nz,
  shape(%c48_i64, %c64_i64), src_layout(%c128_i64),
  dst_group(%c1_i64, %c1_i64, %c48_i64, %c0_i64),
  ctrl(%c0_i64, %false)
```

**tmatmul_mx (f8, 失败)**:
```mlir
pto.mte_gm_l1 %a_gm, %l1_a_data, %c1024_i64 nburst(%c8_i64, %c0_i64, %c0_i64)
```

**关键差异**:
- tmatmul 使用 `mte_gm_l1_frac` with `nd2nz` 模式，将 row-major 数据从 GM 加载到 L1 并转换为 NZ 格式
- tmatmul_mx 使用 `mte_gm_l1` 做原始数据拷贝，不进行格式转换

#### pto-isa TLoad.hpp 源码分析

从 `/usr/local/CANN/cann-9.0.0-beta.1/x86_64-linux/include/pto/npu/a5/TLoad.hpp` 第 216-303 行：

```cpp
template <typename TileData, typename GlobalData, Layout Layout = Layout::ND>
PTO_INTERNAL void TLoadCubeInstr(__cbuf__ typename TileData::DType *dst, 
                                  typename GlobalData::DType *src,
                                  uint64_t loop1SrcStride, uint16_t nValue, uint32_t dValue) {
  if constexpr (Layout == Layout::ND) {
    if constexpr (sizeof(typename TileData::DType) == 1) {
      copy_gm_to_cbuf_multi_nd2nz(reinterpret_cast<__cbuf__ uint8_t *>(dst),
          reinterpret_cast<__gm__ uint8_t *>(src), 0 /*sid*/, loop1SrcStride, 0, 
          nValue, dValue, 0, false);
    }
  }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadCubeND2NZ(__cbuf__ typename TileData::DType *dst, 
                                  typename GlobalData::DType *src, 
                                  int gShape0, int gShape1, int gShape2, 
                                  int gShape3, int gShape4, ...) {
  uint16_t nValue = gShape3;
  uint32_t dValue = validCol;
  
  uint64_t loop1SrcStride = GetByteSize<typename TileData::DType>(gStride3);
  
  constexpr uint16_t ndNum = 1;
  uint16_t loop2DstStride = 1;
  uint16_t loop3DstStride = TileData::Rows;  // unit is 32B
  uint16_t loop4DstStride = 0;
  
  uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48;
  mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;
  mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;
  mte2NzPara |= static_cast<uint64_t>(ndNum);
  set_mte2_nz_para(mte2NzPara);
  
  TLoadCubeInstr<TileData, GlobalData, GlobalData::layout>(dst, src, loop1SrcStride, nValue, dValue);
}
```

**关键参数**:
- `nValue`: 对应 `dst_group` 的第 4 个参数（gShape3）
- `dValue`: 对应 `shape` 的第 2 个参数（validCol）
- `loop1SrcStride`: 对应 `src_layout` 参数
- `mte2NzPara`: 由 `dst_group` 的 4 个参数打包

#### PTOAS mte_gm_l1_frac 展开逻辑

从 `lib/PTO/Transforms/VPTOExpandWrapperOps.cpp` 第 1219-1252 行：

```cpp
LogicalResult matchAndRewrite(pto::MteGmL1FracOp op, PatternRewriter &rewriter) {
  Value mte2NzPara = packMte2NzPara(
      loc, op.getGroupCount(),      // dst_group[0]
      op.getDstLoop2Stride(),       // dst_group[1]
      op.getDstLoop3Stride(),       // dst_group[2]
      op.getDstLoop4Stride(),       // dst_group[3]
      rewriter);
  rewriter.create<pto::SetMte2NzParaOp>(loc, mte2NzPara);
  
  rewriter.create<pto::CopyGmToCbufMultiNd2NzOp>(
      loc, source, destination, zero, 
      op.getSrcInnerStride(),  // src_layout[0]
      op.getL2CacheCtrl(),     // ctrl[0]
      op.getNValue(),          // shape[0]
      op.getDValue(),          // shape[1]
      srcOuterStride,          // src_layout[1] (optional)
      op.getSmallc0En());      // ctrl[1]
}
```

**packMte2NzPara 函数** (第 491-506 行):
```cpp
static Value packMte2NzPara(Location loc, Value groupCount, Value dstLoop2Stride,
                            Value dstLoop3Stride, Value dstLoop4Stride,
                            PatternRewriter &rewriter) {
  Value loop2Bits = rewriter.create<arith::ShLIOp>(loc, dstLoop2Stride, shift16);
  Value loop3Bits = rewriter.create<arith::ShLIOp>(loc, dstLoop3Stride, shift32);
  Value loop4Bits = rewriter.create<arith::ShLIOp>(loc, dstLoop4Stride, shift48);
  Value low = rewriter.create<arith::OrIOp>(loc, groupCount, loop2Bits);
  Value high = rewriter.create<arith::OrIOp>(loc, loop3Bits, loop4Bits);
  return rewriter.create<arith::OrIOp>(loc, low, high);
}
```

**打包格式**:
```
mte2NzPara[63:48] = dstLoop4Stride  (dst_group[3])
mte2NzPara[47:32] = dstLoop3Stride  (dst_group[2])
mte2NzPara[31:16] = dstLoop2Stride  (dst_group[1])
mte2NzPara[15:0]  = groupCount      (dst_group[0])
```

### 3.3 已尝试的解决方案

#### 尝试 1: 使用 mte_gm_l1_frac with nd2nz (ndNum=1)

**参数**: 
- A: `shape(128, 64), src_layout(64), dst_group(1, 1, 128, 0)`
- B: `shape(64, 64), src_layout(64), dst_group(1, 1, 64, 0)`

**结果**: 前 89 行正确，后面 39 行全部为零

**分析**: 
- 89 = 5 * 16 + 9，说明前 5 个 M-block (80 行) 完全正确，第 6 个 M-block 部分正确
- 这比之前测试的"前 32 行正确"要好得多
- 可能是之前的测试配置有误，或者 gen_data.py 的修改改善了数据布局

#### 尝试 2: 不同的 dst_group 参数

**参数 1**: `dst_group(4, 1, 128, 0)` (ndNum=4)
- **结果**: 更差，连前 32 行都出错 (index 782 = row 12)
- **结论**: ndNum 不是控制 M-block 数量的参数

**参数 2**: 只有 A 用 nd2nz，B 用 mte_gm_l1
- **结果**: 前 6 行正确 (index 397 = row 6)
- **结论**: 两个矩阵都需要用 nd2nz

#### 尝试 3: 使用 mte_gm_l1 加载 column-major 数据

**修改 gen_data.py**: 将数据保存为 column-major 格式
```python
x1_col_major = np.ascontiguousarray(x1_padded.T)
x2_col_major = np.ascontiguousarray(x2_padded.T)
```

**结果**: 更差，只有 1021 个非零元素（应该 8192 个）

#### 尝试 4: 调整 mte_gm_l1 的 len_burst 和 nburst

**参数**: `len_burst=64, nburst=128` (A 矩阵)
**结果**: 更差，只有 63 个非零元素

---

## 4. 最新发现 (2024-06-XX)

### 4.1 FP4 vs FP8 对比测试

**测试配置：**
- FP4 kernels: 使用 mte_gm_l1 (原始拷贝)
- FP8 kernels: 使用 mte_gm_l1_frac with nd2nz (ndNum=1)

**测试结果：**

| Kernel | M | K | N | 正确行数 | 总行数 | 正确率 |
|--------|---|---|---|---------|--------|--------|
| fp4_e2m1_128x64x64 | 128 | 64 | 64 | 118 | 128 | 92% ✨ |
| fp4_e1m2_e2m1_117x64x60 | 117 | 64 | 60 | 101 | 117 | 86% ✨ |
| fp4_e2m1_e1m2_115x64x30 | 115 | 64 | 30 | 113 | 115 | 98% ✨ |
| fp8_e5m2_128x64x64 | 128 | 64 | 64 | 43-89 | 128 | 34-70% ⚠️ |
| fp8_e4m3_127x72x64 | 127 | 128 | 64 | 78 | 127 | 61% ⚠️ |
| fp8_e4m3_e5m2_128x110x63 | 128 | 128 | 63 | 63 | 128 | 49% ⚠️ |

**关键发现：**
1. FP4 cases 工作得非常好（86-98% 正确率），即使使用 mte_gm_l1
2. FP8 cases 只能部分工作（34-70% 正确率），即使使用 nd2nz
3. 问题是 **FP8 特有的**，不是通用的 DMA 问题

### 4.2 FP4 数据布局分析

FP4 使用 `pack_two_fp4` 函数将两个 fp4 值打包成一个字节：
```python
def pack_two_fp4(x):
    # x is [row, col] with col being even
    # Returns [row, col//2] with two fp4 values per byte
    flat = x.reshape(-1)
    high = flat[0::2].view(np.uint8)
    low = flat[1::2].view(np.uint8)
    low_bits = (low & 0x0F) << 4
    high_bits = high & 0x0F
    combined = low_bits | high_bits
    return combined.reshape(row, col // 2)
```

这意味着 FP4 的内存布局与 FP8 不同：
- FP8: 1 byte per element, row-major
- FP4: 0.5 bytes per element, packed into bytes

### 4.3 假设

**假设 1: mad_mx 对 FP8 有行数限制**
- mad_mx 可能只能计算前 N 行（N < M）
- 需要检查 mad_mx 的硬件规格

**假设 2: L1→L0A 传输对 FP8 有限制**
- mte_l1_l0a 可能只能传输前 N 行
- 需要尝试多次传输

**假设 3: NZ 格式对 FP8 有特殊要求**
- FP8 的 NZ 格式可能需要特殊的对齐或填充
- 需要检查 NZ 格式的规格

## 5. 编译器代码分析

### 5.1 DMA Intrinsic 选择逻辑

**文件**: `lib/PTO/Transforms/VPTOLLVMEmitter.cpp` 和 `VPTOCANN900LLVMEmitter.cpp`

**函数**: `getCopyElementFragment`

```cpp
static std::string getCopyElementFragment(Type elementType) {
  // ... FP16, BF16, FP32 处理 ...
  
  // FP8 使用特定 intrinsic
  if (StringRef(lower).contains("e4m3"))
    return "e4m3";
  if (StringRef(lower).contains("e5m2"))
    return "e5m2";
  
  // FP4 使用通用 u8 intrinsic
  if (StringRef(lower).contains("e1m2x2") || StringRef(lower).contains("e2m1x2"))
    return "u8";
}
```

**生成的 Intrinsic**:
- FP8 (e4m3): `llvm.hivm.MOV.OUT.TO.L1.ALIGN.V2.e4m3.DV`
- FP8 (e5m2): `llvm.hivm.MOV.OUT.TO.L1.ALIGN.V2.e5m2.DV`
- FP4: `llvm.hivm.MOV.OUT.TO.L1.ALIGN.V2.u8.DV`

### 5.2 尝试的修复方案

**方案**: 让 FP8 也使用 "u8" intrinsic（与 FP4 相同）

**结果**: ❌ 准确率从 69.5% 降到 0.07%

**结论**: FP8 确实需要使用特定的 intrinsic，不能使用通用的 u8。问题不在 intrinsic 选择。

### 5.3 关键观察

1. **FP4 工作得很好**（86-98% 准确率）
2. **FP8 只有部分正确**（34-70% 准确率）
3. **K 越大，FP8 正确率越低**：
   - K=64: 69.5%
   - K=72: 61.4%
   - K=110: 49.2%

4. **正确的行数与 K 成反比**，这暗示可能是：
   - L1 buffer 溢出问题
   - DMA 传输的边界问题
   - mad_mx 计算的 K 维度限制

## 6. 根因分析：MMAD.MX Intrinsic 的类型组合限制

### 6.1 关键发现

通过分析所有 FP8 测试用例的输出模式，发现了**根本原因**：

**不同 FP8 类型组合的行为差异：**

| 类型组合 | Intrinsic | N=64 时的输出列数 | 状态 |
|---------|-----------|------------------|------|
| e5m2 x e5m2 | MMAD.MX.e5m2e5m2 | 32/64 | ❌ 只有前 32 列 |
| e4m3 x e4m3 | MMAD.MX.e4m3e4m3 | 32/64 | ❌ 只有前 32 列 |
| e4m3 x e5m2 | MMAD.MX.e4m3e5m2 | 63-64/64 | ✅ 几乎全部正确 |
| e4m3 x e4m3 (N=16) | MMAD.MX.e4m3e4m3 | 16/16 | ✅ 全部正确 |

**测试用例详细结果：**
- `fp8_e5m2_128x64x64` (e5m2 x e5m2): 只有列 0-31 有值
- `fp8_e4m3_127x72x64` (e4m3 x e4m3): 只有列 0-31 有值
- `fp8_e4m3_e5m2_128x110x63` (e4m3 x e5m2): 63/64 列有值（只缺列 32）
- `fp8_e4m3_16x32x16` (e4m3 x e4m3, N=16): 16/16 列全部正确
- `fp8_e4m3_e5m2_10x50x54` (e4m3 x e5m2): 64/64 列全部正确

### 6.2 结论

**问题不在 padding 或 TFILLPAD，而在 MMAD.MX intrinsic 本身！**

硬件的 MMAD.MX intrinsic 对于**相同类型**的 FP8 组合（e5m2 x e5m2, e4m3 x e4m3）有 N 维度限制：
- 当 N > 32 时，只能计算前 32 列
- 当 N <= 32 时，可以正常计算所有列

但对于**混合类型**的 FP8 组合（e4m3 x e5m2），没有这个限制，可以计算完整的 N=64。

### 6.3 为什么 FP4 工作正常？

FP4 使用的是不同的 intrinsic（MMAD.MX.e2m1x2e2m1x2 等），这些 intrinsic 没有 N 维度限制，可以正常计算 N=64。

### 6.4 验证假设

这个假设可以解释所有观察到的现象：
1. ✅ FP8 e5m2 x e5m2 只有 32 列正确
2. ✅ FP8 e4m3 x e4m3 只有 32 列正确（当 N >= 32）
3. ✅ FP8 e4m3 x e5m2 几乎全部正确
4. ✅ FP4 全部工作正常
5. ✅ K 越大正确率越低（因为 K 影响 M 维度的正确行数）

## 7. 解决方案尝试

### 7.1 尝试方案一：分块计算（失败）

**方法**：将 N=64 分成两次 N=32 的计算

**实现**：
1. 修改 `mte_gm_l1` 加载 B 矩阵的前 32 列（len_burst=512）
2. 调用 `mad_mx` 计算前 32 列
3. 写入 GM

**结果**：❌ 失败
- 前 32 列的计算结果仍然完全错误（0% 正确率）
- 后 32 列为零（符合预期，因为还没实现）

**结论**：问题不在 N 维度的分块，而在计算本身。

### 7.2 重新分析

回到最初的观察：
- 原始 kernel（N=64）：前 32 列有非零输出，但值错误
- 修改后的 kernel（N=32）：前 32 列有非零输出，但值仍然错误

这说明：
1. ✅ MMAD.MX intrinsic 确实在计算（有非零输出）
2. ❌ 但计算结果是错误的
3. ❌ 问题不在 N 维度的限制，而在其他地方

### 7.3 可能的原因

1. **Scale 数据问题**：MX scale 的加载或应用可能有误
2. **数据布局问题**：FP8 数据在 L1/L0 中的布局可能与预期不同
3. **mad_mx 参数问题**：m, n, k 参数可能不正确
4. **硬件 bug**：MMAD.MX intrinsic 对于 FP8 可能有 bug

## 8. 下一步行动

### 8.1 立即行动

1. **恢复原始 kernel**：撤销分块修改，回到原始实现
2. **检查 scale 数据**：验证 scale 的加载和应用是否正确
3. **对比 FP4 和 FP8**：详细对比 FP4（工作正常）和 FP8（失败）的差异

### 8.2 长期方案

1. **查阅硬件文档**：确认 MMAD.MX 对于 FP8 的具体要求
2. **联系硬件团队**：如果怀疑是硬件 bug，需要确认
3. **考虑 workaround**：如果无法修复，可能需要避免使用 FP8 same-type 组合

## 9. 总结

**核心问题**：FP8 的 tmatmul_mx 计算结果完全错误，不仅仅是部分列缺失。

**已尝试的方案**：
- ❌ 分块计算（N=64 → 2×N=32）：失败，计算结果仍然错误

**已排除的原因**：
- ❌ DMA intrinsic 选择错误
- ❌ L1 buffer 溢出
- ❌ DMA 传输边界问题
- ❌ 缺少 TFILLPAD 操作
- ❌ Padding 区域未初始化为零
- ❌ N 维度限制（分块后仍然错误）

**可能的原因**：
- ⚠️ Scale 数据加载或应用错误
- ⚠️ FP8 数据布局问题
- ⚠️ mad_mx 参数错误
- ⚠️ 硬件 bug

**下一步**：
1. 恢复原始 kernel
2. 详细检查 scale 数据的处理
3. 对比 FP4 和 FP8 的实现差异

---

## 5. 相关文件路径

### 5.1 测试文件
- PTO kernel: `/home/wujiajun/llvm-workspace/00_pto/PTOAS/test/tilelang_st/npu/a5/src/st/testcase/tmatmul_mx/tmatmul_mx.pto`
- 数据生成: `/home/wujiajun/llvm-workspace/00_pto/PTOAS/test/tilelang_st/npu/a5/src/st/testcase/tmatmul_mx/gen_data.py`
- 主程序: `/home/wujiajun/llvm-workspace/00_pto/PTOAS/test/tilelang_st/npu/a5/src/st/testcase/tmatmul_mx/main.cpp`
- 比较脚本: `/home/wujiajun/llvm-workspace/00_pto/PTOAS/test/tilelang_st/npu/a5/src/st/testcase/tmatmul_mx/compare.py`
- 测试用例: `/home/wujiajun/llvm-workspace/00_pto/PTOAS/test/tilelang_st/npu/a5/src/st/testcase/tmatmul_mx/cases.py`

### 5.2 编译器文件
- VPTO.cpp: `/home/wujiajun/llvm-workspace/00_pto/PTOAS/lib/PTO/IR/VPTO.cpp`
- VPTOLLVMEmitter.cpp: `/home/wujiajun/llvm-workspace/00_pto/PTOAS/lib/PTO/Transforms/VPTOLLVMEmitter.cpp`
- VPTOCANN900LLVMEmitter.cpp: `/home/wujiajun/llvm-workspace/00_pto/PTOAS/lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp`
- VPTOExpandWrapperOps.cpp: `/home/wujiajun/llvm-workspace/00_pto/PTOAS/lib/PTO/Transforms/VPTOExpandWrapperOps.cpp`

### 5.3 参考文件
- pto-isa TLoad: `/usr/local/CANN/cann-9.0.0-beta.1/x86_64-linux/include/pto/npu/a5/TLoad.hpp`
- pto-isa 测试: `/home/wujiajun/llvm-workspace/pto-isa/tests/npu/a5/src/st/testcase/tmatmul_mx/`
- TileLang 模板: `/home/wujiajun/llvm-workspace/00_pto/PTOAS/lib/TileOps/tmatmul_mx_template.py`

### 5.4 构建目录
- Build 目录: `/home/wujiajun/llvm-workspace/00_pto/PTOAS/test/tilelang_st/npu/a5/src/st/build/testcase/tmatmul_mx/`
- 生成的数据: `fp8_e5m2_128x64x64/input1.bin`, `input2.bin`, `scale1.bin`, `scale2.bin`, `golden.bin`, `output.bin`

---

## 6. 运行测试命令

```bash
cd /home/wujiajun/llvm-workspace/00_pto/PTOAS
python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t tmatmul_mx
```

单独测试第一个 case:
```bash
cd /home/wujiajun/llvm-workspace/00_pto/PTOAS/test/tilelang_st/npu/a5/src/st/build/testcase/tmatmul_mx
export LD_LIBRARY_PATH=/usr/local/CANN/cann-9.0.0-beta.1/tools/simulator/Ascend950PR_9599/lib:/usr/local/CANN/cann-9.0.0-beta.1/runtime/lib64/stub:/usr/local/CANN/cann-9.0.0-beta.1/lib64:...
../../bin/tmatmul_mx fp8_e5m2_128x64x64
```

---

## 7. 关键发现总结

1. **数据生成正确**: 使用 pto-isa 的相同随机种子和算法生成的数据，golden 结果一致
2. **kernel 执行成功**: 无崩溃、无 DMA 断言错误
3. **DMA 加载不完整**: 只正确加载了 32×32 的数据块（128×64 矩阵的前 1/4）
4. **mte_gm_l1 不够**: 简单的 `mte_gm_l1` 无法正确加载数据到 L0A/L0B
5. **mte_gm_l1_frac 参数不对**: 尝试了多种 dst_group 参数组合，都无法正确加载完整数据

**核心问题**: 需要找到正确的 `mte_gm_l1_frac` dst_group 参数，或者使用 TileLang 编译器自动生成正确的 kernel。

---

## 8. 待办事项

- [ ] 查阅 CANN 文档，理解 mte_gm_l1_frac dst_group 参数的正确含义
- [ ] 分析 pto-isa 的 TLoad 调用，找到正确的参数映射
- [ ] 尝试使用 TileLang 编译器从模板生成 kernel
- [ ] 调试 fp4 类型的 case（当前所有 fp4 case 也失败）
- [ ] 调试 bias case（当前所有 bias case 也失败）
- [ ] 调试 gemv case（当前所有 gemv case 也失败）

---

**备注**: 本文档记录了 tmatmul_mx 测试的完整调试过程。下次继续调试时，阅读本文档可以快速恢复上下文。
