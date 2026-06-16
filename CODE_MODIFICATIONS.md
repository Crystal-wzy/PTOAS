# PTOAS 仓库修改记录

**文档创建日期**: 2026-06-16  
**修改目的**: 支持 FP8 混合类型和 FP4 数据类型的 tmatmul_mx 算子

## 概述

本文档记录了 PTOAS 仓库中除测试用例目录和 TileOps 模板目录外的所有代码修改。这些修改主要是为了扩展编译器对低精度数据类型（FP8、FP4）的支持，特别是解决 tmatmul_mx 算子在使用这些数据类型时的问题。

## 修改文件列表

| 文件路径 | 修改类型 | 主要目的 |
|---------|---------|---------|
| `lib/PTO/IR/PTO.cpp` | 类型验证 | 支持 FP8 混合类型和 HiF8 |
| `lib/PTO/IR/VPTO.cpp` | MX 类型扩展 | 支持 FP4 和 FP8 E5M2 类型 |
| `lib/PTO/Transforms/VPTOLLVMEmitter.cpp` | LLVM 代码生成 | 支持 FP4/FP8 的 DMA 和计算指令 |
| `lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp` | CANN900 代码生成 | 支持 FP4/FP8 的 DMA 和计算指令 |
| `tilelang-dsl/python/tilelang_dsl/semantic.py` | DSL 语义分析 | 允许 FP8 混合类型 |

## 详细修改说明


---

### 2. lib/PTO/IR/PTO.cpp

#### 修改 1: 允许低精度类型的矩阵操作数验证

**修改位置**: 第 4432-4433 行

**修改内容**:
```cpp
// 修改前
if (failed(verifyTileBufCommon(op, lhsTy, "lhs")) ||
    failed(verifyTileBufCommon(op, rhsTy, "rhs")) ||

// 修改后
if (failed(verifyTileBufCommon(op, lhsTy, "lhs", /*allowLowPrecision=*/true)) ||
    failed(verifyTileBufCommon(op, rhsTy, "rhs", /*allowLowPrecision=*/true)) ||
```

**修改目的**: 允许 FP8 和 FP4 等低精度类型作为矩阵乘法的操作数

**解决的问题**: 编译器拒绝接受 FP8/FP4 类型的矩阵乘法操作数

#### 修改 2: 支持 HiF8 类型

**修改位置**: 第 4694-4696 行

**修改内容**:
```cpp
// 新增代码
if (isa<HiF8Type>(lhsElemTy))
  return success();
```

**修改目的**: 在类型验证中添加对 HiF8（High-precision Float8）类型的支持

**解决的问题**: HiF8 类型无法通过矩阵乘法的类型验证

#### 修改 3: 支持 A5 架构下的 FP8 混合类型

**修改位置**: 第 4698-4705 行

**修改内容**:
```cpp
// 新增代码
// A5: allow mixed fp8 pairs (e.g., f8e4m3 x f8e5m2 -> f32)
if (isA5 && dstElemTy.isF32()) {
  auto lt = mlir::dyn_cast<FloatType>(lhsElemTy);
  auto rt = mlir::dyn_cast<FloatType>(rhsElemTy);
  if (lt && rt && lt.getWidth() == 8 && rt.getWidth() == 8)
    return success();
}
```

**修改目的**: 在 A5 架构下允许 FP8 混合类型组合（如 f8e4m3 x f8e5m2）

**解决的问题**: A5 硬件支持 FP8 混合类型，但编译器拒绝这种组合

#### 修改 4: 更新错误消息

**修改位置**: 第 4709 行

**修改内容**:
```cpp
// 修改前
<< (isA5 ? ", or an A5-supported fp8 pair" : "");

// 修改后
<< (isA5 ? ", or an A5-supported fp8/hif8 pair" : "");
```

**修改目的**: 在错误消息中明确提到 HiF8 支持

**解决的问题**: 错误消息不完整，未提及 HiF8 支持

---

### 3. lib/PTO/IR/VPTO.cpp

#### 修改 1: 扩展 MX 元素类型识别

**修改位置**: 第 786-789 行

**修改内容**:
```cpp
// 修改前
static bool isMxElementType(Type type) { return isa<Float8E4M3FNType>(type); }

// 修改后
static bool isMxElementType(Type type) {
  return isa<Float8E4M3FNType, Float8E5M2Type>(type) ||
         isa<pto::F4E1M2x2Type, pto::F4E2M1x2Type>(type);
}
```

**修改目的**: 扩展 MX（Matrix eXtension）元素类型的识别范围，包括：
- Float8E5M2Type（FP8 E5M2 格式）
- F4E1M2x2Type（FP4 E1M2 打包格式）
- F4E2M1x2Type（FP4 E2M1 打包格式）

**解决的问题**: 编译器无法识别 FP8 E5M2 和 FP4 类型为有效的 MX 类型

#### 修改 2: 更新错误消息

**修改位置**: 第 3718 行

**修改内容**:
```cpp
// 修改前
"requires MX lhs/rhs element types (currently f8E4M3FN)"

// 修改后
"requires MX lhs/rhs element types (f8E4M3FN, f8E5M2, f4E1M2x2, or f4E2M1x2)"
```

**修改目的**: 在错误消息中列出所有支持的 MX 类型

**解决的问题**: 错误消息不准确，未反映实际支持的类型

---

### 4. lib/PTO/Transforms/VPTOLLVMEmitter.cpp

#### 修改 1: 扩展 MX 元素类型识别（LLVM 发射器）

**修改位置**: 第 293-294 行

**修改内容**:
```cpp
// 新增代码
if (isa<pto::F4E1M2x2Type, pto::F4E2M1x2Type>(ty))
  return true;
```

**修改目的**: 在 LLVM 代码生成阶段识别 FP4 类型为 MX 类型

**解决的问题**: LLVM 发射器无法处理 FP4 类型的 MX 操作

#### 修改 2: 添加 E5M2 类型识别函数

**修改位置**: 第 372-374 行

**修改内容**:
```cpp
// 新增函数
static bool isMadE5M2ElementType(Type type) {
  return type.isFloat8E5M2() || type.isFloat8E5M2FNUZ();
}
```

**修改目的**: 添加专门的函数来识别 FP8 E5M2 类型（包括 FNUZ 变体）

**解决的问题**: 缺少对 E5M2 类型的识别能力

#### 修改 3: 支持 FP8 混合类型的 MAD 指令

**修改位置**: 第 409-417 行

**修改内容**:
```cpp
// 新增代码
if (isMadE4M3ElementType(lhsElem) && isMadE5M2ElementType(rhsElem) &&
    dst == "f32")
  return StringAttr::get(context, "llvm.hivm.MAD.e4m3e5m2.c310").getValue();
if (isMadE5M2ElementType(lhsElem) && isMadE4M3ElementType(rhsElem) &&
    dst == "f32")
  return StringAttr::get(context, "llvm.hivm.MAD.e5m2e4m3.c310").getValue();
if (isMadE5M2ElementType(lhsElem) && isMadE5M2ElementType(rhsElem) &&
    dst == "f32")
  return StringAttr::get(context, "llvm.hivm.MAD.e5m2e5m2.c310").getValue();
```

**修改目的**: 为 FP8 混合类型组合生成正确的 LLVM intrinsic 调用：
- e4m3 x e5m2 → `llvm.hivm.MAD.e4m3e5m2.c310`
- e5m2 x e4m3 → `llvm.hivm.MAD.e5m2e4m3.c310`
- e5m2 x e5m2 → `llvm.hivm.MAD.e5m2e5m2.c310`

**解决的问题**: 编译器无法为 FP8 混合类型生成正确的计算指令

#### 修改 4: 支持 FP4 的 L0 加载

**修改位置**: 第 616-618 行

**修改内容**:
```cpp
// 修改前
if (StringRef(lower).contains("e4m3") ||
    StringRef(lower).contains("e5m2") ||
    StringRef(lower).contains("e8m0") ||
    StringRef(lower).contains("hif8"))

// 修改后
if (StringRef(lower).contains("e4m3") ||
    StringRef(lower).contains("e5m2") ||
    StringRef(lower).contains("e8m0") ||
    StringRef(lower).contains("hif8") ||
    StringRef(lower).contains("e1m2x2") ||
    StringRef(lower).contains("e2m1x2"))
```

**修改目的**: 在 L0（Level 0）缓存加载时识别 FP4 类型，返回 "s8"（8位有符号整数）作为元素片段

**解决的问题**: FP4 数据无法正确加载到 L0 缓存

#### 修改 5: 支持 FP4 的 DMA 传输

**修改位置**: 第 818-819 行

**修改内容**:
```cpp
// 新增代码
if (StringRef(lower).contains("e1m2x2") || StringRef(lower).contains("e2m1x2"))
  return "u8";
```

**修改目的**: 在 DMA（Direct Memory Access）传输时识别 FP4 类型，返回 "u8"（8位无符号整数）作为元素片段

**解决的问题**: FP4 数据无法通过 DMA 正确传输

---

### 5. lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp

#### 修改 1: 扩展 MX 元素类型识别（CANN900 发射器）

**修改位置**: 第 292-293 行

**修改内容**:
```cpp
// 新增代码
if (isa<pto::F4E1M2x2Type, pto::F4E2M1x2Type>(ty))
  return true;
```

**修改目的**: 在 CANN900 LLVM 代码生成阶段识别 FP4 类型为 MX 类型

**解决的问题**: CANN900 发射器无法处理 FP4 类型的 MX 操作

#### 修改 2: 支持 FP4 的 L0 加载（CANN900）

**修改位置**: 第 648-650 行

**修改内容**:
```cpp
// 修改前
if (StringRef(lower).contains("e4m3") ||
    StringRef(lower).contains("e5m2") ||
    StringRef(lower).contains("e8m0") ||
    StringRef(lower).contains("hif8"))

// 修改后
if (StringRef(lower).contains("e4m3") ||
    StringRef(lower).contains("e5m2") ||
    StringRef(lower).contains("e8m0") ||
    StringRef(lower).contains("hif8") ||
    StringRef(lower).contains("e1m2x2") ||
    StringRef(lower).contains("e2m1x2"))
```

**修改目的**: 在 CANN900 的 L0 缓存加载时识别 FP4 类型

**解决的问题**: CANN900 后端无法正确处理 FP4 数据的 L0 加载

#### 修改 3: 支持 FP4 的 DMA 传输（CANN900）

**修改位置**: 第 837-838 行

**修改内容**:
```cpp
// 新增代码
if (StringRef(lower).contains("e1m2x2") || StringRef(lower).contains("e2m1x2"))
  return "u8";
```

**修改目的**: 在 CANN900 的 DMA 传输时识别 FP4 类型

**解决的问题**: CANN900 后端无法正确处理 FP4 数据的 DMA 传输

---

### 6. tilelang-dsl/python/tilelang_dsl/semantic.py

#### 修改 1: 添加 FP8 混合类型支持参数

**修改位置**: 第 3876 行

**修改内容**:
```python
# 修改前
def _require_matching_element_dtypes(
    self,
    lhs: SemanticExpr,
    rhs: SemanticExpr,
    context: str,
) -> None:

# 修改后
def _require_matching_element_dtypes(
    self,
    lhs: SemanticExpr,
    rhs: SemanticExpr,
    context: str,
    allow_mixed_fp8: bool = False,
) -> None:
```

**修改目的**: 添加 `allow_mixed_fp8` 参数来控制是否允许 FP8 混合类型

**解决的问题**: 语义分析器过于严格，拒绝所有类型不匹配的情况

#### 修改 2: 实现 FP8 混合类型检查逻辑

**修改位置**: 第 3884-3889 行

**修改内容**:
```python
# 新增代码
if allow_mixed_fp8:
    lhs_name = getattr(lhs_dtype, "name", "")
    rhs_name = getattr(rhs_dtype, "name", "")
    fp8_set = {"f8e4m3", "f8e5m2"}
    if lhs_name in fp8_set and rhs_name in fp8_set:
        return
```

**修改目的**: 当 `allow_mixed_fp8=True` 时，检查两个操作数是否都是 FP8 类型（f8e4m3 或 f8e5m2），如果是则允许类型不匹配

**解决的问题**: TileLang DSL 无法表达 FP8 混合类型的矩阵乘法

#### 修改 3: 在矩阵乘法操作中启用 FP8 混合类型

**修改位置**: 第 4000 行

**修改内容**:
```python
# 修改前
self._require_matching_element_dtypes(
    lhs,
    rhs,
    f"pto.{name}",
)

# 修改后
self._require_matching_element_dtypes(
    lhs,
    rhs,
    f"pto.{name}",
    allow_mixed_fp8=True,
)
```

**修改目的**: 在处理 `pto.tmatmul` 等矩阵乘法操作时，启用 FP8 混合类型支持

**解决的问题**: 用户在 TileLang DSL 中无法使用 FP8 混合类型进行矩阵乘法

---

## 修改总结

### 核心目标

这些修改的核心目标是**扩展 PTOAS 编译器对低精度数据类型的支持**，特别是：

1. **FP8 混合类型支持**: 允许 f8e4m3 和 f8e5m2 的混合使用
2. **FP4 类型支持**: 添加对 f4e1m2x2 和 f4e2m1x2 打包格式的完整支持
3. **HiF8 类型支持**: 添加对高精度 Float8 类型的支持

### 修改层次

修改涉及编译器的多个层次：

1. **IR 层** (`lib/PTO/IR/`): 类型验证和识别
2. **Transforms 层** (`lib/PTO/Transforms/`): LLVM 代码生成
3. **DSL 层** (`tilelang-dsl/`): 用户友好的前端语法

### 解决的问题

| 问题类型 | 具体表现 | 解决方案 |
|---------|---------|---------|
| 类型验证失败 | 编译器拒绝 FP8/FP4 操作数 | 扩展类型验证规则 |
| 代码生成失败 | 无法生成正确的 LLVM intrinsic | 添加新的 intrinsic 映射 |
| DMA 传输失败 | FP4 数据无法正确传输 | 添加 FP4 的 DMA 支持 |
| 缓存加载失败 | FP4 数据无法加载到 L0 | 添加 FP4 的 L0 加载支持 |
| DSL 语法限制 | 无法表达 FP8 混合类型 | 添加 `allow_mixed_fp8` 参数 |

### 影响范围

这些修改影响以下功能：

- ✅ `tmatmul` 算子：支持 FP8 混合类型和 FP4
- ✅ `tmatmul_mx` 算子：支持所有 MX 类型（FP8 E4M3/E5M2、FP4 E1M2/E2M1）
- ✅ `tmatmul_bias` 算子：支持 FP8 混合类型
- ✅ `tgemv` 算子：支持 FP8 混合类型
- ✅ `tgemv_mx` 算子：支持所有 MX 类型

### 测试验证

这些修改已通过以下测试用例验证：

- FP8 混合类型：`fp8_e4m3_e5m2_128x110x63` ✅
- FP4 类型：`fp4_e2m1_128x64x64`, `fp4_e1m2_e2m1_117x64x60` ✅
- HiF8 类型：相关测试用例 ✅

---


