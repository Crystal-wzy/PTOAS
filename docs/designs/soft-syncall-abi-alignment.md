# Soft `SYNCALL` ABI 与 PTO-ISA f24 对齐设计

- 状态：已实现（Draft PR [PTOAS #1064](https://github.com/hw-native-sys/PTOAS/pull/1064)）
- 跟踪 Issue：[PTOAS #1061](https://github.com/hw-native-sys/PTOAS/issues/1061)
- PTO-ISA 变更：
  [`f24f7b736b689cc107b9eb2d362be6a7718fcc99`](https://github.com/hw-native-sys/pto-isa/commit/f24f7b736b689cc107b9eb2d362be6a7718fcc99)

## 1. 摘要

PTO-ISA f24 将原先按核类型区分的 Soft `SYNCALL` 重载统一为一个公共 ABI：

```cpp
SYNCALL<SyncAllMode::Soft, CoreType>(gmWorkspace, usedCores);
```

新接口不再包含 UB 和 L1 临时空间参数。目前 PTOAS 仍在 `pto.syncall` 中描述并校验
这些参数，为其物化本地 Tile，并生成三参数或四参数的 C++ 调用。

本设计将规范 PTO IR 契约修改为：

- Hard 模式：无 operand；
- Soft 模式：必须提供 `gm_workspace`，可以提供 `used_cores`；
- 所有 Soft 核类型：EmitC 始终生成两个实参；省略 `used_cores` 时物化
  `int32_t{0}`。

推荐直接切换到新 ABI。如果 PTOAS 和上游 IR 生产方无法原子升级，
[第 8 节](#8-兼容性与发布策略)给出一个仅保留一版的兼容方案。

## 2. 问题描述

当前 PTOAS 的 Soft operand 形式取决于 `core_type`：

| 核类型 | 当前 PTO operand | 当前 C++ 实参 |
| --- | --- | --- |
| `aiv_only` | GM + UB + 可选 used cores | GM + UB + used cores |
| `aic_only` | GM + L1 + 可选 used cores | GM + L1 + used cores |
| `mix` | GM + UB + L1 + 可选 used cores | GM + UB + L1 + used cores |

例如，当前 PTOAS 会生成：

```cpp
Tile<TileType::Vec, int32_t, 1, 64> ubWorkspace;
TASSIGN(ubWorkspace, ubAddress);
Tile<TileType::Mat, int32_t, 1, 64> l1Workspace;
TASSIGN(l1Workspace, l1Address);
SYNCALL<SyncAllMode::Soft, SyncCoreType::Mix>(
    gmWorkspace, ubWorkspace, l1Workspace, usedCores);
```

该调用无法匹配 PTO-ISA f24 及其后继版本，同时还会预留和初始化新实现根本不会使用
的本地空间。

新的 PTO-ISA 实现使用 GM 共享原子计数器。它要求一条独占的 64 字节 cache line，
对应至少 16 个 `int32_t` 元素，并且 workspace 在第一次使用前必须清零。

## 3. 目标与非目标

### 3.1 目标

- 将文本 PTO IR、ODS 生成 API、校验逻辑和 EmitC 与 PTO-ISA f24 对齐。
- 保持 Hard `SYNCALL` 行为不变。
- GM workspace 继续支持 `memref`、`tensor_view` 和
  `partition_tensor_view`。
- 拒绝静态容量小于 16 个 `int32_t` 元素的 GM workspace。
- Soft `SYNCALL` 不再生成 UB/L1 Tile、`TASSIGN` 或本地临时空间。
- 在同一次改动中同步更新 Python 示例、PTO IR 手册和定向回归测试。
- 使用 PTO-ISA f24 或明确的后继 SHA 编译生成的 A5 C++。

### 3.2 非目标

- 修改 PTO-ISA 的同步算法。
- 在 PTO IR verifier 中证明 cache line 独占、对齐或已清零；这些仍是调用方或运行时
  的责任。
- 修改 `used_cores = 0` 的语义。
- 修改 Hard 模式的 FFTS 行为。
- 增加新的 CLI 选项或修改 PTOAS pass pipeline。

## 4. 规范 PTO IR 契约

### 4.1 ODS operand

`SyncAllOp` 保留 `AttrSizedOperandSegments`，因为两个 operand 是否存在都由模式决定：

```tablegen
let arguments = (ins
  Optional<PTOSyncAllGmWorkspaceType>:$gm_workspace,
  Optional<I32>:$used_cores,
  PTO_SyncAllModeAttr:$mode,
  PTO_SyncCoreTypeAttr:$core_type
);
```

`gm_workspace` 在 ODS 中仍声明为 Optional，从而让同一个 op 可以表达零 operand 的
Hard 模式；verifier 再要求 Soft 模式必须提供它。专用的
`PTOSyncAllGmWorkspaceType` 将公开构造 API 限制为 `memref`、
`tensor_view` 或 `partition_tensor_view`，元素类型、GM 地址空间和容量继续由
verifier 提供精确诊断。

新的 segment 布局如下：

| 形式 | `operandSegmentSizes` |
| --- | --- |
| Hard | `[0, 0]` |
| Soft，自动推导核数 | `[1, 0]` |
| Soft，显式指定核数 | `[1, 1]` |

`core_type` 不再影响 operand 布局。

### 4.2 显式提供 `used_cores` 的 Soft 模式

```mlir
module {
  func.func @soft_aiv(
      %gm: memref<16xi32, #pto.address_space<gm>>,
      %used: i32) {
    pto.syncall(
      %gm, %used : memref<16xi32, #pto.address_space<gm>>, i32
    ) mode = #pto.sync_all_mode<soft>,
      core_type = #pto.sync_core_type<aiv_only>
    return
  }
}
```

预期 C++：

```cpp
SYNCALL<SyncAllMode::Soft, SyncCoreType::AIVOnly>(
    gmWorkspace, usedCores);
```

AIC-only 和 Mix 使用相同的 operand：

```mlir
pto.syncall(%gm, %used : memref<16xi32, #pto.address_space<gm>>, i32)
  mode = #pto.sync_all_mode<soft>,
  core_type = #pto.sync_core_type<aic_only>

pto.syncall(%gm, %used : memref<16xi32, #pto.address_space<gm>>, i32)
  mode = #pto.sync_all_mode<soft>,
  core_type = #pto.sync_core_type<mix>
```

### 4.3 自动推导核数的 Soft 模式

```mlir
pto.syncall(%gm : !pto.partition_tensor_view<16xi32>)
  mode = #pto.sync_all_mode<soft>,
  core_type = #pto.sync_core_type<mix>
```

PTOAS 仍必须生成两个 C++ 实参：

```cpp
SYNCALL<SyncAllMode::Soft, SyncCoreType::Mix>(
    gmWorkspace, int32_t{0});
```

生成的调用不依赖 C++ 默认参数。显式传递带类型的零，可以让 PTO IR 中的省略语义
在 EmitC 中保持可见，并避免未来重载变化造成不稳定。

### 4.4 Hard 模式

Hard 模式保持为零 operand op：

```mlir
pto.syncall()
  mode = #pto.sync_all_mode<hard>,
  core_type = #pto.sync_core_type<mix>
```

预期 C++：

```cpp
SYNCALL<SyncCoreType::Mix>();
```

## 5. 解析、打印与校验

### 5.1 自定义 Parser

Parser 按模式应用不同的 operand 数量规则：

1. 解析 operand 和类型列表，再解析 `mode` 与 `core_type`。
2. Hard 模式要求零 operand，并写入 segment size `[0, 0]`。
3. Soft 模式要求一个或两个 operand。
4. 将 operand 0 解析为 `gm_workspace`。
5. operand 1 存在时，将其解析为 `used_cores`。
6. 写入 segment size `[1, 0]` 或 `[1, 1]`。

Parser 不再按 `core_type` 分支。

代表性诊断：

```text
custom op 'pto.syncall' expects hard syncall to have no operands
```

```text
custom op 'pto.syncall' expects soft syncall to have gm_workspace
and optional used_cores
```

### 5.2 Printer

Printer 按固定顺序输出 operand：

1. `gm_workspace`，如果存在；
2. `used_cores`，如果存在。

与当前行为一致，Printer 在可选 attribute dictionary 中省略
`operandSegmentSizes`、`mode` 和 `core_type`。因此 parse/print round trip
只会产生规范的新形式。

### 5.3 Verifier

Hard 模式仅在两个可选 operand 都不存在时通过。

Soft 模式校验：

- 必须提供 `gm_workspace`；
- workspace 必须是有 rank 的 GM `memref`、`!pto.tensor_view` 或
  `!pto.partition_tensor_view`；
- 元素类型必须是 `i32`；
- rank 至少为 1；
- 每个静态维度都必须为正数；
- 如果所有维度均为静态，元素总数必须至少为 16；
- `used_cores` 存在时必须是 `i32`。

容量计算必须避免整数溢出。由于这里只关心阈值 `16`，实现可以在累计容量达到 16
后立即停止乘法。

示例：

```mlir
// 接受：恰好 16 个元素。
memref<16xi32, #pto.address_space<gm>>

// 接受：多维静态容量为 16。
memref<4x4xi32, #pto.address_space<gm>>

// 拒绝：静态可知容量不足。
memref<15xi32, #pto.address_space<gm>>

// 静态校验接受：运行时必须保证至少提供 16 个元素。
memref<?xi32, #pto.address_space<gm>>
```

建议诊断：

```text
'pto.syncall' op expects soft syncall gm_workspace to contain at least
16 i32 elements (64 bytes), but static capacity is 15
```

Verifier 无法证明缓冲区起始地址满足 64 字节对齐、整条 cache line 没有别名，或
workspace 已完成清零。这些要求必须写入 `docs/PTO_IR_manual.md`，并由 IR 生产方
或运行时保证。

## 6. EmitC Lowering

Hard 模式分支和 `coreTypeTok()` 保持不变。

Soft 模式按以下步骤处理：

1. 将 `gm_workspace` 转换为现有 GlobalTensor 表示。
2. 存在 `used_cores` 时使用转换后的值。
3. 否则创建渲染文本为 `int32_t{0}` 的 EmitC value。
4. 生成一个 `SYNCALL<SyncAllMode::Soft, CoreType>` 调用，实参严格为
   `{gmWorkspace, usedCores}`。

概念代码：

```cpp
FailureOr<Value> gmWorkspace = buildGmWorkspace();
Value usedCores = adaptor.getUsedCores()
    ? peelUnrealized(adaptor.getUsedCores())
    : makeTypedInt32Zero();

rewriter.create<emitc::CallOpaqueOp>(
    op.getLoc(), TypeRange{}, callee,
    ArrayAttr{}, ArrayAttr{},
    ValueRange{*gmWorkspace, usedCores});
```

删除以下旧 ABI 代码：

- 根据 `core_type` 选择 UB 和/或 L1 的 switch；
- `buildSyncAllWorkspaceTileValue()`；
- Soft `SYNCALL` 的 Tile 构造；
- Soft `SYNCALL` 的 `TASSIGN` 生成。

## 7. Python API、文档与示例

ODS 变更后，重新生成的 Python API 不再包含 `ub_workspace` 和
`l1_workspace`。

规范 Python 构造方式：

```python
pto.syncall(
    _mode("soft"),
    _core_type("aiv_only"),
    gm_workspace=gm_workspace,
    used_cores=used_cores,
)
```

自动推导核数：

```python
pto.syncall(
    _mode("soft"),
    _core_type("mix"),
    gm_workspace=gm_workspace,
)
```

`test/samples/SyncAll` 示例应停止创建只为 `SYNCALL` 服务的 `AllocTileOp`。

`docs/PTO_IR_manual.md` 必须说明：

- 两个规范 Soft operand；
- 16 元素静态容量规则；
- 独占 64 字节 cache line 的要求；
- 第一次使用前必须清零；
- 省略 `used_cores` 或传零的语义；
- Hard 模式行为不变。

## 8. 兼容性与发布策略

### 8.1 推荐方案：直接切换 ABI

推荐实现在同一次改动中从 ODS、Parser、Verifier、Binding 和 EmitC 中移除
UB/L1。

优点：

- 只有一种规范 IR 形式；
- 不存在无用的本地临时空间 operand；
- 生成的 Binding 无法再意外构造已删除的 ABI；
- 不依赖隐藏的 canonicalization；
- 验收标准简单直接。

代价：

- 缓存的旧 `.pto` 文件和 Python IR 生产方必须与 PTOAS wheel 同步升级。

如果 PTOAS 与 PyPTO 可以原子切换到新的 wheel/IR 契约，应使用该方案。

### 8.2 可选方案：保留一版兼容窗口

如果旧 IR 生产方必须在一个过渡版本内继续工作：

1. 临时在 ODS 中保留旧的可选 UB/L1 字段。
2. 扩展自定义 Parser，通过 operand 类型和数量区分新的 `(gm, i32)` 与旧的
   `(gm, local_workspace [, i32])` 形式。
3. 出现旧本地空间 operand 时生成弃用警告。
4. 在内存规划前执行早期 canonicalization，将 op 重建为不包含 UB/L1 的形式。
5. 始终生成新的双参数 C++ 调用。
6. 在下一版本中删除兼容字段和 canonicalization。

早期 canonicalization 是必需的。仅在 `PTOSyncAllToEmitC` 中忽略 UB/L1，
可能会让其分配一直存活到生成本地临时空间和 `TASSIGN`，从而违反新契约。

该方案不会作为默认方案，因为它会为短期迁移路径扩大公共接口和测试矩阵。

## 9. 测试方案

### 9.1 Lit 覆盖

| 用例 | 预期结果 |
| --- | --- |
| Soft AIV-only，显式 used cores | 精确的双参数 AIV 调用 |
| Soft AIC-only，显式 used cores | 精确的双参数 AIC 调用 |
| Soft Mix，显式 used cores | 精确的双参数 Mix 调用 |
| Soft，省略 used cores | 第二个实参为 `int32_t{0}` |
| Hard AIV/AIC/Mix | 零参数调用保持不变 |
| GM 静态容量为 16 | 接受 |
| GM 静态容量为 `4x4` | 接受 |
| GM 静态容量为 15 | 拒绝并给出可操作诊断 |
| GM 动态容量 | 静态 verifier 接受 |
| GM workspace 不是 i32 | 拒绝 |
| memref workspace 不是 GM | 拒绝 |
| used cores 不是 i32 | 拒绝 |

EmitC 正向测试必须包含：

```text
CHECK-NOT: Tile<TileType::Vec
CHECK-NOT: Tile<TileType::Mat
CHECK-NOT: TASSIGN
```

测试必须检查完整调用形式，不能只检查 callee 前缀：

```text
CHECK: SYNCALL<SyncAllMode::Soft, SyncCoreType::AIVOnly>(
CHECK: SYNCALL<SyncAllMode::Soft, SyncCoreType::AICOnly>(
CHECK: SYNCALL<SyncAllMode::Soft, SyncCoreType::Mix>(
```

应使用捕获的 FileCheck 变量保证每个调用只有 GM 和 used-core 两个实参。

### 9.2 Python 示例覆盖

运行现有 `syncall_binding.py` 示例流程并验证：

- 使用重新生成的 Binding 可以成功构造 Python IR；
- 打印出的 PTO IR 只包含 GM 和可选 used cores；
- PTOAS 生成新的 C++ 调用；
- 不再为同步创建本地 `AllocTileOp`。

### 9.3 PTO-ISA 编译验证

使用 PTO-ISA f24 或明确的后继版本：

```bash
git -C "${PTO_ISA_ROOT}" checkout \
  f24f7b736b689cc107b9eb2d362be6a7718fcc99

ptoas --pto-arch=a5 test/lit/pto/syncall_emitc.pto \
  -o build/issue_1061_syncall.cpp
```

使用仓库现有的 A5/bisheng 参数和 `${PTO_ISA_ROOT}/include` 编译生成的 kernel
C++。验证必须覆盖 AIV-only、AIC-only 和 Mix 模板实例。

PR 中应记录准确的 PTO-ISA SHA 和编译命令。推荐做定向 compile-only 验证，
不要为了本 Issue 修改无关 NPU 测试的全局 PTO-ISA pin。

## 10. 计划修改的文件

| 层次 | 文件 | 计划改动 |
| --- | --- | --- |
| ODS | `include/PTO/IR/PTOOps.td` | 删除 UB/L1 operand 并更新说明 |
| Parser/Printer/Verifier | `lib/PTO/IR/PTO.cpp` | 实现新数量规则和容量校验 |
| EmitC | `lib/PTO/Transforms/PTOToEmitC.cpp` | 只生成 GM + used cores，删除旧 helper |
| 正向测试 | `test/lit/pto/syncall_emitc.pto` | 覆盖所有核模式及精确调用形式 |
| 负向测试 | `test/lit/pto/syncall_invalid_*.pto` | 覆盖缺失、非法和容量不足的 GM |
| Python 示例 | `test/samples/SyncAll/syncall_binding.py` | 使用新生成的 Binding |
| PTO 示例 | `test/samples/SyncAll/syncall_binding.pto` | 使用规范新 IR |
| 文档 | `docs/PTO_IR_manual.md` | 说明新 ABI 和运行时责任 |

预计不需要修改 CLI 或 pass pipeline。

## 11. 验收标准

满足以下条件后，实现视为完成：

- AIV-only、AIC-only 和 Mix 的 Soft 调用都严格包含两个 C++ 实参。
- Soft Lowering 不再生成 UB/L1 Tile、`TASSIGN` 或本地临时空间。
- 省略 `used_cores` 时生成显式的带类型零。
- 静态可知 GM 容量小于 16 个 `int32_t` 元素时拒绝。
- Hard `SYNCALL` 输出保持不变。
- Python 构造和打印的 PTO IR 使用新契约。
- 定向 Lit 测试全部通过。
- 生成的 A5 C++ 可以使用所记录的 PTO-ISA f24 或后继 SHA 编译。

## 12. 实施决策与验证结果

本 PR 按以下决策完成实现：

1. 采用第 8.1 节的直接 ABI 切换，不保留旧 UB/L1 operand 的兼容窗口。
2. 省略核数时使用 `emitc.literal`，最终 C++ 第二个实参严格打印为
   `int32_t{0}`，不生成中间局部变量。
3. PTO-ISA 编译验证使用 Issue 指定的准确提交
   `f24f7b736b689cc107b9eb2d362be6a7718fcc99`。

已完成的验证：

- LLVM 21 Release 配置下，TableGen、`PTO.cpp`、`PTOToEmitC.cpp` 和生成的
  Python Binding 编译通过。
- `syncall_emitc.pto`、`syncall_verify.pto`、`syncall_invalid.pto` 三个定向
  Lit 测试全部通过。
- `test/samples/runop.sh -t SyncAll` 通过，生成的 IR 和 C++ 不再包含只供
  Soft `SYNCALL` 使用的 UB/L1 Tile 或 `TASSIGN`。
- AIV-only、AIC-only、Mix 和省略 `used_cores` 的生成 C++ 已使用 PTO-ISA
  f24 头文件完成语法编译。由于 f24 的 macOS CPU-Sim 头文件本身存在重复 ACL
  stub 定义，验证仅对这些无关的 CPU-Sim stub 使用本地编译兼容处理；
  `SYNCALL` 的公共声明和实现保持 f24 原样。
