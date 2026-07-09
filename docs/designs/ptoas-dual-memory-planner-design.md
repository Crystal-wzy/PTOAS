# PTOAS 双内存规划器设计说明

## 背景

PTOAS 之前曾临时删除旧版 `pto-plan-memory` pass，以便让 level1/level2 在没有完整 memplan 的情况下继续编译。同时，`pto.alloc_tile` 的 lowering 路径也从旧的 `pto.alloc_tile -> memref.alloc -> pto.pointer_cast -> pto.bind_tile` 调整为 tile-native 路径：无显式地址的 `pto.alloc_tile` 保留到 memplan，由 memplan 直接补充常量 `addr`，后续 `pto.t*` tile op 继续消费 `!pto.tile_buf`。

随着旧 memplan 恢复，需要同时满足两个需求：

- 默认行为保持稳定，继续使用旧 memplan，避免影响现有 pipeline 和测试基线。
- 保留重写后的 modern memplan，便于继续开发 alias-aware、SPEC_LEVEL_0 复用等新策略。

因此，本次提交将旧版和新版 memplan 拆成两个实现，由 `tools/ptoas/ptoas.cpp` 通过 CLI 参数选择使用哪一个。

## 目标

本次改动的目标如下：

- 默认使用 legacy memplan，保证既有行为和兼容性。
- 通过 `--plan-memory-impl=modern` 显式启用 modern memplan。
- `pto-plan-memory` 仍保持 module 级 pass，pass 内部遍历 `func::FuncOp` 做规划。
- `pto.alloc_tile(no addr)` 不再经过 `memref.alloc` 中间路径，而是由所选 memplan 直接补 `addr`。
- 尽量减少对 legacy memplan 原有代码的结构性改动，只补充 tile-native alloc 所需的最小适配。
- 将原先删除的 memplan lit/sample 用例恢复，并让现有 `plan_memory_*.pto` lit 同时覆盖 legacy 和 modern。

## 非目标

本次改动不试图完成 modern memplan 的全部设计目标：

- 不实现 SPEC_LEVEL_1 / SPEC_LEVEL_2 的完整投机复用策略。
- 不替换 legacy memplan 的 pipeline conflict、double-buffer 相关逻辑。
- 不改变 level3 显式地址模式的规则。
- 不把 `reserve_buffer` 手动 base 支持扩展到 level1/level2。

## 设计方案

### 文件组织

旧 memplan 仍保留在原文件：

```text
lib/PTO/Transforms/PTOPlanMemory.cpp
lib/PTO/Transforms/PTOPlanMemory.h
```

modern memplan 放在独立文件：

```text
lib/PTO/Transforms/PTOPlanMemoryModern.cpp
```

`PTOPlanMemory.cpp` 继续定义默认 pass factory：

```cpp
createPlanMemoryPass(const PlanMemoryOptions &options = {})
```

`PTOPlanMemoryModern.cpp` 定义 modern factory：

```cpp
createPlanMemoryModernPass(const PlanMemoryOptions &options)
```

`lib/PTO/Transforms/CMakeLists.txt` 同时编译两个实现文件。

### Pass 级别

`pto-plan-memory` 保持 module 级 pass：

```td
def PlanMemory : Pass<"pto-plan-memory", "ModuleOp">
```

legacy 和 modern 都在 module pass 的 `runOnOperation()` 中遍历 module 内的 `func::FuncOp`，分别对每个函数做静态 local memory planning。这样保留旧 memplan 的 module-level 调用形态，也避免在 pass manager pipeline 中暴露两个不同的 nested func pass。

### 实现选择

实现选择不放在 `PTOPlanMemory.cpp` 内部，而是在 `tools/ptoas/ptoas.cpp` 中完成：

```cpp
if (planMemoryImpl == "legacy") {
  pm.addPass(pto::createPlanMemoryPass(planMemoryOptions));
} else if (planMemoryImpl == "modern") {
  pm.addPass(pto::createPlanMemoryModernPass(planMemoryOptions));
}
```

CLI 参数为：

```text
--plan-memory-impl=legacy  # 默认
--plan-memory-impl=modern
```

这样 `PTOPlanMemory.cpp` 可以继续表示 legacy memplan 本体，而不是变成 dispatcher。

### `memMode`

`PlanMemoryOptions::memMode` 当前仍由 `ptoas.cpp` 固定设置为：

```cpp
planMemoryOptions.memMode = "local";
```

含义是规划 local memory buffer。legacy memplan 内部仍保留 `MemPlanMode` 枚举，用于尽量少改旧代码：

```cpp
enum class MemPlanMode {
  LOCAL_MEM_PLAN,
  GLOBAL_WORKSPACE_PLAN,
};
```

当前 PTOAS 主 pipeline 只使用 `"local"`。`GLOBAL_WORKSPACE_PLAN` 是 legacy 代码的历史模式保留，不作为当前 level1/level2 local memplan 的正式入口。

### `pto.alloc_tile(no addr)` 处理

`PTOViewToMemref` 不再把无地址的 `pto.alloc_tile` 降成 `memref.alloc`。因此，两个 memplan 都需要直接处理 tile-native allocation root：

```mlir
%tile = pto.alloc_tile : !pto.tile_buf<...>
```

memplan 完成规划后直接补常量地址：

```mlir
%c0_i64 = arith.constant 0 : i64
%tile = pto.alloc_tile addr = %c0_i64 : !pto.tile_buf<...>
```

legacy memplan 为此增加了最小适配：

- liveness 收集阶段识别 `pto.alloc_tile(no addr)`。
- `BufferInfo` 支持从 `TileBufType` 计算 shape、element type 和字节数。
- materialize 阶段新增 `pto.alloc_tile -> pto.alloc_tile addr` 的 rewrite。
- `pto.tile_buf_addr` 被建模为读取 tile source，避免被未知 local-buffer op 检查误判。

### `memref.alloc` 处理

非 tile-native 的 `memref.alloc` root 仍沿用 legacy/modern 各自的既有 materialization 方式：

```mlir
memref.alloc -> pto.pointer_cast(...)
```

也就是说，本次改动只要求 `pto.alloc_tile` 不再经过 `memref.alloc` 中间链路；并不禁止普通 memref root 继续使用 `pto.pointer_cast` 表达规划地址。

## 测试方案

测试覆盖分为三类。

### 恢复旧 memplan 测试

恢复 `test/samples/planmemory` 下之前删除的 sample case，覆盖 loop、if、nested loop、fragmentation、peak capacity、reuse 等场景。

### lit 双实现覆盖

现有 `test/lit/pto/plan_memory_*.pto` 增加 modern RUN：

```mlir
// RUN: ptoas ... | FileCheck %s
// RUN: ptoas --plan-memory-impl=modern ... | FileCheck %s
```

`order_by_size` 用例同时覆盖 legacy/modern 的默认顺序和 `--plan-memory-order-by-size` 顺序。负例也同时覆盖 legacy/modern。

### 回归命令

本次提交使用以下命令验证：

```bash
cmake --build build --target ptoas -j8

PATH=/Users/fangrui/workspace/huawei/llvm21-workspace/llvm-project/llvm/build-assert/bin:$PATH \
  /Users/fangrui/workspace/huawei/llvm21-workspace/llvm-project/llvm/build-assert/bin/llvm-lit \
  -sv build/test/lit \
  --filter 'plan_memory|reserve_buffer|alloc_tile_addr|alloc_tile_plan_memory_no_memref_alloc|multi_tile_get_const_slot_lowering|multi_tile_.*planmem'

ctest --test-dir build --output-on-failure -L PTODSL
```

同时对恢复的 `test/samples/planmemory/*.py` 做逐个生成和 `ptoas --emit-pto-ir` 编译验证。

## 后续工作

- modern memplan 后续继续补齐 SPEC_LEVEL_1 / SPEC_LEVEL_2。
- 根据 modern 策略差异，为必要用例拆分 legacy/modern 的独立 FileCheck 前缀。
- 如果确认 global workspace planning 不再需要，可单独清理 legacy 内部的 `GLOBAL_WORKSPACE_PLAN` 历史模式。
- 继续减少 legacy 文件中的非必要 diff，保持旧实现可对照、可回退。
