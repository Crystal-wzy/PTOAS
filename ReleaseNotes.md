# PTOAS (PTO Assembler & Optimizer)

## 未发布

### 不兼容变更

- Soft `pto.syncall` 的 operand ABI 与 PTO-ISA 新接口对齐，从按核类型区分的
  `gm_workspace + ub_workspace/l1_workspace [+ used_cores]` 统一为
  `gm_workspace [+ used_cores]`。
- 旧的三/四 operand `.pto` 文本及包含旧 `operandSegmentSizes` 的 `.ptobc`
  不再兼容。使用方需升级 PTOAS 与 PyPTO/其他 IR 生产者，并重新生成已落盘的
  PTO IR 和 PTO Bytecode。
- 生成的新 C++ 需要包含两参数 Soft `SYNCALL` ABI 的 PTO-ISA。GitHub 基准提交为
  `f24f7b736b689cc107b9eb2d362be6a7718fcc99`；GitCode 的
  `ce3262e3825a235f951917eeada30e52910b6a84` 已通过等价提交
  `d56d42db6a3c14eb195de85392a69b68b862a87c` 包含该接口。

## 版本
- 版本号：v0.51
- 发布日期：2026-02-14

## 变更摘要
- PTOAS 首次发布

## 概述
PTOAS（PTO Assembler & Optimizer）是面向 PTO Bytecode 的编译器工具链，基于 LLVM/MLIR LLVM21 VPTO 分支 `vpto-dev/llvm-project:feature-vpto-llvm21` 构建。它提供 PTO Dialect 的定义、解析、验证、优化与代码生成能力，并输出可调用 `pto-isa` 的 C++ 代码。

PTOAS很快将集成到以下框架中，敬请期待
- PyPTO
- TileLang

## 本仓库的目标用户
PTOAS 主要面向：
- 编译器与框架后端开发者
- 高性能算子/内核开发者
- 需要进行 PTO Bytecode 生成、调试与落地的工程团队

## 主要能力
- PTO Dialect 全流程（定义、解析、验证、打印）
- 与 Tile 抽象/地址空间/同步模型配套的 IR 支撑
- PTO Bytecode → C++ 生成
- Python 端的 Dialect 构建与测试样例

## 平台与依赖最低配置
- **操作系统**：macOS (Darwin) 或 Linux (Ubuntu 20.04+)
- **编译器**：Clang >= 12 或 GCC >= 9（支持 C++17）
- **构建工具**：CMake >= 3.20，Ninja
- **Python**：Python 3.8+

## 如何使用PTOAS以及PTO IR的详细描述
- 构建与环境配置：`README.md`
- PTO Bytecode 定义：`docs/PTO_IR_manual.md`
