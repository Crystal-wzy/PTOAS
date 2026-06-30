// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "ptoas.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOLLVMEmitter.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/BufferizableOpInterfaceImpl.h"
#include "VPTOHostStubEmission.h"
#include "TilelangDaemon.h"
#include "PTO/Transforms/CppPostprocess.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include <cctype>
#include <cstring>
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/FileSystem.h" // [Fix] Required for OF_None
#include "llvm/Support/Path.h"
#include "ptobc/ptobc_decode.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <chrono>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>

extern "C" {
extern char **environ;
}

using namespace mlir;
using namespace pto;

#ifndef PTOAS_RELEASE_VERSION
#define PTOAS_RELEASE_VERSION "unknown"
#endif

namespace {

constexpr unsigned kSeenCalleeInlineCapacity = 8;
constexpr int kDefaultGraphSyncSolverEventIdMax = 8;
constexpr unsigned kStringRefInlineCapacity = 4;
constexpr unsigned kEmptyExpressionInlineCapacity = 8;
constexpr unsigned kBranchInlineCapacity = 16;
constexpr size_t kMarkerCallReserveExtra = 16;
constexpr size_t kRewriteOutputReserveExtra = 64;
constexpr size_t kMarkerRewriteMinArgCount = 2;
constexpr size_t kMarkerRewriteTernaryArgCount = 3;

using StringRefVector =
    llvm::SmallVector<llvm::StringRef, kStringRefInlineCapacity>;

} // namespace

int main(int argc, char **argv);

void mlir::pto::registerPTOASDialects(DialectRegistry &registry) {
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::math::MathDialect>();

  registry.insert<mlir::pto::PTODialect>();
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  pto::registerBufferizableOpInterfaceExternalModels(registry);

  registry.insert<emitc::EmitCDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
}

void mlir::pto::registerPTOASPassesAndCLOptions() {
  mlir::registerAllPasses();
  mlir::pto::registerPTOPasses();
  mlir::pto::registerPTOViewToMemrefPass();
  mlir::pto::registerPTOInlineLibCall();
  mlir::pto::registerFoldTileBufIntrinsics();
  mlir::pto::registerExpandTileOp();
  mlir::registerPassManagerCLOptions();
}

void mlir::pto::loadPTOASDialects(MLIRContext &context) {
  context.getOrLoadDialect<emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::pto::PTODialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<math::MathDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<affine::AffineDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
}

static std::string getParentDir(llvm::StringRef path) {
  llvm::SmallString<256> parent(path);
  llvm::sys::path::remove_filename(parent);
  llvm::sys::path::remove_dots(parent, true);
  return std::string(parent);
}

static bool pathExists(llvm::StringRef path) {
  return !path.empty() && llvm::sys::fs::exists(path);
}

static std::string joinPath(llvm::StringRef lhs, llvm::StringRef rhs) {
  llvm::SmallString<256> joined(lhs);
  llvm::sys::path::append(joined, rhs);
  llvm::sys::path::remove_dots(joined, true);
  return std::string(joined);
}

static std::string detectInstalledTilelangPath(const char *argv0) {
  std::string exePath = llvm::sys::fs::getMainExecutable(argv0, (void *)&main);
  if (exePath.empty())
    return {};

  const std::string exeDir = getParentDir(exePath);
  const std::string prefixDir = getParentDir(exeDir);
  const std::string installedTileOps = joinPath(prefixDir, "share/ptoas/TileOps");
  if (pathExists(installedTileOps))
    return installedTileOps;
  return {};
}

static std::string detectInstalledTilelangPkgPath(const char *argv0) {
  std::string exePath = llvm::sys::fs::getMainExecutable(argv0, (void *)&main);
  if (exePath.empty())
    return {};

  const std::string exeDir = getParentDir(exePath);
  const std::string prefixDir = getParentDir(exeDir);
  const std::string installedPkgRoot = prefixDir;
  const std::string installedPkg = joinPath(installedPkgRoot, "tilelang_dsl");
  if (pathExists(installedPkg))
    return installedPkgRoot;
  return {};
}

static bool hasCLIOption(int argc, char **argv, llvm::StringRef option) {
  const std::string optionWithValue = (option + "=").str();
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg == option || arg.starts_with(optionWithValue))
      return true;
  }
  return false;
}

static LogicalResult applyConfiguredPassManagerCLOptions(
    PassManager &pm, llvm::StringRef pipelineName,
    llvm::raw_ostream &diagOS = llvm::errs()) {
  if (succeeded(mlir::applyPassManagerCLOptions(pm)))
    return success();
  diagOS << "Error: failed to apply MLIR pass manager command-line options for "
         << pipelineName << ".\n";
  return failure();
}

static LogicalResult reorderEmitCFunctions(ModuleOp module) {
  SmallVector<emitc::FuncOp> declarations;
  SmallVector<emitc::FuncOp> definitions;
  llvm::DenseMap<StringAttr, emitc::FuncOp> definitionsByName;

  for (auto func : module.getOps<emitc::FuncOp>()) {
    if (func.isDeclaration()) {
      declarations.push_back(func);
      continue;
    }
    definitions.push_back(func);
    definitionsByName[func.getSymNameAttr()] = func;
  }

  llvm::DenseMap<Operation *, unsigned> indegree;
  llvm::DenseMap<Operation *, SmallVector<Operation *>> outgoing;
  for (auto func : definitions)
    indegree[func.getOperation()] = 0;

  for (auto caller : definitions) {
    Operation *callerOp = caller.getOperation();
    llvm::SmallPtrSet<Operation *, kSeenCalleeInlineCapacity> seenCallees;
    bool hasCycle = false;
    caller.walk([&](emitc::CallOp call) {
      auto calleeAttr = call.getCalleeAttr();
      if (!calleeAttr)
        return;
      auto it = definitionsByName.find(calleeAttr.getLeafReference());
      if (it == definitionsByName.end())
        return;
      Operation *calleeOp = it->second.getOperation();
      if (calleeOp == callerOp) {
        hasCycle = true;
        return;
      }
      if (!seenCallees.insert(calleeOp).second)
        return;
      outgoing[calleeOp].push_back(callerOp);
      ++indegree[callerOp];
    });
    if (hasCycle) {
      return caller.emitOpError()
             << "recursive function calls are not supported for EmitC C++ "
                "emission";
    }
  }

  SmallVector<Operation *> ready;
  for (auto func : definitions) {
    if (indegree[func.getOperation()] == 0)
      ready.push_back(func.getOperation());
  }

  SmallVector<emitc::FuncOp> sortedDefinitions;
  while (!ready.empty()) {
    Operation *next = ready.front();
    ready.erase(ready.begin());
    auto nextFunc = cast<emitc::FuncOp>(next);
    sortedDefinitions.push_back(nextFunc);

    for (Operation *user : outgoing[next]) {
      unsigned &userIndegree = indegree[user];
      if (--userIndegree == 0)
        ready.push_back(user);
    }
  }

  if (sortedDefinitions.size() != definitions.size()) {
    return module.emitError()
           << "cyclic function call graph is not supported for EmitC C++ emission";
  }

  if (declarations.empty() && definitions.size() <= 1)
    return success();

  SmallVector<emitc::FuncOp> desiredOrder;
  desiredOrder.append(declarations.begin(), declarations.end());
  desiredOrder.append(sortedDefinitions.begin(), sortedDefinitions.end());

  Block &body = module.getBodyRegion().front();
  Operation *anchor = nullptr;
  for (Operation &op : body.getOperations()) {
    if (isa<emitc::FuncOp>(op)) {
      anchor = &op;
      break;
    }
  }
  if (!anchor)
    return success();

  auto advanceAnchor = [&]() {
    while (anchor) {
      anchor = anchor->getNextNode();
      if (!anchor || isa<emitc::FuncOp>(anchor))
        return;
    }
  };

  for (auto func : desiredOrder) {
    if (func.getOperation() == anchor) {
      advanceAnchor();
      continue;
    }
    if (anchor)
      func->moveBefore(anchor);
    else
      func->moveBefore(&body, body.end());
  }

  return success();
}

// --------------------------------------------------------------------------
// Command Line Options
// --------------------------------------------------------------------------
static llvm::cl::opt<bool> enableInsertSync("enable-insert-sync",
                                            llvm::cl::desc("Enable automatic synchronization insertion pass"),
                                            llvm::cl::init(false));

static llvm::cl::opt<bool> enableBufidSync(
    "enable-bufid_sync",
    llvm::cl::desc("Enable A5 buffer-id synchronization insertion pass"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> enableBufidSyncDebug(
    "enable-bufid-sync-debug",
    llvm::cl::desc("Enable verbose debug printing for --enable-bufid_sync"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> enableInjectBarrierAllSync(
    "enable-inject-barrier-all-sync",
    llvm::cl::desc("Enable conservative synchronization by inserting "
                   "pto.barrier PIPE_ALL before memory-effecting PTO pipe ops"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> enableGraphSyncSolver(
    "enable-graph-sync-solver",
    llvm::cl::desc("Enable the graph-based intra-core sync solver "
                   "(experimental). Mutually exclusive with "
                   "--enable-insert-sync, --enable-bufid_sync, and "
                   "--enable-inject-barrier-all-sync."),
    llvm::cl::init(false));

static llvm::cl::opt<int> graphSyncSolverEventIdMax(
    "graph-sync-solver-event-id-max",
    llvm::cl::desc(
        "Maximum EVENT_ID slots for the graph sync solver (default 8). "
        "Lower values exercise the PIPE_ALL coloring fallback sooner."),
    llvm::cl::init(kDefaultGraphSyncSolverEventIdMax));

static llvm::cl::opt<bool> enableTileOpExpand(
    "enable-tile-op-expand",
    llvm::cl::desc(
        "Deprecated compatibility flag. TileOp expansion is controlled by "
        "--pto-backend=vpto."),
    llvm::cl::init(false));

#ifndef PTOAS_DEFAULT_TILELANG_PATH
#define PTOAS_DEFAULT_TILELANG_PATH ""
#endif
#ifndef PTOAS_DEFAULT_TILELANG_PKG_PATH
#define PTOAS_DEFAULT_TILELANG_PKG_PATH ""
#endif

static llvm::cl::opt<std::string> tilelangPath(
    "tilelang-path",
    llvm::cl::desc("Path to directory of .py tilelang DSL template files "
                   "(default: <source>/lib/TileOps, baked in at build time)"),
    llvm::cl::init(PTOAS_DEFAULT_TILELANG_PATH));

static llvm::cl::opt<std::string> tilelangPkgPath(
    "tilelang-pkg-path",
    llvm::cl::desc("PYTHONPATH for tilelang_dsl package "
                   "(default: <source>/tilelang-dsl/python, baked in at build time)"),
    llvm::cl::init(PTOAS_DEFAULT_TILELANG_PKG_PATH));

static llvm::cl::opt<std::string> daemonSocketPath(
    "daemon-socket-path",
    llvm::cl::desc("Path to Unix domain socket for daemon RPC "
                   "(default: /tmp/tilelang_daemon_{pid}.sock)"),
    llvm::cl::init(""));

static pto::ExpandTileOpOptions resolveExpandTileOpOptions(int argc,
                                                           char **argv) {
  pto::ExpandTileOpOptions expandOpts;
  expandOpts.tilelangPath = tilelangPath;
  expandOpts.tilelangPkgPath = tilelangPkgPath;

  if (!hasCLIOption(argc, argv, "--tilelang-path")) {
    std::string detectedTilelangPath = detectInstalledTilelangPath(argv[0]);
    if (!detectedTilelangPath.empty())
      expandOpts.tilelangPath = detectedTilelangPath;
  }

  if (!hasCLIOption(argc, argv, "--tilelang-pkg-path")) {
    std::string detectedTilelangPkgPath = detectInstalledTilelangPkgPath(argv[0]);
    if (!detectedTilelangPkgPath.empty())
      expandOpts.tilelangPkgPath = detectedTilelangPkgPath;
  }

  // Daemon mode is default (no CLI option needed)
  // Automatically start daemon for instance caching
  if (!expandOpts.tilelangPath.empty()) {
    std::string socket = daemonSocketPath;
    if (socket.empty())
      socket = ptoas::DaemonManager::generateSocketPath();

    // Register cleanup handler (daemon will be stopped on PTOAS exit)
    ptoas::registerDaemonCleanup();

    // Try to start daemon automatically
    if (ptoas::DaemonManager::start(socket, expandOpts.tilelangPath, expandOpts.tilelangPkgPath)) {
      expandOpts.daemonSocketPath = socket;
      llvm::errs() << "Info: TileLang daemon started successfully\n";
    } else {
      // Fallback: daemon failed, use subprocess mode (current approach)
      expandOpts.daemonSocketPath = "";
      llvm::errs() << "Warning: Failed to start daemon, using subprocess mode (fallback)\n";
    }
  }

  return expandOpts;
}

static llvm::cl::opt<bool> enableOpFusion(
    "enable-op-fusion",
    llvm::cl::desc("Enable A5 tile fusion on level2/level3. EmitC uses "
                   "last-use annotation; VPTO uses fusion-region lifecycle."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> enableShapeInference(
    "enable-shape-inference",
    llvm::cl::desc("Enable shape inference (ShapeConstraintSolver) for A5 tile "
                  "fusion. Off by default: falls back to static/direct-bound "
                  "iteration-domain inference."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> disableInferLayout(
    "disable-infer-layout",
    llvm::cl::desc("Disable PTO layout inference pass (static-only)"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> emitAddPtrTrace(
    "emit-addptr-trace",
    llvm::cl::desc("Emit addptr trace comments in generated C++ output"),
    llvm::cl::init(false));

llvm::cl::opt<bool> mlir::pto::emitMlirIR(
    "emit-pto-ir",
    llvm::cl::desc("Emit PTO IR after lowering instead of C++"),
    llvm::cl::init(false));

llvm::cl::opt<std::string> mlir::pto::ptoTargetArch(
    "pto-arch",
    llvm::cl::desc("Target Ascend architecture for codegen: a3 or a5 (default: a3)"),
    llvm::cl::value_desc("a3|a5"),
    llvm::cl::init("a3"));

static llvm::cl::opt<std::string> ptoBuildLevel(
    "pto-level",
    llvm::cl::desc("Build level for pass pipeline: level1, level2, or level3 (default: level2)"),
    llvm::cl::value_desc("level1|level2|level3"),
    llvm::cl::init("level2"));

llvm::cl::opt<std::string> mlir::pto::ptoBackend(
    "pto-backend",
    llvm::cl::desc("Final PTOAS backend: emitc or vpto (default: emitc)"),
    llvm::cl::value_desc("emitc|vpto"), llvm::cl::init("emitc"));

llvm::cl::opt<bool> mlir::pto::emitVPTO(
    "emit-vpto",
    llvm::cl::desc("Write final post-pass VPTO IR to -o"),
    llvm::cl::init(false));

llvm::cl::opt<bool> mlir::pto::emitVPTOLLVMDialect(
    "emit-vpto-llvm-ir",
    llvm::cl::desc("Write translated VPTO LLVM IR to -o"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> vptoPrintIR(
    "vpto-print-ir",
    llvm::cl::desc("Print post-pass VPTO backend IR to stderr"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> vptoLoweringStrategy(
    "vpto-lowering-strategy",
    llvm::cl::desc("VPTO vector lowering strategy: post-update or no-post-update"),
    llvm::cl::value_desc("post-update|no-post-update"),
    llvm::cl::init("post-update"));

static llvm::cl::opt<bool> dumpVPTOIR(
    "dump-vpto-ir",
    llvm::cl::desc("Print post-pass VPTO backend IR to stderr"),
    llvm::cl::init(false));

llvm::cl::opt<bool> mlir::pto::ptoPrintSeamIR(
    "pto-print-seam-ir",
    llvm::cl::desc("Print shared pre-backend seam IR to stderr"),
    llvm::cl::init(false));

llvm::cl::opt<std::string> mlir::pto::ptoSeamIRFile(
    "pto-seam-ir-file",
    llvm::cl::desc("Write shared pre-backend seam IR to a file"),
    llvm::cl::value_desc("path"),
    llvm::cl::init(""));

llvm::cl::opt<std::string> mlir::pto::cannOutputVersion(
    "cann-output-version",
    llvm::cl::desc("Override the CANN version used for lowering and public ABI output selection; examples: 9.0.0, 9.0.0-beta.1"),
    llvm::cl::value_desc("version"), llvm::cl::init(""));

enum class PTOBuildLevel {
  Level1,
  Level2,
  Level3,
};

static PTOBuildLevel defaultBuildLevel() {
  return PTOBuildLevel::Level2;
}

static bool parseBuildLevel(llvm::StringRef levelStr, PTOBuildLevel &out) {
  std::string s = levelStr.str();
  for (char &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "level1") {
    out = PTOBuildLevel::Level1;
    return true;
  }
  if (s == "level2") {
    out = PTOBuildLevel::Level2;
    return true;
  }
  if (s == "level3") {
    out = PTOBuildLevel::Level3;
    return true;
  }
  return false;
}

static constexpr llvm::StringLiteral kAutoSyncTailPolicyBarrierAll =
    "barrier_all";
static constexpr llvm::StringLiteral kAutoSyncTailPolicyMte3ToSEvent0 =
    "setwait_mte3_to_s_event0";

static bool parseAutoSyncTailHint(llvm::StringRef hintStr, std::string &normalized) {
  std::string s = hintStr.str();
  for (char &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "barrier-all" || s == "barrier_all" || s == "default") {
    normalized = kAutoSyncTailPolicyBarrierAll.str();
    return true;
  }
  if (s == "mte3-to-s-event0" || s == "mte3_to_s_event0" ||
      s == "setwait-mte3-to-s-event0" ||
      s == "setwait_mte3_to_s_event0") {
    normalized = kAutoSyncTailPolicyMte3ToSEvent0.str();
    return true;
  }
  return false;
}

static LogicalResult emitSharedPreBackendSeamIR(ModuleOp module,
                                                llvm::StringRef outputPath) {
  if (outputPath.empty())
    return success();

  if (outputPath == "-") {
    module->print(llvm::outs());
    llvm::outs() << "\n";
    llvm::outs().flush();
    return success();
  }

  std::error_code ec;
  llvm::ToolOutputFile outputFile(outputPath, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "Error: failed to open seam IR file '" << outputPath
                 << "': " << ec.message() << "\n";
    return failure();
  }

  module->print(outputFile.os());
  outputFile.os() << "\n";
  outputFile.keep();
  return success();
}

static bool hasUnexpandedTileOps(ModuleOp module) {
  bool found = false;
  module.walk([&](Operation *op) {
    if (found)
      return;
    if (isa<pto::OpPipeInterface>(op))
      found = true;
  });
  return found;
}

struct ParsedTextualNameHints {
  llvm::SmallVector<llvm::SmallVector<std::string, 4>, 8> functionArgHints;
  llvm::SmallVector<llvm::SmallVector<std::string, 4>, 8> blockArgHints;
  llvm::SmallVector<llvm::SmallVector<std::string, 4>, 32> opResultHints;
};

using FunctionArgHintMap =
    llvm::StringMap<llvm::SmallVector<std::string, 4>>;
using FunctionBlockArgHintMap =
    llvm::StringMap<llvm::SmallVector<llvm::SmallVector<std::string, 4>, 4>>;

static bool isGeneratedValueName(llvm::StringRef name);
static SmallVector<std::string, 4> getValueNameHints(Value value);

static bool isMlirValueNameChar(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' ||
         c == '$' || c == '-';
}

static bool isCppIdentifierStart(char c) {
  return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

static bool isCppIdentifierChar(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

static bool parseMlirValueName(llvm::StringRef text, size_t &pos,
                               std::string &name) {
  if (pos >= text.size() || text[pos] != '%')
    return false;
  size_t begin = ++pos;
  while (pos < text.size() && isMlirValueNameChar(text[pos]))
    ++pos;
  if (pos == begin)
    return false;
  name = text.slice(begin, pos).str();
  return true;
}

static llvm::SmallVector<std::string, 4>
parseMlirNamedArguments(llvm::StringRef text) {
  llvm::SmallVector<std::string, 4> names;
  for (size_t pos = 0; pos < text.size();) {
    if (text[pos] != '%') {
      ++pos;
      continue;
    }
    std::string name;
    size_t namePos = pos;
    if (!parseMlirValueName(text, namePos, name)) {
      ++pos;
      continue;
    }
    while (namePos < text.size() &&
           std::isspace(static_cast<unsigned char>(text[namePos])))
      ++namePos;
    if (namePos < text.size() && text[namePos] == ':')
      names.push_back(std::move(name));
    pos = namePos;
  }
  return names;
}

static bool parseLeadingOpResultNames(
    llvm::StringRef line, llvm::SmallVectorImpl<std::string> &names) {
  size_t pos = 0;
  while (pos < line.size() &&
         std::isspace(static_cast<unsigned char>(line[pos])))
    ++pos;
  if (pos >= line.size() || line[pos] != '%')
    return false;

  while (true) {
    std::string name;
    if (!parseMlirValueName(line, pos, name))
      return false;
    names.push_back(std::move(name));

    while (pos < line.size() &&
           std::isspace(static_cast<unsigned char>(line[pos])))
      ++pos;
    if (pos < line.size() && line[pos] == ',') {
      ++pos;
      while (pos < line.size() &&
             std::isspace(static_cast<unsigned char>(line[pos])))
        ++pos;
      continue;
    }
    break;
  }

  while (pos < line.size() &&
         std::isspace(static_cast<unsigned char>(line[pos])))
    ++pos;
  if (pos < line.size() && line[pos] == ':') {
    ++pos;
    size_t countBegin = pos;
    while (pos < line.size() &&
           std::isdigit(static_cast<unsigned char>(line[pos])))
      ++pos;
    if (pos == countBegin)
      return false;
    while (pos < line.size() &&
           std::isspace(static_cast<unsigned char>(line[pos])))
      ++pos;
  }
  return pos < line.size() && line[pos] == '=';
}

static std::string stripMlirLineComments(llvm::StringRef text) {
  std::string stripped;
  stripped.reserve(text.size());
  llvm::StringRef remaining = text;
  while (!remaining.empty()) {
    auto split = remaining.split('\n');
    llvm::StringRef line = split.first;
    llvm::StringRef rest = split.second;
    llvm::StringRef body = line;
    if (size_t commentPos = line.find("//"); commentPos != llvm::StringRef::npos)
      body = line.take_front(commentPos);
    stripped.append(body.begin(), body.end());
    if (!rest.empty())
      stripped.push_back('\n');
    remaining = rest;
  }
  return stripped;
}

static ParsedTextualNameHints extractTextualNameHints(llvm::StringRef text) {
  ParsedTextualNameHints hints;
  std::string stripped = stripMlirLineComments(text);
  llvm::StringRef source(stripped);

  size_t searchPos = 0;
  while (true) {
    size_t funcPos = source.find("func.func", searchPos);
    if (funcPos == llvm::StringRef::npos)
      break;

    size_t atPos = source.find('@', funcPos);
    if (atPos == llvm::StringRef::npos)
      break;
    size_t lParenPos = source.find('(', atPos);
    if (lParenPos == llvm::StringRef::npos)
      break;

    int depth = 0;
    size_t rParenPos = llvm::StringRef::npos;
    for (size_t i = lParenPos; i < source.size(); ++i) {
      if (source[i] == '(') {
        ++depth;
      } else if (source[i] == ')') {
        --depth;
        if (depth == 0) {
          rParenPos = i;
          break;
        }
      }
    }
    if (rParenPos == llvm::StringRef::npos)
      break;

    hints.functionArgHints.push_back(
        parseMlirNamedArguments(source.slice(lParenPos + 1, rParenPos)));
    searchPos = rParenPos + 1;
  }

  llvm::StringRef remaining(source);
  while (!remaining.empty()) {
    auto split = remaining.split('\n');
    llvm::StringRef line = split.first;
    llvm::StringRef rest = split.second;
    llvm::StringRef trimmed = line.ltrim();

    if (trimmed.starts_with("^")) {
      size_t lParenPos = trimmed.find('(');
      size_t rParenPos = trimmed.rfind(')');
      size_t colonPos = trimmed.rfind(':');
      if (lParenPos != llvm::StringRef::npos &&
          rParenPos != llvm::StringRef::npos && colonPos != llvm::StringRef::npos &&
          lParenPos < rParenPos && rParenPos < colonPos) {
        hints.blockArgHints.push_back(
            parseMlirNamedArguments(trimmed.slice(lParenPos + 1, rParenPos)));
      }
    }

    llvm::SmallVector<std::string, 4> resultNames;
    if (parseLeadingOpResultNames(trimmed, resultNames))
      hints.opResultHints.push_back(std::move(resultNames));

    remaining = rest;
  }

  return hints;
}

static std::string sanitizeCppIdentifier(llvm::StringRef name) {
  std::string sanitized;
  sanitized.reserve(name.size() + 4);

  auto appendUnderscore = [&]() {
    if (sanitized.empty() || sanitized.back() != '_')
      sanitized.push_back('_');
  };

  for (char c : name) {
    if (isCppIdentifierChar(c))
      sanitized.push_back(c);
    else
      appendUnderscore();
  }

  while (!sanitized.empty() && sanitized.back() == '_')
    sanitized.pop_back();

  if (sanitized.empty())
    return {};
  if (!isCppIdentifierStart(sanitized.front()))
    sanitized.insert(sanitized.begin(), '_');
  return sanitized;
}

static bool isReservedCppIdentifier(llvm::StringRef name) {
  static const std::set<std::string> kReserved = {
      "alignas",         "alignof",        "and",            "and_eq",
      "asm",             "auto",           "bitand",         "bitor",
      "bool",            "break",          "case",           "catch",
      "char",            "char8_t",        "char16_t",       "char32_t",
      "class",           "co_await",       "co_return",      "co_yield",
      "compl",           "concept",        "const",          "const_cast",
      "consteval",       "constexpr",      "constinit",      "continue",
      "decltype",        "default",        "delete",         "do",
      "double",          "dynamic_cast",   "else",           "enum",
      "explicit",        "export",         "extern",         "false",
      "float",           "for",            "friend",         "goto",
      "if",              "import",         "inline",         "int",
      "long",            "module",         "mutable",        "namespace",
      "new",             "noexcept",       "not",            "not_eq",
      "nullptr",         "operator",       "or",             "or_eq",
      "private",         "protected",      "public",         "register",
      "reinterpret_cast","requires",       "return",         "short",
      "signed",          "sizeof",         "static",         "static_assert",
      "static_cast",     "struct",         "switch",         "template",
      "this",            "thread_local",   "throw",          "true",
      "try",             "typedef",        "typeid",         "typename",
      "union",           "unsigned",       "using",          "virtual",
      "void",            "volatile",       "wchar_t",        "while",
      "xor",             "xor_eq"};
  return kReserved.count(name.str()) != 0;
}

static std::string makeUniqueCppIdentifier(llvm::StringRef baseName,
                                           std::set<std::string> &usedNames) {
  std::string candidate = sanitizeCppIdentifier(baseName);
  if (candidate.empty())
    return {};
  if (isReservedCppIdentifier(candidate))
    candidate.append("_v");

  std::string unique = candidate;
  unsigned suffix = 1;
  while (!usedNames.insert(unique).second) {
    unique = candidate + "_" + std::to_string(suffix++);
  }
  return unique;
}

static void appendLocationNameHints(Location loc,
                                    SmallVectorImpl<std::string> &hints) {
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    std::string sanitized = sanitizeCppIdentifier(nameLoc.getName().getValue());
    if (!sanitized.empty())
      hints.push_back(std::move(sanitized));
    return;
  }

  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    if (Attribute metadata = fusedLoc.getMetadata()) {
      if (auto strAttr = dyn_cast<StringAttr>(metadata)) {
        std::string sanitized = sanitizeCppIdentifier(strAttr.getValue());
        if (!sanitized.empty())
          hints.push_back(std::move(sanitized));
        return;
      }
      if (auto arrayAttr = dyn_cast<ArrayAttr>(metadata)) {
        for (Attribute attr : arrayAttr) {
          auto strAttr = dyn_cast<StringAttr>(attr);
          if (!strAttr)
            continue;
          std::string sanitized = sanitizeCppIdentifier(strAttr.getValue());
          if (!sanitized.empty())
            hints.push_back(std::move(sanitized));
        }
        if (!hints.empty())
          return;
      }
    }

    for (Location childLoc : fusedLoc.getLocations())
      appendLocationNameHints(childLoc, hints);
    return;
  }

  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc)) {
    appendLocationNameHints(callSiteLoc.getCallee(), hints);
    if (hints.empty())
      appendLocationNameHints(callSiteLoc.getCaller(), hints);
  }
}

static bool hasLocationNameHints(Location loc) {
  SmallVector<std::string, 4> hints;
  appendLocationNameHints(loc, hints);
  return !hints.empty();
}

// Read the *raw* (unsanitized) source SSA name hints carried in the Location
// metadata. Unlike appendLocationNameHints, this preserves the original textual
// form (e.g. "0", "24", "query_tile") so that issue #337's "pto: %N" provenance
// comments can map a generated C++ variable back to its input .pto SSA name,
// even for pure-digit names that would otherwise be sanitized to "_0".
static void appendRawLocationProvenance(Location loc,
                                        SmallVectorImpl<std::string> &hints) {
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    std::string raw = nameLoc.getName().getValue().str();
    if (!raw.empty())
      hints.push_back(std::move(raw));
    return;
  }

  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    if (Attribute metadata = fusedLoc.getMetadata()) {
      if (auto strAttr = dyn_cast<StringAttr>(metadata)) {
        std::string raw = strAttr.getValue().str();
        if (!raw.empty())
          hints.push_back(std::move(raw));
        return;
      }
      if (auto arrayAttr = dyn_cast<ArrayAttr>(metadata)) {
        for (Attribute attr : arrayAttr) {
          auto strAttr = dyn_cast<StringAttr>(attr);
          if (!strAttr)
            continue;
          std::string raw = strAttr.getValue().str();
          if (!raw.empty())
            hints.push_back(std::move(raw));
        }
        if (!hints.empty())
          return;
      }
    }

    for (Location childLoc : fusedLoc.getLocations())
      appendRawLocationProvenance(childLoc, hints);
    return;
  }

  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc)) {
    appendRawLocationProvenance(callSiteLoc.getCallee(), hints);
    if (hints.empty())
      appendRawLocationProvenance(callSiteLoc.getCaller(), hints);
  }
}

// Recover the raw provenance (input .pto SSA name) for an op's results.
// Returns one raw name per result when available, mirroring getResultNameHints
// but without sanitization.
static SmallVector<std::string, 4> getRawResultProvenance(Operation *op) {
  SmallVector<std::string, 4> hints;
  if (!op || op->getNumResults() == 0)
    return hints;
  appendRawLocationProvenance(op->getLoc(), hints);
  if (hints.empty())
    return hints;
  hints.erase(std::remove_if(hints.begin(), hints.end(),
                              [](const std::string &name) {
                                return name.empty();
                              }),
              hints.end());
  if (hints.empty())
    return hints;
  if (op->getNumResults() == 1) {
    if (hints.size() > 1)
      hints.resize(1);
    return hints;
  }
  if (hints.size() > op->getNumResults())
    hints.resize(op->getNumResults());
  return hints;
}

static Location attachLocationNameHints(Location baseLoc,
                                        llvm::ArrayRef<std::string> hints,
                                        MLIRContext *context) {
  SmallVector<Attribute, 4> attrs;
  attrs.reserve(hints.size());
  for (llvm::StringRef hint : hints) {
    if (!hint.empty())
      attrs.push_back(StringAttr::get(context, hint));
  }
  if (attrs.empty())
    return baseLoc;
  if (attrs.size() == 1)
    return NameLoc::get(cast<StringAttr>(attrs.front()), baseLoc);
  return FusedLoc::get(ArrayRef<Location>{baseLoc}, ArrayAttr::get(context, attrs),
                       context);
}

static void applyValueNameHints(Value value, llvm::ArrayRef<std::string> hints) {
  if (!value || hints.empty() || hasLocationNameHints(value.getLoc()))
    return;
  value.setLoc(attachLocationNameHints(value.getLoc(), hints, value.getContext()));
}

static void applyOperationResultNameHints(Operation *op,
                                          llvm::ArrayRef<std::string> hints) {
  if (!op || op->getNumResults() == 0 || hints.empty() ||
      hasLocationNameHints(op->getLoc()))
    return;

  SmallVector<std::string, 4> limitedHints;
  limitedHints.reserve(std::min<size_t>(op->getNumResults(), hints.size()));
  for (size_t i = 0, e = std::min<size_t>(op->getNumResults(), hints.size());
       i < e; ++i)
    limitedHints.push_back(hints[i]);
  if (limitedHints.empty())
    return;

  op->setLoc(attachLocationNameHints(op->getLoc(), limitedHints, op->getContext()));
}

static void collectNonEntryBlocksInSourceOrder(
    Operation *op, SmallVectorImpl<Block *> &blocks) {
  for (Region &region : op->getRegions()) {
    bool isEntryBlock = true;
    for (Block &block : region) {
      if (!isEntryBlock && block.getNumArguments() != 0)
        blocks.push_back(&block);
      isEntryBlock = false;
      for (Operation &nestedOp : block)
        collectNonEntryBlocksInSourceOrder(&nestedOp, blocks);
    }
  }
}

static void applyParsedTextualNameHints(ModuleOp module,
                                        const ParsedTextualNameHints &hints) {
  size_t funcHintIndex = 0;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (funcHintIndex < hints.functionArgHints.size()) {
      auto argHints = hints.functionArgHints[funcHintIndex];
      for (auto [index, arg] : llvm::enumerate(func.getArguments())) {
        if (index < argHints.size())
          applyValueNameHints(arg, llvm::ArrayRef<std::string>{argHints[index]});
      }
    }
    ++funcHintIndex;
  }

  SmallVector<Block *, 16> nonEntryBlocks;
  collectNonEntryBlocksInSourceOrder(module.getOperation(), nonEntryBlocks);
  for (auto [index, block] : llvm::enumerate(nonEntryBlocks)) {
    if (index >= hints.blockArgHints.size())
      break;
    auto blockHints = hints.blockArgHints[index];
    for (auto [argIndex, arg] : llvm::enumerate(block->getArguments())) {
      if (argIndex < blockHints.size())
        applyValueNameHints(arg,
                            llvm::ArrayRef<std::string>{blockHints[argIndex]});
    }
  }

  size_t opHintIndex = 0;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->getNumResults() == 0)
      return WalkResult::advance();
    if (opHintIndex < hints.opResultHints.size())
      applyOperationResultNameHints(op, hints.opResultHints[opHintIndex]);
    ++opHintIndex;
    return WalkResult::advance();
  });
}

void mlir::pto::applyTextualNameHintsToModule(ModuleOp module,
                                              llvm::StringRef sourceText) {
  if (!module)
    return;
  ParsedTextualNameHints hints = extractTextualNameHints(sourceText);
  applyParsedTextualNameHints(module, hints);
}

static FunctionArgHintMap collectFunctionArgNameHints(ModuleOp module) {
  FunctionArgHintMap hintsByFunction;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    SmallVector<std::string, 4> argHints;
    bool hasAllHints = func.getNumArguments() != 0;
    for (BlockArgument arg : func.getArguments()) {
      SmallVector<std::string, 4> hints = getValueNameHints(arg);
      if (hints.empty()) {
        hasAllHints = false;
        break;
      }
      argHints.push_back(std::move(hints.front()));
    }
    if (hasAllHints)
      hintsByFunction[func.getSymNameAttr()] = std::move(argHints);
  }
  return hintsByFunction;
}

static FunctionBlockArgHintMap collectFunctionBlockArgNameHints(ModuleOp module) {
  FunctionBlockArgHintMap hintsByFunction;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    SmallVector<Block *, 8> nonEntryBlocks;
    collectNonEntryBlocksInSourceOrder(func.getOperation(), nonEntryBlocks);
    if (nonEntryBlocks.empty())
      continue;

    SmallVector<SmallVector<std::string, 4>, 4> blockHints;
    blockHints.reserve(nonEntryBlocks.size());
    for (Block *block : nonEntryBlocks) {
      SmallVector<std::string, 4> argHints;
      bool hasAllHints = block->getNumArguments() != 0;
      for (BlockArgument arg : block->getArguments()) {
        SmallVector<std::string, 4> hints = getValueNameHints(arg);
        if (hints.empty()) {
          hasAllHints = false;
          break;
        }
        argHints.push_back(std::move(hints.front()));
      }
      if (hasAllHints)
        blockHints.push_back(std::move(argHints));
    }

    if (!blockHints.empty())
      hintsByFunction[func.getSymNameAttr()] = std::move(blockHints);
  }
  return hintsByFunction;
}

static void applyFunctionBlockArgNameHintsToEmitC(
    ModuleOp module, const FunctionBlockArgHintMap &blockArgHints) {
  for (emitc::FuncOp func : module.getOps<emitc::FuncOp>()) {
    auto it = blockArgHints.find(func.getSymNameAttr());
    if (it == blockArgHints.end() || it->second.empty())
      continue;

    SmallVector<Block *, 8> nonEntryBlocks;
    collectNonEntryBlocksInSourceOrder(func.getOperation(), nonEntryBlocks);
    for (size_t blockIndex = 0,
                e = std::min(nonEntryBlocks.size(), it->second.size());
         blockIndex < e; ++blockIndex) {
      Block *block = nonEntryBlocks[blockIndex];
      auto argHints = it->second[blockIndex];
      for (auto [argIndex, arg] : llvm::enumerate(block->getArguments())) {
        if (argIndex < argHints.size())
          applyValueNameHints(arg, llvm::ArrayRef<std::string>{argHints[argIndex]});
      }
    }
  }
}

static SmallVector<std::string, 4>
getResultNameHints(Operation *op) {
  SmallVector<std::string, 4> hints;
  if (!op || op->getNumResults() == 0)
    return hints;

  appendLocationNameHints(op->getLoc(), hints);
  if (hints.empty())
    return hints;

  hints.erase(std::remove_if(hints.begin(), hints.end(),
                             [](const std::string &name) {
                               return name.empty();
                             }),
              hints.end());
  if (hints.empty())
    return hints;

  if (op->getNumResults() == 1) {
    if (hints.size() > 1)
      hints.resize(1);
    return hints;
  }

  if (hints.size() > op->getNumResults())
    hints.resize(op->getNumResults());
  return hints;
}

static SmallVector<std::string, 4> getValueNameHints(Value value) {
  SmallVector<std::string, 4> hints;
  if (!value)
    return hints;
  appendLocationNameHints(value.getLoc(), hints);
  if (hints.size() > 1)
    hints.resize(1);
  return hints;
}

static std::string buildHintMarker(llvm::StringRef prefix,
                                   llvm::ArrayRef<std::string> hints) {
  std::string marker = ("/* " + prefix + ":").str();
  for (size_t i = 0; i < hints.size(); ++i) {
    if (i != 0)
      marker.push_back(',');
    marker.append(hints[i]);
  }
  marker.append(" */\n");
  return marker;
}

static void annotateEmitCNameHints(ModuleOp module) {
  llvm::SmallVector<Operation *, 32> opsToAnnotate;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->getNumResults() == 0 || isa<emitc::VerbatimOp>(op))
      return WalkResult::advance();
    if (op->getParentOfType<emitc::ExpressionOp>())
      return WalkResult::advance();
    // Annotate any op that has either a semantic name hint (for renaming) or a
    // raw provenance (for the `// pto: %N` comment, which applies even to
    // pure-digit SSA names like %0 that have no usable semantic name).
    if (getResultNameHints(op).empty() && getRawResultProvenance(op).empty())
      return WalkResult::advance();
    opsToAnnotate.push_back(op);
    return WalkResult::advance();
  });

  OpBuilder builder(module.getContext());
  for (Operation *op : opsToAnnotate) {
    SmallVector<std::string, 4> hints = getResultNameHints(op);
    if (!hints.empty()) {
      builder.setInsertionPoint(op);
      builder.create<emitc::VerbatimOp>(
          op->getLoc(),
          builder.getStringAttr(buildHintMarker("PTOAS_NAME_HINTS", hints)));
    }
    // Emit a provenance marker carrying the raw input SSA name. This is
    // consumed by the C++ post-processor to emit `// pto: %N` comments so a
    // reader can map a generated variable back to its .pto source (issue #337
    // point 1: locatability without strict number alignment).
    SmallVector<std::string, 4> provenance = getRawResultProvenance(op);
    if (!provenance.empty()) {
      builder.setInsertionPoint(op);
      builder.create<emitc::VerbatimOp>(
          op->getLoc(),
          builder.getStringAttr(buildHintMarker("PTOAS_PROVENANCE", provenance)));
    }
  }
}

// --------------------------------------------------------------------------
// Post-process C++ output: rewrite marker calls into Tile member calls.
// We emit marker calls in EmitC IR because EmitC currently does not provide a
// first-class op for member-function invocation. After translation, we rewrite:
//   PTOAS__TILE_SET_VALUE(dst, offset, val) -> dst.SetValue(offset, val)
//   PTOAS__TILE_GET_VALUE(src, offset)      -> src.GetValue(offset)
//   PTOAS__TILE_DATA(obj)                   -> obj.data()
//   PTOAS__TILE_SET_VALIDSHAPE(obj, r, c)   -> obj.SetValidShape(r, c)
//   PTOAS__TILE_GET_VALID_ROW(obj)          -> obj.GetValidRow()
//   PTOAS__TILE_GET_VALID_COL(obj)          -> obj.GetValidCol()
//   PTOAS__PTR_LOAD(ptr, offset)            -> ptr[offset]
//   PTOAS__PTR_STORE(ptr, offset, val)      -> ptr[offset] = val
//   PTOAS__EVENTID_ARRAY_LOAD(arr, idx)     -> arr[idx]
//   PTOAS__EVENTID_ARRAY_STORE(arr, idx, v) -> arr[idx] = v
// --------------------------------------------------------------------------
struct ParsedMarkerCall {
  size_t markerPos = std::string::npos;
  size_t rparenPos = std::string::npos;
  StringRefVector args;
};

struct MarkerRewriteSpec {
  llvm::StringRef marker;
  llvm::StringRef memberName;
  unsigned expectedNumArgs = 0;
};

struct MarkerSubscriptRewriteSpec {
  llvm::StringRef marker;
  unsigned expectedNumArgs = 0;
  bool isStore = false;
};

static bool parseMarkerArgs(llvm::StringRef argsRef,
                            llvm::SmallVectorImpl<llvm::StringRef> &args) {
  size_t partBegin = 0;
  int parenDepth = 0;
  for (size_t i = 0; i < argsRef.size(); ++i) {
    char c = argsRef[i];
    if (c == '(') {
      ++parenDepth;
      continue;
    }
    if (c == ')') {
      if (parenDepth > 0)
        --parenDepth;
      continue;
    }
    if (c == ',' && parenDepth == 0) {
      args.push_back(argsRef.slice(partBegin, i).trim());
      partBegin = i + 1;
    }
  }
  if (partBegin > argsRef.size())
    return false;
  args.push_back(argsRef.drop_front(partBegin).trim());
  return true;
}

static std::optional<ParsedMarkerCall>
findNextMarkerCall(const std::string &cpp, llvm::StringRef marker,
                   size_t searchPos) {
  ParsedMarkerCall call;
  call.markerPos = cpp.find(marker.str(), searchPos);
  if (call.markerPos == std::string::npos)
    return std::nullopt;

  size_t lparenPos = call.markerPos + marker.size();
  if (lparenPos >= cpp.size() || cpp[lparenPos] != '(')
    return ParsedMarkerCall{call.markerPos, std::string::npos, {}};

  size_t argsBegin = lparenPos + 1;
  int parenDepth = 0;
  for (size_t i = argsBegin; i < cpp.size(); ++i) {
    char c = cpp[i];
    if (c == '(') {
      ++parenDepth;
      continue;
    }
    if (c != ')')
      continue;
    if (parenDepth == 0) {
      call.rparenPos = i;
      break;
    }
    --parenDepth;
  }
  if (call.rparenPos == std::string::npos)
    return call;

  llvm::StringRef argsRef(cpp.data() + argsBegin, call.rparenPos - argsBegin);
  if (!parseMarkerArgs(argsRef, call.args))
    call.args.clear();
  return call;
}

template <typename BuildReplacementFn>
static bool rewriteMarkerCalls(std::string &cpp, llvm::StringRef marker,
                               BuildReplacementFn buildReplacement) {
  size_t searchPos = 0;
  bool changed = false;
  for (auto call = findNextMarkerCall(cpp, marker, searchPos); call;
       call = findNextMarkerCall(cpp, marker, searchPos)) {
    if (call->rparenPos == std::string::npos) {
      searchPos = call->markerPos + marker.size();
      continue;
    }

    std::optional<std::string> replacement = buildReplacement(*call);
    if (!replacement) {
      searchPos = call->rparenPos + 1;
      continue;
    }

    cpp.replace(call->markerPos, (call->rparenPos - call->markerPos) + 1,
                *replacement);
    changed = true;
    searchPos = call->markerPos + replacement->size();
  }
  return changed;
}

static bool rewriteMarkerCallToMember(std::string &cpp, llvm::StringRef marker,
                                      llvm::StringRef memberName,
                                      unsigned expectedNumArgs) {
  return rewriteMarkerCalls(
      cpp, marker, [&](const ParsedMarkerCall &call) -> std::optional<std::string> {
        if (call.args.size() != expectedNumArgs)
          return std::nullopt;

        std::string replacement;
        replacement.reserve(marker.size() + kMarkerCallReserveExtra);
        replacement.append(call.args[0].str());
        replacement.push_back('.');
        replacement.append(memberName.str());
        replacement.push_back('(');
        if (expectedNumArgs >= kMarkerRewriteMinArgCount)
          replacement.append(call.args[1].str());
        if (expectedNumArgs == kMarkerRewriteTernaryArgCount) {
          replacement.append(", ");
          replacement.append(call.args[2].str());
        }
        replacement.push_back(')');
        return replacement;
      });
}

static void rewriteMarkerCallsToMembers(
    std::string &cpp, llvm::ArrayRef<MarkerRewriteSpec> rewrites) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (const MarkerRewriteSpec &rewrite : rewrites) {
      changed |= rewriteMarkerCallToMember(cpp, rewrite.marker,
                                           rewrite.memberName,
                                           rewrite.expectedNumArgs);
    }
  }
}

static bool rewriteMarkerCallToField(std::string &cpp, llvm::StringRef marker,
                                     llvm::StringRef fieldName,
                                     size_t expectedNumArgs) {
  return rewriteMarkerCalls(
      cpp, marker, [&](const ParsedMarkerCall &call) -> std::optional<std::string> {
        if (call.args.size() != expectedNumArgs)
          return std::nullopt;
        if (call.args.empty())
          return std::nullopt;
        std::string replacement;
        replacement.reserve(call.args.front().size() + fieldName.size() + 1);
        replacement.append(call.args.front().str());
        replacement.push_back('.');
        replacement.append(fieldName.str());
        return replacement;
      });
}

static void rewriteTileGetSetValueMarkers(std::string &cpp) {
  static const MarkerRewriteSpec kTileMarkerRewrites[] = {
      {"PTOAS__TILE_SET_VALUE", "SetValue", 3},
      {"PTOAS__TILE_GET_VALUE", "GetValue", 2},
      {"PTOAS__TILE_DATA", "data", 1},
      {"PTOAS__TILE_SET_VALIDSHAPE", "SetValidShape", 3},
      {"PTOAS__TILE_GET_VALID_ROW", "GetValidRow", 1},
      {"PTOAS__TILE_GET_VALID_COL", "GetValidCol", 1},
  };
  rewriteMarkerCallsToMembers(cpp, kTileMarkerRewrites);
}

static void rewriteAsyncEventMarkers(std::string &cpp) {
  static const MarkerRewriteSpec kAsyncEventMarkerRewrites[] = {
      {"PTOAS__ASYNC_EVENT_WAIT", "Wait", 2},
      {"PTOAS__ASYNC_EVENT_TEST", "Test", 2},
  };
  rewriteMarkerCallsToMembers(cpp, kAsyncEventMarkerRewrites);
  (void)rewriteMarkerCallToField(cpp, "PTOAS__PREFETCH_CTX_SESSION",
                                 "session", 1);
}

// --------------------------------------------------------------------------
// EmitC cleanup: drop trivial emitc.expression ops.
// After FormExpressions + CSE, EmitC expressions can become invalid in two
// ways:
//   1. the root op is CSE'd away, leaving an empty expression region
//   2. the region degenerates to `emitc.yield %outer_value`, i.e. the yielded
//      value is defined outside the expression body
// Both cases crash mlir::emitc::translateToCpp because ExpressionOp expects a
// root op defined within the region.
// --------------------------------------------------------------------------
static void dropEmptyEmitCExpressions(Operation *rootOp) {
  llvm::SmallVector<emitc::ExpressionOp, kEmptyExpressionInlineCapacity>
      toErase;
  rootOp->walk([&](emitc::ExpressionOp expr) {
    Block *body = expr.getBody();
    if (!body)
      return;
    auto yield = dyn_cast<emitc::YieldOp>(body->getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return;
    Value yielded = yield.getOperand(0);
    Operation *defOp = yielded.getDefiningOp();
    bool yieldedFromOutside = !defOp || defOp->getBlock() != body;
    if (!yieldedFromOutside && expr.getRootOp())
      return;
    expr.getResult().replaceAllUsesWith(yielded);
    toErase.push_back(expr);
  });
  for (emitc::ExpressionOp expr : llvm::reverse(toErase))
    expr.erase();
}

static Attribute getDefaultEmitCVariableInitAttr(OpBuilder &builder, Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return builder.getIntegerAttr(intTy, 0);
  if (isa<IndexType>(type))
    return builder.getIndexAttr(0);
  if (auto floatTy = dyn_cast<FloatType>(type))
    return builder.getFloatAttr(floatTy, 0.0);
  if (isa<emitc::OpaqueType, emitc::PointerType>(type))
    return emitc::OpaqueAttr::get(builder.getContext(), "");
  return Attribute{};
}

// FormExpressions may inline conditions into emitc.expression, but the C++
// emitter prints cf.br/cf.cond_br operands by variable name rather than by
// recursively emitting an expression. Materialize such operands so CFG-based
// lowering (e.g. scf.while -> cf.*) stays valid.
static void materializeControlFlowOperands(Operation *rootOp) {
  llvm::SmallVector<Operation *, kBranchInlineCapacity> branches;
  rootOp->walk([&](Operation *op) {
    if (isa<cf::BranchOp, cf::CondBranchOp>(op))
      branches.push_back(op);
  });

  OpBuilder builder(rootOp->getContext());
  for (Operation *op : branches) {
    builder.setInsertionPoint(op);
    for (OpOperand &operand : op->getOpOperands()) {
      Value value = operand.get();
      auto expr = dyn_cast_or_null<emitc::ExpressionOp>(value.getDefiningOp());
      if (!expr)
        continue;

      Attribute initAttr =
          getDefaultEmitCVariableInitAttr(builder, value.getType());
      if (!initAttr)
        continue;

      Location valueLoc = value.getLoc();

      Value tmp =
          builder.create<emitc::VariableOp>(valueLoc, value.getType(),
                                            initAttr)
              .getResult();
      builder.create<emitc::AssignOp>(valueLoc, tmp, value);
      operand.set(tmp);
    }
  }
}

static bool rewriteMarkerCallToSubscript(std::string &cpp, llvm::StringRef marker,
                                         unsigned expectedNumArgs,
                                         bool isStore) {
  return rewriteMarkerCalls(
      cpp, marker, [&](const ParsedMarkerCall &call) -> std::optional<std::string> {
        if (call.args.size() != expectedNumArgs)
          return std::nullopt;
        if (isStore) {
          return (call.args[0] + "[" + call.args[1] + "] = " + call.args[2])
              .str();
        }
        return (call.args[0] + "[" + call.args[1] + "]").str();
      });
}

static void rewriteMarkerCallsToSubscripts(
    std::string &cpp, llvm::ArrayRef<MarkerSubscriptRewriteSpec> rewrites) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (const MarkerSubscriptRewriteSpec &rewrite : rewrites) {
      changed |= rewriteMarkerCallToSubscript(cpp, rewrite.marker,
                                              rewrite.expectedNumArgs,
                                              rewrite.isStore);
    }
  }
}

static void rewritePtrScalarMarkers(std::string &cpp) {
  static const MarkerSubscriptRewriteSpec kPtrMarkerRewrites[] = {
      {"PTOAS__PTR_LOAD", 2, false},
      {"PTOAS__PTR_STORE", 3, true},
  };
  rewriteMarkerCallsToSubscripts(cpp, kPtrMarkerRewrites);
}

static std::string getLineIndent(llvm::StringRef line) {
  size_t firstNonSpace = line.find_first_not_of(" \t");
  if (firstNonSpace == llvm::StringRef::npos)
    return line.str();
  return line.take_front(firstNonSpace).str();
}

static bool isAICOREFunctionStart(llvm::StringRef trimmed) {
  if (trimmed.empty() || trimmed.starts_with("#") || trimmed.starts_with("//"))
    return false;
  if (!trimmed.contains("AICORE"))
    return false;
  return trimmed.contains("(");
}

static int countBraceDelta(llvm::StringRef line) {
  int delta = 0;
  for (char c : line) {
    if (c == '{')
      ++delta;
    else if (c == '}')
      --delta;
  }
  return delta;
}

static void appendScalarGMFlush(std::string &out, llvm::StringRef indent) {
  out.append(indent.str());
  out.append("pipe_barrier(PIPE_ALL);\n");
  out.append(indent.str());
  out.append("dcci((__gm__ void*)0, ENTIRE_DATA_CACHE, CACHELINE_OUT);\n");
  out.append(indent.str());
  out.append("dsb((mem_dsb_t)0);\n");
}

static bool stripScalarGMFlushMarkersFromLine(std::string &line) {
  static constexpr llvm::StringLiteral kMarker =
      "PTOAS__SCALAR_GM_STORE_FLUSH";

  bool changed = false;
  size_t searchPos = 0;
  while (true) {
    auto call = findNextMarkerCall(line, kMarker, searchPos);
    if (!call)
      break;
    if (call->rparenPos == std::string::npos) {
      searchPos = call->markerPos + kMarker.size();
      continue;
    }

    size_t eraseBegin = call->markerPos;
    while (eraseBegin > 0 &&
           (line[eraseBegin - 1] == ' ' || line[eraseBegin - 1] == '\t'))
      --eraseBegin;

    size_t eraseEnd = call->rparenPos + 1;
    while (eraseEnd < line.size() &&
           (line[eraseEnd] == ' ' || line[eraseEnd] == '\t'))
      ++eraseEnd;
    if (eraseEnd < line.size() && line[eraseEnd] == ';')
      ++eraseEnd;
    while (eraseEnd < line.size() &&
           (line[eraseEnd] == ' ' || line[eraseEnd] == '\t'))
      ++eraseEnd;

    line.erase(eraseBegin, eraseEnd - eraseBegin);
    changed = true;
    searchPos = eraseBegin;
  }
  return changed;
}

static bool previousSignificantLineIsTailFlushPoint(
    llvm::ArrayRef<std::string> lines, size_t index) {
  for (size_t i = index; i > 0; --i) {
    llvm::StringRef prev = llvm::StringRef(lines[i - 1]).trim();
    if (prev.empty())
      continue;
    return prev.starts_with("#endif // __DAV_") ||
           prev.starts_with("ptoas_auto_sync_tail(");
  }
  return false;
}

static bool previousSignificantLineIsExitOrTailFlushPoint(
    llvm::ArrayRef<std::string> lines, size_t index) {
  for (size_t i = index; i > 0; --i) {
    llvm::StringRef prev = llvm::StringRef(lines[i - 1]).trim();
    if (prev.empty())
      continue;
    return prev.starts_with("return") ||
           prev.starts_with("#endif // __DAV_") ||
           prev.starts_with("ptoas_auto_sync_tail(");
  }
  return false;
}

static std::string rewriteScalarGMStoreFlushMarkersInFunction(
    llvm::ArrayRef<std::string> functionLines, bool hasTrailingNewline) {
  bool needsScalarGMFlush = false;
  llvm::SmallVector<std::string, 32> lines;
  lines.reserve(functionLines.size());

  for (const std::string &rawLine : functionLines) {
    std::string line = rawLine;
    bool hadMarker = stripScalarGMFlushMarkersFromLine(line);
    needsScalarGMFlush |= hadMarker;
    if (hadMarker && llvm::StringRef(line).trim().empty()) {
      continue;
    }
    lines.push_back(std::move(line));
  }

  if (!needsScalarGMFlush) {
    std::string unchanged;
    unchanged.reserve(kRewriteOutputReserveExtra);
    for (size_t i = 0; i < lines.size(); ++i) {
      unchanged.append(lines[i]);
      if (i + 1 < lines.size() || hasTrailingNewline)
        unchanged.push_back('\n');
    }
    return unchanged;
  }

  std::string out;
  out.reserve(kRewriteOutputReserveExtra);
  bool inserted = false;
  size_t fallbackIndex = lines.size();
  for (size_t i = lines.size(); i > 0; --i) {
    llvm::StringRef trimmed = llvm::StringRef(lines[i - 1]).trim();
    if (trimmed.empty())
      continue;
    if (trimmed.starts_with("}"))
      fallbackIndex = i - 1;
    break;
  }

  for (size_t i = 0; i < lines.size(); ++i) {
    llvm::StringRef lineRef(lines[i]);
    llvm::StringRef trimmed = lineRef.trim();
    bool insertHere = false;
    if (trimmed.starts_with("return")) {
      insertHere = !previousSignificantLineIsTailFlushPoint(lines, i);
    } else {
      insertHere = trimmed.starts_with("#endif // __DAV_") ||
                   trimmed.starts_with("ptoas_auto_sync_tail(");
    }
    if (i == fallbackIndex &&
        !previousSignificantLineIsExitOrTailFlushPoint(lines, i))
      insertHere = true;
    if (insertHere) {
      appendScalarGMFlush(out, getLineIndent(lineRef));
      inserted = true;
    }
    out.append(lines[i]);
    if (i + 1 < lines.size() || hasTrailingNewline)
      out.push_back('\n');
  }

  if (!inserted)
    appendScalarGMFlush(out, "  ");
  return out;
}

static void rewriteScalarGMStoreFlushMarkers(std::string &cpp) {
  std::string out;
  out.reserve(cpp.size() + kRewriteOutputReserveExtra);

  llvm::SmallVector<std::string, 32> functionLines;
  bool inFunction = false;
  bool sawFunctionBrace = false;
  int braceDepth = 0;

  auto flushFunction = [&](bool hasTrailingNewline) {
    out.append(rewriteScalarGMStoreFlushMarkersInFunction(functionLines,
                                                         hasTrailingNewline));
    functionLines.clear();
    inFunction = false;
    sawFunctionBrace = false;
    braceDepth = 0;
  };

  llvm::StringRef ref(cpp);
  while (!ref.empty()) {
    auto split = ref.split('\n');
    std::string line = split.first.str();
    bool hadNewline = !split.second.empty();
    ref = split.second;

    llvm::StringRef trimmed = llvm::StringRef(line).trim();
    if (!inFunction && isAICOREFunctionStart(trimmed))
      inFunction = true;

    if (!inFunction) {
      out.append(line);
      if (hadNewline)
        out.push_back('\n');
      continue;
    }

    functionLines.push_back(std::move(line));
    int delta = countBraceDelta(functionLines.back());
    if (delta != 0)
      sawFunctionBrace = true;
    braceDepth += delta;
    if (sawFunctionBrace && braceDepth == 0)
      flushFunction(hadNewline);
  }

  if (!functionLines.empty())
    flushFunction(false);
  cpp.swap(out);
}

static void rewriteEventIdArrayMarkers(std::string &cpp) {
  static const MarkerSubscriptRewriteSpec kEventIdMarkerRewrites[] = {
      {"PTOAS__EVENTID_ARRAY_LOAD", 2, false},
      {"PTOAS__EVENTID_ARRAY_STORE", 3, true},
  };
  rewriteMarkerCallsToSubscripts(cpp, kEventIdMarkerRewrites);
}

static bool isPreprocessorDirectiveLine(llvm::StringRef trimmedLine) {
  return trimmedLine.starts_with("#");
}

// Nested emitc.verbatim ops inside emitc.for / emitc.if regions currently
// pick up an extra trailing semicolon from EmitC C++ emission, which produces
// invalid lines such as `#if defined(__DAV_VEC__);` and `set_mask_norm();;`.
// Trim only those malformed suffixes here so bisheng can compile the emitted
// source until the upstream printer behavior is fixed.
static void rewriteMalformedVerbatimSemicolons(std::string &cpp) {
  if (cpp.empty())
    return;

  llvm::StringRef input(cpp);
  std::string rewritten;
  rewritten.reserve(cpp.size());

  bool prevWasPreprocessorDirective = false;
  size_t offset = 0;
  while (offset < input.size()) {
    size_t newlinePos = input.find('\n', offset);
    bool hasNewline = newlinePos != llvm::StringRef::npos;
    llvm::StringRef line =
        hasNewline ? input.slice(offset, newlinePos) : input.drop_front(offset);
    std::string current(line.str());
    llvm::StringRef trimmed = llvm::StringRef(current).trim();

    if (trimmed == ";" && prevWasPreprocessorDirective) {
      // `#endif ...` in nested verbatim blocks currently materializes as the
      // directive line followed by a standalone `;` on the next line.
      prevWasPreprocessorDirective = false;
    } else {
      if (isPreprocessorDirectiveLine(trimmed) && trimmed.ends_with(";")) {
        size_t semicolonPos = current.find_last_of(';');
        if (semicolonPos != std::string::npos)
          current.erase(semicolonPos, 1);
      } else if (!trimmed.empty() && !trimmed.starts_with("//") &&
                 !trimmed.starts_with("/*") && trimmed.ends_with(";;")) {
        size_t semicolonPos = current.find_last_of(';');
        if (semicolonPos != std::string::npos)
          current.erase(semicolonPos, 1);
      }

      rewritten.append(current);
      if (hasNewline)
        rewritten.push_back('\n');
      prevWasPreprocessorDirective =
          isPreprocessorDirectiveLine(llvm::StringRef(current).trim());
    }

    if (!hasNewline)
      break;
    offset = newlinePos + 1;
  }

  cpp.swap(rewritten);
}

static bool rewriteAddPtrTraceMarkers(std::string &cpp, bool showTrace) {
  size_t searchPos = 0;
  bool changed = false;
  for (auto call = findNextMarkerCall(cpp, "PTOAS__ADDPTR_TRACE", searchPos);
       call; call = findNextMarkerCall(cpp, "PTOAS__ADDPTR_TRACE", searchPos)) {
    if (call->rparenPos == std::string::npos) {
      searchPos = call->markerPos + 1;
      continue;
    }
    if (call->args.size() != kMarkerRewriteTernaryArgCount) {
      searchPos = call->rparenPos + 1;
      continue;
    }

    std::string replacement;
    if (showTrace) {
      replacement.reserve(kRewriteOutputReserveExtra);
      replacement.append("/* ADDPTR_TRACE: ");
      replacement.append(call->args[0].str());
      replacement.append(" = ");
      replacement.append(call->args[1].str());
      replacement.append(" + ");
      replacement.append(call->args[2].str());
      replacement.append(" */");
    }

    size_t replaceEnd = call->rparenPos;
    if (!showTrace) {
      size_t i = call->rparenPos + 1;
      while (i < cpp.size() && std::isspace(static_cast<unsigned char>(cpp[i])))
        ++i;
      if (i < cpp.size() && cpp[i] == ';')
        replaceEnd = i;
    }

    cpp.replace(call->markerPos, (replaceEnd - call->markerPos) + 1,
                replacement);
    changed = true;
    searchPos = call->markerPos + replacement.size();
  }
  return changed;
}

static bool isGeneratedGlobalTensorDecl(llvm::StringRef trimmed,
                                        llvm::StringRef &decl,
                                        llvm::StringRef &varName) {
  if (!trimmed.starts_with("GlobalTensor<") || !trimmed.ends_with(";") ||
      trimmed.contains('=') || trimmed.contains('(')) {
    return false;
  }

  decl = trimmed.drop_back().rtrim();
  size_t lastWs = decl.find_last_of(" \t");
  if (lastWs == llvm::StringRef::npos)
    return false;
  varName = decl.drop_front(lastWs + 1);
  if (!varName.starts_with("v") || varName.size() <= 1)
    return false;
  return llvm::all_of(varName.drop_front(1),
                      [](char c) { return std::isdigit(c); });
}

static void rewriteHoistedGlobalTensorDecls(std::string &cpp) {
  // When `declareVariablesAtTop` is enabled, the C++ emitter hoists SSA value
  // declarations to the top of the function and emits assignments later. This
  // requires the C++ type to be default-constructible.
  //
  // `GlobalTensor<...>` from pto-isa does NOT have a default constructor, so
  // hoisted declarations of that type must be rewritten with a null-pointer
  // initializer before the later assignment remains in place.
  // We keep the assignment later; the null-initialized value is never used.
  std::string out;
  out.reserve(cpp.size() + kRewriteOutputReserveExtra);

  llvm::StringRef ref(cpp);
  while (!ref.empty()) {
    auto split = ref.split('\n');
    llvm::StringRef line = split.first;
    llvm::StringRef rest = split.second;

    llvm::StringRef trimmed = line.trim();
    bool rewritten = false;
    llvm::StringRef decl;
    llvm::StringRef varName;
    if (isGeneratedGlobalTensorDecl(trimmed, decl, varName)) {
      size_t indentLen = line.find_first_not_of(" \t");
      if (indentLen == std::string::npos)
        indentLen = 0;
      llvm::StringRef indent = line.take_front(indentLen);

      out.append(indent.str());
      out.append(decl.str());
      out.append("(nullptr);");
      rewritten = true;
    }

    if (!rewritten)
      out.append(line.str());
    if (!rest.empty())
      out.push_back('\n');
    ref = rest;
  }

  cpp.swap(out);
}

static std::optional<llvm::SmallVector<std::string, 4>>
parseNameHintMarker(llvm::StringRef markerBody) {
  llvm::SmallVector<std::string, 4> hints;
  markerBody = markerBody.trim();
  if (markerBody.empty())
    return std::nullopt;

  size_t start = 0;
  while (start <= markerBody.size()) {
    size_t comma = markerBody.find(',', start);
    llvm::StringRef token = markerBody.slice(
        start, comma == llvm::StringRef::npos ? markerBody.size() : comma);
    token = token.trim();
    if (!token.empty())
      hints.push_back(token.str());
    if (comma == llvm::StringRef::npos)
      break;
    start = comma + 1;
  }

  if (hints.empty())
    return std::nullopt;
  return hints;
}

static std::optional<llvm::SmallVector<std::string, 4>>
findNextHintedGeneratedParams(llvm::StringRef snippet) {
  size_t lParenPos = snippet.find('(');
  if (lParenPos == llvm::StringRef::npos)
    return std::nullopt;

  int parenDepth = 0;
  size_t rParenPos = llvm::StringRef::npos;
  for (size_t i = lParenPos; i < snippet.size(); ++i) {
    char c = snippet[i];
    if (c == '(') {
      ++parenDepth;
    } else if (c == ')') {
      --parenDepth;
      if (parenDepth == 0) {
        rParenPos = i;
        break;
      }
    }
  }
  if (rParenPos == llvm::StringRef::npos)
    return std::nullopt;

  llvm::StringRef params = snippet.slice(lParenPos + 1, rParenPos);
  llvm::SmallVector<std::string, 4> names;
  size_t partBegin = 0;
  int angleDepth = 0;
  int bracketDepth = 0;
  parenDepth = 0;
  for (size_t i = 0; i <= params.size(); ++i) {
    char c = i < params.size() ? params[i] : ',';
    if (c == '<') {
      ++angleDepth;
    } else if (c == '>' && angleDepth > 0) {
      --angleDepth;
    } else if (c == '[') {
      ++bracketDepth;
    } else if (c == ']' && bracketDepth > 0) {
      --bracketDepth;
    } else if (c == '(') {
      ++parenDepth;
    } else if (c == ')' && parenDepth > 0) {
      --parenDepth;
    }

    bool atSeparator =
        (i == params.size()) ||
        (c == ',' && angleDepth == 0 && bracketDepth == 0 && parenDepth == 0);
    if (!atSeparator)
      continue;

    llvm::StringRef param = params.slice(partBegin, i).trim();
    partBegin = i + 1;
    if (param.empty())
      continue;

    size_t end = param.size();
    while (end > 0 && std::isspace(static_cast<unsigned char>(param[end - 1])))
      --end;
    size_t begin = end;
    while (begin > 0 && isCppIdentifierChar(param[begin - 1]))
      --begin;
    llvm::StringRef token = param.slice(begin, end);
    if (isGeneratedValueName(token))
      names.push_back(token.str());
  }

  if (names.empty())
    return std::nullopt;
  return names;
}

static std::optional<llvm::SmallVector<std::string, 4>>
findNextHintedGeneratedNames(llvm::StringRef snippet) {
  static const llvm::Regex kTieRegex(
      R"re(std::tie\(([[:space:]]*v[0-9]+([[:space:]]*,[[:space:]]*v[0-9]+)*)\)[[:space:]]*=)re");
  static const llvm::Regex kSingleRegex(
      R"re((^|[^[:alnum:]_])(v[0-9]+)[[:space:]]*(=|;))re");

  llvm::SmallVector<llvm::StringRef, 4> matches;
  if (kTieRegex.match(snippet, &matches) && matches.size() >= 2) {
    llvm::SmallVector<std::string, 4> names;
    llvm::StringRef tieNames = matches[1];
    size_t start = 0;
    while (start < tieNames.size()) {
      size_t comma = tieNames.find(',', start);
      llvm::StringRef token = tieNames.slice(
          start, comma == llvm::StringRef::npos ? tieNames.size() : comma);
      token = token.trim();
      if (!token.empty())
        names.push_back(token.str());
      if (comma == llvm::StringRef::npos)
        break;
      start = comma + 1;
    }
    if (!names.empty())
      return names;
  }

  matches.clear();
  if (kSingleRegex.match(snippet, &matches) && matches.size() >= 3)
    return llvm::SmallVector<std::string, 4>{matches[2].str()};

  return std::nullopt;
}

static void rewriteIdentifiersWithMap(
    std::string &cpp, const llvm::StringMap<std::string> &replacements) {
  if (replacements.empty())
    return;
  std::string rewritten;
  rewritten.reserve(cpp.size());
  enum class LexState {
    Normal,
    LineComment,
    BlockComment,
    StringLiteral,
    CharLiteral,
  };
  LexState state = LexState::Normal;

  auto appendCurrent = [&](char c) { rewritten.push_back(c); };

  for (size_t i = 0; i < cpp.size();) {
    char c = cpp[i];
    char next = i + 1 < cpp.size() ? cpp[i + 1] : '\0';

    switch (state) {
    case LexState::Normal:
      if (c == '/' && next == '/') {
        state = LexState::LineComment;
        appendCurrent(c);
        appendCurrent(next);
        i += 2;
        continue;
      }
      if (c == '/' && next == '*') {
        state = LexState::BlockComment;
        appendCurrent(c);
        appendCurrent(next);
        i += 2;
        continue;
      }
      if (c == '"') {
        state = LexState::StringLiteral;
        appendCurrent(c);
        ++i;
        continue;
      }
      if (c == '\'') {
        state = LexState::CharLiteral;
        appendCurrent(c);
        ++i;
        continue;
      }
      if (isCppIdentifierStart(c)) {
        size_t end = i + 1;
        while (end < cpp.size() && isCppIdentifierChar(cpp[end]))
          ++end;
        llvm::StringRef token(cpp.data() + i, end - i);
        auto it = replacements.find(token);
        if (it != replacements.end())
          rewritten.append(it->second);
        else
          rewritten.append(token.begin(), token.end());
        i = end;
        continue;
      }
      appendCurrent(c);
      ++i;
      continue;

    case LexState::LineComment:
      appendCurrent(c);
      ++i;
      if (c == '\n')
        state = LexState::Normal;
      continue;

    case LexState::BlockComment:
      appendCurrent(c);
      ++i;
      if (c == '*' && next == '/') {
        appendCurrent(next);
        ++i;
        state = LexState::Normal;
      }
      continue;

    case LexState::StringLiteral:
      appendCurrent(c);
      ++i;
      if (c == '\\' && i < cpp.size()) {
        appendCurrent(cpp[i]);
        ++i;
      } else if (c == '"') {
        state = LexState::Normal;
      }
      continue;

    case LexState::CharLiteral:
      appendCurrent(c);
      ++i;
      if (c == '\\' && i < cpp.size()) {
        appendCurrent(cpp[i]);
        ++i;
      } else if (c == '\'') {
        state = LexState::Normal;
      }
      continue;
    }
  }

  cpp.swap(rewritten);
}

static void stripHintMarkersWithPrefix(std::string &cpp,
                                       llvm::StringRef markerPrefix) {
  size_t searchPos = 0;
  while (true) {
    size_t markerPos = cpp.find(markerPrefix.str(), searchPos);
    if (markerPos == std::string::npos)
      break;

    size_t markerEnd = cpp.find("*/", markerPos + markerPrefix.size());
    if (markerEnd == std::string::npos)
      break;
    markerEnd += 2;
    while (markerEnd < cpp.size() &&
           (cpp[markerEnd] == '\r' || cpp[markerEnd] == '\n'))
      ++markerEnd;

    cpp.erase(markerPos, markerEnd - markerPos);
    searchPos = markerPos;
  }
}

static void stripAllHintMarkers(std::string &cpp) {
  stripHintMarkersWithPrefix(cpp, "/* PTOAS_NAME_HINTS:");
  stripHintMarkersWithPrefix(cpp, "/* PTOAS_PARAM_HINTS:");
  stripHintMarkersWithPrefix(cpp, "/* PTOAS_PROVENANCE:");
}

static bool isHintMarkerLine(llvm::StringRef trimmed) {
  return trimmed.starts_with("/* PTOAS_NAME_HINTS:") ||
         trimmed.starts_with("/* PTOAS_PARAM_HINTS:") ||
         trimmed.starts_with("/* PTOAS_PROVENANCE:");
}

static std::optional<std::string>
extractFunctionNameFromSegment(llvm::StringRef segment) {
  size_t lParenPos = segment.find('(');
  if (lParenPos == llvm::StringRef::npos)
    return std::nullopt;
  size_t end = lParenPos;
  while (end > 0 && std::isspace(static_cast<unsigned char>(segment[end - 1])))
    --end;
  size_t begin = end;
  while (begin > 0 && isCppIdentifierChar(segment[begin - 1]))
    --begin;
  if (begin == end)
    return std::nullopt;
  return segment.slice(begin, end).str();
}

static bool isTopLevelFunctionStartLine(llvm::StringRef trimmed) {
  if (trimmed.empty() || trimmed.starts_with("#") || !trimmed.ends_with("{"))
    return false;
  if (!trimmed.contains('(') || !trimmed.contains(')'))
    return false;
  if (trimmed.starts_with("if ") || trimmed.starts_with("if(") ||
      trimmed.starts_with("for ") || trimmed.starts_with("for(") ||
      trimmed.starts_with("while ") || trimmed.starts_with("while(") ||
      trimmed.starts_with("switch ") || trimmed.starts_with("switch(") ||
      trimmed.starts_with("catch ") || trimmed.starts_with("catch("))
    return false;
  return true;
}

static std::optional<std::string>
parseAnyDeclaredIdentifierName(llvm::StringRef line);

static llvm::SmallVector<std::string, 4>
findTopLevelGeneratedDeclarations(llvm::StringRef segment);

static std::optional<std::string>
parseGeneratedDeclarationName(llvm::StringRef line) {
  auto declaredName = parseAnyDeclaredIdentifierName(line);
  if (!declaredName || !isGeneratedValueName(*declaredName))
    return std::nullopt;
  return declaredName;
}

static std::set<std::string> collectDeclaredIdentifiersInFunctionSegment(
    llvm::StringRef segment) {
  std::set<std::string> declaredNames;

  if (size_t lParenPos = segment.find('('); lParenPos != llvm::StringRef::npos) {
    int parenDepth = 0;
    size_t rParenPos = llvm::StringRef::npos;
    for (size_t i = lParenPos; i < segment.size(); ++i) {
      char c = segment[i];
      if (c == '(') {
        ++parenDepth;
      } else if (c == ')') {
        --parenDepth;
        if (parenDepth == 0) {
          rParenPos = i;
          break;
        }
      }
    }
    if (rParenPos != llvm::StringRef::npos) {
      llvm::StringRef params = segment.slice(lParenPos + 1, rParenPos);
      size_t partBegin = 0;
      int angleDepth = 0;
      int bracketDepth = 0;
      parenDepth = 0;
      for (size_t i = 0; i <= params.size(); ++i) {
        char c = i < params.size() ? params[i] : ',';
        if (c == '<') {
          ++angleDepth;
        } else if (c == '>' && angleDepth > 0) {
          --angleDepth;
        } else if (c == '[') {
          ++bracketDepth;
        } else if (c == ']' && bracketDepth > 0) {
          --bracketDepth;
        } else if (c == '(') {
          ++parenDepth;
        } else if (c == ')' && parenDepth > 0) {
          --parenDepth;
        }

        bool atSeparator =
            (i == params.size()) ||
            (c == ',' && angleDepth == 0 && bracketDepth == 0 &&
             parenDepth == 0);
        if (!atSeparator)
          continue;

        llvm::StringRef param = params.slice(partBegin, i).trim();
        partBegin = i + 1;
        if (param.empty())
          continue;

        size_t end = param.size();
        while (end > 0 &&
               std::isspace(static_cast<unsigned char>(param[end - 1])))
          --end;
        size_t begin = end;
        while (begin > 0 && isCppIdentifierChar(param[begin - 1]))
          --begin;
        llvm::StringRef name = param.slice(begin, end);
        if (!name.empty() && isCppIdentifierStart(name.front()) &&
            llvm::all_of(name, isCppIdentifierChar))
          declaredNames.insert(name.str());
      }
    }
  }

  llvm::StringRef remaining = segment;
  while (!remaining.empty()) {
    auto split = remaining.split('\n');
    llvm::StringRef line = split.first;
    llvm::StringRef rest = split.second;
    if (auto declaredName = parseAnyDeclaredIdentifierName(line))
      declaredNames.insert(*declaredName);
    remaining = rest;
  }

  return declaredNames;
}

struct PendingIdentifierRename {
  std::string oldName;
  std::string baseHint;
};

static llvm::SmallVector<PendingIdentifierRename, 8>
collectPendingIdentifierRenames(
    llvm::StringRef segment, llvm::ArrayRef<std::string> functionParamHints,
    llvm::ArrayRef<llvm::SmallVector<std::string, 4>> blockArgHints) {
  static constexpr llvm::StringLiteral kResultMarkerPrefix =
      "/* PTOAS_NAME_HINTS:";
  llvm::SmallVector<PendingIdentifierRename, 8> pendingRenames;

  if (!functionParamHints.empty()) {
    if (auto generatedParams = findNextHintedGeneratedParams(segment)) {
      size_t pairCount =
          std::min(functionParamHints.size(), generatedParams->size());
      for (size_t i = 0; i < pairCount; ++i) {
        pendingRenames.push_back(
            PendingIdentifierRename{(*generatedParams)[i], functionParamHints[i]});
      }
    }
  }

  size_t searchPos = 0;
  while (true) {
    size_t markerPos = segment.find(kResultMarkerPrefix.str(), searchPos);
    if (markerPos == std::string::npos)
      break;

    size_t bodyBegin = markerPos + kResultMarkerPrefix.size();
    size_t markerEnd = segment.find("*/", bodyBegin);
    if (markerEnd == std::string::npos)
      break;

    auto hints = parseNameHintMarker(
        llvm::StringRef(segment).slice(bodyBegin, markerEnd));
    searchPos = markerEnd + 2;
    if (!hints)
      continue;

    size_t windowEnd =
        std::min(searchPos + static_cast<size_t>(2048), segment.size());
    llvm::StringRef searchWindow =
        llvm::StringRef(segment).slice(searchPos, windowEnd);
    auto generatedNames = findNextHintedGeneratedNames(searchWindow);
    if (!generatedNames)
      continue;

    size_t pairCount = std::min(hints->size(), generatedNames->size());
    for (size_t i = 0; i < pairCount; ++i) {
      pendingRenames.push_back(
          PendingIdentifierRename{(*generatedNames)[i], (*hints)[i]});
    }
  }

  if (!blockArgHints.empty()) {
    llvm::SmallVector<std::string, 4> generatedDecls =
        findTopLevelGeneratedDeclarations(segment);
    llvm::SmallVector<std::string, 4> flattenedBlockHints;
    for (auto blockHints : blockArgHints)
      flattenedBlockHints.append(blockHints.begin(), blockHints.end());
    if (!flattenedBlockHints.empty() &&
        generatedDecls.size() >= flattenedBlockHints.size()) {
      size_t startIndex = generatedDecls.size() - flattenedBlockHints.size();
      for (size_t i = 0; i < flattenedBlockHints.size(); ++i) {
        pendingRenames.push_back(PendingIdentifierRename{
            generatedDecls[startIndex + i], flattenedBlockHints[i]});
      }
    }
  }

  return pendingRenames;
}

static std::optional<std::string>
parseAnyDeclaredIdentifierName(llvm::StringRef line) {
  llvm::StringRef trimmed = line.trim();
  if (trimmed.empty() || trimmed.starts_with("#") || trimmed.starts_with("//") ||
      !trimmed.ends_with(";"))
    return std::nullopt;
  llvm::StringRef body = trimmed.drop_back().rtrim();
  if (body.starts_with("return") || body.starts_with("goto ") ||
      body.starts_with("if ") || body.starts_with("if(") ||
      body.starts_with("switch ") || body.starts_with("switch(") ||
      body.starts_with("for ") || body.starts_with("for(") ||
      body.starts_with("while ") || body.starts_with("while(") ||
      body.starts_with("using namespace "))
    return std::nullopt;

  llvm::StringRef lhs = body;
  if (size_t eqPos = body.find('='); eqPos != llvm::StringRef::npos)
    lhs = body.take_front(eqPos).rtrim();
  size_t lastWs = lhs.find_last_of(" \t");
  if (lastWs == llvm::StringRef::npos)
    return std::nullopt;
  llvm::StringRef name = lhs.drop_front(lastWs + 1).trim();
  if (name.empty() || !isCppIdentifierStart(name.front()) ||
      !llvm::all_of(name, isCppIdentifierChar))
    return std::nullopt;
  return name.str();
}

static llvm::SmallVector<std::string, 4>
findTopLevelGeneratedDeclarations(llvm::StringRef segment) {
  llvm::SmallVector<std::string, 4> names;
  size_t lBracePos = segment.find('{');
  if (lBracePos == llvm::StringRef::npos)
    return names;

  llvm::StringRef body = segment.drop_front(lBracePos + 1);
  llvm::StringRef remaining = body;
  while (!remaining.empty()) {
    auto split = remaining.split('\n');
    llvm::StringRef line = split.first;
    llvm::StringRef rest = split.second;
    llvm::StringRef trimmed = line.trim();
    if (trimmed.empty() || trimmed.starts_with("#") || trimmed.starts_with("//") ||
        isHintMarkerLine(trimmed)) {
      remaining = rest;
      continue;
    }
    if (trimmed.starts_with("using ") || trimmed.starts_with("constexpr ") ||
        trimmed.starts_with("typedef ")) {
      remaining = rest;
      continue;
    }
    if (auto generatedName = parseGeneratedDeclarationName(trimmed))
      names.push_back(*generatedName);
    else if (!parseAnyDeclaredIdentifierName(trimmed))
      break;
    remaining = rest;
  }
  return names;
}

// Convert `/* PTOAS_PROVENANCE:rawname,... */` markers into trailing
// `// pto: %rawname` comments on the next generated-value declaration line.
// This gives issue #337 point 1 locatability: each generated C++ local that
// traces back to an input .pto SSA value is annotated with its original name,
// even when the identifier itself was renamed or sanitized (e.g. pure-digit
// %0 -> _0 still carries `// pto: %0`). The marker is consumed (removed) here.
static void emitProvenanceComments(std::string &segment) {
  static constexpr llvm::StringLiteral kProvenancePrefix =
      "/* PTOAS_PROVENANCE:";
  size_t searchPos = 0;
  llvm::SmallVector<std::pair<size_t, size_t>, 16> markers;
  // First collect marker spans and their raw names.
  struct ProvenanceMarker {
    size_t begin;
    size_t end; // exclusive of "*/"
    llvm::SmallVector<std::string, 4> names;
  };
  llvm::SmallVector<ProvenanceMarker, 16> found;
  while (true) {
    size_t markerPos = segment.find(kProvenancePrefix.str(), searchPos);
    if (markerPos == std::string::npos)
      break;
    size_t bodyBegin = markerPos + kProvenancePrefix.size();
    size_t markerEnd = segment.find("*/", bodyBegin);
    if (markerEnd == std::string::npos)
      break;
    auto names = parseNameHintMarker(
        llvm::StringRef(segment).slice(bodyBegin, markerEnd));
    size_t spanEnd = markerEnd + 2;
    // consume trailing newline so the marker line disappears cleanly.
    while (spanEnd < segment.size() &&
           (segment[spanEnd] == '\r' || segment[spanEnd] == '\n'))
      ++spanEnd;
    if (names) {
      found.push_back({markerPos, spanEnd, std::move(*names)});
    } else {
      // No recoverable names; just drop the marker.
      markers.push_back({markerPos, spanEnd});
    }
    searchPos = spanEnd;
  }

  // For each provenance marker, locate the next generated declaration line
  // and append a `// pto: %n` comment to it.
  for (const auto &m : found) {
    size_t windowEnd =
        std::min(m.end + static_cast<size_t>(2048), segment.size());
    llvm::StringRef searchWindow =
        llvm::StringRef(segment).substr(m.end, windowEnd - m.end);
    auto generatedNames = findNextHintedGeneratedNames(searchWindow);
    if (!generatedNames)
      continue;
    size_t pairCount =
        std::min(m.names.size(), generatedNames->size());
    if (pairCount == 0)
      continue;
    // Build the comment text: `// pto: %n0[, %n1, ...]`.
    std::string comment = " // pto: ";
    for (size_t i = 0; i < pairCount; ++i) {
      if (i != 0)
        comment += ", ";
      comment += "%";
      comment += m.names[i];
    }
    // Find the end of the declaration line that mentions the first generated
    // name, and insert the comment right before the newline.
    std::string firstName = (*generatedNames)[0];
    size_t namePos = segment.find(firstName, m.end);
    if (namePos == std::string::npos || namePos >= windowEnd)
      continue;
    size_t lineEnd = segment.find('\n', namePos);
    if (lineEnd == std::string::npos)
      lineEnd = segment.size();
    // Skip a trailing '\r'.
    size_t insertAt = lineEnd;
    // Avoid double-commenting if a provenance comment already sits there.
    llvm::StringRef tail =
        llvm::StringRef(segment).substr(namePos, lineEnd - namePos);
    if (tail.contains("// pto:"))
      continue;
    segment.insert(insertAt, comment);
  }

  // Remove all provenance marker spans (offsets captured before insertions
  // shifted; re-scan from scratch to be safe).
  std::string out;
  out.reserve(segment.size());
  size_t i = 0;
  while (i < segment.size()) {
    size_t mp = segment.find(kProvenancePrefix.str(), i);
    if (mp == std::string::npos) {
      out.append(segment, i, std::string::npos);
      break;
    }
    out.append(segment, i, mp - i);
    size_t me = segment.find("*/", mp + kProvenancePrefix.size());
    if (me == std::string::npos) {
      out.append(segment, i, std::string::npos);
      break;
    }
    me += 2;
    while (me < segment.size() &&
           (segment[me] == '\r' || segment[me] == '\n'))
      ++me;
    i = me;
  }
  segment.swap(out);
}

static void rewriteNameHintsInFunctionSegment(
    std::string &segment, llvm::ArrayRef<std::string> functionParamHints,
    llvm::ArrayRef<llvm::SmallVector<std::string, 4>> blockArgHints) {
  llvm::StringMap<std::string> replacements;
  llvm::SmallVector<PendingIdentifierRename, 8> pendingRenames =
      collectPendingIdentifierRenames(segment, functionParamHints, blockArgHints);
  std::set<std::string> usedNames =
      collectDeclaredIdentifiersInFunctionSegment(segment);
  for (const PendingIdentifierRename &rename : pendingRenames)
    usedNames.erase(rename.oldName);

  for (const PendingIdentifierRename &rename : pendingRenames) {
    llvm::StringRef oldName = rename.oldName;
    if (replacements.count(oldName))
      continue;
    std::string newName = makeUniqueCppIdentifier(rename.baseHint, usedNames);
    if (newName.empty() || newName == oldName)
      continue;
    replacements[oldName] = std::move(newName);
  }

  // Emit `// pto: %N` provenance comments from PTOAS_PROVENANCE markers
  // before the markers are stripped. Done before renaming so the comment is
  // attached to the correct declaration line; the comment body uses raw input
  // names and is unaffected by the vN->semantic rename below.
  emitProvenanceComments(segment);
  stripAllHintMarkers(segment);
  rewriteIdentifiersWithMap(segment, replacements);
}

static void rewriteNameHintMarkers(std::string &cpp,
                                   const FunctionArgHintMap &functionArgHints,
                                   const FunctionBlockArgHintMap &functionBlockArgHints) {
  llvm::SmallVector<std::string, 0> lines;
  for (llvm::StringRef ref(cpp); !ref.empty();) {
    auto split = ref.split('\n');
    lines.push_back(split.first.str());
    ref = split.second;
  }

  std::string rewritten;
  rewritten.reserve(cpp.size());
  size_t cursor = 0;
  int topLevelBraceDepth = 0;

  auto appendLines = [&](size_t begin, size_t end, bool stripMarkers) {
    if (begin >= end)
      return;
    std::string chunk;
    for (size_t i = begin; i < end; ++i) {
      chunk.append(lines[i]);
      if (i + 1 != end || end != lines.size())
        chunk.push_back('\n');
    }
    if (stripMarkers)
      stripAllHintMarkers(chunk);
    rewritten.append(chunk);
  };

  size_t i = 0;
  while (i < lines.size()) {
    llvm::StringRef trimmed = llvm::StringRef(lines[i]).trim();
    if (topLevelBraceDepth == 0 && isTopLevelFunctionStartLine(trimmed)) {
      size_t segmentBegin = i;
      while (segmentBegin > cursor &&
             isHintMarkerLine(llvm::StringRef(lines[segmentBegin - 1]).trim()))
        --segmentBegin;

      appendLines(cursor, segmentBegin, true);

      size_t segmentEnd = i;
      int segmentBraceDepth = 0;
      bool sawOpeningBrace = false;
      for (; segmentEnd < lines.size(); ++segmentEnd) {
        int lineDelta = countBraceDelta(lines[segmentEnd]);
        if (lines[segmentEnd].find('{') != std::string::npos)
          sawOpeningBrace = true;
        segmentBraceDepth += lineDelta;
        if (sawOpeningBrace && segmentBraceDepth == 0) {
          ++segmentEnd;
          break;
        }
      }

      std::string segment;
      for (size_t lineIndex = segmentBegin; lineIndex < segmentEnd; ++lineIndex) {
        segment.append(lines[lineIndex]);
        if (lineIndex + 1 != segmentEnd || segmentEnd != lines.size())
          segment.push_back('\n');
      }
      llvm::SmallVector<std::string, 4> paramHints;
      llvm::SmallVector<llvm::SmallVector<std::string, 4>, 4> blockHints;
      if (auto functionName = extractFunctionNameFromSegment(segment)) {
        auto it = functionArgHints.find(*functionName);
        if (it != functionArgHints.end())
          paramHints = it->second;
        auto blockIt = functionBlockArgHints.find(*functionName);
        if (blockIt != functionBlockArgHints.end())
          blockHints = blockIt->second;
      }
      rewriteNameHintsInFunctionSegment(segment, paramHints, blockHints);
      rewritten.append(segment);

      cursor = segmentEnd;
      i = segmentEnd;
      topLevelBraceDepth = 0;
      continue;
    }

    topLevelBraceDepth += countBraceDelta(lines[i]);
    ++i;
  }

  appendLines(cursor, lines.size(), true);
  cpp.swap(rewritten);
}

namespace {
struct ConstantDeclCandidate {
  size_t declLine = 0;
  std::string indent;
  std::string type;
  bool hasInitializer = false;
  std::string initializer;
  size_t assignmentCount = 0;
  size_t assignmentLine = 0;
  std::string assignmentRhs;
};
} // namespace

static bool isGeneratedValueName(llvm::StringRef name) {
  if (!name.consume_front("v") || name.empty())
    return false;
  return llvm::all_of(name, [](char c) { return std::isdigit(c); });
}

static bool isConstFoldableScalarType(llvm::StringRef type) {
  type = type.trim();
  if (type.starts_with("const ") || type.starts_with("constexpr "))
    return false;
  return llvm::StringSwitch<bool>(type)
      .Cases("bool", "float", "double", "half", "bfloat16_t", true)
      .Cases("int8_t", "uint8_t", "int16_t", "uint16_t", true)
      .Cases("int32_t", "uint32_t", "int64_t", "uint64_t", true)
      .Default(false);
}

static bool isLiteralInitializer(llvm::StringRef rhs) {
  rhs = rhs.trim();
  if (rhs.empty())
    return false;
  if (rhs == "true" || rhs == "false" || rhs == "nullptr")
    return true;

  static const llvm::Regex kIntLiteral(
      R"(^[+-]?(0[xX][0-9A-Fa-f]+|[0-9]+)[uUlL]*$)");
  static const llvm::Regex kFloatLiteral(
      R"(^[+-]?(([0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)([eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)[fF]?$)");
  static const llvm::Regex kHexFloatLiteral(
      R"(^[+-]?0[xX]([0-9A-Fa-f]+\.[0-9A-Fa-f]*|[0-9A-Fa-f]+|\.[0-9A-Fa-f]+)[pP][+-]?[0-9]+[fF]?$)");
  static const llvm::Regex kSpecialFloatLiteral(
      R"(^[+-]?(nan|inf)[fF]?$)");

  return kIntLiteral.match(rhs) || kFloatLiteral.match(rhs) ||
         kHexFloatLiteral.match(rhs) || kSpecialFloatLiteral.match(rhs);
}

static std::string normalizeConstInitializer(llvm::StringRef type,
                                             llvm::StringRef rhs) {
  type = type.trim();
  rhs = rhs.trim();
  if (type == "bool") {
    if (rhs == "0" || rhs == "false")
      return "false";
    if (rhs == "1" || rhs == "-1" || rhs == "true")
      return "true";
  }
  return rhs.str();
}

static bool parseConstantDeclarationLine(llvm::StringRef line,
                                         ConstantDeclCandidate &candidate,
                                         std::string &valueName) {
  llvm::StringRef trimmed = line.trim();
  if (trimmed.empty() || trimmed.starts_with("#") || trimmed.starts_with("//") ||
      !trimmed.ends_with(";"))
    return false;

  llvm::StringRef body = trimmed.drop_back().rtrim();
  if (body.starts_with("return") || body.starts_with("goto ") ||
      body.starts_with("if ") || body.starts_with("if(") ||
      body.starts_with("switch ") || body.starts_with("switch(") ||
      body.starts_with("for ") || body.starts_with("for(") ||
      body.starts_with("while ") || body.starts_with("while(") ||
      body.starts_with("case ") || body == "default")
    return false;

  llvm::StringRef lhs = body;
  llvm::StringRef rhs;
  if (size_t eqPos = body.find('='); eqPos != llvm::StringRef::npos) {
    lhs = body.take_front(eqPos).rtrim();
    rhs = body.drop_front(eqPos + 1).trim();
  }

  size_t lastWs = lhs.find_last_of(" \t");
  if (lastWs == llvm::StringRef::npos)
    return false;

  llvm::StringRef type = lhs.take_front(lastWs).rtrim();
  llvm::StringRef name = lhs.drop_front(lastWs + 1).trim();
  if (!isGeneratedValueName(name) || !isConstFoldableScalarType(type))
    return false;

  size_t indentLen = line.find_first_not_of(" \t");
  if (indentLen == llvm::StringRef::npos)
    indentLen = 0;
  candidate.indent = line.take_front(indentLen).str();
  candidate.type = type.str();
  valueName = name.str();

  if (!rhs.empty()) {
    if (!isLiteralInitializer(rhs))
      return false;
    candidate.hasInitializer = true;
    candidate.initializer = normalizeConstInitializer(type, rhs);
  }

  return true;
}

static bool parseGeneratedValueAssignment(llvm::StringRef line,
                                          llvm::StringRef &valueName,
                                          llvm::StringRef &rhs) {
  llvm::StringRef trimmed = line.trim();
  if (trimmed.empty() || trimmed.starts_with("#") || trimmed.starts_with("//") ||
      !trimmed.ends_with(";"))
    return false;

  llvm::StringRef body = trimmed.drop_back().rtrim();
  size_t eqPos = body.find('=');
  if (eqPos == llvm::StringRef::npos)
    return false;

  llvm::StringRef lhs = body.take_front(eqPos).rtrim();
  rhs = body.drop_front(eqPos + 1).trim();
  if (!isGeneratedValueName(lhs))
    return false;
  valueName = lhs;
  return true;
}

static void rewriteScalarConstantDecls(std::string &cpp) {
  llvm::SmallVector<std::string, 0> lines;
  for (llvm::StringRef ref(cpp); !ref.empty(); ref = ref.split('\n').second) {
    auto split = ref.split('\n');
    lines.push_back(split.first.str());
  }

  llvm::SmallVector<bool, 0> eraseLine(lines.size(), false);
  auto rewriteSegment = [&](size_t beginLine, size_t endLine) {
    llvm::StringMap<ConstantDeclCandidate> candidates;

    for (size_t i = beginLine; i <= endLine; ++i) {
      ConstantDeclCandidate candidate;
      std::string valueName;
      if (parseConstantDeclarationLine(lines[i], candidate, valueName)) {
        candidate.declLine = i;
        candidates[valueName] = std::move(candidate);
        continue;
      }

      llvm::StringRef assignedName;
      llvm::StringRef rhs;
      if (!parseGeneratedValueAssignment(lines[i], assignedName, rhs))
        continue;

      auto it = candidates.find(assignedName);
      if (it == candidates.end())
        continue;

      ConstantDeclCandidate &info = it->second;
      ++info.assignmentCount;
      info.assignmentLine = i;
      info.assignmentRhs = rhs.str();
    }

    for (auto &entry : candidates) {
      llvm::StringRef valueName = entry.getKey();
      ConstantDeclCandidate &info = entry.getValue();

      std::string initializer;
      if (info.hasInitializer) {
        if (info.assignmentCount != 0)
          continue;
        initializer = info.initializer;
      } else {
        if (info.assignmentCount != 1)
          continue;
        if (!isLiteralInitializer(info.assignmentRhs))
          continue;
        initializer = normalizeConstInitializer(
            info.type, llvm::StringRef(info.assignmentRhs));
        eraseLine[info.assignmentLine] = true;
      }

      lines[info.declLine] = (info.indent + "const " + info.type + " " +
                              valueName.str() + " = " + initializer + ";");
    }
  };

  int braceDepth = 0;
  size_t segmentStart = 0;
  for (size_t i = 0; i < lines.size(); ++i) {
    int depthBefore = braceDepth;
    for (char c : lines[i]) {
      if (c == '{')
        ++braceDepth;
      else if (c == '}')
        --braceDepth;
    }

    if (depthBefore == 0 && braceDepth > 0)
      segmentStart = i;
    if (depthBefore > 0 && braceDepth == 0)
      rewriteSegment(segmentStart, i);
  }

  std::string out;
  out.reserve(cpp.size());
  for (size_t i = 0; i < lines.size(); ++i) {
    if (eraseLine[i])
      continue;
    out.append(lines[i]);
    if (i + 1 != lines.size())
      out.push_back('\n');
  }
  cpp.swap(out);
}

static bool shouldDeclareVariablesAtTop(ModuleOp module) {
  auto hasMultiBlockFunc = [](auto func) { return func.getBlocks().size() > 1; };
  return llvm::any_of(module.getOps<func::FuncOp>(), hasMultiBlockFunc) ||
         llvm::any_of(module.getOps<emitc::FuncOp>(), hasMultiBlockFunc);
}

static void prepareVPTOForEmission(PassManager &pm) {
  auto &kernelModulePM = pm.nest<ModuleOp>();
  // Issue #485 workaround: unroll small constant-trip-count scf.for loops
  // inside pto.simt_entry functions, then constant-fold the induction-variable
  // dependent scf.if branches so subsequent canonicalize/cse eliminate them.
  kernelModulePM.addNestedPass<func::FuncOp>(
      pto::createPTOUnrollSIMTForPass());
  kernelModulePM.addPass(createSCCPPass());
  kernelModulePM.addPass(createCanonicalizerPass());
  kernelModulePM.addPass(createCSEPass());
  kernelModulePM.addPass(pto::createVPTOPtrNormalizePass());
  kernelModulePM.addPass(pto::createVPTOPtrCastCleanupPass());
  kernelModulePM.addPass(createReconcileUnrealizedCastsPass());
  kernelModulePM.addNestedPass<func::FuncOp>(
      createVPTOExpandWrapperOpsPass());
  kernelModulePM.addPass(createCSEPass());
  kernelModulePM.addNestedPass<func::FuncOp>(
      pto::createPTOInferVPTOVecScopePass());
  kernelModulePM.addPass(createCanonicalizerPass());
  kernelModulePM.addPass(createCSEPass());
  kernelModulePM.addPass(pto::createPTOValidateVPTOEmissionIRPass());
}

static void lowerPTOToVPTOBackend(PassManager &pm, ModuleOp module, int argc,
                                  char **argv) {
  auto &kernelModulePM = pm.nest<ModuleOp>();
  auto moduleArchAttr =
      module->getAttrOfType<mlir::StringAttr>("pto.target_arch");
  const bool enableA5VPTOPostLoweringFusionLifecycle =
      enableOpFusion && moduleArchAttr && moduleArchAttr.getValue() == "a5";

  pto::ExpandTileOpOptions expandOpts = resolveExpandTileOpOptions(argc, argv);
  kernelModulePM.addPass(pto::createExpandTileOpPass(expandOpts));

  kernelModulePM.addPass(pto::createPTOInlineLibCallPass());
  kernelModulePM.addNestedPass<mlir::func::FuncOp>(
      pto::createFoldTileBufIntrinsicsPass("shape-only"));
  if (enableA5VPTOPostLoweringFusionLifecycle) {
    kernelModulePM.addPass(pto::createPTOLowLevelLoopFusionPass());
    kernelModulePM.addPass(mlir::createCanonicalizerPass());
    kernelModulePM.addPass(mlir::createCSEPass());
    kernelModulePM.addNestedPass<mlir::func::FuncOp>(
        pto::createPTOFusionPredicateElisionPass());
    kernelModulePM.addNestedPass<mlir::func::FuncOp>(
        pto::createPTOFusionLoadStoreElisionPass());
    kernelModulePM.addNestedPass<mlir::func::FuncOp>(
        pto::createPTOFlattenFusionRegionPass());
    kernelModulePM.addPass(mlir::createCSEPass());
  }
  kernelModulePM.addNestedPass<mlir::func::FuncOp>(
      pto::createFoldTileBufIntrinsicsPass("addr-only"));
  kernelModulePM.addPass(mlir::createSCCPPass());
  kernelModulePM.addPass(mlir::createCanonicalizerPass());
}

static pto::VPTOEmissionOptions
buildVPTOEmissionOptions(const pto::CANNVersion &cannVersion) {
  pto::VPTOEmissionOptions options;
  options.dumpVPTOIR = false;
  options.targetTriple = "hiipu64-hisilicon-cce";
  options.cannVersion = cannVersion;
  return options;
}

static int emitVPTOBackendResult(ModuleOp module, PTOASCompileResult &result,
                                 bool emitHostStub,
                                 const pto::CANNVersion &cannVersion) {
  if (emitVPTO) {
    result.kind = PTOASCompileResultKind::Text;
    llvm::raw_string_ostream os(result.textOutput);
    module.print(os);
    os << "\n";
    os.flush();
    return 0;
  }

  if (emitVPTOLLVMDialect) {
    result.kind = PTOASCompileResultKind::Text;
    pto::VPTOEmissionOptions options = buildVPTOEmissionOptions(cannVersion);
    if (failed(pto::lowerVPTOModuleToLLVMIRText(
            module, options, result.textOutput, llvm::errs()))) {
      llvm::errs() << "Error: Failed to lower VPTO to LLVM IR.\n";
      return 1;
    }
    return 0;
  }

  pto::VPTOEmissionOptions options = buildVPTOEmissionOptions(cannVersion);
  std::string stubSource;
  if (emitHostStub) {
    if (failed(pto::emitVPTOHostStubSource(module, stubSource, llvm::errs()))) {
      llvm::errs() << "Error: Failed to emit VPTO host stub source.\n";
      return 1;
    }
  }

  if (failed(
          pto::lowerVPTOModuleToLLVMModules(module, options,
                                            result.vptoCubeModule,
                                            result.vptoVectorModule,
                                            llvm::errs()))) {
    llvm::errs() << "Error: Failed to lower VPTO to LLVM modules.\n";
    return 1;
  }

  result.vptoStubSource = std::move(stubSource);
  result.kind = PTOASCompileResultKind::VPTOObject;
  return 0;
}

static LogicalResult runVPTOBackendPipeline(OwningOpRef<ModuleOp> &module,
                                            int argc, char **argv,
                                            bool hasTileOpsToExpand) {
  PassManager pm(module->getContext());
  pm.enableVerifier();
  pm.addPass(pto::createVPTOSplitCVModulePass());
  pm.addPass(pto::createVPTONormalizeContainerPass());
  if (hasTileOpsToExpand)
    lowerPTOToVPTOBackend(pm, module.get(), argc, argv);
  prepareVPTOForEmission(pm);
  if (failed(applyConfiguredPassManagerCLOptions(
          pm, "VPTO unified emission pipeline")))
    return failure();
  if (failed(pm.run(module.get()))) {
    llvm::errs() << "Error: VPTO emission pipeline failed.\n";
    return failure();
  }
  return success();
}

int mlir::pto::compilePTOASModule(
    OwningOpRef<ModuleOp> &module, PTOASContext &context,
    PTOBackend effectiveBackend, PTOASCompileResult &result,
    bool emitVPTOHostStub) {
  result.reset();
  llvm::StringRef arch = context.getArch();
  int argc = context.getArgc();
  char **argv = context.getArgv();

  // Name-hint provenance: textual .pto inputs had their SSA/arg/block-arg names
  // attached to op Locations by the driver right after parsing. Collect the
  // function-level arg/block-arg hint maps now, before any lowering, so they
  // survive CSE and emitc.variable hoisting.
  FunctionArgHintMap functionArgHints;
  FunctionBlockArgHintMap functionBlockArgHints;
  if (module) {
    functionArgHints = collectFunctionArgNameHints(*module);
    functionBlockArgHints = collectFunctionBlockArgNameHints(*module);
  }

  if (effectiveBackend != PTOBackend::VPTO &&
      (emitVPTO || emitVPTOLLVMDialect || ptoPrintSeamIR ||
       !ptoSeamIRFile.empty())) {
    llvm::errs() << "Error: VPTO-specific flags require "
                    "--pto-backend=vpto or pto.backend = \"vpto\".\n";
    return 1;
  }

  PTOBuildLevel effectiveLevel = defaultBuildLevel();
  if (!parseBuildLevel(ptoBuildLevel, effectiveLevel)) {
    llvm::errs() << "Error: invalid --pto-level='" << ptoBuildLevel
                 << "'. Expected 'level1', 'level2', or 'level3'.\n";
    return 1;
  }
  if (enableBufidSync && arch != "a5") {
    llvm::errs() << "Error: --enable-bufid_sync requires --pto-arch=a5.\n";
    return 1;
  }

  if (enableOpFusion) {
    if (arch != "a5") {
      llvm::errs() << "Warning: --enable-op-fusion is ignored because "
                      "--pto-arch=a5 is required.\n";
    } else if (effectiveLevel == PTOBuildLevel::Level1) {
      llvm::errs() << "Warning: --enable-op-fusion is ignored because "
                      "--pto-level=level2 or level3 is required.\n";
    }
  }

  const bool enableA5FusionPath =
      enableOpFusion && arch == "a5" &&
      effectiveLevel != PTOBuildLevel::Level1;
  const bool enableA5EmitCFusionPath =
      enableA5FusionPath && effectiveBackend == PTOBackend::EmitC;
  const bool enableA5VPTOFusionPath =
      enableA5FusionPath && effectiveBackend == PTOBackend::VPTO;

  bool invalidAutoSyncTailHint = false;
  module->walk([&](mlir::func::FuncOp func) {
    auto hintAttr =
        func->getAttrOfType<mlir::StringAttr>("pto.auto_sync_tail_hint");
    if (!hintAttr)
      return;

    std::string normalizedHint;
    if (!parseAutoSyncTailHint(hintAttr.getValue(), normalizedHint)) {
      func.emitError("invalid pto.auto_sync_tail_hint '")
          << hintAttr.getValue()
          << "'. Expected 'barrier-all' (or 'default') or "
             "'mte3-to-s-event0'.";
      invalidAutoSyncTailHint = true;
      return;
    }
    func->setAttr("pto.auto_sync_tail_hint",
                  mlir::StringAttr::get(module->getContext(), normalizedHint));
  });
  if (invalidAutoSyncTailHint)
    return 1;

  bool hasTAssign = false;
  module->walk([&](pto::TAssignOp) { hasTAssign = true; });

  if (hasTAssign && effectiveLevel != PTOBuildLevel::Level3) {
    llvm::errs() << "Error: pto.tassign is only supported when "
                    "--pto-level=level3.\n";
    return 1;
  }

  if (hasTAssign && enableInsertSync) {
    llvm::errs() << "Error: pto.tassign requires --enable-insert-sync to be "
                    "disabled.\n";
    return 1;
  }

  int enabledAutoSyncModes =
      (enableInsertSync ? 1 : 0) + (enableBufidSync ? 1 : 0) +
      (enableInjectBarrierAllSync ? 1 : 0) + (enableGraphSyncSolver ? 1 : 0);
  if (enabledAutoSyncModes > 1) {
    llvm::errs() << "Error: --enable-insert-sync, --enable-bufid_sync, "
                    "--enable-inject-barrier-all-sync, and "
                    "--enable-graph-sync-solver are mutually exclusive.\n";
    return 1;
  }
  if (hasTAssign && enableInjectBarrierAllSync) {
    llvm::errs() << "Error: pto.tassign requires "
                    "--enable-inject-barrier-all-sync to be disabled.\n";
    return 1;
  }
  if (hasTAssign && enableGraphSyncSolver) {
    llvm::errs() << "Error: pto.tassign requires --enable-graph-sync-solver "
                    "to be disabled.\n";
    return 1;
  }
  if (hasTAssign && enableBufidSync) {
    llvm::errs() << "Error: pto.tassign requires --enable-bufid_sync to be "
                    "disabled.\n";
    return 1;
  }

  if (effectiveLevel == PTOBuildLevel::Level3) {
    bool missing = false;
    module->walk([&](pto::AllocTileOp op) {
      if (!op.getAddr()) {
        op.emitError("requires 'addr' operand when --pto-level=level3");
        missing = true;
      }
    });
    if (missing)
      return 1;
  } else {
    bool hasAddr = false;
    module->walk([&](pto::AllocTileOp op) {
      if (op.getAddr()) {
        op.emitError(
            "unexpected 'addr' operand: only supported when --pto-level=level3");
        hasAddr = true;
      }
    });
    if (hasAddr)
      return 1;
  }

  {
    PassManager preBackendPM(module->getContext());
    preBackendPM.enableVerifier();
    preBackendPM.addPass(pto::createPTONormalizeUncoveredTileSectionsPass());
    if (failed(preBackendPM.run(module.get()))) {
      llvm::errs() << "Error: failed to normalize uncovered PTO tile sections.\n";
      return 1;
    }
  }

  const bool hasTileOpsToExpand = hasUnexpandedTileOps(*module);

  if (effectiveBackend == PTOBackend::VPTO && !hasTileOpsToExpand) {
    if (ptoPrintSeamIR || !ptoSeamIRFile.empty()) {
      llvm::errs() << "Error: shared pre-backend seam IR is unavailable when "
                      "skipping the shared PTO-to-VPTO lowering pipeline.\n";
      return 1;
    }
    if (failed(runVPTOBackendPipeline(module, argc, argv, hasTileOpsToExpand)))
      return 1;
    return emitVPTOBackendResult(*module, result, emitVPTOHostStub,
                                 context.getCANNVersionOrDefault());
  }

  // Main PassManager
  PassManager pm(module->getContext());

  if (failed(applyPassManagerCLOptions(pm)))
    return 1;

  // Rank-2 → rank-5 view canonicalization is currently gated on the VPTO
  // backend to limit blast radius.  A3/A5 EmitC codegen already pads strides
  // to rank-5 via InferPTOLayout and buildGlobalTensorShapeAndStride, so it
  // does not need the canonicalization pass at the IR level.  When VPTO
  // validation is complete and the pass is proven stable, the gate can be
  // lifted to make it unconditional for all backends.
  if (effectiveBackend == PTOBackend::VPTO)
    pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOCanonicalizeIRPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      pto::createPTOAssignDefaultFrontendPipeIdPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      pto::createPTOLowerFrontendPipeOpsPass());
  //pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOVerifyTFreePass());
  pm.addPass(pto::createPTOInferValidatePipeInitPass());
  pm.addNestedPass<mlir::func::FuncOp>(pto::createLoweringSyncToPipePass());
  if (!disableInferLayout)
    pm.addNestedPass<mlir::func::FuncOp>(pto::createInferPTOLayoutPass());
  pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOA5NormalizeTMovPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      pto::createPTOValidateIntToPtrUsesPass());

  // Keep frontend fusion on tile-native PTO IR and annotate last_use directly
  // on scheduled block-local spans before the shared mainline lowers tiles.
  // The shape-inference switch drives FusionPlan only: that is where the
  // iteration-domain decisions (static vs ShapeConstraintSolver) are made.
  // FusionRegionGen consumes only the shared pre-fusion dataflow graph (cached
  // by the analysis manager and built once by FusionPlan) plus the resulting
  // pto.fusion.group_id/order metadata; it never consults the domain classes,
  // so it takes no option here.
  pto::FusionPlanOptions fusionPlanOpts;
  fusionPlanOpts.enableShapeInference = enableShapeInference;
  if (enableA5EmitCFusionPath) {
    pm.addNestedPass<mlir::func::FuncOp>(
        pto::createFusionPlanPass(fusionPlanOpts));
    pm.addNestedPass<mlir::func::FuncOp>(pto::createOpSchedulingPass());
    pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOMarkLastUsePass());
  } else if (enableA5VPTOFusionPath) {
    pm.addNestedPass<mlir::func::FuncOp>(
        pto::createFusionPlanPass(fusionPlanOpts));
    pm.addNestedPass<mlir::func::FuncOp>(pto::createOpSchedulingPass());
    pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOFusionRegionGenPass());
  }

  pm.addPass(pto::createPTOViewToMemrefPass());

  if (effectiveLevel != PTOBuildLevel::Level3) {
    PlanMemoryOptions planMemoryOption;
    planMemoryOption.memMode = MemPlanMode::LOCAL_MEM_PLAN;
    planMemoryOption.enableGlobalReuse = false;
    planMemoryOption.enablePrintMemoryAllocatedSize = false;
    pm.addPass(pto::createPlanMemoryPass(planMemoryOption));
  }
  pm.addPass(pto::createPTOResolveReservedBuffersPass());

  // Conditionally add one automatic synchronization mode. Barrier-all is a
  // conservative standalone pass; InsertSync and GraphSyncSolver are set/wait
  // solvers, while BufidSync is A5-only get_buf/rls_buf synchronization.
  pm.addNestedPass<mlir::func::FuncOp>(
      pto::createPTOVerifySubkernelPipeContractPass());
  if (enableInsertSync)
    pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOInsertSyncPass());
  else if (enableBufidSync) {
    PTOBufidSyncOptions bufidOptions;
    bufidOptions.enableBufidSyncDebug = enableBufidSyncDebug;
    pm.addNestedPass<mlir::func::FuncOp>(
        pto::createPTOBufidSyncPass(bufidOptions));
  } else if (enableInjectBarrierAllSync)
    pm.addNestedPass<mlir::func::FuncOp>(
        pto::createPTOInjectBarrierAllSyncPass());
  else if (enableGraphSyncSolver) {
    PTOGraphSyncSolverOptions graphSyncOpts;
    graphSyncOpts.eventIdNumMax = graphSyncSolverEventIdMax;
    pm.addNestedPass<mlir::func::FuncOp>(
        pto::createPTOGraphSyncSolverPass(graphSyncOpts));
  }

  if (emitMlirIR) {
    if (failed(pm.run(*module))) {
      llvm::errs() << "Error: Pass execution failed.\n";
      return 1;
    }
    result.kind = PTOASCompileResultKind::Text;
    llvm::raw_string_ostream os(result.textOutput);
    module->print(os);
    os.flush();
    return 0;
  }

  // Reintroduce tile-native handles once on the shared mainline so both
  // backends consume the same post-planning seam IR.
  pm.addPass(pto::createPTOMaterializeTileHandlesPass());
  pm.addPass(createCSEPass());
  // Inline PTODSL backend helpers only after the shared mainline has
  // materialized tile-native handles, so helper arguments are restored to the
  // tile_buf ABI before qk.as_ptr()-style bridges are cloned into callers.
  pm.addPass(pto::createPTOInlineBackendHelpersPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (failed(applyConfiguredPassManagerCLOptions(pm, "main PTOAS pipeline")))
    return 1;

  module->getOperation()->setAttr("pto.target_arch",
                                  mlir::StringAttr::get(module->getContext(), arch));

  if (effectiveBackend == PTOBackend::VPTO) {
    if (failed(pm.run(*module))) {
      llvm::errs() << "Error: Pass execution failed.\n";
      return 1;
    }

    if (ptoPrintSeamIR) {
      module->print(llvm::errs());
      llvm::errs() << "\n";
    }
    if (failed(emitSharedPreBackendSeamIR(*module, ptoSeamIRFile)))
      return 1;

    if (failed(runVPTOBackendPipeline(module, argc, argv, hasTileOpsToExpand)))
      return 1;
    return emitVPTOBackendResult(*module, result, emitVPTOHostStub,
                                 context.getCANNVersionOrDefault());
  }

  if (arch == "a3") {
    pm.addPass(pto::createEmitPTOManualPass(pto::PTOArch::A3));
  } else {
    pm.addPass(pto::createEmitPTOManualPass(pto::PTOArch::A5));
  }
  pm.addPass(emitc::createFormExpressionsPass());
  pm.addPass(mlir::createCSEPass());

  if (failed(pm.run(*module))) {
    llvm::errs() << "Error: Pass execution failed.\n";
    return 1;
  }

  applyFunctionBlockArgNameHintsToEmitC(*module, functionBlockArgHints);
  dropEmptyEmitCExpressions(module.get());
  materializeControlFlowOperands(module.get());
  if (failed(reorderEmitCFunctions(module.get()))) {
    llvm::errs() << "Error: Failed to order emitted functions for C++ emission.\n";
    return 1;
  }
  annotateEmitCNameHints(*module);

  // Emit C++ to string, then post-process, then write to output file.
  std::string cppOutput;
  llvm::raw_string_ostream cppOS(cppOutput);
  // CFG-style lowering (e.g. scf.while -> cf.br/cf.cond_br) may introduce
  // multiple blocks, requiring variables to be declared at the top for valid
  // C++ emission.
  bool declareVariablesAtTop = shouldDeclareVariablesAtTop(*module);
  if (failed(emitc::translateToCpp(*module, cppOS,
                                  /*declareVariablesAtTop=*/declareVariablesAtTop))) {
    llvm::errs() << "Error: Failed to emit C++.\n";
    return 1;
  }
  cppOS.flush();
  rewriteTileGetSetValueMarkers(cppOutput);
  rewriteAsyncEventMarkers(cppOutput);
  rewritePtrScalarMarkers(cppOutput);
  rewriteScalarGMStoreFlushMarkers(cppOutput);
  rewriteEventIdArrayMarkers(cppOutput);
  pto::rewriteLastUseMarkersInCpp(cppOutput);
  rewriteAddPtrTraceMarkers(cppOutput, emitAddPtrTrace);
  rewriteMalformedVerbatimSemicolons(cppOutput);
  rewriteScalarConstantDecls(cppOutput);
  rewriteHoistedGlobalTensorDecls(cppOutput);
  rewriteNameHintMarkers(cppOutput, functionArgHints, functionBlockArgHints);

  result.kind = PTOASCompileResultKind::Text;
  result.textOutput = std::move(cppOutput);
  return 0;
}
