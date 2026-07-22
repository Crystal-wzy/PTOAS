# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(PTOAS_RUNTIME_SOURCES
  ptoas.cpp
  driver.cpp
  VPTOHostStubEmission.cpp
  ObjectEmission.cpp
  TilelangDaemon.cpp
)

set(PTOAS_WRAPPER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/ptoas_wrapper.py")
set(PTOAS_WRAPPER_BINARY "${CMAKE_CURRENT_BINARY_DIR}/ptoas")

configure_file(
  "${PTOAS_WRAPPER_SOURCE}"
  "${PTOAS_WRAPPER_BINARY}"
  @ONLY
)
file(CHMOD "${PTOAS_WRAPPER_BINARY}"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

function(ptoas_configure_runtime_target target_name)
  target_compile_definitions(${target_name} PRIVATE
    PTOAS_RELEASE_VERSION="${PTOAS_CLI_VERSION}"
    # Source-tree defaults for TileLib expansion. These let ptoas run directly
    # from the build tree without passing --tilelang-path /
    # --tilelang-pkg-path / --ptodsl-pkg-path. Installed layouts are expected
    # to launch through the wrapper/launcher flow, which injects explicit
    # runtime paths instead of relying on executable-path probing.
    PTOAS_DEFAULT_TILELANG_PATH="${CMAKE_SOURCE_DIR}/lib/TileOps"
    PTOAS_DEFAULT_TILELANG_PKG_PATH="${CMAKE_SOURCE_DIR}/tilelang-dsl/python"
    PTOAS_DEFAULT_PTODSL_PKG_PATH="${CMAKE_SOURCE_DIR}/ptodsl"
  )
  target_link_libraries(${target_name} PUBLIC
    PTOIR
    PTOTransforms
    ptobc_lib
    MLIRMlirOptMain
    MLIRBufferizationTransforms
    MLIRArithTransforms
    MLIRTensorTransforms
    MLIRFuncTransforms
    MLIRIR
    MLIRParser
    MLIRPass
    MLIREmitCDialect
    MLIRSCFDialect
    MLIRGPUDialect
    MLIRTransforms
    MLIRTargetCpp
    MLIRAffineDialect
    MLIRArithDialect
    MLIRTensorDialect
    MLIRControlFlowDialect
    MLIRBufferizationDialect
    MLIRFuncDialect
    MLIRFuncInlinerExtension
    MLIRLLVMIRTransforms
    MLIRSupport
    LLVMOption
    MLIRTransformUtils
  )
  add_dependencies(${target_name}
    PTOOpsIncGen
  )
endfunction()

# Keep the LLVM-based driver sources under LLVM's RTTI/exception compile mode.
# The pybind11 translation unit must retain the Python extension's normal RTTI
# and exception mode, so it is compiled by the module target separately.
add_library(ptoas_runtime_objects OBJECT ${PTOAS_RUNTIME_SOURCES})
llvm_update_compile_flags(ptoas_runtime_objects)
ptoas_configure_runtime_target(ptoas_runtime_objects)

# Build the normal Python extension used by wheel, build-tree, install-tree,
# and Python-based standalone distributions. Linking the object library pulls
# in the driver implementation and propagates its native link dependencies.
pybind11_add_module(ptoas_runtime MODULE NativeModule.cpp)
target_link_libraries(ptoas_runtime PRIVATE ptoas_runtime_objects)
set_target_properties(ptoas_runtime PROPERTIES
  OUTPUT_NAME "_native"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python-module"
)

set(PTOAS_BUILD_PYTHON_PACKAGE_DIR "${CMAKE_BINARY_DIR}/python/ptoas")
add_custom_target(PTOASPythonPackage ALL
  COMMAND ${CMAKE_COMMAND} -E remove_directory
          "${PTOAS_BUILD_PYTHON_PACKAGE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_SOURCE_DIR}/ptodsl/ptoas"
          "${PTOAS_BUILD_PYTHON_PACKAGE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:ptoas_runtime>"
          "${PTOAS_BUILD_PYTHON_PACKAGE_DIR}/$<TARGET_FILE_NAME:ptoas_runtime>"
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${PTOAS_BUILD_PYTHON_PACKAGE_DIR}/_runtime/share/ptoas"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_SOURCE_DIR}/lib/TileOps"
          "${PTOAS_BUILD_PYTHON_PACKAGE_DIR}/_runtime/share/ptoas/TileOps"
  COMMAND ${CMAKE_COMMAND} -E remove_directory
          "${PTOAS_BUILD_PYTHON_PACKAGE_DIR}/_runtime/share/ptoas/TileOps/__pycache__"
  DEPENDS ptoas_runtime
  COMMENT "Staging the ptoas Python package"
  VERBATIM
)
if(TARGET PTODSLPackage)
  add_dependencies(PTOASPythonPackage PTODSLPackage)
endif()

# Keep the historical aggregate target name for existing developer commands;
# the actual runtime payload now lives in the staged Python package.
add_custom_target(ptoas_runtime_staging ALL
  DEPENDS PTOASPythonPackage
)

if(TARGET ptoas_runtime_deps)
  add_dependencies(ptoas_runtime_deps ptoas_runtime_staging)
endif()

install(PROGRAMS "${PTOAS_WRAPPER_BINARY}"
  DESTINATION bin
  COMPONENT PTOAS_Runtime
)

install(TARGETS ptoas_runtime
  LIBRARY DESTINATION ptoas
  COMPONENT PTOAS_Python
)

install(
  DIRECTORY "${CMAKE_SOURCE_DIR}/lib/TileOps"
  DESTINATION "ptoas/_runtime/share/ptoas"
  COMPONENT PTOAS_Python
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)
