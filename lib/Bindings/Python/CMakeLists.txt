# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

include(AddMLIRPython)

# Do not carry toolchain-injected paths, such as a Conda compiler RPATH, into
# installed Python modules. Distribution builds keep only MLIR's relative
# package RPATHs; editable builds add their external link paths below.
set(CMAKE_INSTALL_REMOVE_ENVIRONMENT_RPATH ON)

# scikit-build-core editable installs use the external LLVM build directly.
# Keep this developer-only fallback local to the MLIR Python targets; wheel
# builds use package-relative RPATHs and let auditwheel/delocate collect LLVM.
if(SKBUILD_STATE STREQUAL "editable")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
endif()

# This directory materializes C++ sources exported by upstream MLIR. PTOAS
# keeps -Werror enabled globally, but upstream sources are not part of the
# project warning contract. Scope the exception to this directory; the
# project-owned ptoas._core extension is built in tools/ptoas and therefore
# remains under -Werror.
if(PTOAS_ENABLE_WERROR)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wno-error>")
  elseif(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/WX->")
  endif()
endif()

# Build the `mlir` Python package from MLIR's exported source declarations and
# add PTOAS's dialect to the same package. This is the composition mechanism
# supported by AddMLIRPython and avoids copying or overlaying another build's
# package tree.
set(PTOAS_MLIR_PYTHON_PACKAGE_DIR "${CMAKE_BINARY_DIR}/python/mlir")

declare_mlir_python_sources(PTOASPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT PTOASPythonSources
  ROOT_DIR "${CMAKE_SOURCE_DIR}/python/pto"
  TD_FILE dialects/PTOOps.td
  SOURCES dialects/pto.py
  DIALECT_NAME pto
)

# Use MLIR's standard site-initializer hook to seed each Python Context with
# the selected upstream dialects shipped by PTOAS. Declaring the extension in
# the Python source graph lets AddMLIRPython own its build, installation, and
# common-CAPI dependency collection.
set(PTOAS_PYTHON_REGISTRATION_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/PythonRegistration.cpp")
declare_mlir_python_extension(PTOASPythonSources.Registration
  MODULE_NAME _site_initialize_0
  ADD_TO_PARENT PTOASPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES PythonRegistration.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIArith
    MLIRCAPIFunc
    MLIRCAPILLVM
    MLIRCAPIMath
    MLIRCAPIMemRef
    MLIRCAPISCF
)

# This directory otherwise materializes only upstream MLIR/nanobind C++
# sources under -Wno-error. Keep the project-owned registration source under
# PTOAS's default strict warning policy while retaining one declarative Python
# package graph.
if(PTOAS_ENABLE_WERROR)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set_property(SOURCE "${PTOAS_PYTHON_REGISTRATION_SOURCE}" APPEND PROPERTY
      COMPILE_OPTIONS -Werror)
  elseif(MSVC)
    set_property(SOURCE "${PTOAS_PYTHON_REGISTRATION_SOURCE}" APPEND PROPERTY
      COMPILE_OPTIONS /WX)
  endif()
endif()

# Package only the upstream Python APIs used by PTOAS and PTODSL. In
# particular, do not instantiate MLIRPythonSources or RegisterEverything:
# downstream MLIR packages are expected to select their public dialect set
# explicitly rather than linking every upstream dialect, translation and pass.
set(PTOAS_MLIR_PYTHON_SOURCE_SETS
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.llvm
  MLIRPythonSources.Dialects.math
  MLIRPythonSources.Dialects.memref
  MLIRPythonSources.Dialects.scf
)

# Keep one common CAPI DSO for the upstream MLIR Python package. PTOAS owns a
# single ptoas._core extension, built in tools/ptoas, which links this common
# CAPI and contains both the PTO dialect implementation and compiler driver.
add_mlir_python_common_capi_library(PTOASPythonCAPI
  INSTALL_COMPONENT PTOAS_Python
  INSTALL_DESTINATION "mlir/_mlir_libs"
  OUTPUT_DIRECTORY "${PTOAS_MLIR_PYTHON_PACKAGE_DIR}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_HEADERS
    MLIRPythonCAPI.HeaderSources
  DECLARED_SOURCES
    ${PTOAS_MLIR_PYTHON_SOURCE_SETS}
    PTOASPythonSources
  # MLIRPythonExtension.Core currently obtains its rewrite CAPI through
  # RegisterEverything. PTOAS deliberately omits RegisterEverything, so retain
  # the missing direct dependency until upstream Core declares it itself.
  EMBED_LIBS
    MLIRCAPITransforms
)
# Imported MLIR aggregate objects do not expose a source language from which
# CMake can infer the linker driver. The common CAPI is C++, matching MLIR's
# own in-tree target.
set_property(TARGET PTOASPythonCAPI PROPERTY LINKER_LANGUAGE CXX)

add_mlir_python_modules(PTOAS_Python
  ROOT_PREFIX "${PTOAS_MLIR_PYTHON_PACKAGE_DIR}"
  INSTALL_PREFIX mlir
  DECLARED_SOURCES
    ${PTOAS_MLIR_PYTHON_SOURCE_SETS}
    PTOASPythonSources
  COMMON_CAPI_LINK_LIBS
    PTOASPythonCAPI
)
