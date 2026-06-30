# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
TileLib render runtime.

Traces a tilelang-style template body into a standalone ``func.func`` whose MLIR is on
par with the legacy tilelang-dsl render (``tile_buf_addr`` -> memref, ``memref.subview``,
``pto.vlds/vadd/vsts``, dynamic ``pto.tile_valid_rows/cols``).

Control flow is handled by the engine's AST rewrite (``rewrite_jit_function``): plain
``for x in range(...)`` in the template body is rewritten at trace time to
``pto.for_(...).carry(...)`` (the ``_control_flow`` surface, re-exported via :mod:`author`),
with loop-carried variables detected by liveness. This module therefore owns only the
entry tile_buf typing and the golden-shaped module/func container; loops, slicing,
valid-shape, mask and vector ops all come from the existing ptodsl engine.
"""

from __future__ import annotations

import inspect

from .metadata import TileSpec, scalar_descriptor
from .._ast_rewrite import rewrite_jit_function
from .._bootstrap import make_context
from .._surface_types import Tile
from .._surface_values import TileValue
from .._tracing import KernelModuleSpec, ModuleStyle, TracingRuntime
from .._tracing.active import activate_runtime, activate_session
from .._types import _resolve

from mlir.dialects import func
from mlir.ir import Attribute, InsertionPoint, Location, Module, StringAttr, UnitAttr


# ── tile handle handed to the template body ────────────────────────────────────────

class _TemplateTile(TileValue):
    """Engine ``TileValue`` with the template-author alias ``element_type`` and forced
    dynamic ``valid_shape`` (emit ``pto.tile_valid_rows/cols`` rather than folding the
    static ``v_row/v_col`` carried in the tile_buf type).

    Metadata (shape/dtype/memory_space) is supplied from the ``TileSpec`` because a raw
    entry-block ``tile_buf`` type is not introspectable by ``parse_tile_type_metadata``;
    supplying it explicitly takes the fast path in ``infer_memref_type_from_surface_value``.
    """

    def __init__(self, value, spec: TileSpec):
        elem = _resolve(scalar_descriptor(spec.dtype))
        super().__init__(
            value,
            shape=tuple(spec.shape),
            physical_shape=tuple(spec.shape),
            dtype=elem,
            memory_space=spec.memory_space,
            valid_shape=None,
        )
        # Force the dynamic valid-shape ops to match the tilelang render.
        self.static_valid_shape = None
        self._valid_shape._cache.clear()

    @property
    def element_type(self):
        return self.dtype


# ── tracing runtime ────────────────────────────────────────────────────────────────

class _TemplateTrace(TracingRuntime):
    def __init__(self, descriptor, tile_specs: dict):
        super().__init__(
            KernelModuleSpec(
                function_name=descriptor.name,
                target_arch=descriptor.target,
                kernel_kind="vector",
                mode="auto",
                module_style=ModuleStyle.NESTED,
                source_file=inspect.getsourcefile(descriptor.py_fn) or inspect.getfile(descriptor.py_fn),
                source_line=getattr(descriptor.py_fn.__code__, "co_firstlineno", None),
            )
        )
        self.descriptor = descriptor
        self.tile_specs = tile_specs
        self._ordered_specs: list = []
        self._signature_parameters = tuple(inspect.signature(descriptor.py_fn).parameters.items())

    def compute_argument_types(self):
        arg_types = []
        ordered = []
        for param_name, param in self._signature_parameters:
            if not _is_tile_annotation(param.annotation):
                raise TypeError(
                    f"tile-template parameters must be annotated Tile; {param_name!r} is {param.annotation!r}"
                )
            spec = self.tile_specs.get(param_name)
            if spec is None:
                raise ValueError(f"missing TileSpec for parameter {param_name!r}")
            ordered.append((param_name, spec))
            arg_types.append(spec.mlir_type())
        self._ordered_specs = ordered
        return arg_types

    def bind_entry_arguments(self, entry_arguments):
        return tuple(
            _TemplateTile(arg, spec) for arg, (_, spec) in zip(entry_arguments, self._ordered_specs)
        )

    def trace_entry(self, *args):
        # Apply the engine's AST control-flow rewrite so the template body can use plain
        # `for x in range(...)` (rewritten to pto.for_(...).carry(...)) like tilelang.
        rewritten = rewrite_jit_function(self.descriptor.py_fn)
        rewritten(*args)

    # Custom golden-shaped container: single module(target_arch) + func(instance, kernel_kind).
    def build_module(self):
        ctx = make_context()
        with ctx, Location.unknown():
            arg_types = list(self.compute_argument_types())
            module, ir_fn = self._create_instance_module(arg_types)
            session = self.create_session(module, ir_fn)
            entry = ir_fn.add_entry_block()
            with InsertionPoint(entry), activate_runtime(self), activate_session(session):
                self.initialize_session(session, entry)
                args = self.bind_entry_arguments(entry.arguments)
                self.trace_entry(*args)
                self.validate_trace_state()
                self.emit_return()
                self.finalize_session(session)
                session.validate_final_state()
            self.verify_module(module)
            return module

    def _create_instance_module(self, arg_types):
        module = Module.create()
        module.operation.attributes["pto.target_arch"] = StringAttr.get(self.descriptor.target)
        with InsertionPoint(module.body):
            fn_ty = func.FunctionType.get(arg_types, [])
            ir_fn = func.FuncOp(self.descriptor.name, fn_ty)
            ir_fn.attributes["pto.tilelang.instance"] = UnitAttr.get()
            ir_fn.attributes["pto.kernel_kind"] = Attribute.parse("#pto.kernel_kind<vector>")
        return module, ir_fn


def _is_tile_annotation(annotation) -> bool:
    if annotation is Tile:
        return True
    if isinstance(annotation, str):
        return annotation == "Tile" or annotation.endswith(".Tile")
    return getattr(annotation, "__name__", None) == "Tile"


__all__ = ["_TemplateTrace", "_TemplateTile"]
