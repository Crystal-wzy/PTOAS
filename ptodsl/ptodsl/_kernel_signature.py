# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Declarative PTODSL kernel-signature parsing and entry-ABI binding."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from ._diagnostics import (
    jit_illegal_formal_annotation_error,
    jit_missing_annotation_error,
)
from ._host_tensors import bind_host_tensor_argument, infer_jit_host_tensor_spec
from ._surface_values import wrap_surface_value
from ._surface_types import constexpr as _constexpr_marker
from ._types import _DType, _MaskDescriptor, _PtrDescriptor, _VRegDescriptor, _resolve


@dataclass(frozen=True)
class KernelSpecializationKey:
    kernel_identity: int
    abi_signature: tuple
    constexpr_signature: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class DeviceParameterSpec:
    name: str
    annotation: object

    def entry_arg_types(self):
        return (_resolve(self.annotation),)

    def bind_entry_arguments(self, entry_arguments):
        if not entry_arguments:
            raise RuntimeError(f"entry ABI for device parameter '{self.name}' is incomplete")
        return wrap_surface_value(entry_arguments[0]), entry_arguments[1:]

    def abi_signature(self):
        return ("device", self.name, _hashable_signature_atom(self.annotation))


@dataclass(frozen=True)
class RuntimeScalarParameterSpec:
    name: str
    annotation: object

    def entry_arg_types(self):
        return (_resolve(self.annotation),)

    def bind_entry_arguments(self, entry_arguments):
        if not entry_arguments:
            raise RuntimeError(f"entry ABI for runtime scalar parameter '{self.name}' is incomplete")
        return wrap_surface_value(entry_arguments[0]), entry_arguments[1:]

    def abi_signature(self):
        return ("scalar", self.name, _hashable_signature_atom(self.annotation))


@dataclass(frozen=True)
class TensorSpecParameterSpec:
    name: str
    tensor_spec: object

    def entry_arg_types(self):
        return tuple(self.tensor_spec.entry_arg_types())

    def bind_entry_arguments(self, entry_arguments):
        return bind_host_tensor_argument(self.name, self.tensor_spec, entry_arguments)

    def abi_signature(self):
        return ("tensor", self.name, self.tensor_spec.abi_signature())


@dataclass(frozen=True)
class ConstexprParameterSpec:
    name: str
    default: object

    def bind_specialization(self, provided_bindings):
        value = provided_bindings.get(self.name, self.default)
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(
                f"@pto.jit constexpr parameter '{self.name}' must be hashable so it can "
                "participate in the specialization cache"
            ) from exc
        return value


def _hashable_signature_atom(value):
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _is_supported_runtime_scalar_annotation(annotation) -> bool:
    return (
        isinstance(annotation, _DType)
        and not isinstance(annotation, (_PtrDescriptor, _VRegDescriptor, _MaskDescriptor))
    )


@dataclass(frozen=True)
class KernelSignature:
    positional_parameters: tuple
    constexpr_parameters: tuple[ConstexprParameterSpec, ...]

    def compute_entry_arg_types(self):
        arg_types = []
        for param in self.positional_parameters:
            arg_types.extend(param.entry_arg_types())
        return tuple(arg_types)

    def bind_entry_arguments(self, entry_arguments):
        remaining = tuple(entry_arguments)
        bound_args = []
        for param in self.positional_parameters:
            bound_value, remaining = param.bind_entry_arguments(remaining)
            bound_args.append(bound_value)
        if remaining:
            raise RuntimeError(f"unexpected trailing entry arguments in PTODSL kernel ABI: {len(remaining)}")
        return tuple(bound_args)

    def default_constexpr_bindings(self):
        return {param.name: param.default for param in self.constexpr_parameters}

    def bind_constexpr_bindings(self, provided_bindings):
        provided = dict(provided_bindings)
        expected_names = {param.name for param in self.constexpr_parameters}
        unknown = sorted(name for name in provided if name not in expected_names)
        if unknown:
            raise TypeError(
                f"unknown @pto.jit constexpr parameter(s): {', '.join(unknown)}"
            )

        bound = {}
        for param in self.constexpr_parameters:
            bound[param.name] = param.bind_specialization(provided)
        return bound

    def abi_signature(self):
        return tuple(param.abi_signature() for param in self.positional_parameters)

    def specialization_key(self, kernel_identity, constexpr_bindings):
        return KernelSpecializationKey(
            kernel_identity=kernel_identity,
            abi_signature=self.abi_signature(),
            constexpr_signature=tuple(
                (param.name, constexpr_bindings[param.name])
                for param in self.constexpr_parameters
            ),
        )


def parse_jit_kernel_signature(py_fn) -> KernelSignature:
    """Parse one authored ``@pto.jit`` function signature."""
    sig = inspect.signature(py_fn)
    positional_parameters = []
    constexpr_parameters = []

    for param in sig.parameters.values():
        if param.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            if param.annotation is inspect.Parameter.empty:
                raise jit_missing_annotation_error(param.name)
            host_tensor_spec = infer_jit_host_tensor_spec(param)
            if host_tensor_spec is not None:
                positional_parameters.append(
                    TensorSpecParameterSpec(param.name, host_tensor_spec)
                )
            elif _is_supported_runtime_scalar_annotation(param.annotation):
                positional_parameters.append(
                    RuntimeScalarParameterSpec(param.name, param.annotation)
                )
            else:
                raise jit_illegal_formal_annotation_error(param.name, param.annotation)
            continue

        if param.kind is inspect.Parameter.KEYWORD_ONLY:
            if param.annotation is not _constexpr_marker:
                raise TypeError(
                    f"@pto.jit keyword-only parameter '{param.name}' must be annotated "
                    "with pto.constexpr in PTODSL v1"
                )
            if param.default is inspect.Parameter.empty:
                raise TypeError(
                    f"@pto.jit constexpr parameter '{param.name}' must declare a default "
                    "value until explicit compile-time specialization is implemented"
                )
            constexpr_parameters.append(ConstexprParameterSpec(param.name, param.default))
            continue

        raise TypeError(
            f"@pto.jit parameter '{param.name}' uses unsupported parameter kind "
            f"{param.kind!r}"
        )

    return KernelSignature(
        positional_parameters=tuple(positional_parameters),
        constexpr_parameters=tuple(constexpr_parameters),
    )


__all__ = [
    "ConstexprParameterSpec",
    "DeviceParameterSpec",
    "KernelSpecializationKey",
    "KernelSignature",
    "RuntimeScalarParameterSpec",
    "TensorSpecParameterSpec",
    "parse_jit_kernel_signature",
]
