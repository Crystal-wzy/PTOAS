# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib daemon for the ExpandTileOp Unix-socket RPC contract.

The daemon owns template discovery, selection, specialization, rendering, and an
in-memory instance cache. PTODSL templates are loaded from the Python package, so
the daemon does not scan or depend on an external template directory.

Run it with:

    python3 -m ptodsl.tilelib.serving.daemon --socket <path>
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socketserver
import threading

from .. import registry as _registry
from ..metadata import ScalarType, TileSpec
from ..templates import load_template
from .wire import recv_message, send_message


def _build_tile_specs(descriptor, operand_specs: list) -> dict:
    """Map positional daemon operands onto a template's parameter names."""
    if not isinstance(operand_specs, list):
        raise TypeError("operand_specs must be a list")
    if len(operand_specs) != len(descriptor.param_names):
        raise ValueError(
            f"template {descriptor.name!r} expects {len(descriptor.param_names)} "
            f"operands, got {len(operand_specs)}"
        )

    tile_specs = {}
    for index, (name, spec) in enumerate(zip(descriptor.param_names, operand_specs)):
        if not isinstance(spec, dict):
            raise TypeError(f"operand_specs[{index}] must be an object")

        kind = spec.get("kind")
        if kind != "tile":
            raise NotImplementedError(
                "PTODSL TileLib daemon currently supports only tile operands; "
                f"operand {index} ({name!r}) has kind {kind!r}"
            )

        config = spec.get("config") or {}
        if not isinstance(config, dict):
            raise TypeError(f"operand_specs[{index}].config must be an object")

        try:
            shape = tuple(spec["shape"])
            dtype = ScalarType(spec["dtype"])
        except KeyError as exc:
            raise ValueError(
                f"tile operand {index} ({name!r}) is missing {exc.args[0]!r}"
            ) from exc

        valid_shape = spec.get("valid_shape")
        tile_specs[name] = TileSpec(
            shape=shape,
            dtype=dtype,
            memory_space=spec.get("memory_space", "ub"),
            valid_shape=tuple(valid_shape) if valid_shape is not None else None,
            b_layout=config.get("b_layout", "row_major"),
            s_layout=config.get("s_layout", "none_box"),
        )
    return tile_specs


def _constraint_name(predicate) -> str:
    return getattr(predicate, "__name__", repr(predicate))


def _metadata_value(value):
    if callable(value):
        return {"callable": _constraint_name(value)}
    return value


def _metadata_for_descriptor(descriptor) -> dict:
    metadata = descriptor.metadata
    return {
        "op": metadata.op,
        "target": metadata.target,
        "name": metadata.name,
        "dtypes": [list(signature) for signature in metadata.dtypes],
        "layouts": list(metadata.layouts),
        "memory_spaces": list(metadata.memory_spaces),
        "constraints": [
            _constraint_name(predicate) for predicate in metadata.constraints
        ],
        "priority": metadata.priority,
        "fusible": metadata.fusible,
        "loop_depth": metadata.loop_depth,
        "id": metadata.id,
        "Tail": _metadata_value(metadata.Tail),
        "is_post_update": metadata.is_post_update,
        "tags": list(metadata.tags),
    }


def _tile_specs_for_request(target: str, op: str, operand_specs: list) -> dict:
    # Import only this op's template module. Registration happens as an import
    # side effect and repeated requests are no-ops because the loader is cached.
    load_template(op, target)
    candidates = _registry.default_registry().lookup(op, target)
    if not candidates:
        raise _registry.NoMatchingTemplate(
            f"no template registered for op={op!r} target={target!r}"
        )

    # All versions of an op share one parameter order. Any candidate can map the
    # positional wire operands before legality filtering chooses a version.
    return _build_tile_specs(candidates[0], operand_specs)


def metadata_request(
    target: str,
    op: str,
    operand_specs: list,
    context_attrs: dict | None = None,
) -> dict:
    """Return every legal candidate and its selection metadata."""
    tile_specs = _tile_specs_for_request(target, op, operand_specs)
    legal = _registry.legal_candidates(op, target, tile_specs, context_attrs)
    return {
        "target": target,
        "op": op,
        "candidates": {
            descriptor.name: _metadata_for_descriptor(descriptor)
            for descriptor in legal
        },
    }


def render_request(
    target: str,
    op: str,
    operand_specs: list,
    context_attrs: dict | None = None,
    candidate_id: str | None = None,
) -> str:
    """Select and render one PTODSL template as MLIR text."""
    tile_specs = _tile_specs_for_request(target, op, operand_specs)
    descriptor = _registry.select(
        op,
        target,
        tile_specs,
        context_attrs,
        candidate_id,
    )
    return descriptor.specialize(**tile_specs).mlir_text()


class TileLibDaemonServer(socketserver.ThreadingUnixStreamServer):
    """Threaded Unix-socket RPC server with an in-memory render cache."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, socket_path: str, max_entries: int = 1000):
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than zero")
        super().__init__(socket_path, _Handler)
        self._cache: dict[str, str] = {}
        self._state_lock = threading.Lock()
        self._max_entries = max_entries
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    @property
    def stats(self) -> dict:
        """Return a snapshot of cache counters for diagnostics and tests."""
        with self._state_lock:
            return dict(self._stats)

    def dispatch(self, request: dict) -> dict:
        if not isinstance(request, dict):
            return {"success": False, "error": "request must be a JSON object"}

        method = request.get("method")
        params = request.get("params") or {}
        if not isinstance(params, dict):
            return {"success": False, "error": "request params must be a JSON object"}

        try:
            if method == "instantiate":
                result = self._instantiate(**params)
            elif method == "get_metadata":
                result = self._get_metadata(**params)
            elif method == "ping":
                result = "pong"
            elif method == "get_stats":
                result = self._get_stats()
            elif method == "clear":
                result = self._clear()
            else:
                return {"success": False, "error": f"unknown method {method!r}"}
            return {"success": True, "result": result}
        except Exception as exc:
            return {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _get_metadata(self, target, op, operand_specs, context_attrs=None):
        return metadata_request(target, op, operand_specs, context_attrs)

    def _get_stats(self):
        with self._state_lock:
            requests = self._stats["hits"] + self._stats["misses"]
            total_entries = len(self._cache)
            return {
                **self._stats,
                "entries": total_entries,
                "total_entries": total_entries,
                "max_entries": self._max_entries,
                "hit_rate": self._stats["hits"] / requests if requests else 0.0,
            }

    def _clear(self):
        with self._state_lock:
            self._cache.clear()
        return {"cleared": True}

    def _instantiate(
        self,
        target,
        op,
        operand_specs,
        context_attrs=None,
        candidate_id=None,
    ):
        key = json.dumps(
            {
                "target": target,
                "op": op,
                "operand_specs": operand_specs,
                "context_attrs": context_attrs,
                "candidate_id": candidate_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

        with self._state_lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._stats["hits"] += 1
                return cached
            self._stats["misses"] += 1

        mlir_text = render_request(
            target,
            op,
            operand_specs,
            context_attrs,
            candidate_id,
        )

        with self._state_lock:
            if len(self._cache) >= self._max_entries:
                self._cache.pop(next(iter(self._cache)))
                self._stats["evictions"] += 1
            self._cache[key] = mlir_text
        return mlir_text


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            request = recv_message(self.request)
        except (ConnectionError, UnicodeDecodeError, ValueError):
            return
        send_message(self.request, self.server.dispatch(request))


def _parse_args(argv):
    parser = argparse.ArgumentParser(prog="ptodsl.tilelib.serving.daemon")
    parser.add_argument("--socket", required=True)
    parser.add_argument(
        "--template-dir",
        default=None,
        help="accepted during migration but ignored; PTODSL templates are in-package",
    )
    parser.add_argument("--max-entries", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if os.path.exists(args.socket):
        os.unlink(args.socket)

    server = TileLibDaemonServer(args.socket, max_entries=args.max_entries)
    stop = threading.Event()

    def _request_shutdown(*_):
        stop.set()

    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    if args.verbose:
        print(f"PTODSL TileLib daemon listening on {args.socket}", flush=True)

    try:
        stop.wait()
    finally:
        server.shutdown()
        server.server_close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)


if __name__ == "__main__":
    main()


__all__ = ["TileLibDaemonServer", "main", "metadata_request", "render_request"]
