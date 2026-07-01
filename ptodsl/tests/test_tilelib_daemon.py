# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""End-to-end tests for the PTODSL TileLib daemon's Unix-socket RPC."""

import os
import socket
import stat
import tempfile
import threading
import unittest

from ptodsl.tilelib.serving.client import DaemonClient, DaemonError
from ptodsl.tilelib.serving.daemon import (
    TileLibDaemonServer,
    _remove_socket_path,
)
from ptodsl.tilelib.serving.wire import MAX_MESSAGE_SIZE, recv_message


def _tile_spec(dtype="f32", shape=(8, 64)):
    return {
        "kind": "tile",
        "dtype": dtype,
        "shape": list(shape),
        "valid_shape": list(shape),
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }


# ExpandTileOp sends tadd as ins(src0, src1), outs(dst), matching the
# template parameter order (src0, src1, dst).
TADD_OPERANDS = [_tile_spec(), _tile_spec(), _tile_spec()]
TADD_2D_NO_POST_UPDATE = "template_tadd_2d_no_post_update"


class TileLibDaemonTest(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.socket_path = os.path.join(
            self._temporary_directory.name,
            "ptodsl_tilelib.sock",
        )
        self.server = TileLibDaemonServer(self.socket_path)
        self._thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self._thread.start()
        self.client = DaemonClient(self.socket_path)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self._thread.join()
        self._temporary_directory.cleanup()

    def test_ping(self):
        self.assertEqual(self.client.ping(), "pong")

    def test_socket_is_accessible_only_by_owner(self):
        mode = stat.S_IMODE(os.stat(self.socket_path).st_mode)
        self.assertEqual(mode, 0o600)

    def test_instantiate_named_candidate_returns_structured_mlir(self):
        mlir = self.client.instantiate(
            "a5",
            "pto.tadd",
            TADD_OPERANDS,
            candidate_id=TADD_2D_NO_POST_UPDATE,
        )
        self.assertIn(f"func.func @{TADD_2D_NO_POST_UPDATE}", mlir)
        for operation in (
            "pto.tile_buf_addr",
            "memref.subview",
            "pto.vlds",
            "pto.vadd",
            "pto.vsts",
            "pto.plt_b32",
            "pto.tilelang.instance",
        ):
            self.assertIn(operation, mlir)
        self.assertNotIn("pto.castptr", mlir)

    def test_instantiate_requires_candidate_when_top_priority_ties(self):
        with self.assertRaises(DaemonError):
            self.client.instantiate("a5", "pto.tadd", TADD_OPERANDS)

    def test_get_metadata_returns_legal_candidates(self):
        metadata = self.client.get_metadata("a5", "pto.tadd", TADD_OPERANDS)
        candidates = metadata["candidates"]
        self.assertEqual(
            set(candidates),
            {
                TADD_2D_NO_POST_UPDATE,
                "template_tadd_1d_no_post_update",
                "template_tadd_2d_post_update",
                "template_tadd_1d_post_update",
            },
        )

        selected = candidates[TADD_2D_NO_POST_UPDATE]
        self.assertEqual(selected["loop_depth"], 2)
        self.assertEqual(selected["Tail"], {"callable": "has_tail"})
        self.assertFalse(selected["has_tail"])
        self.assertFalse(selected["is_post_update"])
        self.assertEqual(selected["tags"], ["binop", "2d", "no_post_update"])

    def test_get_metadata_evaluates_tail_for_each_request(self):
        tail_operands = [
            _tile_spec(shape=(7, 65)),
            _tile_spec(shape=(7, 65)),
            _tile_spec(shape=(7, 65)),
        ]

        metadata = self.client.get_metadata("a5", "pto.tadd", tail_operands)

        self.assertTrue(
            metadata["candidates"][TADD_2D_NO_POST_UPDATE]["has_tail"]
        )

    def test_cache_stats_and_clear_are_available_over_rpc(self):
        arguments = (
            "a5",
            "pto.tadd",
            TADD_OPERANDS,
        )
        self.client.instantiate(
            *arguments,
            candidate_id=TADD_2D_NO_POST_UPDATE,
        )
        self.client.instantiate(
            *arguments,
            candidate_id=TADD_2D_NO_POST_UPDATE,
        )

        stats = self.client.get_stats()
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["entries"], 1)

        self.assertEqual(self.client.clear(), {"cleared": True})
        self.assertEqual(self.client.get_stats()["entries"], 0)

    def test_cache_key_includes_context_attributes(self):
        self.client.instantiate(
            "a5",
            "pto.tadd",
            TADD_OPERANDS,
            context_attrs={"variant": 0},
            candidate_id=TADD_2D_NO_POST_UPDATE,
        )
        self.client.instantiate(
            "a5",
            "pto.tadd",
            TADD_OPERANDS,
            context_attrs={"variant": 1},
            candidate_id=TADD_2D_NO_POST_UPDATE,
        )
        self.assertEqual(self.client.get_stats()["misses"], 2)

    def test_oversized_wire_message_is_rejected_before_payload_read(self):
        receiver, sender = socket.socketpair()
        self.addCleanup(receiver.close)
        self.addCleanup(sender.close)
        sender.sendall((MAX_MESSAGE_SIZE + 1).to_bytes(4, byteorder="big"))

        with self.assertRaisesRegex(ValueError, "exceeds limit"):
            recv_message(receiver)

    def test_socket_cleanup_removes_broken_symlink(self):
        missing_target = os.path.join(
            self._temporary_directory.name,
            "missing.sock",
        )
        broken_link = os.path.join(
            self._temporary_directory.name,
            "broken.sock",
        )
        os.symlink(missing_target, broken_link)

        _remove_socket_path(broken_link)

        self.assertFalse(os.path.lexists(broken_link))

    def test_non_tile_operand_is_rejected_explicitly(self):
        operands = list(TADD_OPERANDS)
        operands[0] = {"kind": "scalar", "dtype": "f32", "value": 1.0}

        with self.assertRaisesRegex(
            DaemonError,
            "currently supports only tile operands",
        ):
            self.client.instantiate(
                "a5",
                "pto.tadd",
                operands,
                candidate_id=TADD_2D_NO_POST_UPDATE,
            )

    def test_unknown_op_errors(self):
        with self.assertRaises(DaemonError):
            self.client.instantiate("a5", "pto.tnope", TADD_OPERANDS)


if __name__ == "__main__":
    unittest.main()
