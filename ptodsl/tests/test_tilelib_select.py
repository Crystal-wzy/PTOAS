# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TileLib selection test: metadata-driven legality and descriptor metadata."""

import unittest
from types import SimpleNamespace

from ptodsl.tilelib import (
    AmbiguousTemplate,
    ScalarType,
    TemplateMetadata,
    TileSpec,
    TileTemplateRegistry,
    legal_candidates,
    select,
)
from ptodsl.tilelib import constraints as _constraints
from ptodsl.tilelib.registry import NoMatchingTemplate


def _f32_specs():
    spec = TileSpec(shape=(8, 64), dtype=ScalarType("f32"))
    return {"src0": spec, "src1": spec, "dst": spec}


def _f32_single_row_specs():
    spec = TileSpec(shape=(1, 64), dtype=ScalarType("f32"))
    return {"src0": spec, "src1": spec, "dst": spec}


class TileLibSelectTest(unittest.TestCase):
    def test_context_attributes_are_available_to_constraints(self):
        registry = TileTemplateRegistry()
        registry.register(SimpleNamespace(
            op="pto.test_context",
            target="a5",
            name="context_candidate",
            param_names=("src0", "src1", "dst"),
            metadata=TemplateMetadata.build(
                op="pto.test_context",
                target="a5",
                name="context_candidate",
                constraints=(lambda mode: mode == "enabled",),
            ),
        ))

        legal = registry.legal_candidates(
            "pto.test_context",
            "a5",
            _f32_specs(),
            context_attrs={"mode": "enabled"},
        )
        self.assertEqual([candidate.name for candidate in legal], ["context_candidate"])

        with self.assertRaises(NoMatchingTemplate):
            registry.legal_candidates(
                "pto.test_context",
                "a5",
                _f32_specs(),
                context_attrs={"mode": "disabled"},
            )

    def test_four_tadd_versions_registered(self):
        names = {
            candidate.name
            for candidate in legal_candidates("pto.tadd", "a5", _f32_specs())
        }
        self.assertEqual({
            "template_tadd_2d_no_post_update",
            "template_tadd_1d_no_post_update",
            "template_tadd_2d_post_update",
            "template_tadd_1d_post_update",
        }, names)

    def test_plain_tadd_select_is_ambiguous(self):
        with self.assertRaises(AmbiguousTemplate):
            select("pto.tadd", "a5", _f32_specs())

    def test_named_tadd_selects_2d_no_post_update_template(self):
        chosen = select(
            "pto.tadd",
            "a5",
            _f32_specs(),
            candidate_id="template_tadd_2d_no_post_update",
        )
        self.assertEqual(chosen.name, "template_tadd_2d_no_post_update")
        self.assertFalse(chosen.metadata.is_post_update)
        self.assertEqual(chosen.metadata.loop_depth, 2)
        self.assertTrue(callable(chosen.metadata.Tail))
        self.assertEqual(chosen.metadata.tags, ("binop", "2d", "no_post_update"))

    def test_single_row_tadd_candidates_are_still_all_visible(self):
        candidates = legal_candidates("pto.tadd", "a5", _f32_single_row_specs())
        self.assertEqual(len(candidates), 4)

    def test_legal_candidates_include_loop_depth_metadata(self):
        candidates = legal_candidates("pto.tadd", "a5", _f32_specs())
        by_name = {candidate.name: candidate for candidate in candidates}
        self.assertEqual(set(by_name), {
            "template_tadd_2d_no_post_update",
            "template_tadd_1d_no_post_update",
            "template_tadd_2d_post_update",
            "template_tadd_1d_post_update",
        })
        self.assertEqual(by_name["template_tadd_2d_no_post_update"].metadata.loop_depth, 2)
        self.assertFalse(by_name["template_tadd_2d_no_post_update"].metadata.is_post_update)
        self.assertTrue(callable(by_name["template_tadd_2d_no_post_update"].metadata.Tail))
        self.assertEqual(by_name["template_tadd_1d_no_post_update"].metadata.loop_depth, 1)
        self.assertTrue(callable(by_name["template_tadd_1d_no_post_update"].metadata.Tail))
        self.assertTrue(by_name["template_tadd_2d_post_update"].metadata.is_post_update)
        self.assertTrue(callable(by_name["template_tadd_2d_post_update"].metadata.Tail))
        self.assertTrue(by_name["template_tadd_1d_post_update"].metadata.is_post_update)
        self.assertTrue(callable(by_name["template_tadd_1d_post_update"].metadata.Tail))
        context = _constraints.build_context(_f32_specs(), "a5", "pto.tadd")
        self.assertFalse(by_name["template_tadd_1d_post_update"].metadata.Tail(**context))

    def test_can_select_named_legal_candidate(self):
        chosen = select(
            "pto.tadd",
            "a5",
            _f32_specs(),
            candidate_id="template_tadd_2d_no_post_update",
        )
        self.assertEqual(chosen.name, "template_tadd_2d_no_post_update")

    def test_no_matching_dtype_raises(self):
        spec = TileSpec(shape=(8, 64), dtype=ScalarType("i8"))
        with self.assertRaises(NoMatchingTemplate):
            select("pto.tadd", "a5", {"src0": spec, "src1": spec, "dst": spec})

    def test_unknown_op_raises(self):
        with self.assertRaises(NoMatchingTemplate):
            select("pto.tnope", "a5", _f32_specs())


if __name__ == "__main__":
    unittest.main()
