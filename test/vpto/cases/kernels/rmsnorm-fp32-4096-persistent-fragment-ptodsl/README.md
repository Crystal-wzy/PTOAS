# RMSNorm Persistent SIMT Fragment PTODSL Case

This directory stores a TileLang-generated PTODSL source variant used to
develop and validate persistent SIMT fragment materialization in PTOAS.

## Files

- `kernel.py`: RMSNorm fp32 4096x4096 PTODSL source adapted from a
  TileLang PTO codegen dump. The weight fragment is allocated outside the
  per-token SIMT section, initialized once from UB, and reused by subsequent
  SIMT sections.

## Purpose

The case is a compile fixture for the optimization described in
`docs/designs/ptoas_persistent_simt_fragment_plan.md`. Current PTODSL marks the
SIMT-external fragment allocation with `pto.persistent` and preserves inline
SIMT sections. PTOAS materializes `pto.keep` / `pto.resume`, outlines the
sections, and compiles the resulting module through the VPTO pipeline.

The kernel sets `ast_rewrite=False` for compatibility with the explicit PTODSL
source, but uses `pto.for_` for the token loop. The loop therefore remains an
`scf.for` in the pass input instead of being copied 64 times: one persistent
top-level `llvm.alloca`, one init section, and one inline carry section in the
loop body. The directory has no host launcher or reference check, so it remains
a compile-only fixture rather than a runnable kernel case.

## Source

The source was derived from:

`~/.tilelang/cache/0.1.11_cpu_gitdd4f9aa0-aarch64/kernels/ab0771e55dcd836a25047f82c5e906a2b772be48c86efd19bd47e7c228b15b54/device_kernel.cu`

and manually adapted into the target persistent-fragment shape.
