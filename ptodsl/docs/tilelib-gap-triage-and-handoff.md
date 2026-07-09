# PTODSL TileLib Gap Triage And Backend Handoff

This note collects three views of the remaining PTODSL TileLib work on the
`mani/ptodsl` branch:

1. ST testcase triage for both smoke and non-smoke validation.
2. Remaining TileLangDSL template-version or mode coverage still missing in
   PTODSL.
3. The PTODSL daemon -> `InsertTemplateAttributes` -> `ExpandTileOp` handoff:
   what metadata is already flowing, what is missing, and what looks redundant.

Data sources used for this snapshot:

- `ptodsl/docs/tilelib-st-failures.md`
- `ptodsl/docs/tilelib-template-parity-gaps.md`
- `mani_log/full_nonsmoke_isolated_20260708-023856/summary.tsv`
- `mani_log/full_nonsmoke_isolated_20260708-023856/result.txt`
- current PTODSL TileLib metadata/backend code in
  `ptodsl/ptodsl/tilelib/*`,
  `lib/PTO/Transforms/InsertTemplateAttributes.cpp`, and
  `lib/PTO/Transforms/ExpandTileOp.cpp`

## Topline Counts

| Metric | Count | Notes |
|---|---:|---|
| Known current smoke failures | `8` | Based on the active smoke-blocker list below, after the reduction/arg-reduction smoke batch plus `tmrgsort` and `tsort32` were rechecked and passed on 2026-07-09. |
| Full non-smoke failures | `37 / 105` | Exact count from `mani_log/full_nonsmoke_isolated_20260708-023856`. This is the last exact isolated full snapshot and still includes failures from before the 2026-07-08 reduction-family smoke fix. |
| Tileops still tracked in the implementation / parity table | `30` | Unique tileops in Section 3. Most are version / mode gaps; this count also still includes the two row-arg semantic holdouts because they remain closely tied to the same planning bucket. |
 
## Snapshot Summary

| Area | Current read |
|---|---|
| Smoke ST | Known current blocker count is `8`. The reduction/arg-reduction smoke batch plus `tmrgsort` and `tsort32` now pass. |
| Non-smoke ST | Latest isolated full run is exact: `68 passed`, `37 failed`, `105 total`, but it predates the latest reduction-family smoke fix. |
| Gap shape | The remaining work is split across three buckets: wrong-output bugs, missing template versions/modes/dtypes, and backend metadata/selection gaps. |
| Best next wins | `tcmp`, `trandom`, `tinsert`, `tcvt`, `tlog`, `tsel`, and the high-precision math family. |

## Recent Updates

Recent branch-tip changes that are newer than the last exact isolated full
non-smoke rerun:

- The reduction/arg-reduction smoke batch now passes:
  `tcolmax`, `tcolmin`, `tcolprod`, `tcolargmax`, `tcolargmin`,
  `trowargmax`, `trowargmin`.
- `tmrgsort` and `tsort32` smoke now pass after aligning PTODSL callable
  forms with ST, forwarding `exhausted`, preserving template candidates
  through `PTOViewToMemref`, and making the PTODSL sort templates use
  runtime-safe scalar/control-flow constructs.
- Because of that, the smoke blocker list below no longer treats those nine as
  active smoke failures.
- The last exact full non-smoke number (`37 / 105`) is still useful for
  prioritization, but it should be read as a dated snapshot rather than current
  branch-tip truth for this family.

## 1. Smoke ST Triage

Important note: the smoke tracker has some stale rows because several focused
fixes landed later without a fresh all-cases rerun at branch tip. The table
below is still useful for prioritization, but the `confidence` column should be
read literally.

### 1.1 Likely Still-Active Smoke Blockers

| Testcase | Group | Failure type | Current symptom | Likely root cause | Suggested fix | Priority | Confidence |
|---|---|---|---|---|---|---|---|
| `tcmp` | compare / predicate pack | Wrong output | compare mismatch on 32-bit and 16-bit forms | PTODSL packing/store offsets do not match legacy `TCmp.hpp` behavior | Re-check 32-bit two-compare path, `pdintlv_b8`, predicate dist, and row/col store offsets against TileLang | P0 | High |
| `tcmps` | compare / scalar predicate | Build / no legal template | custom constraints not satisfied | Scalar compare operand shape/kind or predicate store layout does not match the legacy callable form | Revisit `tcmps` operand binding and then port legacy PK/NORM packing split | P0 | High |
| `tcolexpanddiv` | broadcast divide | Build / unsupported signature | no legal template for integer/non-default forms | PTODSL only covers a subset; legacy also has high-precision divide behavior | Add missing `i32` path and keep the later high-precision path on the same family | P1 | High |
| `tdiv` | divide | Build / unsupported signature | `('f16','f16','f16')` unsupported | PTODSL default implementation is much narrower than legacy | Add f16 default divide first; port high-precision later | P1 | High |
| `tdivs` | scalar divide | Build / constraint failure | custom constraints not satisfied | Scalar operand order / tmp / value typing still narrower than TileLang | Relax legality to accept both real ST scalar forms, then revisit high-precision parity | P1 | High |
| `textract` | movement / extract | Build / constraint failure | vec-to-vec constraint fails in smoke | PTODSL only covers the UB ND path and may still be too narrow even there | First fix the UB vec-to-vec legality; then add the cube/mat forms from legacy | P1 | High |
| `tlog` | unary math | Build / no candidate | `ExpandTileOp requires at least one template candidate` | Selection path is not finding a legal candidate; high-precision parity is also still missing | Confirm candidate insertion for the default path, then port `precisionType=high_precision` | P1 | High |
| `tsel` | select / mask | Build / constraint failure | custom constraints are not satisfied | PTODSL mask/data layout acceptance is narrower than legacy | Relax legality to the ST mask tile layout, then validate select mask unpacking | P1 | High |

### 1.2 Smoke Rows That Are Likely Stale But Still Point At Real Gaps

| Testcase | Why it is probably stale | Real gap still underneath | Suggested next step | Priority |
|---|---|---|---|---|
| `tcvt` | later focused rerun passed smoke | PTODSL still lacks a lot of the TileLang conversion matrix; non-smoke currently asks for `f32 -> f8e4m3` | Treat smoke as cleared for the old case; keep `tcvt` in the version-gap table | P1 |
| `textract_fp` | failure was recorded before later `tmatmul`/candidate fixes | pre-quant extract variants still need rerun and parity review | Rerun smoke and non-smoke after the next extract/fixpipe pass | P2 |
| `trowargmax` | smoke now passes after the shared row-arg rewrite on 2026-07-08 | non-smoke semantics still need confirmation on the full ST cases | Keep the smoke status as fixed; rerun non-smoke before calling full parity closed | P0 |
| `trowargmin` | same as `trowargmax` | same | same | P0 |
| `trowsum` | non-smoke now passes | legacy/ST used an i16 path; PTODSL likely gained enough for current ST, but full dtype parity still needs review | Move from smoke blocker to version-watch item only | P2 |
| `tsels` | non-smoke now passes | scalar/mask dtype matrix is still narrower than TileLang in principle | Keep only as a version-watch item unless new failures reappear | P2 |

## 2. Non-Smoke ST Triage

This section is based on the exact isolated full run:

- source: `mani_log/full_nonsmoke_isolated_20260708-023856/summary.tsv`
- result: `68 passed`, `37 failed`, `105 total`

### 2.1 Priority Groups

| Group | Meaning | Testcases | Priority |
|---|---|---|---|
| Wrong output | Template expands and runs, but semantics do not match golden | `tcmp`, `tfillpad`, `tpartmax`, `tpartmin`, `trandom`, `trowargmax`, `trowargmin` | P0 |
| Missing version / dtype / mode | No legal PTODSL template or wrong callable form | `tcmps`, `tcolargmax`, `tcolargmin`, `tcolexpanddiv`, `tcolmax`, `tcolmin`, `tcolprod`, `tcvt`, `tdiv`, `tdivs`, `textract`, `tinsert`, `tinsert_acc2vec`, `tload`, `tlrelu`, `tmov`, `tmrgsort`, `tsel`, `tsort32`, `txors` | P1 |
| Candidate / backend-sensitive cube or fixpipe path | `ExpandTileOp` sees no template candidates even though PTODSL has module coverage | `tgemv`, `tgemv_mx`, `tlog`, `tmatmul_acc`, `tmatmul_bias`, `tmatmul_bias_mx`, `tmatmul_mx`, `tmov2bias` | P1 |
| Probably not TileLib-first | verifier or surrounding PTO IR/testcase issue before TileLib parity becomes the first blocker | `tload_mat`, `tstore_acc2gm` | P2 |

### 2.2 Detailed Non-Smoke Table

| Testcase | Group | Failure type | Current symptom | Suggested fix | Expected change layer |
|---|---|---|---|---|---|
| `tcmp` | Wrong output | compare mismatch | packed predicate outputs differ badly from golden, often yielding `-128` or zero | Reconcile 32-bit and 16-bit pack/store behavior with legacy `TCmp.hpp`; verify `pdintlv_b8`, `PredicateDist`, and per-row byte offsets | TileLib implementation |
| `tcmps` | Missing mode | no legal template | custom constraints are not satisfied | Fix scalar compare operand legality first, then port legacy width-specific packing | TileLib implementation |
| `tcolargmax` | Missing dtype | no legal template | legacy/ST asks for unsigned index/data combinations not accepted by PTODSL | Add the missing unsigned signature family, then validate index writeback packing | TileLib implementation |
| `tcolargmin` | Missing dtype | no legal template | same shape of issue as `tcolargmax` | same | TileLib implementation |
| `tcolexpanddiv` | Missing dtype/version | no legal template | current run asks for unsupported dtype signature | Add missing integer/default variants; later add high-precision branch | TileLib implementation |
| `tcolmax` | Missing dtype | no legal template | ST asks for unsigned reductions not currently registered | Add unsigned column-extreme signatures | TileLib implementation |
| `tcolmin` | Missing dtype | no legal template | same family as `tcolmax` | same | TileLib implementation |
| `tcolprod` | Missing dtype | no legal template | ST asks for unsigned / different element width than PTODSL registers | Extend column-product dtype matrix to match legacy/ST | TileLib implementation |
| `tcvt` | Missing conversion version | no legal template | current non-smoke wants `('f32', 'f8e4m3')` | Port the missing fp8 conversion family and continue expanding toward the legacy matrix | TileLib implementation, maybe PTODSL surface/dtype support |
| `tdiv` | Missing dtype/version | no legal template | current run wants f16 divide | Add f16 default divide; keep high-precision work as the next step | TileLib implementation |
| `tdivs` | Missing callable form | no legal template | custom constraints are not satisfied | Match the legacy scalar-divide operand form(s), then enable high-precision parity later | TileLib implementation |
| `textract` | Missing mode / too-narrow constraint | no legal template | current vec-to-vec constraint still does not accept the ST form | Relax UB vec-to-vec legality first; then port mat->left/right same/cross fractal variants | TileLib implementation |
| `tfillpad` | Wrong output | compare mismatch | edge-pad lanes use fill extrema where valid tail data should survive | Re-check aligned-boundary fill / tail-restore order against legacy on full non-smoke cases | TileLib implementation |
| `tgemv` | Candidate/backend-sensitive | no candidate | `ExpandTileOp requires at least one template candidate` | Audit candidate attr insertion/survival for level3 cube GEMV; confirm op name and operand kinds still reach PTODSL unchanged | Backend first, then template review if needed |
| `tgemv_mx` | Candidate/backend-sensitive | no candidate | same failure shape as `tgemv` | same, but also validate MX operand signatures and scaling operands | Backend first, then template review |
| `tinsert` | Missing mode | no legal template | current ST form does not match the basic UB vec->vec PTODSL subset | Port the broader TileLang mode matrix, starting with the ST-emitted form | TileLib implementation plus new context attrs |
| `tinsert_acc2vec` | Missing mode | no legal template | acc->vec mode is not implemented in PTODSL | Port `acc_to_vec_mode` families from TileLang | TileLib implementation plus new context attrs |
| `tload` | Missing dtype | no legal template | current run asks for `f8e4m3` load/store coverage | Add fp8 load/store signatures if the underlying PTODSL types/ops support them cleanly | TileLib implementation, maybe PTODSL dtype support |
| `tload_mat` | Upstream verifier/testcase | PTO IR verifier failure | `pto.tmatmul` address-space verifier fails before TileLib | Fix the testcase/frontend/layout setup first; TileLib is not the first blocker here | PTO IR/testcase, not TileLib-first |
| `tlog` | Candidate/backend-sensitive | no candidate | no candidate survives by the time `ExpandTileOp` runs | First confirm candidate attr flow for default `tlog`; then port high-precision `precisionType` branch | Backend first, then TileLib implementation |
| `tlrelu` | Missing dtype | no legal template | current ST asks for a dtype/signature not registered by PTODSL | Extend `tlrelu` dtype matrix to match legacy/ST | TileLib implementation |
| `tmatmul_acc` | Candidate/backend-sensitive | no candidate | `ExpandTileOp` cannot find any surviving candidate | Investigate level3/cube candidate propagation for `.acc` variants specifically | Backend first, then template review |
| `tmatmul_bias` | Candidate/backend-sensitive | no candidate | same family as `tmatmul_acc` | same | Backend first, then template review |
| `tmatmul_bias_mx` | Candidate/backend-sensitive | no candidate | same family, MX flavor | same plus MX signature review | Backend first, then template review |
| `tmatmul_mx` | Candidate/backend-sensitive | no candidate | same family, MX flavor | same plus MX signature review | Backend first, then template review |
| `tmov` | Missing dtype | no legal template | current non-smoke asks for `ui8` movement forms not accepted today | Extend movement dtype matrix where the legacy path allows it | TileLib implementation |
| `tmov2bias` | Candidate/backend-sensitive | no candidate | build actually fails while instantiating `tmatmul.bias` downstream | Treat this as a downstream cube/bias candidate-propagation issue rather than a standalone `tmov2bias` template gap | Backend first |
| `tmrgsort` | Missing mode / attr | no legal template | wrong operand count and wrong `ex_vec` dtype/signature for PTODSL callable forms | Match ST operand forms and add the missing `exhausted` context attr forwarding | TileLib implementation plus `InsertTemplateAttributes` |
| `tpartmax` | Wrong output | compare mismatch | output uses `-inf` instead of the finite legacy pad minimum | Change pad constants to match legacy finite extrema, not mathematical infinities | TileLib implementation |
| `tpartmin` | Wrong output | compare mismatch | output uses `+inf` instead of the finite legacy pad maximum | same | TileLib implementation |
| `trandom` | Wrong output | compare mismatch | generated Philox sequence does not match golden | Reconcile round loop, counter increment cadence, and lane ordering against legacy | TileLib implementation |
| `trowargmax` | Wrong output | compare mismatch | many indices stay zero or wrong offsets are written | Rework row-arg index accumulation/writeback for `ui32` output and wide rows | TileLib implementation |
| `trowargmin` | Wrong output | compare mismatch | outputs look bit-pattern wrong for `ui32` and wide rows | same | TileLib implementation |
| `tsel` | Missing mode / too-narrow constraint | no legal template | custom constraints are not satisfied | Relax legality to the ST mask/data layout and verify predicate unpacking width by dtype | TileLib implementation |
| `tsort32` | Missing callable form | no legal template | aligned path wants the wrong operand form; tmp path rejects current dtype | Align PTODSL operand order and index dtype coverage with legacy/ST | TileLib implementation |
| `tstore_acc2gm` | Upstream verifier/testcase | PTO IR verifier failure | `pto.tmatmul` address-space verifier fails before TileLib, plus alloc-tile shape issues | Fix upstream testcase/frontend shape/layout first; TileLib is downstream | PTO IR/testcase, not TileLib-first |
| `txors` | Missing callable form | no legal template | custom constraints are not satisfied | Revisit scalar logical-op legality and signed/unsigned acceptance | TileLib implementation |

## 3. Remaining TileLangDSL Template Versions / Modes Still Missing In PTODSL

This table is about missing legacy coverage, not just current failing tests.
The counts below are grouped by implementation family rather than by individual
ST testcase.

| Group | Tileops | Missing legacy coverage | Why it matters | Priority | Effort | Expected layers |
|---|---|---|---|---|---|---|
| High-precision math family | `tdiv`, `tdivs`, `trecip`, `tcolexpanddiv`, `trowexpanddiv`, `tlog`, `texp`, `tsqrt` | `precisionType=high_precision` algorithms from `div_hp.py`, `exp_hp.py`, `sqrt_hp.py` | Needed for true TileLang parity even when default paths compile today | P1 | Medium-high | TileLib implementation; PTODSL surface wrappers only if helper ops are missing; `InsertTemplateAttributes` already forwards `precisionType` |
| Conversion matrix | `tcvt` | fp8/fp4/i64/more unsigned and mixed-width conversions, plus more version combinations | Current PTODSL covers only a small subset of TileLang `tcvt_template.py` | P1 | High | TileLib implementation, possibly PTODSL dtype/surface support; no new pass fields expected |
| Insert mode matrix | `tinsert` | acc->mat, acc->vec ND/DN/NZ, vec->vec NZ, vec->mat ND/NZ, pre-quant + relu variants | Large user-visible feature gap versus TileLang | P0 | High | TileLib implementation plus new context attrs (`acc_to_vec_mode`, `relu_pre_mode`) in `InsertTemplateAttributes` |
| Extract mode matrix | `textract`, `textract_fp` | mat->left/right same-fractal and cross-fractal forms, plus any still-missing fixpipe forms | Current PTODSL mostly covers only the UB vec->vec path and split fp path | P1 | Medium-high | Mostly TileLib implementation; current operand metadata/layout flow likely sufficient |
| Sort / merge callable forms | `tmrgsort`, `tsort32` | correct operand count/order, `ex_vec` dtype, tmp/no-tmp split, and `exhausted` attr handling | These are structural mismatches, not just missing dtypes | P0 | Medium | TileLib implementation plus `InsertTemplateAttributes` for `exhausted` |
| Row arg-reduction semantics | `trowargmax`, `trowargmin` | shared implementation now compiles and passes smoke again, but full ST semantic parity still needs rerun/confirmation | These are no longer missing-version issues; they are now remaining semantic-validation items | P0 | Low-medium | TileLib implementation |
| Reduction dtype matrix | `trowsum` | legacy/ST used an i16 form that should stay covered, even though the previous smoke blocker is now cleared | Mostly a parity-watch item after the 2026-07-08 reduction-family smoke fix | P2 | Low-medium | TileLib implementation |
| Default divide dtype coverage | `tdiv`, `tcolexpanddiv`, maybe `tdivs` forms | f16 default divide, integer default variants where legacy supports them | These block practical ST cases before high-precision work even starts | P1 | Medium | TileLib implementation |
| f16 remainder parity | `tfmod`, `tfmods`, `trem`, `trems` | TileLang widens f16 halves to f32 before remainder math; PTODSL does not | Likely numerical differences in full validation even if smoke passes | P2 | Medium | TileLib implementation |
| Movement/load dtype coverage | `tload`, `tmov`, possibly related move helpers | fp8 and unsigned movement forms present in ST/legacy but not all registered in PTODSL | Small but real catalog parity gap | P2 | Low-medium | TileLib implementation, maybe PTODSL dtype support |
| Cube / GEMV non-default variants | `tmatmul.acc`, `tmatmul.bias`, `tmatmul.mx*`, `tgemv`, `tgemv_mx` | candidate selection / non-default forms still not proven in PTODSL | Some failures may be backend-sensitive rather than missing template code, but the family is not yet closed | P1 | Medium-high | Backend first, then TileLib review |
| Multi-candidate elementwise parity | `tadd`, `tmul` | exact TileLang tail/version-selection behavior | Not the highest risk today, but still a correctness/selection parity item | P3 | Low-medium | TileLib implementation only |

### 3.1 Best Next Version-Work Order

| Order | Family | Why first |
|---|---|---|
| 1 | `tinsert` | biggest single feature gap and needs explicit attr plumbing decisions |
| 2 | `tmrgsort` / `tsort32` | blocking ST with structural callable-form mismatches |
| 3 | reduction / arg-reduction dtype matrix | good payoff, mostly TileLib-only |
| 4 | `tcvt` conversion matrix | broad real-world coverage gap |
| 5 | high-precision math family | large parity gap, but the existing `precisionType` plumbing is already in place |
| 6 | `textract` / `textract_fp` | fewer ops, but more mode-heavy |
| 7 | f16 remainder parity | less likely to block build, more likely to show up as numeric drift later |

## 4. PTODSL Daemon -> InsertTemplateAttributes -> ExpandTileOp Handoff

### 4.1 Current Flow

| Stage | What it does today |
|---|---|
| PTODSL template registration | `@tile_template(...)` records hard constraints and selection hints in `TemplateMetadata` |
| PTODSL daemon legality | daemon builds concrete operand specs and context attrs, selects legal candidates in Python, and returns candidate metadata JSON |
| `InsertTemplateAttributes` | rebuilds operand/context JSON from MLIR, asks the daemon for metadata, and stores only a compact `candidates` array attr on the TileOp |
| `ExpandTileOp` | rebuilds its own specialization key from operand types, reads the compact `candidates` attr, and expands using the first still-present candidate |

### 4.2 What PTODSL Already Sends

#### 4.2.1 Template Metadata Available In Python / Daemon

| Metadata field | Available in PTODSL metadata? | Persisted on IR by `InsertTemplateAttributes`? | Used today |
|---|---|---|---|
| `op`, `target`, `name` | Yes | only `name` | selection / diagnostics |
| `dtypes` | Yes | No | Python legality only |
| `layouts`, `memory_spaces` | Yes | No | Python legality only |
| `constraints` | Yes | No | Python legality only |
| `priority` | Yes | No | Python-side legal candidate ranking |
| `fusible` | Yes | No | currently unused by C++ passes |
| `loop_depth` | Yes | Yes | `ExpandTileOp` candidate metadata |
| `id` | Yes | Yes | stable candidate identity |
| `Tail` / `has_tail` | Yes | `has_tail` only | candidate metadata |
| `is_post_update` | Yes | Yes | candidate metadata |
| `iteration_axis`, `op_engine`, `op_class` | Yes | No | docs / analytics only today |
| `tags` | Yes | No | docs / analytics only today |

#### 4.2.2 Operand Specs Already Reconstructed By `InsertTemplateAttributes`

| Operand kind | Fields reconstructed today | Used by |
|---|---|---|
| tile | `dtype`, `shape`, `valid_shape`, `memory_space`, `b_layout`, `s_layout`, `s_fractal_size`, `pad_value` | PTODSL legality selection |
| view | `dtype`, `shape`, `strides`, `memory_space`, optional `layout` | PTODSL legality selection |
| vector | `dtype`, `shape` | PTODSL legality selection |
| scalar | `dtype`, optional constant integer `value` | PTODSL legality selection |

#### 4.2.3 Context Attrs Already Forwarded

| Attr | Used by tileops today |
|---|---|
| `round_mode` | `tcvt` |
| `rounds` | `trandom` |
| `cmp_mode` | `tcmp`, `tcmps` |
| `mask_pattern` | future / gather-side work |
| `precisionType` | `tdiv`, `tdivs`, `trecip`, `tcolexpanddiv`, `trowexpanddiv`, `tlog`, `texp`, `tsqrt` |

### 4.3 What Is Missing And What It Would Unblock

| Missing field | Where to add it | Why it is needed | Which tests it helps | Which tileops would use it |
|---|---|---|---|---|
| `exhausted` | `InsertTemplateAttributes` context-attr forwarding | TileLang `tmrgsort_template.py` reads it with `get_op_attr("exhausted", "0")`; PTODSL currently cannot see it | `tmrgsort` smoke and non-smoke | `tmrgsort` |
| `acc_to_vec_mode` | `InsertTemplateAttributes` context-attr forwarding | needed to port the TileLang `tinsert` acc->vec family instead of only the basic UB path | `tinsert_acc2vec`, future extended `tinsert` ST | `tinsert` |
| `relu_pre_mode` | `InsertTemplateAttributes` context-attr forwarding | needed for TileLang `tinsert` pre-relu / pre-quant mat insertion behavior | future extended `tinsert` ST and lit cases like `tinsert_a5_extended_modes` | `tinsert` |
| richer scalar literal capture for non-integers | operand-spec JSON builder | low priority, but would help if future legality depends on float scalar values rather than only dtype | no current blocker | any future scalar-mode legality |

### 4.4 What Looks Redundant Right Now

| Field | Why it is effectively redundant today | Keep / drop recommendation |
|---|---|---|
| daemon-returned `dtypes`, `layouts`, `memory_spaces`, `constraints` | useful for Python selection and debugging, but not needed on the IR once the legal candidate list is already computed | Keep in daemon responses; do not add to IR attrs unless debugging or audit tooling needs them |
| daemon-returned `priority` | candidate ranking already happens in Python before the compact IR attr is built | Keep in daemon metadata; no need to store on IR today |
| daemon-returned `fusible` | not consumed by current C++ passes | Keep in PTODSL metadata only until a pass actually wants it |
| daemon-returned `iteration_axis`, `op_engine`, `op_class`, `tags` | useful for docs, analytics, and future test filtering, but not needed by `ExpandTileOp` | Keep in PTODSL metadata only; add to IR attrs only if a pass or test harness starts using them |

### 4.5 Current Candidate Attr Is Intentionally Tiny

Today the IR-side `candidates` attr only stores:

- `id`
- `name`
- `loop_depth`
- `postupdate`
- `tail`

That is enough for the current expansion flow because:

1. legality already happened in Python;
2. `ExpandTileOp` only needs a stable candidate id/name plus a small amount of
   selection metadata; and
3. all operand details are reconstructed again from the current MLIR op when
   building the specialization key.

This compactness is a good default. The only additions that currently look
worthwhile are the missing context attrs above, not a large expansion of the IR
candidate payload.

## 5. Recommended Next Actions

| Order | Action | Why |
|---|---|---|
| 1 | Fix `tcmp` and `trandom` wrong-output cases | highest correctness risk |
| 2 | Add `exhausted` to context attrs and fix `tmrgsort` callable forms | small backend change, high payoff |
| 3 | Port the missing `tinsert` mode matrix and add `acc_to_vec_mode` / `relu_pre_mode` | biggest feature gap |
| 4 | Close the reduction/arg-reduction dtype gaps (`tcol*`, `trowarg*`, `trowsum`) | mostly TileLib-only wins |
| 5 | Expand `tcvt` and the default divide family | blocks broad non-smoke coverage |
| 6 | Revisit cube/fixpipe candidate loss (`tgemv*`, `tmatmul*.acc/bias/mx`, `tmov2bias`, `tlog`) | likely backend-sensitive and affects several families at once |
