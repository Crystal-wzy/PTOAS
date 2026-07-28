# PTOAS Memplan 对比记录

这份说明记录 `pypto-lib/models/**/*.py` 模型用例在不同 PTOAS memplan 配置下的结果。旧版历史测试已删除，此前 latest `main` 的全量结果作为新的 Round 1；此前 latest `feature_memplan` 全量结果保留为 Round 2；本次新增最新 `feature_memplan` 二进制对 Round 1 PASS case 的定向复测结果作为 Round 3。

## 测试轮次

| 轮次 | PTOAS 版本 / 分支 | memplan 来源 | `pto-level` | 测试范围 | 结果 |
|---|---|---|---|---|---|
| 1 | latest `main` commit `625f3fdd` / `ptoas 0.52` | PTOAS memplan | `level2` | `pypto-lib/models/**/*.py` 全量，排除 `build_output` | `15 pass / 35 fail / 0 timeout` |
| 2 | older `feature_memplan` local build / `ptoas 0.52` | PTOAS modern memplan (`--plan-memory-impl=modern`) | `level2` | `pypto-lib/models/**/*.py` 全量，排除 `build_output` | `7 pass / 43 fail / 0 timeout` |
| 3 | latest `feature_memplan` local build / `ptoas 0.54` | PTOAS modern memplan (`--plan-memory-impl=modern`) | `level2` | 仅复测 Round 1 PASS 的 15 个 case | `14 pass / 1 fail / 0 timeout` |

Round 3 运行信息：

- 远端机器：`101.245.68.6`
- PTOAS 二进制：`/home/zhongxuan/fangrui/PTOAS-feature-memplan-latest-round2/build-round2/tools/ptoas/ptoas`
- pypto-lib：`/home/zhongxuan/fangrui/pypto-lib`
- 结果目录：`/home/zhongxuan/fangrui/model_sweep_feature_memplan_latest_round2_15/20260728_003234`
- 补充复测目录：`/home/zhongxuan/fangrui/selected_modern_level2_20260728_014715`

## 跨轮次通过矩阵

说明：

- `PASS`：该轮通过
- `FAIL`：该轮失败
- `NOT RUN`：该轮未运行
- Round 1 / Round 2 表格列出全部 50 个 case；Round 3 仅更新 Round 1 中通过的 15 个 case。

| Case | Round 1 | Round 2 | Round 3 |
|---|---|---|---|
| `models/deepseek/v3_2/deepseek_v3_2_decode_back.py` | PASS | PASS | PASS |
| `models/deepseek/v3_2/deepseek_v3_2_decode_front.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v3_2/deepseek_v3_2_prefill_back.py` | PASS | PASS | PASS |
| `models/deepseek/v4/decode_attention_csa.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/decode_attention_hca.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/decode_attention_swa.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/decode_compressor_ratio128.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/decode_compressor_ratio4.py` | PASS | FAIL | PASS |
| `models/deepseek/v4/decode_fwd.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/decode_indexer.py` | PASS | FAIL | PASS |
| `models/deepseek/v4/decode_indexer_compressor.py` | PASS | FAIL | PASS |
| `models/deepseek/v4/decode_layer.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/decode_mtp.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/decode_sparse_attn.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/decode_sparse_attn_hca.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/decode_sparse_attn_swa.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/expert_routed.py` | PASS | PASS | PASS |
| `models/deepseek/v4/expert_shared.py` | PASS | FAIL | PASS |
| `models/deepseek/v4/gate.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/hc_head.py` | PASS | PASS | PASS |
| `models/deepseek/v4/hc_post.py` | PASS | PASS | PASS |
| `models/deepseek/v4/hc_pre.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/lm_head.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/moe.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/mtp_projection.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/prefill_attention_csa.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/prefill_attention_hca.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/prefill_attention_swa.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/prefill_compressor_ratio128.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/prefill_compressor_ratio4.py` | PASS | FAIL | PASS |
| `models/deepseek/v4/prefill_fwd.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/prefill_indexer.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/prefill_indexer_compressor.py` | PASS | FAIL | PASS |
| `models/deepseek/v4/prefill_layer.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/prefill_mtp.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/prefill_sparse_attn.py` | FAIL | FAIL | NOT RUN |
| `models/deepseek/v4/qkv_proj_rope.py` | PASS | FAIL | PASS |
| `models/deepseek/v4/rmsnorm.py` | PASS | PASS | PASS |
| `models/qwen3/14b/decode_fwd.py` | FAIL | FAIL | NOT RUN |
| `models/qwen3/14b/decode_layer_a8w8.py` | FAIL | FAIL | NOT RUN |
| `models/qwen3/14b/greedy_sample.py` | FAIL | FAIL | NOT RUN |
| `models/qwen3/14b/prefill_fwd.py` | FAIL | FAIL | NOT RUN |
| `models/qwen3/14b/qwen3_14b_decode_ssn_draft.py` | FAIL | FAIL | NOT RUN |
| `models/qwen3/14b/qwen3_14b_decode_tq_draft.py` | FAIL | FAIL | NOT RUN |
| `models/qwen3/14b/qwen3_14b_prefill_tq_draft.py` | FAIL | FAIL | NOT RUN |
| `models/qwen3/14b/test_paged_attention_cce.py` | PASS | FAIL | FAIL |
| `models/qwen3/14b/topk_select.py` | FAIL | FAIL | NOT RUN |
| `models/qwen3/32b/qwen3_32b_decode.py` | PASS | PASS | PASS |
| `models/qwen3/32b/qwen3_32b_decode_4d.py` | FAIL | FAIL | NOT RUN |
| `models/qwen3/32b/qwen3_32b_prefill_draft.py` | FAIL | FAIL | NOT RUN |

## Round 2 失败摘要

latest main pass 但 older `feature_memplan` + `--plan-memory-impl=modern` fail 的 case 共 8 个。

| Case | 错误摘要 | 类型 |
|---|---|---|
| `models/deepseek/v4/decode_compressor_ratio4.py` | `rmsnorm_rope_cache_write.pto:40:3: error: 'pto.tmov' op expects A2/A3 non-mat tmov to use matching src/dst shapes` | 编译失败，`tmov` shape 校验 |
| `models/deepseek/v4/decode_indexer.py` | `qr_rope.pto:77:5`、`rmsnorm_rope.pto:30:3` 都报 `pto.tmov ... matching src/dst shapes` | 编译失败，`tmov` shape 校验 |
| `models/deepseek/v4/decode_indexer_compressor.py` | `rmsnorm_rope.pto:30:3: error: 'pto.tmov' op expects A2/A3 non-mat tmov to use matching src/dst shapes` | 编译失败，`tmov` shape 校验 |
| `models/deepseek/v4/expert_shared.py` | `sh_gate_up_act_q.pto:2:3: error: vec overflow, requires 1605632 bits while 1572864 bits available` | 编译失败，UB 容量溢出 |
| `models/deepseek/v4/prefill_compressor_ratio4.py` | `prefill_c4_rmsnorm_rope.pto:52:7`、`prefill_c4_softmax_pool.pto:54:7` 都报 `pto.tmov ... matching src/dst shapes` | 编译失败，`tmov` shape 校验 |
| `models/deepseek/v4/prefill_indexer_compressor.py` | `prefill_idx_c4_rmsnorm_rope.pto:50:7`、`prefill_idx_c4_softmax_pool.pto:51:7` 都报 `pto.tmov ... matching src/dst shapes` | 编译失败，`tmov` shape 校验 |
| `models/deepseek/v4/qkv_proj_rope.py` | `q_rope_prepare.pto:70:5`、`qproj_dequant_rms_nope_rope.pto:97:9`、`kv_rms_norm_rope.pto:175:5` 都报 `pto.tmov ... matching src/dst shapes` | 编译失败，`tmov` shape 校验 |
| `models/qwen3/14b/test_paged_attention_cce.py` | `aclrtSynchronizeStreamWithTimeout (AICPU) failed: 507018`，随后 `PTO2 runtime failed` / `run_prepared failed with code 507018` | 运行期 AICore/AICPU fatal |

Round 2 错误原因总结：

- 主要回退是 `pto.tmov` 的 A2/A3 non-mat shape 约束：8 个回退 case 中 6 个都是这个错误。
- `expert_shared.py` 是 modern memplan 规划后 UB 需求超过 vec 容量，说明该 case 当时 modern planner 的复用/生命周期压缩能力不够，或者新增 tile/tmp/root 让容量压力变大。
- `test_paged_attention_cce.py` 不是编译失败，而是运行期设备错误，日志显示 stream sync 超时/运行时 fatal。

## Round 3 失败摘要

latest main pass 但 latest `feature_memplan` + `--plan-memory-impl=modern` fail 的 case 共 1 个。

| Case | 错误摘要 | 类型 |
|---|---|---|
| `models/qwen3/14b/test_paged_attention_cce.py` | `aclrtSynchronizeStreamWithTimeout (AICPU) failed: 507018`，随后 `PTO2 runtime failed` / `runtime_status=-100` | 运行期 AICore/AICPU fatal |

Round 3 相比 Round 2 的变化：

- 之前因 `pto.tmov` shape verifier 失败的 `decode_compressor_ratio4.py`、`decode_indexer.py`、`decode_indexer_compressor.py`、`prefill_compressor_ratio4.py`、`qkv_proj_rope.py` 在最新 `feature_memplan` 上已通过。
- 之前因 modern memplan vec overflow 失败的 `expert_shared.py` 在最新 `feature_memplan` 上已通过。
- `deepseek_v3_2_prefill_back.py`、`hc_post.py`、`prefill_indexer_compressor.py`、`qwen3_32b_decode.py` 经补充复测已通过。
- `test_paged_attention_cce.py` 仍是运行期设备 fatal，不是 PTOAS 编译期失败。
- 当前 Round 3 仅剩 `test_paged_attention_cce.py` 的运行期设备 fatal。

## Round 1 失败原因归类

latest `main` 全量测试中的 35 个失败 case 可粗略归类如下。

| 错误类型 | 数量 | 涉及 case |
|---|---:|---|
| PTOAS memplan/资源检查：UB/pipe 容量溢出 | 10 | `decode_attention_hca.py`、`decode_attention_swa.py`、`decode_compressor_ratio128.py`、`decode_sparse_attn_hca.py`、`decode_sparse_attn_swa.py`、`prefill_attention_csa.py`、`prefill_attention_hca.py`、`prefill_attention_swa.py`、`prefill_sparse_attn.py`、`qwen3/14b/prefill_fwd.py` |
| 测试环境设备数不足 | 8 | `decode_fwd.py`、`decode_layer.py`、`decode_mtp.py`、`lm_head.py`、`moe.py`、`prefill_fwd.py`、`prefill_layer.py`、`prefill_mtp.py` |
| pto-isa C++ 接口/Tile 类型不匹配 | 4 | `mtp_projection.py`、`decode_layer_a8w8.py`、`qwen3_14b_decode_tq_draft.py`、`qwen3_14b_prefill_tq_draft.py` |
| 运行结果精度不匹配 | 3 | `hc_pre.py`、`prefill_compressor_ratio128.py`、`qwen3/14b/decode_fwd.py` |
| `pto.tmov` verifier 失败 | 4 | `deepseek_v3_2_decode_front.py`、`qwen3_32b_decode_4d.py` 是 src/dst shape 不匹配；`decode_attention_csa.py`、`decode_sparse_attn.py` 是地址空间组合不支持 |
| MLIR tile_buf 类型不一致 | 2 | `greedy_sample.py`、`topk_select.py`，`pad=3` 和无 pad 的 `!pto.tile_buf` 类型冲突 |
| 其他 PTOAS verifier/memplan 问题 | 2 | `gate.py` 是 `set_validshape` 要求动态 validShape；`prefill_indexer.py` 是 `reserve_buffer` 找不到 local memory hole |
| 测试脚本/前端配置问题 | 2 | `qwen3_14b_decode_ssn_draft.py` 不接受 `-p a2a3 -d` 参数；`qwen3_32b_prefill_draft.py` 使用了当前 PyPTO 不支持的 `pl.auto_chunk` |

## 性能测试

本次性能采集在远端机器 `101.245.68.6` 上完成，复用了两套 PTOAS 二进制：

- Round 1：`/home/zhongxuan/fangrui/PTOAS-main-latest/build/tools/ptoas/ptoas`
- Round 3：`/home/zhongxuan/fangrui/PTOAS-feature-memplan-latest-round2/build-round2/tools/ptoas/ptoas`

运行方式遵循 `pypto-lib/docs/performance-tuning.md` 和 `pypto-lib-command-guide.zh-CN.md`：

- 启用 L2 swimlane 采集
- 使用 `level2` / PTOAS memplan 或 modern memplan
- 结果目录下会生成 `dfx_outputs/l2_swimlane_records.json` 和 `merged_swimlane_*.json`

原始结果目录：

- `/home/zhongxuan/fangrui/memplan_perf_rounds/20260728_045510`
- `/home/zhongxuan/fangrui/memplan_perf_rounds/20260728_053407_explicit_l2`

说明：

- 下表给的是端到端 wall time，不是单纯 kernel time。
- `FAIL` 表示本次性能采样没有跑通；原因保留在右侧说明里。
- `decode_indexer.py` 和 `qkv_proj_rope.py` 需要显式传入 `--enable-l2-swimlane 2`；下表已使用该参数补充复测。
- `qkv_proj_rope.py` 和 `rmsnorm.py` 仍是运行期失败，但 `qkv_proj_rope.py` 已采集到 AICore timing。
- `expert_shared.py` 在 Round 3 下单独重试后仍在 PTOAS compile 阶段报 `vec overflow, requires 1605632 bits while 1572864 bits available`，没有生成 `dfx_outputs`，因此无法采集 AICore timing。

| Case | Round 1（main + legacy level2） | Round 3（feature + modern level2） |
|---|---|---|
| `models/deepseek/v3_2/deepseek_v3_2_decode_back.py` | PASS 39.29s | PASS 44.371s |
| `models/deepseek/v3_2/deepseek_v3_2_prefill_back.py` | PASS 28.19s | PASS 29.117s |
| `models/deepseek/v4/decode_compressor_ratio4.py` | PASS 15.343s | PASS 16.133s |
| `models/deepseek/v4/decode_indexer.py` | PASS 16.851s | PASS 16.243s |
| `models/deepseek/v4/decode_indexer_compressor.py` | PASS 14.728s | PASS 15.444s |
| `models/deepseek/v4/expert_routed.py` | PASS 26.275s | PASS 27.203s |
| `models/deepseek/v4/expert_shared.py` | PASS 14.237s | FAIL 3.32s: `Pass execution failed` |
| `models/deepseek/v4/hc_head.py` | PASS 15.268s | PASS 13.179s |
| `models/deepseek/v4/hc_post.py` | PASS 13.209s | PASS 13.14s |
| `models/deepseek/v4/prefill_compressor_ratio4.py` | PASS 18.219s | PASS 18.145s |
| `models/deepseek/v4/prefill_indexer_compressor.py` | PASS 21.192s | PASS 22.043s |
| `models/deepseek/v4/qkv_proj_rope.py` | FAIL 23.993s: `run_prepared failed with code 13` | FAIL 22.973s: `run_prepared failed with code 13` |
| `models/deepseek/v4/rmsnorm.py` | FAIL 19.412s: `run_prepared failed with code 13` | FAIL 19.814s: `run_prepared failed with code 13` |
| `models/qwen3/14b/test_paged_attention_cce.py` | FAIL 108.223s: `run_prepared failed with code 507018` | NOT RUN |
| `models/qwen3/32b/qwen3_32b_decode.py` | PASS 27.272s | PASS 32.151s |

### AICore kernel time

下面的数据从 `l2_swimlane_records.json` 的 AICore task start/end tick 计算得到，`clock_freq_hz=50MHz`：

- Round 1：ptoas main 分支 + legacy memplan。
- Round 3：ptoas 删除 memref + modern memplan。
- `makespan`：所有 AICore task 从最早 start 到最晚 end 的跨度。
- `exec sum`：所有 AICore task 的执行时间求和；由于多 core 并行，该值通常大于 `makespan`。
- `Round 3 vs Round 1`：按耗时计算，正数表示 Round 3 更慢，负数表示 Round 3 更快。

| Case | Round 1 makespan(us) | Round 1 exec sum(us) | Round 3 makespan(us) | Round 3 exec sum(us) | makespan Δ | exec sum Δ |
|---|---:|---:|---:|---:|---:|---:|
| `models/deepseek/v3_2/deepseek_v3_2_decode_back.py` | 2330.50 | 34807.56 | 2318.84 | 34657.86 | -0.50% | -0.43% |
| `models/deepseek/v3_2/deepseek_v3_2_prefill_back.py` | 11309.64 | 68752.78 | 11482.32 | 68739.84 | +1.53% | -0.02% |
| `models/deepseek/v4/decode_compressor_ratio4.py` | 62.26 | 427.18 | 67.22 | 437.20 | +7.97% | +2.35% |
| `models/deepseek/v4/decode_indexer.py` | 166.54 | 965.20 | 139.90 | 982.46 | -16.00% | +1.79% |
| `models/deepseek/v4/decode_indexer_compressor.py` | 60.06 | 125.14 | 63.82 | 131.86 | +6.26% | +5.37% |
| `models/deepseek/v4/expert_routed.py` | 547.34 | 13463.08 | 582.58 | 13664.08 | +6.44% | +1.49% |
| `models/deepseek/v4/hc_head.py` | 65.90 | 177.14 | 73.70 | 162.82 | +11.84% | -8.08% |
| `models/deepseek/v4/hc_post.py` | 21.86 | 154.92 | 21.82 | 155.68 | -0.18% | +0.49% |
| `models/deepseek/v4/prefill_compressor_ratio4.py` | 318.92 | 6035.98 | 337.10 | 8167.12 | +5.70% | +35.31% |
| `models/deepseek/v4/prefill_indexer_compressor.py` | 369.26 | 3792.46 | 458.26 | 3817.62 | +24.10% | +0.66% |
| `models/qwen3/32b/qwen3_32b_decode.py` | 2156.82 | 53562.68 | 2127.18 | 53170.68 | -1.37% | -0.73% |
