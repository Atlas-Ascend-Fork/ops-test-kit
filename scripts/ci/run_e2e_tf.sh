#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# =============================================================================
# run_e2e_tf.sh —— E2E TensorFlow 通路（examples/case_store/e2e/tf_cases.csv，14 例）
#
# 关键环境：tf39 conda 环境（tf 2.6.5 + npu-device 2.6.5 配套）；
#           与 torch 框架进程互斥，须独立执行。
#
# 用法:
#   ./run_e2e_tf.sh                # 全量（14 例 = 13 例常规 + tf_inplace_add_unique --run=3）
#   ./run_e2e_tf.sh <分组>         # 只跑指定分组（--list 查看分组）
#   ./run_e2e_tf.sh --list         # 列出全部分组
#   ./run_e2e_tf.sh --dry-run      # 只打印命令不执行
#
# 变体经 TTK_EXTRA_ARGS 追加，如:
#   TTK_EXTRA_ARGS="--eager -c" ./run_e2e_tf.sh    # eager + 静态图（-c 固定 input_signature）
# =============================================================================
set -o pipefail

_SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "$_SDIR/_common.sh"

TF_ENV="${TF_ENV:-$HOME/miniconda3/envs/tf39}"
PY="$TF_ENV/bin/python"
E="$REPO_ROOT/examples/case_store/e2e"
CASES="$E/tf_cases.csv"

# ---- 分组定义（与 examples/case_store/e2e/README.md TF 段分组一致）----
G_BASIC="tf_add_f32,tf_abs,tf_add_broadcast,tf_math_add"
G_DTYPE="tf_add_f16,tf_add_bf16,tf_add_int32"
G_MATMUL="tf_matmul"
G_TF_NN="tf_relu_f32,tf_relu_f16"
G_TF_SPECIFIC="tf_scatter_div_stateful,tf_dequantize_scaled,tf_range_valuedepend"
G_TF_DET="tf_inplace_add_unique"
# 全量① = 12 例常规精度 + ③ matmul_v3(perf-only);② tf_inplace_add_unique 单列 --run=3
G_TF12="$G_BASIC,$G_DTYPE,$G_TF_NN,$G_TF_SPECIFIC"

usage_groups() {
    cat <<'EOF'
分组（按测试项分类，与 examples/case_store/e2e/README.md TF 段分组一致）:
  full           全量 = 13 例常规（--pc 2）+ tf_inplace_add_unique（--run=3）【默认】
  basic        基础链路（raw_ops 双输入+属性/单输入/广播/tf.math 命名空间，4 例）
  dtype        dtype 覆盖（f16/bf16/int32，3 例）+ tf_matmul（perf-only）
  tf-nn        tf.nn 公共 API（属性传递，2 例）
  tf-specific  TF 专属链路（Resource 有状态/量化 quint8+字符串属性/value-depend，3 例）
  deterministic  别名写确定性（tf_inplace_add_unique，1 例；InplaceAdd 的 x
                 与输出共享 buffer，逐轮复位三轮一致）             --run=3
EOF
}

# ---- 参数解析 ----
ARGS=()
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=1 ;;
        --list)    usage_groups; exit 0 ;;
        -h|--help) awk 'NR>9{if(/^set -/)exit;print}' "$0"; echo; usage_groups; exit 0 ;;
        *)         ARGS+=("$a") ;;
    esac
done
GROUP="${ARGS[0]:-full}"

# ---- 环境准备 ----
source_cann
check_exec "$PY"
check_file "$CASES"
cd "$REPO_ROOT" || die "无法进入仓库根目录: $REPO_ROOT"

# ---- 分发（--pc 2 为受限内存容器的官方建议，全部分组保持一致）----
case "$GROUP" in
    full|"")
        run_step "E2E tf 全量①：12 例常规精度用例（--pc 2）" \
            "$PY" -m ttk e2e -i "$CASES" --pc 2 -t "$G_TF12" "${EXTRA[@]}"
        run_step "E2E tf 全量②：tf_inplace_add_unique（别名写确定性，--run=3 三轮一致，e2e/README.md 记载口径）" \
            "$PY" -m ttk e2e -i "$CASES" --pc 2 --run=3 -t "$G_TF_DET" "${EXTRA[@]}"
        run_step "E2E tf 全量③：tf_matmul（perf-only：--golden-mode Disable，判据=Performance 数据）" \
            "$PY" -m ttk e2e -i "$CASES" --pc 2 --golden-mode Disable -t "$G_MATMUL" "${EXTRA[@]}"
        ;;
    basic)
        run_step "E2E tf 基础链路（4 例）" \
            "$PY" -m ttk e2e -i "$CASES" --pc 2 -t "$G_BASIC" "${EXTRA[@]}"
        ;;
    dtype)
        run_step "E2E tf dtype 覆盖（f16/bf16/int32，3 例）" \
            "$PY" -m ttk e2e -i "$CASES" --pc 2 -t "$G_DTYPE" "${EXTRA[@]}"
        run_step "E2E tf matmul（perf-only）" \
            "$PY" -m ttk e2e -i "$CASES" --pc 2 --golden-mode Disable -t "$G_MATMUL" "${EXTRA[@]}"
        ;;
    tf-nn)
        run_step "E2E tf.nn 公共 API（2 例）" \
            "$PY" -m ttk e2e -i "$CASES" --pc 2 -t "$G_TF_NN" "${EXTRA[@]}"
        ;;
    tf-specific)
        run_step "E2E tf TF 专属链路（3 例）" \
            "$PY" -m ttk e2e -i "$CASES" --pc 2 -t "$G_TF_SPECIFIC" "${EXTRA[@]}"
        ;;
    deterministic)
        run_step "E2E tf 别名写确定性（tf_inplace_add_unique，--run=3）" \
            "$PY" -m ttk e2e -i "$CASES" --pc 2 --run=3 -t "$G_TF_DET" "${EXTRA[@]}"
        ;;
    *)
        echo "[ERROR] 未知分组: $GROUP" >&2
        usage_groups
        exit 2
        ;;
esac

summary
