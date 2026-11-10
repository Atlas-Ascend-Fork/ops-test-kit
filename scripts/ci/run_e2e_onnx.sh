#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# =============================================================================
# run_e2e_onnx.sh —— E2E ONNX 通路（examples/case_store/e2e/onnx_cases.csv，7 例）
#
# 关键环境：onnx 1.13.1 装在 torchnpu 环境（max opset=18 恰为 atc 上限；
#           onnx≥1.14 的 max opset>18 会被解析期拒绝）；链路：TTK 组模 →
#           atc 编译 .om（缓存 onnx_om/<用例名>/<opset>/）→ NPU aclmdl 执行。
#           onnx 的 eager ≡ 静态图（om 恒为静态图）。
#
# 用法:
#   ./run_e2e_onnx.sh               # 全量（7 例）
#   ./run_e2e_onnx.sh <分组>        # 只跑指定分组（--list 查看分组）
#   ./run_e2e_onnx.sh --list        # 列出全部分组
#   ./run_e2e_onnx.sh --dry-run     # 只打印命令不执行
#
# 变体经 TTK_EXTRA_ARGS 追加，如:
#   TTK_EXTRA_ARGS="--eager -c" ./run_e2e_onnx.sh   # 双开：单次执行，EAGER/GRAPH CST 两列同值
# =============================================================================
set -o pipefail

_SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "$_SDIR/_common.sh"

TORCHNPU_ENV="${TORCHNPU_ENV:-$HOME/miniconda3/envs/torchnpu}"
PY="$TORCHNPU_ENV/bin/python"
E="$REPO_ROOT/examples/case_store/e2e"
CASES="$E/onnx_cases.csv"

# ---- 分组定义（与 examples/case_store/e2e/README.md ONNX 段分组一致）----
G_BASIC="onnx_add_f32,onnx_add_broadcast"
G_DTYPE="onnx_add_f16,onnx_matmul_bf16"
G_STRUCTURE="onnx_concat_i32,onnx_split_f32"
G_OPSET="onnx_add_v13"

usage_groups() {
    cat <<'EOF'
分组（按测试项分类，与 examples/case_store/e2e/README.md ONNX 段分组一致）:
  full       全量（7 例；eager ≡ 静态图）【默认】
  basic      基础链路（f32 双输入基线/广播 shape 推导，2 例）
  dtype      dtype 覆盖（f16 输出槽/bf16 精度档，2 例）
  structure  结构特性（Variadic 输入+必选属性/Optional 省略+多输出槽，2 例）
  opset      显式 opset 段（onnx.13.Add，1 例）
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

# ---- 环境准备（PATH 前置：atc 与环境内 onnx 的解释器一致）----
source_cann
check_exec "$PY"
check_file "$CASES"
export PATH="$TORCHNPU_ENV/bin:$PATH"
cd "$REPO_ROOT" || die "无法进入仓库根目录: $REPO_ROOT"

# ---- 分发 ----
case "$GROUP" in
    full|"")
        run_step "E2E onnx 全量（7 例；eager ≡ 静态图）" \
            "$PY" -m ttk e2e -i "$CASES" "${EXTRA[@]}"
        ;;
    basic)
        run_step "E2E onnx 基础链路（2 例）" \
            "$PY" -m ttk e2e -i "$CASES" -t "$G_BASIC" "${EXTRA[@]}"
        ;;
    dtype)
        run_step "E2E onnx dtype 覆盖（2 例）" \
            "$PY" -m ttk e2e -i "$CASES" -t "$G_DTYPE" "${EXTRA[@]}"
        ;;
    structure)
        run_step "E2E onnx 结构特性（2 例）" \
            "$PY" -m ttk e2e -i "$CASES" -t "$G_STRUCTURE" "${EXTRA[@]}"
        ;;
    opset)
        run_step "E2E onnx 显式 opset 段（1 例）" \
            "$PY" -m ttk e2e -i "$CASES" -t "$G_OPSET" "${EXTRA[@]}"
        ;;
    *)
        echo "[ERROR] 未知分组: $GROUP" >&2
        usage_groups
        exit 2
        ;;
esac

summary
