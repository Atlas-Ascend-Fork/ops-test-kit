#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# =============================================================================
# run_geir.sh —— GEIR 通路（examples/case_store/geir，13 例）执行脚本
#
# 关键环境：PATH 须前置含 sympy 的 Python 环境 bin——GE harness 内嵌 python
#           （TeFusion→AOE→InitCannKB）按 PATH 找 python3，缺 sympy 会报
#           "GEINITIALIZE FAILED"。本脚本默认前置 torchnpu 环境 bin。
#
# 用法:
#   ./run_geir.sh                  # 全量 = 12 例 golden（-d false -c -b release）+ 1 例 perf-only
#   ./run_geir.sh <分组>           # 只跑指定分组（--list 查看分组）
#   ./run_geir.sh --list           # 列出全部分组
#   ./run_geir.sh --dry-run        # 只打印命令不执行
#
# 环境变量:
#   TORCHNPU_ENV   Python 环境根（默认 ~/miniconda3/envs/torchnpu；需 sympy/onnx 等）
#   TTK_EXTRA_ARGS 追加到每条命令的额外参数
#   CANN_ENV       CANN set_env.sh 路径
# =============================================================================
set -o pipefail

_SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "$_SDIR/_common.sh"

TORCHNPU_ENV="${TORCHNPU_ENV:-$HOME/miniconda3/envs/torchnpu}"
PY="$TORCHNPU_ENV/bin/python"
G="$REPO_ROOT/examples/case_store/geir"
KKP="$REPO_ROOT/examples/case_store/kernel"   # 复用 Kernel 矩阵
CASES="$G/cases.csv"

# ---- 分组定义（与 examples/case_store/geir/README.md 分组一致）----
G_OPTATTR="conv2d_nchw_optional,arg_max_v2_range_pin,arg_max_v2_dtype_attr"
G_TENSORLIST="split_dyn_output,foreach_div_tensorlist,foreach_add_inplace_tensorlist,avgpool_update_int4"
G_REGRESSION="smoothl1_loss_inf_attr,arg_max_v2_dtype_attr,split_dyn_output"
G_PERFONLY="scatter_list_perf_only"
K_REUSE="add_fp16_broadcast,abs_int16_3d,add_complex64,pad_float8,pad_float4"

usage_groups() {
    cat <<'EOF'
分组（按测试项分类，与 examples/case_store/geir/README.md 分组一致）:
  full           全量 = 12 例 golden（-d false -c -b release）+ scatter_list_perf_only（--golden-mode Disable）【默认】
  optional-attrs 可选输入与复杂属性（conv2d/arg_max_v2，3 例）  -d false -c -b release
  tensorlist     TensorList 算子（split/foreach/avgpool，4 例）  -d false -c -b release
  regression     历史缺陷回归（与前两组用例交叉复用，3 例）  -d false -c -b release
  perf-only      scatter_list_perf_only（无 golden；--golden-mode Disable，
                 判据=Performance 数据，状态记 SKIP）                -d false -c -b release --golden-mode Disable
  reuse-kernel   复用 Kernel 用例矩阵（5 例 + concat_d，-d false -c -b release）
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

# ---- 环境准备（PATH 前置：GE 内嵌 python 需要 sympy）----
source_cann
check_exec "$PY"
check_file "$CASES"
export PATH="$TORCHNPU_ENV/bin:$PATH"
cd "$REPO_ROOT" || die "无法进入仓库根目录: $REPO_ROOT"

# ---- 分发 ----
case "$GROUP" in
    full|"")
        run_step "GEIR 全量 golden（12 例，-d false -c -b release）" \
            "$PY" -m ttk geir -i "$CASES" --plugin "$G/golden.py" -d false -c -b release --no-op scatter_list "${EXTRA[@]}"
        run_step "GEIR perf-only（scatter_list；--golden-mode Disable，判据=Performance 数据，状态记 SKIP）" \
            "$PY" -m ttk geir -i "$CASES" -d false -c -b release --golden-mode Disable -t "$G_PERFONLY" "${EXTRA[@]}"
        ;;
    optional-attrs)
        run_step "GEIR 可选输入与复杂属性（3 例，-d false -c -b release）" \
            "$PY" -m ttk geir -i "$CASES" --plugin "$G/golden.py" -d false -c -b release -t "$G_OPTATTR" "${EXTRA[@]}"
        ;;
    tensorlist)
        run_step "GEIR TensorList 算子（4 例，-d false -c -b release）" \
            "$PY" -m ttk geir -i "$CASES" --plugin "$G/golden.py" -d false -c -b release -t "$G_TENSORLIST" "${EXTRA[@]}"
        ;;
    regression)
        run_step "GEIR 历史缺陷回归（3 例，-d false -c -b release）" \
            "$PY" -m ttk geir -i "$CASES" --plugin "$G/golden.py" -d false -c -b release -t "$G_REGRESSION" "${EXTRA[@]}"
        ;;
    perf-only)
        run_step "GEIR perf-only（scatter_list；--golden-mode Disable，判据=Performance 数据，状态记 SKIP）" \
            "$PY" -m ttk geir -i "$CASES" -d false -c -b release --golden-mode Disable -t "$G_PERFONLY" "${EXTRA[@]}"
        ;;
    reuse-kernel)
        run_step "GEIR 复用 Kernel 矩阵（5 例，-d false -c -b release）" \
            "$PY" -m ttk geir -i "$KKP/cases.csv" --plugin "$KKP/golden.py" -d false -c -b release -t "$K_REUSE" "${EXTRA[@]}"
        run_step "GEIR 复用 Kernel 矩阵：concat_d（-d false -c -b release）" \
            "$PY" -m ttk geir -i "$KKP/cases.csv" --plugin "$KKP/golden.py" -d false -c -b release -t concat_d_tensorlist_in "${EXTRA[@]}"
        ;;
    *)
        echo "[ERROR] 未知分组: $GROUP" >&2
        usage_groups
        exit 2
        ;;
esac

summary
