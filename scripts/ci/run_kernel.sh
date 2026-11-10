#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# =============================================================================
# run_kernel.sh —— Kernel 通路（examples/case_store/kernel，18 例）执行脚本
#
# 模式约定：全部用例统一 -d false 关闭动态编译（CST+BIN 双模式，mixed 判据）；
#           requant_float8 单列专用判据 --compare requant（mixed 对该例数据相关不稳）；
#           matmul_v3 为 perf-only（--golden-mode Disable，判据=Performance 数据）。
#
# 用法:
#   ./run_kernel.sh               # 全量 = 其余 17 例 + matmul_v3（统一 -d false 双模式）
#   ./run_kernel.sh <分组>        # 只跑指定分组（--list 查看分组）
#   ./run_kernel.sh --list        # 列出全部分组
#   ./run_kernel.sh --dry-run     # 只打印命令不执行（可与分组连用）
#
# 环境变量:
#   TTK_PYTHON      解释器（默认 python3 即 PATH 查找；需 sympy + en-dtypes；
#                   可 export TTK_PYTHON=<满足要求的解释器>；本地 venv 丢失时可重建并 pip install -r requirements.txt en-dtypes，

#   TTK_EXTRA_ARGS  追加到每条命令的额外参数，如 "--seed 42"、"--eager -c"
#   CANN_ENV        CANN set_env.sh 路径（默认 ~/Ascend/cann/set_env.sh）
# =============================================================================
set -o pipefail

_SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "$_SDIR/_common.sh"

PY="${TTK_PYTHON:-python3}"
K="$REPO_ROOT/examples/case_store/kernel"
CASES="$K/cases.csv"

# ---- 分组定义（与 examples/case_store/kernel/README.md 分组一致）----
G_BASIC="add_fp16_broadcast,add_fp32_mustappear,add_int32_boundary,add_int8_negative,add_bf16_bcast3d,add_complex64,abs_int16_3d,abs_int64_2d"
G_STRUCTURE="concat_d_tensorlist_in,split_tensorlist_out,zeros_like_constant"
G_SHAPE_UNKNOWN="non_zero_shape_unknown"
G_LOWBIT="pad_float8,pad_float4,quant_int8"
G_REQUANT="requant_float8"
G_MATMUL="matmul_v3_optional_inputs"
G_REDUCE_MIN="reduce_min_valuedepend"
# 全量①的 15 例常规精度清单（= basic + structure + shape_unknown + lowbit）；
# requant / matmul_v3 / reduce_min 各自单列：requant 需专用判据，matmul_v3 与 reduce_min 为 perf-only
G_GOLDEN15="$G_BASIC,$G_STRUCTURE,$G_SHAPE_UNKNOWN,$G_LOWBIT"

usage_groups() {
    cat <<'EOF'
分组（按测试项分类，与 examples/case_store/kernel/README.md 分组一致）:
  full           全量 = 16 例常规精度 + requant（--compare requant）+ matmul_v3
                 （统一 -d false，CST+BIN 双模式）【默认】
  basic          基础精度链路（add/abs，8 例）                        -d false -c -b release
  structure      结构特性（TensorList/ValueDepend/常量输出，4 例）     -d false -c -b release
  shape-unknown  shape-unknown 输出（non_zero，1 例）                  -d false -c -b release
  lowbit-quant   低比特 dtype 与量化判据（3 例）                       -d false -c -b release
  requant        requant 量化判据（requant_float8，1 例）              --compare requant
  matmul-v3      MatMulV3 perf-only（1 例；--golden-mode Disable 跳过比对，
                 判据=Performance 数据，状态记 SKIP）                 -d false -c -b release --golden-mode Disable
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

# ---- 分发 ----
case "$GROUP" in
    full|"")
        run_step "Kernel 全量①：15 例常规精度用例（-d false 双模式，默认 mixed 判据）" \
            "$PY" -m ttk kernel -i "$CASES" --plugin "$K/golden.py" -d false -c -b release -t "$G_GOLDEN15" "${EXTRA[@]}"
        run_step "Kernel 全量②：requant_float8（专用量化判据 --compare requant；mixed 对该例数据相关不稳）" \
            "$PY" -m ttk kernel -i "$CASES" --plugin "$K/golden.py" -t "$G_REQUANT" --compare requant -d false -c -b release "${EXTRA[@]}"
        run_step "Kernel 全量③：matmul_v3（perf-only：--golden-mode Disable 跳过比对，判据=Performance 数据）" \
            "$PY" -m ttk kernel -i "$CASES" --plugin "$K/golden.py" -t "$G_MATMUL" -d false -c -b release --golden-mode Disable "${EXTRA[@]}"
        run_step "Kernel 全量④：reduce_min_valuedepend（perf-only：CANN B046 算子 bug 期间绕过精度比对）" \
            "$PY" -m ttk kernel -i "$CASES" --plugin "$K/golden.py" -t "$G_REDUCE_MIN" -d false -c -b release --golden-mode Disable "${EXTRA[@]}"
        ;;
    basic)
        run_step "Kernel 基础精度链路（8 例）" \
            "$PY" -m ttk kernel -i "$CASES" --plugin "$K/golden.py" -t "$G_BASIC" -d false -c -b release "${EXTRA[@]}"
        ;;
    structure)
        run_step "Kernel 结构特性（4 例）" \
            "$PY" -m ttk kernel -i "$CASES" --plugin "$K/golden.py" -t "$G_STRUCTURE" -d false -c -b release "${EXTRA[@]}"
        ;;
    shape-unknown)
        run_step "Kernel shape-unknown 输出（1 例）" \
            "$PY" -m ttk kernel -i "$CASES" --plugin "$K/golden.py" -t "$G_SHAPE_UNKNOWN" -d false -c -b release "${EXTRA[@]}"
        ;;
    lowbit-quant)
        run_step "Kernel 低比特 dtype 与量化判据（3 例）" \
            "$PY" -m ttk kernel -i "$CASES" --plugin "$K/golden.py" -t "$G_LOWBIT" -d false -c -b release "${EXTRA[@]}"
        ;;
    requant)
        run_step "Kernel requant 量化判据（1 例）" \
            "$PY" -m ttk kernel -i "$CASES" --plugin "$K/golden.py" -t "$G_REQUANT" --compare requant -d false -c -b release "${EXTRA[@]}"
        ;;
    matmul-v3)
        run_step "Kernel MatMulV3 perf-only（仅 -b -c；--golden-mode Disable，判据=Performance 数据）" \
            "$PY" -m ttk kernel -i "$CASES" --plugin "$K/golden.py" -t "$G_MATMUL" -d false -c -b release --golden-mode Disable "${EXTRA[@]}"
        ;;
    *)
        echo "[ERROR] 未知分组: $GROUP" >&2
        usage_groups
        exit 2
        ;;
esac

summary
