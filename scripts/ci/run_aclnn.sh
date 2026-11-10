#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# =============================================================================
# run_aclnn.sh —— ACLNN 通路（examples/case_store/aclnn，25 例）执行脚本
#
# 用法:
#   ./run_aclnn.sh                 # 全量（25 例）
#   ./run_aclnn.sh <分组>          # 只跑指定分组（--list 查看分组）
#   ./run_aclnn.sh --list          # 列出全部分组
#   ./run_aclnn.sh --dry-run       # 只打印命令不执行
#
# 环境变量:
#   TORCHNPU_ENV   Python 环境根（默认 ~/miniconda3/envs/torchnpu；golden.py 顶部
#                  import torch，该环境 torch 2.5.1+cpu 可用）
#   TTK_EXTRA_ARGS 追加到每条命令的额外参数
#   CANN_ENV       CANN set_env.sh 路径
# =============================================================================
set -o pipefail

_SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "$_SDIR/_common.sh"

TORCHNPU_ENV="${TORCHNPU_ENV:-$HOME/miniconda3/envs/torchnpu}"
PY="$TORCHNPU_ENV/bin/python"
A="$REPO_ROOT/examples/case_store/aclnn"
CASES="$A/cases.csv"

# ---- 分组定义（与 examples/case_store/aclnn/README.md 分组一致）----
G_BASIC="add_fp16_broadcast,add_bf16_bcast3d,add_fp32_alpha_attr,add_fp64,add_int16,add_uint8,add_complex64,add_int8_negative,add_noncontig"
G_TENSORLIST="cat_tensorlist_fp32,cat_bf16_dim0,cat_int8_dim_neg1,cat_int64_dim_neg1,split_fp32,split_fp16_dim_neg1,split_int32_tail"
G_SCALAR_INPLACE="clamp_fp16_scalars,inplace_fill_scalar_fp32,inplace_fill_tensor_int32"
G_VALUE_CONSTRAINED="indexselect_fp32_range"
G_DYNAMIC_SHAPE="masked_select_elewise,masked_select_brc,nonzero_dynamic,nonzero_v2_transposed"
G_COMPLEX_ATTRS="convolution_fp16"

usage_groups() {
    cat <<'EOF'
分组（按测试项分类，与 examples/case_store/aclnn/README.md 分组一致）:
  full              全量（25 例）【默认】
  basic             基础精度链路（add/dtype 覆盖/非连续输入，9 例）
  tensorlist        TensorList 输入/输出（cat/split，7 例）
  scalar-inplace    标量参数与原地操作（clamp/inplace_fill，3 例）
  value-constrained 值约束输入（index_select，1 例）
  dynamic-shape     动态输出 shape（masked_select/nonzero，4 例）
  complex-attrs     复杂属性与可选输入（convolution，1 例）
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
export PATH="$TORCHNPU_ENV/bin:$PATH"
cd "$REPO_ROOT" || die "无法进入仓库根目录: $REPO_ROOT"

# ---- 分发（ACLNN 无模式专属选项，分组仅 -t 筛选）----
case "$GROUP" in
    full|"")
        run_step "ACLNN 全量（25 例）" \
            "$PY" -m ttk aclnn -i "$CASES" --plugin "$A/golden.py" "${EXTRA[@]}"
        ;;
    basic)
        run_step "ACLNN 基础精度链路（9 例）" \
            "$PY" -m ttk aclnn -i "$CASES" --plugin "$A/golden.py" -t "$G_BASIC" "${EXTRA[@]}"
        ;;
    tensorlist)
        run_step "ACLNN TensorList 输入/输出（7 例）" \
            "$PY" -m ttk aclnn -i "$CASES" --plugin "$A/golden.py" -t "$G_TENSORLIST" "${EXTRA[@]}"
        ;;
    scalar-inplace)
        run_step "ACLNN 标量参数与原地操作（3 例）" \
            "$PY" -m ttk aclnn -i "$CASES" --plugin "$A/golden.py" -t "$G_SCALAR_INPLACE" "${EXTRA[@]}"
        ;;
    value-constrained)
        run_step "ACLNN 值约束输入（1 例）" \
            "$PY" -m ttk aclnn -i "$CASES" --plugin "$A/golden.py" -t "$G_VALUE_CONSTRAINED" "${EXTRA[@]}"
        ;;
    dynamic-shape)
        run_step "ACLNN 动态输出 shape（4 例）" \
            "$PY" -m ttk aclnn -i "$CASES" --plugin "$A/golden.py" -t "$G_DYNAMIC_SHAPE" "${EXTRA[@]}"
        ;;
    complex-attrs)
        run_step "ACLNN 复杂属性与可选输入（1 例）" \
            "$PY" -m ttk aclnn -i "$CASES" --plugin "$A/golden.py" -t "$G_COMPLEX_ATTRS" "${EXTRA[@]}"
        ;;
    *)
        echo "[ERROR] 未知分组: $GROUP" >&2
        usage_groups
        exit 2
        ;;
esac

summary
