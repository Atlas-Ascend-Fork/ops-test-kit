#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# =============================================================================
# run_e2e_torch.sh —— E2E torch 通路（examples/case_store/e2e/torch_cases.csv，24 例）
#
# 关键环境：解释器（默认 /usr/bin/python3，export TORCH_PY 可覆盖）
#           + PYTHONPATH=$PYENV_CI
#           （torch 2.12.0 + torch_npu 2.12.0.post2，支持 Ascend950PR；
#            注意 torchnpu 环境的 torch_npu 2.5.1 不支持 950，勿用）。
#           ~/.local 提供 en-dtypes（pad_fp8_e8m0）与 ninja（causal_conv1d 扩展）。
#
# 用法:
#   ./run_e2e_torch.sh             # 全量（24 例，仅 eager）
#   ./run_e2e_torch.sh <分组>      # 只跑指定分组（--list 查看分组）
#   ./run_e2e_torch.sh --list      # 列出全部分组
#   ./run_e2e_torch.sh --dry-run   # 只打印命令不执行
#
# 图模式等变体经 TTK_EXTRA_ARGS 追加，如:
#   TTK_EXTRA_ARGS="--eager -c" ./run_e2e_torch.sh          # eager + GE 静态图
#   TTK_EXTRA_ARGS="--backend npusim" ./run_e2e_torch.sh basic  # 无卡仿真（建议仅 eager）
#
# 环境变量:
#   TORCH_PY        解释器（默认 /usr/bin/python3；需 torch 2.12 + torch_npu 2.12，
#                   可 export TORCH_PY=<满足要求的解释器>——与 run_kernel.sh 的
#                   TTK_PYTHON、run_e2e_tf.sh 的 TF_ENV 同为覆盖入口）
#   PYENV_CI        torch/torch_npu 所在 site-packages 目录（默认
#                   ~/pyenv_ci/site-packages），作为 PYTHONPATH 注入
#   TTK_EXTRA_ARGS  追加到每条命令的额外参数
#   CANN_ENV        CANN set_env.sh 路径（默认 ~/Ascend/cann/set_env.sh）
# =============================================================================
set -o pipefail

_SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "$_SDIR/_common.sh"

PYENV_CI="${PYENV_CI:-$HOME/pyenv_ci/site-packages}"
PY="${TORCH_PY:-/usr/bin/python3}"
E="$REPO_ROOT/examples/case_store/e2e"
CASES="$E/torch_cases.csv"

# ---- 分组定义（与 examples/case_store/e2e/README.md torch 段分组一致）----
G_BASIC="add_f32_alpha,add_range_mustappear,add_broadcast,abs_01,relu_f16"
G_DTYPE="add_f16,add_bf16,add_int32,add_int8,add_int16,add_int64,add_uint8,add_bool,add_f64,add_complex64"
G_FP8="pad_fp8_e5m2,pad_fp8_e4m3,pad_fp8_e8m0"
G_LAYOUT="add_out,add_strided_view,cat_tensorlist,relu_inp"
G_NPU_EXT="npu_gelu_f16,causal_conv1d_fn_basic"

usage_groups() {
    cat <<'EOF'
分组（按测试项分类，与 examples/case_store/e2e/README.md torch 段分组一致）:
  full           全量（24 例，仅 eager）【默认】
  basic          基础链路（双输入+属性/值域定制/广播/单输入/functional API，5 例）
  dtype          dtype 覆盖（f16/bf16/int 族/bool/f64/complex64，10 例）
  fp8            低比特 fp8 族常量填充（e8m0 依赖 en-dtypes，3 例）
  layout-output  布局与输出位（out= 槽位/非连续视图/TensorList/原地方法，4 例）
  npu-ext        NPU 专属与扩展命名空间（npu_gelu 用 golden_api 指定 CPU 基准；
                 causal_conv1d 为 CSV golden_api=disable，判据=Performance
                 数据，状态记 SKIP，2 例）
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

# ---- 环境准备（PYTHONPATH 整体替换为 pyenv_ci，与实测通过配置一致）----
source_cann
check_exec "$PY"
check_dir "$PYENV_CI"
check_file "$CASES"
export PYTHONPATH="$PYENV_CI"
export PATH="$HOME/.local/bin:$PATH"   # ninja（causal_conv1d C++ 扩展）/ en-dtypes 所在用户目录
cd "$REPO_ROOT" || die "无法进入仓库根目录: $REPO_ROOT"

# ---- 分发 ----
case "$GROUP" in
    full|"")
        run_step "E2E torch 全量（24 例，仅 eager）" \
            "$PY" -m ttk e2e -i "$CASES" "${EXTRA[@]}"
        ;;
    basic)
        run_step "E2E torch 基础链路（5 例）" \
            "$PY" -m ttk e2e -i "$CASES" -t "$G_BASIC" "${EXTRA[@]}"
        ;;
    dtype)
        run_step "E2E torch dtype 覆盖（10 例）" \
            "$PY" -m ttk e2e -i "$CASES" -t "$G_DTYPE" "${EXTRA[@]}"
        ;;
    fp8)
        run_step "E2E torch 低比特 fp8 族（3 例）" \
            "$PY" -m ttk e2e -i "$CASES" -t "$G_FP8" "${EXTRA[@]}"
        ;;
    layout-output)
        run_step "E2E torch 布局与输出位（4 例）" \
            "$PY" -m ttk e2e -i "$CASES" -t "$G_LAYOUT" "${EXTRA[@]}"
        ;;
    npu-ext)
        run_step "E2E torch NPU 专属与扩展（2 例；causal_conv1d 为 CSV golden_api=disable，判据=Performance 数据，状态记 SKIP）" \
            "$PY" -m ttk e2e -i "$CASES" -t "$G_NPU_EXT" "${EXTRA[@]}"
        ;;
    *)
        echo "[ERROR] 未知分组: $GROUP" >&2
        usage_groups
        exit 2
        ;;
esac

summary
