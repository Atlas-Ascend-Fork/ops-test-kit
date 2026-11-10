#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# =============================================================================
# run_all.sh —— 串行执行全部通路（Kernel/GEIR/ACLNN/E2E torch·tf·onnx）
#
# TTK 同目录禁止并发多条命令，因此六条通路必须串行；每条通路调用对应的
# run_*.sh（各脚本自带环境准备），本脚本只负责编排与汇总。
#
# 用法:
#   ./run_all.sh                    # 全部通路全量（101 例）
#   ./run_all.sh <分组> [--dry-run] # 分组名透传给各通路脚本；各通路分组名不同，
#                                   不识别该分组的通路记为"跳过"（不计失败）
#   ./run_all.sh --dry-run          # 全部通路只打印命令不执行
#
# 退出码约定: 0 = 无失败通路（跳过不计入失败）
#
# 注意：E2E 三框架进程互斥，但各自是独立进程，串行调用天然安全。
# =============================================================================
set -o pipefail

_SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(run_kernel.sh run_geir.sh run_aclnn.sh run_e2e_torch.sh run_e2e_tf.sh run_e2e_onnx.sh)

PASSED=()
SKIPPED=()
FAILED=()
for s in "${SCRIPTS[@]}"; do
    echo
    echo "##################################################"
    echo "##### 通路脚本: $s"
    echo "##################################################"
    if bash "$_SDIR/$s" "$@"; then
        PASSED+=("$s")
    else
        rc=$?
        if [ "$rc" -eq 2 ]; then
            SKIPPED+=("$s")
            echo "[NOTE] $s 不支持该分组/参数，已跳过（退出码 $rc）"
        else
            FAILED+=("$s")
        fi
    fi
done

echo
echo "=================================================="
echo "通路级汇总: 成功 ${#PASSED[@]}，跳过 ${#SKIPPED[@]}，失败 ${#FAILED[@]}"
if [ "${#SKIPPED[@]}" -gt 0 ]; then
    printf '  - 跳过通路(不支持该分组): %s\n' "${SKIPPED[@]}"
fi
if [ "${#FAILED[@]}" -gt 0 ]; then
    printf '  - 失败通路: %s\n' "${FAILED[@]}"
fi
echo "=================================================="
[ "${#FAILED[@]}" -eq 0 ]
