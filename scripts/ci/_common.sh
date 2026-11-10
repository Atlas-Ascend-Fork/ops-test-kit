#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# =============================================================================
# _common.sh —— run_*.sh 公共函数库（被 source 使用，不单独执行）
#
# 提供：仓库定位、CANN 环境加载、解释器/路径检查、run_step 步骤执行与汇总。
# 约定：TTK 同目录禁止并发多条命令，所有步骤在脚本内串行执行。
# =============================================================================
set -o pipefail

# 脚本位于 <repo>/scripts/ci/，仓库根为向上两级
TTK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TTK_SCRIPT_DIR/../.." && pwd)"

# CANN 环境入口（可用环境变量 CANN_ENV 覆盖）
CANN_ENV="${CANN_ENV:-$HOME/Ascend/cann/set_env.sh}"

# 步骤计数（run_step 累计，summary 汇总）
PASSED_STEPS=0
FAILED_STEPS=0
XFAIL_STEPS=0
FAILED_DESCS=()
DRY_RUN=0

# 追加参数（TTK_EXTRA_ARGS，按空白切分为数组，追加到每条 ttk 命令末尾）
EXTRA=()
if [ -n "${TTK_EXTRA_ARGS:-}" ]; then
    read -r -a EXTRA <<< "$TTK_EXTRA_ARGS"
fi

die() { echo "[ERROR] $*" >&2; exit 1; }

# source CANN 环境（所有通路必做：否则主进程 import tbe 失败）
source_cann() {
    [ -f "$CANN_ENV" ] || die "CANN set_env.sh 不存在: $CANN_ENV（可 export CANN_ENV=<路径> 覆盖）"
    # shellcheck disable=SC1090
    source "$CANN_ENV" || die "source CANN 环境失败: $CANN_ENV"
}

check_file() { [ -e "$1" ] || die "文件不存在: $1"; }
check_dir()  { [ -d "$1" ] || die "目录不存在: $1"; }
check_exec() { [ -x "$1" ] || die "解释器不可用: $1（可用环境变量覆盖路径）"; }

# run_step "步骤描述" [--xfail] cmd args...
#   串行执行一条 ttk 命令；失败不中断（TTK 结果表里能看到 FAIL 明细），
#   由 summary 统一汇总并决定脚本退出码。
#   --xfail：该步骤预期以非零退出（如 perf-only 用例结论按设计记 FAIL，
#   ttk 对存在 FAIL 的批次恒返回 1）——不计数为失败，结果表仍完整打印可查。
run_step() {
    local desc="$1"; shift
    local xfail=0
    if [ "${1:-}" = "--xfail" ]; then xfail=1; shift; fi
    echo
    echo "=================================================="
    echo "[$(date '+%F %T')] $desc"
    echo "\$ $*"
    echo "--------------------------------------------------"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY-RUN] 跳过执行"
        return 0
    fi
    if "$@"; then
        PASSED_STEPS=$((PASSED_STEPS + 1))
    else
        local rc=$?
        if [ "$xfail" = "1" ]; then
            XFAIL_STEPS=$((XFAIL_STEPS + 1))
            echo "[NOTE] 步骤按预期以非零退出（$rc）——结论 FAIL 属设计行为（perf-only），不计失败；结果表见上"
        else
            FAILED_STEPS=$((FAILED_STEPS + 1))
            FAILED_DESCS+=("$desc")
            echo "[WARN] 步骤失败（退出码 $rc），继续后续步骤" >&2
        fi
    fi
    return 0
}

# summary —— 打印执行汇总；有失败步骤则以非 0 退出
summary() {
    echo
    echo "=================================================="
    echo "执行汇总: 成功步骤 ${PASSED_STEPS}，预期失败 ${XFAIL_STEPS}，失败步骤 ${FAILED_STEPS}"
    if [ "$FAILED_STEPS" -gt 0 ]; then
        printf '  - 失败步骤: %s\n' "${FAILED_DESCS[@]}"
    fi
    echo "用例日志: $REPO_ROOT/log/<mode>/<用例名>/ttk.log"
    echo "=================================================="
    [ "$FAILED_STEPS" -eq 0 ]
}
