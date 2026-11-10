#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# =============================================================================
# check_results.sh —— 用例结果 vs 预期规格 匹配校验
#
# 读取 log/<mode>/<用例名>/ttk.log 的结果表（STATUS 行），逐例匹配预期：
#   · 未在预期规格中列出的用例 → 预期 PASS
#   · perf-only / 无 CPU 基准用例 → 预期 SKIP，且指定 MODE 行必须有数值型
#     Performance 数据（判据=Performance 数据）
#   · 用例清单自动取自各通路 CSV（testcase_name 列），与用例文件天然同步
#
# 用法:
#   ./check_results.sh                              # 校验全部通路（读日志最新结果）
#   ./check_results.sh kernel geir                  # 只校验指定通路（可多选）
#   ./check_results.sh --since "2026-11-03 09:00"   # 仅认可该时刻之后更新的日志，
#                                                   #   更早的记 STALE、缺失记 NOT-RUN；
#                                                   #   取值不可被 find 解析 → exit 2
#   ./check_results.sh --verbose                    # 逐例打印（默认只打异常与汇总）
#   ./check_results.sh --list                       # 显示预期规格
#
# 通路名: kernel geir aclnn e2e-torch e2e-tf e2e-onnx
# 退出码: 0 = 全部符合预期；1 = 存在 MISMATCH / NOT-RUN / STALE；2 = 用法错误
# =============================================================================
set -o pipefail

_SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "$_SDIR/_common.sh"    # 仅用 REPO_ROOT 等定位；本脚本只读日志，无需 CANN 环境

CS="$REPO_ROOT/examples/case_store"
LOGROOT="$REPO_ROOT/log"
PY3="$(command -v python3 || echo /usr/bin/python3)"

# ---------------- 预期规格（未列出 = 预期 PASS）----------------
# 格式: <通路>:<用例名> <期望状态> [perf模式...]
#   期望=SKIP 时，列出的 MODE 行（如 CST/BIN/DYN/EAGER）必须有数值型性能
EXPECTED_SPEC='
kernel:matmul_v3_optional_inputs SKIP CST BIN
kernel:reduce_min_valuedepend SKIP CST BIN
geir:scatter_list_perf_only SKIP CST_BIN
e2e-torch:causal_conv1d_fn_basic SKIP EAGER
e2e-tf:tf_matmul SKIP EAGER
'

PATHWAYS=(kernel geir aclnn e2e-torch e2e-tf e2e-onnx)

# ---------------- 通路 → 用例清单 CSV / 日志目录 ----------------
pathway_csv() {
    case "$1" in
        kernel)    echo "$CS/kernel/cases.csv" ;;
        geir)      echo "$CS/geir/cases.csv" ;;
        aclnn)     echo "$CS/aclnn/cases.csv" ;;
        e2e-torch) echo "$CS/e2e/torch_cases.csv" ;;
        e2e-tf)    echo "$CS/e2e/tf_cases.csv" ;;
        e2e-onnx)  echo "$CS/e2e/onnx_cases.csv" ;;
    esac
}

pathway_logdir() {
    case "$1" in
        kernel)                 echo "$LOGROOT/kernel" ;;
        geir)                   echo "$LOGROOT/geir" ;;
        aclnn)                  echo "$LOGROOT/aclnn" ;;
        e2e-torch|e2e-tf|e2e-onnx) echo "$LOGROOT/e2e" ;;
    esac
}

# 从 CSV 提取 testcase_name（经 csv 模块，容忍引号/逗号字段）
get_cases() {
    "$PY3" - "$1" <<'PYEOF'
import csv, sys
with open(sys.argv[1], newline="") as f:
    for row in csv.DictReader(f):
        name = (row.get("testcase_name") or "").strip()
        if name:
            print(name)
PYEOF
}

# 预期规格查询: key -> "SKIP CST BIN"（未命中输出空）
spec_lookup() {
    awk -v key="$1" '
        $1 == key { $1 = ""; sub(/^ +/, ""); print; exit }
    ' <<< "$EXPECTED_SPEC"
}

# 单用例状态: 读最后一条 STATUS 行 → PASS | SKIP | FAIL | UNKNOWN
#   含 FAIL 字样（含 INPUT_GEN_FAILURE 等错误文本）→ FAIL
#   含 SKIP/SUPPRESSED → SKIP
case_status_from_log() {
    local st
    st=$(grep -E "^\| *STATUS" "$1" 2>/dev/null | tail -1)
    [ -z "$st" ] && { echo UNKNOWN; return; }
    if grep -qiE "FAIL" <<< "$st"; then echo FAIL
    elif grep -qiE "SKIP|SUPPRESSED" <<< "$st"; then echo SKIP
    elif grep -q "PASS" <<< "$st"; then echo PASS
    else echo UNKNOWN; fi
}

# 性能数据校验: 逐个检查 MODE 行性能列为数值（如 "| CST | 3.701 |..."），
# 输出缺失的 MODE 列表（空 = 数据齐备）
missing_perf_modes() {
    local log=$1; shift
    local m
    for m in "$@"; do
        grep -Eq "^\| *$m *\| *[0-9]" "$log" || printf '%s ' "$m"
    done
}

# ---------------- 参数解析 ----------------
SELECTED=()
SINCE=""
VERBOSE=0
# 用法错误统一 exit 2（与头注释"退出码 2 = 用法错误"一致）
arg_error() { echo "[ERROR] $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
    case "$1" in
        --since)
            # 缺值校验：不足两个参数（shift 2 会失败 → 死循环）或取值形似下一个选项
            [ $# -ge 2 ] || arg_error "--since 需要一个时间参数，如 --since \"2026-11-03 09:00\""
            case "$2" in -*) arg_error "--since 缺少取值（\"$2\" 看起来是另一个选项）";; esac
            SINCE="$2"
            # 取值可解析性预校验（与逐例检查同用 find -newermt，保证解析器一致）：
            # 坏时间串（如 2026-13-03）或本环境 find 不支持 -newermt 时立即判用法
            # 错误 exit 2——否则报错在逐例处被吞掉，每例都误判 [STALE]，
            # 以"业务失败"误导排查方向
            if ! _since_err="$(find . -maxdepth 0 -newermt "$SINCE" 2>&1 >/dev/null)"; then
                arg_error "--since 取值 \"${SINCE}\" 无法被 find 解析（或本环境 find 不支持 -newermt）: ${_since_err:-<find 无报错输出>}"
            fi
            shift 2 ;;
        --verbose|-v) VERBOSE=1; shift ;;
        --list)
            echo "预期规格（未列出 = 预期 PASS）:"
            sed '/^$/d' <<< "$EXPECTED_SPEC" | sed 's/^/  /'
            echo "通路: ${PATHWAYS[*]}"
            exit 0 ;;
        -h|--help) awk 'NR>9{if(/^set -/)exit;print}' "$0"; exit 0 ;;
        kernel|geir|aclnn|e2e-torch|e2e-tf|e2e-onnx) SELECTED+=("$1"); shift ;;
        *) echo "[ERROR] 未知参数/通路: $1（通路名见 -h）" >&2; exit 2 ;;
    esac
done
[ ${#SELECTED[@]} -eq 0 ] && SELECTED=("${PATHWAYS[@]}")

# ---------------- 逐例匹配 ----------------
TOTAL=0; OK=0; MISMATCH=0; NOTRUN=0; STALE=0
MISMATCH_LINES=(); NOTRUN_LINES=(); STALE_LINES=()
for pw in "${SELECTED[@]}"; do
    csv="$(pathway_csv "$pw")"
    logdir="$(pathway_logdir "$pw")"
    if [ ! -f "$csv" ]; then
        echo "[ERROR] 用例清单缺失: $csv" >&2
        exit 2
    fi
    # 先取清单并校验退出码：进程替换会吞掉 get_cases 的失败（python 崩溃/CSV 不可读），
    # 循环零迭代 → 计数全 0 → 误报"全部符合预期"退 0 —— 必须显式失败
    if ! cases_list="$(get_cases "$csv")"; then
        echo "[ERROR] 读取用例清单失败: $csv" >&2
        exit 2
    fi
    [ -n "$cases_list" ] || { echo "[ERROR] 用例清单为空: $csv" >&2; exit 2; }
    while IFS= read -r case_name; do
        [ -z "$case_name" ] && continue
        TOTAL=$((TOTAL + 1))
        key="$pw:$case_name"
        log="$logdir/$case_name/ttk.log"

        spec="$(spec_lookup "$key")"
        if [ -n "$spec" ]; then
            expected="$(awk '{print $1}' <<< "$spec")"
            read -r -a perf_modes <<< "$(cut -d' ' -f2- <<< "$spec")"
        else
            expected="PASS"
            perf_modes=()
        fi

        if [ ! -f "$log" ]; then
            echo "  [NOT-RUN]  $key —— 无结果日志"
            NOTRUN_LINES+=("$key —— 无结果日志")
            NOTRUN=$((NOTRUN + 1)); continue
        fi
        if [ -n "$SINCE" ]; then
            # 单次 find 取合并输出+退出码：执行失败（非"日志不比 SINCE 新"）时
            # 输出即真实报错——直接 exit 2 暴露问题，不再 2>/dev/null 吞掉报错后
            # 把执行失败误判成 [STALE]
            if ! _find_out="$(find "$log" -newermt "$SINCE" 2>&1)"; then
                echo "[ERROR] $key —— find 检查日志时间失败（--since \"$SINCE\"）: ${_find_out:-<find 无报错输出>}" >&2
                exit 2
            fi
            if [ -z "$_find_out" ]; then
                echo "  [STALE]    $key —— 日志早于 $SINCE（非本轮结果）"
                STALE_LINES+=("$key —— 日志早于 $SINCE")
                STALE=$((STALE + 1)); continue
            fi
        fi

        actual="$(case_status_from_log "$log")"
        reason=""
        if [ "$actual" != "$expected" ]; then
            reason="期望 $expected，实际 $actual"
        elif [ "$expected" = "SKIP" ] && [ ${#perf_modes[@]} -gt 0 ]; then
            missing="$(missing_perf_modes "$log" "${perf_modes[@]}")"
            [ -n "$missing" ] && reason="SKIP 达标，但 Performance 数据缺失: ${missing% }"
        fi

        if [ -z "$reason" ]; then
            OK=$((OK + 1))
            [ "$VERBOSE" = "1" ] && echo "  [OK]       $key → $actual"
        else
            MISMATCH=$((MISMATCH + 1))
            echo "  [MISMATCH] $key —— $reason"
            MISMATCH_LINES+=("$key —— $reason")
        fi
    done <<< "$cases_list"
done

# 兜底守卫：任何原因导致 0 例被解析时，不得判为"全部通过"
if [ "$TOTAL" -eq 0 ]; then
    echo "[ERROR] 未解析到任何用例（通路选择: ${SELECTED[*]}）——请检查用例清单/CSV" >&2
    exit 2
fi

# ---------------- 汇总 ----------------
echo "=================================================="
echo "结果匹配汇总: 总数 $TOTAL，符合预期 $OK，不符合 $MISMATCH，未运行 $NOTRUN，陈旧 $STALE"
if [ "$MISMATCH" -gt 0 ]; then
    echo "结论【不符合预期 $MISMATCH 例】:"
    printf '  [MISMATCH] %s\n' "${MISMATCH_LINES[@]}"
fi
if [ "$NOTRUN" -gt 0 ]; then
    echo "结论【未运行 $NOTRUN 例】:"
    printf '  [NOT-RUN]  %s\n' "${NOTRUN_LINES[@]}"
fi
if [ "$STALE" -gt 0 ]; then
    echo "结论【陈旧结果 $STALE 例（日志早于 --since，非本轮）】:"
    printf '  [STALE]    %s\n' "${STALE_LINES[@]}"
fi
if [ $((MISMATCH + NOTRUN + STALE)) -eq 0 ]; then
    echo "结论: 全部 $TOTAL 例用例结果符合预期"
    echo "=================================================="
    exit 0
fi
echo "=================================================="
exit 1
