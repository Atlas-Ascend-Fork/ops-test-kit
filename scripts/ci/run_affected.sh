#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# =============================================================================
# run_affected.sh —— 依据改动代码路径自动判定受影响通路，串行拉起对应 run_*.sh
#
# 输入模式:
#   ./run_affected.sh                        # 自动检测 git 工作区改动（含未跟踪）
#   ./run_affected.sh <路径> [<路径> ...]     # 显式指定改动文件/目录（相对或绝对）
#   ./run_affected.sh --base HEAD~1          # 与指定 git ref 对比（含已提交改动）
#   ./run_affected.sh --paths p1,p2          # 逗号分隔的显式路径（同 --paths 可多次）
#
# 路径判定规则:
#   ttk/core_modules/kernel/**            → Kernel      → run_kernel.sh
#   ttk/core_modules/geir/**              → GEIR        → run_geir.sh
#   ttk/core_modules/aclnn/**             → ACLNN       → run_aclnn.sh
#   ttk/core_modules/framework_api/{torch,tf,onnx}/** → 对应 E2E 框架
#   ttk/core_modules/framework_api/**（其余）→ E2E 全部三框架
#   ttk/core_modules/common/** 等 TTK 公共层 → 全部通路
#   docs/**、*.md、tests/**、运行产物目录    → 无影响（不触发）
#   未识别路径                              → 全部通路（保守，回归门槛要求）
#   完整规则: ./run_affected.sh --list-mapping
#
# 其余参数原样透传给每个 run_*.sh（分组名、--dry-run 等）:
#   ./run_affected.sh ttk/core_modules/kernel --dry-run     # 只打印命令
#   ./run_affected.sh ttk/core_modules/aclnn basic          # aclnn 只跑 basic 分组
#
# 说明: 位置参数中含 "/"、带文件扩展名或仓库内存在的路径按改动路径处理，
#       其余视为透传参数（分组名等）；拿不准时用 --paths 显式声明。
# =============================================================================
set -o pipefail

_SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "$_SDIR/_common.sh"

CANON=(run_kernel.sh run_geir.sh run_aclnn.sh run_e2e_torch.sh run_e2e_tf.sh run_e2e_onnx.sh)

# ---------------- 路径 → 通路 映射（case 自上而下，先精确后宽泛） ----------------
classify_path() {
    case "$1" in
        # —— Kernel 通路 ——
        ttk/core_modules/kernel|ttk/core_modules/kernel/*) echo kernel ;;
        examples/case_store/kernel|examples/case_store/kernel/*) echo kernel ;;
        # —— GEIR 通路（含仓库根 geir/ 的 harness 源与编译产物） ——
        ttk/core_modules/geir|ttk/core_modules/geir/*) echo geir ;;
        examples/case_store/geir|examples/case_store/geir/*) echo geir ;;
        geir|geir/*) echo geir ;;
        # —— ACLNN 通路 ——
        ttk/core_modules/aclnn|ttk/core_modules/aclnn/*) echo aclnn ;;
        examples/case_store/aclnn|examples/case_store/aclnn/*) echo aclnn ;;
        # —— E2E 按框架 ——
        ttk/core_modules/framework_api/torch|ttk/core_modules/framework_api/torch/*) echo e2e-torch ;;
        ttk/core_modules/framework_api/tf|ttk/core_modules/framework_api/tf/*) echo e2e-tf ;;
        ttk/core_modules/framework_api/onnx|ttk/core_modules/framework_api/onnx/*) echo e2e-onnx ;;
        examples/case_store/e2e/torch_cases.csv) echo e2e-torch ;;
        examples/case_store/e2e/tf_cases.csv) echo e2e-tf ;;
        examples/case_store/e2e/onnx_cases.csv) echo e2e-onnx ;;
        # —— E2E 公共（framework_api 其余 + e2e 目录其余资产） ——
        ttk/core_modules/framework_api|ttk/core_modules/framework_api/*) echo e2e-all ;;
        examples/case_store/e2e|examples/case_store/e2e/*) echo e2e-all ;;
        # —— TTK 公共层：全部通路 ——
        ttk/core_modules/common|ttk/core_modules/common/*) echo all ;;
        ttk/core_modules|ttk/core_modules/*) echo all ;;
        ttk/cli|ttk/cli/*) echo all ;;
        ttk/config|ttk/config/*) echo all ;;
        ttk/utilities|ttk/utilities/*) echo all ;;
        ttk/remote|ttk/remote/*) echo all ;;
        ttk/_env.py|ttk/switches.py|ttk/__init__.py|ttk/__main__.py|ttk/_version.py) echo all ;;
        ttk|ttk/*) echo all ;;
        csrc|csrc/*) echo all ;;
        requirements.txt|pyproject.toml) echo all ;;
        # —— 明确无影响：文档 / 自测 / 编排脚本 / 运行产物 ——
        *.md|*.MD|docs|docs/*) echo none ;;
        tests|tests/*) echo none ;;
        scripts|scripts/*) echo none ;;
        log|log/*|msprof|msprof/*|onnx_om|onnx_om/*) echo none ;;
        kernel_meta|kernel_meta/*|oat_reports|oat_reports/*) echo none ;;
        __pycache__|__pycache__/*|*.pyc) echo none ;;
        *.log|*.json|*.xml|combine.csv) echo none ;;
        # —— 未识别：保守起见全部通路 ——
        *) echo all ;;
    esac
}

print_mapping() {
    cat <<'EOF'
路径 → 通路 映射规则（classify_path 自上而下优先匹配）:
  [Kernel]
    ttk/core_modules/kernel/**                    → run_kernel.sh
    examples/case_store/kernel/**
  [GEIR]
    ttk/core_modules/geir/**                       → run_geir.sh
    examples/case_store/geir/**、geir/**（根下 harness 源码/产物）
  [ACLNN]
    ttk/core_modules/aclnn/**                      → run_aclnn.sh
    examples/case_store/aclnn/**
  [E2E 按框架]
    ttk/core_modules/framework_api/torch/**        → run_e2e_torch.sh
    ttk/core_modules/framework_api/tf/**           → run_e2e_tf.sh
    ttk/core_modules/framework_api/onnx/**         → run_e2e_onnx.sh
    examples/case_store/e2e/{torch,tf,onnx}_cases.csv
  [E2E 公共] framework_api 其余 / examples/case_store/e2e 其余 → E2E 全部三框架
  [全部通路] ttk/core_modules/common/**、ttk/{cli,config,utilities,remote}/**、
            ttk 根 *.py、csrc/**、requirements.txt、pyproject.tom
  [无影响]   docs/**、*.md、tests/**、scripts/**（含 scripts/ci 编排脚本）、
            log|msprof|onnx_om|kernel_meta|oat_reports/**、*.log、*.json、*.xml
  [未识别]   → 全部通路（保守：TTK 回归门槛要求任何改动都跑通用例）
EOF
}

# ---------------- 参数形态判定 ----------------
looks_like_path() {
    case "$1" in
        */*) return 0 ;;
        *.py|*.csv|*.sh|*.cpp|*.txt|*.toml|*.md|*.json|*.xml|*.yaml|*.yml) return 0 ;;
    esac
    [ -e "$REPO_ROOT/$1" ] && return 0
    return 1
}

# resolve_path <原始路径>：解析为仓库相对路径（仓库外返回非 0）
resolve_path() {
    local p="$1" rp
    if [[ "$p" == /* ]]; then
        rp="$(realpath -m -- "$p" 2>/dev/null || printf '%s' "$p")"
    elif [ -e "$REPO_ROOT/$p" ] || [[ "$p" == */* ]]; then
        # 带 "/" 或仓库内存在（git 报的删除路径可能不存在）→ 按仓库相对
        rp="$(realpath -m -- "$REPO_ROOT/$p" 2>/dev/null || printf '%s' "$REPO_ROOT/$p")"
    elif [ -e "$PWD/$p" ]; then
        rp="$(realpath -m -- "$PWD/$p" 2>/dev/null || printf '%s' "$PWD/$p")"
    else
        rp="$REPO_ROOT/$p"
    fi
    if [ "$rp" = "$REPO_ROOT" ]; then printf '.'; return 0; fi
    case "$rp" in
        "$REPO_ROOT"/*) printf '%s' "${rp#"$REPO_ROOT/"}"; return 0 ;;
        *) return 1 ;;
    esac
}

# ---------------- 改动路径收集 ----------------
PATHS=()
REPORT=()
CLASSES=()

add_path() {
    local raw="$1" rel cls
    if ! rel="$(resolve_path "$raw")"; then
        REPORT+=("  [仓库外，忽略] $raw")
        return 0
    fi
    cls="$(classify_path "$rel")"
    REPORT+=("  $rel → $cls")
    CLASSES+=("$cls")
}

detect_git_changes() {
    git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1 \
        || die "不在 git 仓库内且未显式传改动路径"
    local entry
    if [ -n "$BASE_REF" ]; then
        # ref 预校验：无效 ref 时 diff 静默零输出 → 误判"无受影响通路"退 0
        git -C "$REPO_ROOT" rev-parse --verify --quiet "$BASE_REF^{commit}" >/dev/null 2>&1 \
            || arg_error "--base '$BASE_REF' 不是有效 git ref"
        while IFS= read -r -d '' f; do add_path "$f"; done \
            < <(git -C "$REPO_ROOT" diff --name-only -z "$BASE_REF" --)
    else
        # "XY " 状态前缀正则：X/Y 均可为空格（仅工作区改动 " M"、仅暂存 "M "）；
        # [[ =~ ]] 内裸空格会破坏分词，须先存入变量再以不带引号的变量匹配
        local status_re='^[MADRCU?!. ][MADRCU?!. ] '
        while IFS= read -r -d '' entry; do
            # status -z 记录为 "XY PATH"；不匹配 "XY " 前缀的记录
            # = rename/copy 的旧路径（PLAIND，无前缀）
            if [[ "$entry" =~ $status_re ]]; then
                add_path "${entry:3}"
            else
                add_path "$entry"
            fi
        done < <(git -C "$REPO_ROOT" status --porcelain -z)
    fi
}

# ---------------- 参数解析 ----------------
BASE_REF=""
PASSTHROUGH=()
# 用法错误统一 exit 2（与未知分组分支一致）
arg_error() { echo "[ERROR] $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
    case "$1" in
        # 缺值校验：$2 不存在时 shift 2 失败且不移动位置参数 → while 死循环
        --base)
            [ $# -ge 2 ] || arg_error "--base 需要一个 git ref 参数，如 --base HEAD~1"
            case "$2" in -*) arg_error "--base 缺少取值（\"$2\" 看起来是另一个选项）";; esac
            BASE_REF="$2"; shift 2 ;;
        --paths)
            [ $# -ge 2 ] || arg_error "--paths 需要一个逗号分隔的路径参数，如 --paths ttk/core_modules/kernel"
            IFS=',' read -r -a _ps <<< "$2"; PATHS+=("${_ps[@]}"); shift 2 ;;
        --list-mapping) print_mapping; exit 0 ;;
        -h|--help) awk 'NR>9{if(/^set -/)exit;print}' "$0"; echo; print_mapping; exit 0 ;;
        -*) PASSTHROUGH+=("$1"); shift ;;
        *)
            if looks_like_path "$1"; then PATHS+=("$1"); else PASSTHROUGH+=("$1"); fi
            shift ;;
    esac
done

if [ "${#PATHS[@]}" -eq 0 ]; then
    detect_git_changes
else
    for p in ${PATHS[@]+"${PATHS[@]}"}; do add_path "$p"; done
fi

# ---------------- 映射汇总 ----------------
echo "=================================================="
echo "改动路径 → 通路 映射（${#PATHS[@]} 个显式路径 / git 自动检测）:"
if [ "${#REPORT[@]}" -eq 0 ]; then
    echo "  （未检测到改动；如需对比某次提交用 --base <ref>，或显式传路径）"
fi
printf '%s\n' "${REPORT[@]:-}"

declare -A WANT=()
for c in ${CLASSES[@]+"${CLASSES[@]}"}; do WANT["$c"]=1; done

SCRIPTS_TO_RUN=()
if [ -n "${WANT[all]:-}" ]; then
    SCRIPTS_TO_RUN=("${CANON[@]}")
else
    if [ -n "${WANT[kernel]:-}" ]; then SCRIPTS_TO_RUN+=(run_kernel.sh); fi
    if [ -n "${WANT[geir]:-}" ]; then SCRIPTS_TO_RUN+=(run_geir.sh); fi
    if [ -n "${WANT[aclnn]:-}" ]; then SCRIPTS_TO_RUN+=(run_aclnn.sh); fi
    if [ -n "${WANT[e2e-all]:-}" ]; then
        SCRIPTS_TO_RUN+=(run_e2e_torch.sh run_e2e_tf.sh run_e2e_onnx.sh)
    else
        if [ -n "${WANT[e2e-torch]:-}" ]; then SCRIPTS_TO_RUN+=(run_e2e_torch.sh); fi
        if [ -n "${WANT[e2e-tf]:-}" ]; then SCRIPTS_TO_RUN+=(run_e2e_tf.sh); fi
        if [ -n "${WANT[e2e-onnx]:-}" ]; then SCRIPTS_TO_RUN+=(run_e2e_onnx.sh); fi
    fi
fi

if [ "${#SCRIPTS_TO_RUN[@]}" -eq 0 ]; then
    echo "受影响通路: 无（改动均为文档/自测/编排脚本/运行产物，或未检测到改动）——不运行测试"
    echo "=================================================="
    exit 0
fi

echo "--------------------------------------------------"
echo "受影响通路（串行执行，ttk 同目录禁止并发）:"
for s in "${SCRIPTS_TO_RUN[@]}"; do
    echo "  - $s ${PASSTHROUGH[*]:+-}"
done
echo "透传参数: ${PASSTHROUGH[*]:-（无）}"
echo "=================================================="

# ---------------- 串行拉起对应 run_*.sh ----------------
PASSED=()
SKIPPED=()
FAILED=()
for s in "${SCRIPTS_TO_RUN[@]}"; do
    echo
    echo "##################################################"
    echo "##### 拉起通路脚本: $s ${PASSTHROUGH[*]:-}"
    echo "##################################################"
    if bash "$_SDIR/$s" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}; then
        PASSED+=("$s")
    else
        rc=$?
        if [ "$rc" -eq 2 ]; then
            SKIPPED+=("$s")
            echo "[NOTE] $s 不支持透传的分组/参数，已跳过（退出码 $rc）"
        else
            FAILED+=("$s")
        fi
    fi
done

echo
echo "=================================================="
echo "通路级汇总: 成功 ${#PASSED[@]}，跳过 ${#SKIPPED[@]}，失败 ${#FAILED[@]}"
if [ "${#SKIPPED[@]}" -gt 0 ]; then
    printf '  - 跳过通路(不支持透传分组): %s\n' "${SKIPPED[@]}"
fi
if [ "${#FAILED[@]}" -gt 0 ]; then
    printf '  - 失败通路: %s\n' "${FAILED[@]}"
fi
echo "=================================================="
[ "${#FAILED[@]}" -eq 0 ]
