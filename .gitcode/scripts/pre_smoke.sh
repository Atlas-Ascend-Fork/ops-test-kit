#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -e
set -o pipefail

# =============================================================================
# pre_smoke.sh —— PreSmoke 冒烟入口（由 .gitcode/workflows/ops-test-kit_action.yml
#                 的 PreSmoke 环节调用，运行于 NPU 冒烟机容器内）
#
# 流程（对齐 ops-math pre_smoke.sh 骨架，测试主体换成 TTK case_store 冒烟）:
#   1. 加载 CANN 环境（CANN_ENV 可覆盖，默认 /usr/local/Ascend/cann/set_env.sh）
#   2. 读取 OBS 下载的 pr_filelist.txt（PR 改动文件清单，每行一个仓库相对路径）
#   3. scripts/ci/run_affected.sh 按改动路径自动判定受影响通路并串行执行
#   4. scripts/ci/check_results.sh 逐例校验本轮结果 vs 预期规格（--since 只认本轮日志）
#   5. 打包上传 CANN slog 与 TTK 用例日志到 OBS（plog 目录）
#   6. 检查 NPU 状态，依据执行/校验退出码给出冒烟结论
#
# 环境变量（流水线注入或运行镜像提供）:
#   WORKSPACE     代码检出目录（必填）
#   repo_name / pr_id / obs_smoke_path / AK / SK   流水线注入
#   CANN_ENV      CANN set_env.sh 路径（默认 /usr/local/Ascend/cann/set_env.sh）
#   TTK_PYTHON    Kernel 通路解释器（run_kernel.sh；需 sympy + en-dtypes）
#   TORCHNPU_ENV  GEIR/ACLNN/onnx 通路 Python 环境根（run_geir.sh 等）
#   TF_ENV        E2E tf 通路 Python 环境根（run_e2e_tf.sh）
#   PYENV_CI      E2E torch 通路 site-packages 路径（run_e2e_torch.sh）
#   以上解释器变量未设置时 run_*.sh 回落到各自内置默认路径，请确保运行镜像可用。
# =============================================================================

echo "repo_name=${repo_name}"
echo "pr_id=${pr_id}"
echo "obs_smoke_path=${obs_smoke_path}"

shopt -s extglob

cd "${WORKSPACE:?}"

log() {
  local dt
  dt=$(date '+%Y%m%d.%H%M%S')
  echo "===================================================================="
  echo "$dt : $*"
  echo "===================================================================="
}

log "init test case, please wait ..."
rm -rf /root/ascend/log
rm -rf ./log
rm -f ./run_test.log

# ==============================
# CANN 环境加载
# export 供 run_*.sh/_common.sh 子进程继承：否则 _common.sh 回退到开发机默认路径
# （冒烟机上该默认不存在，六通路会在环境准备阶段集体失败）
# ==============================
export CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann/set_env.sh}"
echo "CANN_ENV=${CANN_ENV}"
if [ -f "${CANN_ENV}" ]; then
    # shellcheck disable=SC1090
    source "${CANN_ENV}"
else
    echo "[WARN] CANN set_env.sh 不存在: ${CANN_ENV}（可 export CANN_ENV=<路径> 覆盖）"
fi

export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0

# ==============================
# 运行机解释器环境探测与自动定位
# （TTK_PYTHON / TORCHNPU_ENV / TF_ENV / PYENV_CI 未设置时,按 conda env 的
#   site-packages 装配自动定位;找不到则保持未设置,run_*.sh 回退默认路径并在
#   check 处报出明确错误——下方探测输出同时可作为手动配置的依据）
# ==============================
echo "---- runner env discovery ----"
if command -v python3 >/dev/null 2>&1; then
    command -v python3
    python3 --version 2>&1
else
    echo "python3: 不在 PATH"
fi
if command -v conda >/dev/null 2>&1; then
    conda env list 2>/dev/null | head -n 10
fi

# —— 各 conda env 关键包装配探测(纯文件系统检查,不做 import,安全快速)——
for _sp in /opt/conda/lib/python*/site-packages /root/miniconda3/lib/python*/site-packages /opt/conda/envs/*/lib/python*/site-packages /root/miniconda3/envs/*/lib/python*/site-packages; do
    if [ ! -d "$_sp" ]; then
        continue
    fi
    _found=""
    for _pkg in sympy en_dtypes onnx ml_dtypes torch torch_npu tensorflow npu_device mindspore; do
        if [ -d "$_sp/$_pkg" ]; then
            _found="$_found $_pkg"
        fi
    done
    if [ -n "$_found" ]; then
        echo "[site-packages: $_sp]$_found"
    else
        echo "[site-packages: $_sp] (无关键包)"
    fi
done

# —— CANN 核心栈 python ABI 版本(acl/tbe/cann_ops_transformer 按此编译)——
# CANN 9.3 实测:python3.9 导入 acl/tbe 成功;python3.12 加载 cp39 扩展崩溃(GEIR serve)
# 或报版本不匹配(causal_conv1d)。可用环境变量 CANN_PY_VER 覆盖(CANN 版本变更时调整)。
CANN_PY_VER="${CANN_PY_VER:-3.9}"

# —— 按装配自动定位四个通路解释器(显式设置过的变量不覆盖)——
# kernel/GEIR/ACLNN/onnx 通路(TTK_PYTHON/TORCHNPU_ENV)优先取与 CANN python
# 版本匹配的 env;e2e torch(PYENV_CI)取含 torch+torch_npu 的首个 env;tf 取含
# tensorflow 的 env(无则保持未设置,run 脚本在 check 处报出明确错误)。
for _root in /opt/conda/envs/* /root/miniconda3/envs/* /opt/conda /root/miniconda3; do
    if [ ! -d "$_root" ]; then
        continue
    fi
    for _sp in "$_root"/lib/python*/site-packages; do
        if [ ! -d "$_sp" ]; then
            continue
        fi
        case "$_sp" in
            */lib/python${CANN_PY_VER}/site-packages) _cann_match=1 ;;
            *) _cann_match=0 ;;
        esac
        if [ "$_cann_match" = "1" ]; then
            # kernel 通路: sympy(en_dtypes 由下方补装保障)
            if [ -z "${TTK_PYTHON:-}" ] && [ -d "$_sp/sympy" ]; then
                export TTK_PYTHON="$_root/bin/python3"
            fi
            # GEIR/ACLNN/onnx 通路: sympy(其余依赖由下方补装保障)
            if [ -z "${TORCHNPU_ENV:-}" ] && [ -d "$_sp/sympy" ]; then
                export TORCHNPU_ENV="$_root"
            fi
        fi
        # E2E torch 通路: torch + torch_npu(site-packages 目录语义,与系统 python3 配合)
        if [ -z "${PYENV_CI:-}" ] && [ -d "$_sp/torch" ] && [ -d "$_sp/torch_npu" ]; then
            export PYENV_CI="$_sp"
        fi
        # E2E tf 通路: tensorflow
        if [ -z "${TF_ENV:-}" ] && [ -d "$_sp/tensorflow" ]; then
            export TF_ENV="$_root"
        fi
    done
done

# —— 依赖预检(只诊断不安装:流水线不做安装操作,依赖须预装在镜像中)——
if [ -n "${TORCHNPU_ENV:-}" ] && [ -x "${TORCHNPU_ENV}/bin/python3" ]; then
    _cann_py="${TORCHNPU_ENV}/bin/python3"
    for _dep in decorator scipy psutil yaml en_dtypes onnx; do
        if ! "$_cann_py" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$_dep') else 1)" 2>/dev/null; then
            echo "[WARN] ${TORCHNPU_ENV} 缺少依赖: ${_dep}(请预装进镜像: pip install decorator scipy psutil PyYAML en-dtypes onnx==1.13.1)"
        fi
    done
fi
if [ -n "${PYENV_CI:-}" ] && [ -d "${PYENV_CI}" ]; then
    # e2e torch 解释器为系统 python3,PYENV_CI 为其 PYTHONPATH
    _p12_root="$(dirname "$(dirname "${PYENV_CI%/site-packages}")")"
    if [ -x "${_p12_root}/bin/python3" ] && ! "${_p12_root}/bin/python3" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('en_dtypes') else 1)" 2>/dev/null; then
        echo "[WARN] ${_p12_root} 缺少依赖: en_dtypes(e2e torch fp8 用例需要,请预装进镜像)"
    fi
fi
echo "CANN_PY_VER=${CANN_PY_VER}"
echo "TTK_PYTHON=${TTK_PYTHON:-<未设置>}"
echo "TORCHNPU_ENV=${TORCHNPU_ENV:-<未设置>}"
echo "TF_ENV=${TF_ENV:-<未设置>}"
echo "PYENV_CI=${PYENV_CI:-<未设置>}"
echo "------------------------------"

# ==============================
# PR 改动清单（OBS 下载，每行一个仓库相对路径）
# ==============================
FILELIST=pr_filelist.txt
if [ ! -f "${FILELIST}" ]; then
    echo "File ${FILELIST} does not exist, no need to execute smoke test task"
    exit 0
fi

# 去除 CR 与空白行后合并为逗号分隔（run_affected.sh --paths 的入参形态）；
# || true 兜底：全空清单时 grep -v 返回 1，pipefail + set -e 会终止脚本，
# 而此处设计意图是得到空串交给下方 -z 判断后 exit 0
PATHS_CSV="$(tr -d '\r' < "${FILELIST}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -v '^$' | paste -sd, - || true)"
if [ -z "${PATHS_CSV}" ]; then
    echo "No changed files in ${FILELIST}, no need to execute smoke test task"
    exit 0
fi
echo "changed files: ${PATHS_CSV}"

if [ ! -f scripts/ci/run_affected.sh ]; then
    echo "[ERROR] scripts/ci/run_affected.sh 不存在——PR 基线需包含 scripts/ci 冒烟脚本套件（请 rebase 到最新 master）"
    exit 1
fi

# 只认可本轮日志（check_results --since 的基准时刻）
SMOKE_START="$(date '+%Y-%m-%d %H:%M:%S')"

# ==============================
# 运行受影响通路（改动路径 → 通路 → 串行执行）
# ==============================
log "start run test case, please wait ..."
run_rc=0
bash scripts/ci/run_affected.sh --paths "${PATHS_CSV}" 2>&1 | tee -a ./run_test.log || run_rc=$?

# ==============================
# 用例结果 vs 预期规格校验（仅本轮受影响通路）
# ==============================
# 从 run_affected.sh 输出解析实际拉起的通路脚本（"  - run_*.sh" 计划清单行与
# "##### 拉起通路脚本: run_*.sh" 头部行两种形态均认），映射为 check_results.sh 通路名
chk_rc=0
# 注意末尾 || true 兜底：纯文档/测试/脚本类 PR 时 run_test.log 中无通路脚本行，
# grep 无匹配返回 1，pipefail + set -e 会在赋值处终止脚本，导致下方 elif（跳过
# 校验）与 else（保守失败提示）成为死代码，且日志上传/NPU 检查被跳过
PW_ARGS="$(grep -oE '(拉起通路脚本: |^  - )run_[a-z0-9_]+\.sh' ./run_test.log 2>/dev/null \
    | grep -oE 'run_[a-z0-9_]+\.sh' | sort -u \
    | sed 's/^run_//; s/\.sh$//' | tr '_' '-' || true)"
if [ -n "${PW_ARGS}" ]; then
    log "checking test results ..."
    # shellcheck disable=SC2086
    bash scripts/ci/check_results.sh --since "${SMOKE_START}" ${PW_ARGS} 2>&1 | tee -a ./run_test.log || chk_rc=$?
elif grep -q "受影响通路: 无" ./run_test.log 2>/dev/null; then
    echo "no affected pathway (docs/tests/scripts only), skip result check"
else
    echo "[ERROR] 无法从 run_test.log 解析受影响通路（run_affected.sh 输出格式变化？），保守判定失败"
    chk_rc=1
fi

# ==============================
# 打包并上传日志（CANN slog + TTK 用例日志）
# ==============================
log "packing and uploading logs ..."
mkdir -p /root/ascend
if [ -d /root/ascend/log ]; then
    tar -zcf slog.tar.gz -C /root/ascend log
    obs_key=${obs_smoke_path}/plog/slog.tar.gz
    # upload plog
    if python3 /home/upload.py --bucket-name "ascend-ci" --action upload \
        --local-file "slog.tar.gz" --obs-object-key "${obs_key}"; then
        echo "::set-output var=plog_url:https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_key}"
    fi
else
    echo "[WARN] /root/ascend/log 不存在，跳过 slog 打包上传"
fi

TTKLOG_ITEMS=()
if [ -d ./log ]; then TTKLOG_ITEMS+=(./log); fi
if [ -f ./run_test.log ]; then TTKLOG_ITEMS+=(./run_test.log); fi
if [ "${#TTKLOG_ITEMS[@]}" -gt 0 ]; then
    tar -zcf ttklog.tar.gz "${TTKLOG_ITEMS[@]}"
    obs_key=${obs_smoke_path}/plog/ttklog.tar.gz
    if python3 /home/upload.py --bucket-name "ascend-ci" --action upload \
        --local-file "ttklog.tar.gz" --obs-object-key "${obs_key}"; then
        echo "::set-output var=ttklog_url:https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_key}"
    fi
fi

# ==============================
# 检查 NPU 状态
# ==============================
log "checking NPU status ..."
mkdir -p ./npu_log
# || true 兜底：NPU 异常(dcmi 初始化失败)时 npu-smi 返回非零，set -e 会在状态
# 采集处终止脚本，导致最终冒烟结论段不执行——状态检查本身是诊断信息，
# 只落盘不参与判定，不允许它掩盖 verdict
npu-smi info 2>&1 | tee ./npu_log/npu_info.log || true
if grep "dcmi module initialize failed" "./npu_log/npu_info.log"; then
    date_time=$(date '+%Y%m%d.%H%M%S')
    echo "$date_time : ${repo_name}_${pr_id} dcmi module initialize failed" >> "./npu_log/$(date +%Y%m%d).log"
fi

# ==============================
# 冒烟结论
# ==============================
log "smoke verdict ..."
date_time=$(date '+%Y%m%d.%H%M%S')
if [ "${run_rc}" -ne 0 ] || [ "${chk_rc}" -ne 0 ]; then
    echo "$date_time : run test case failed (run_rc=${run_rc}, check_rc=${chk_rc})"
    exit 1
else
    echo "$date_time : run test case success"
fi
