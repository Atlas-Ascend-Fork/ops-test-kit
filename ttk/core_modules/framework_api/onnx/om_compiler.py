#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
atc 离线编译（onnx → om）+ 编译缓存。

工程约束全部来自实测：
  - 命令钉 --precision_mode=must_keep_origin_dtype：atc 默认 force_fp16 静默
    降精度（f32 值 >65504 可得 inf 而 desc 仍报 float32）；
  - 失败载荷合并捕获 stdout+stderr 尾部：atc 诊断走 stdout、stderr 常为空；
  - 子进程 PYTHONPATH 显式归一化：保证 $ASCEND_TOOLKIT_HOME/python/site-packages
    在列（防 EC0010 No module named 'tbe'），外部条目可保留；
  - 产物路径探测实际文件名：动态 shape om 带 _linux_aarch64 后缀，不可假设
    <output>.om；
  - soc_version 由调用方注入 DSMI 完整芯片名（如 Ascend910_9362，不走
    soc_series 短名——short_soc 会被 atc 拒绝）。

缓存照抄 geir/compiler.py compile_op 范式：flock 独占 + stamp 内容比对
（md5(onnx 字节 + soc_version + atc 参数串)），缓存目录
<work_dir>/<md5>/（调用方传 <root_path>/onnx_cache/<op_type>）。
"""

__all__ = ["OmniCompileError", "compile_om"]

import fcntl
import glob
import hashlib
import logging
import os
import subprocess

_ATC_TIMEOUT_S = 300
_FAILURE_TAIL_BYTES = 4096


class OmniCompileError(RuntimeError):
    """atc 编译失败（失败码前缀 ATC_*，载荷原文透传）。"""


def _find_atc():
    """atc 定位：ttk/_env._find_ascend_root() + bin/atc。"""
    from ttk._env import _find_ascend_root

    root = _find_ascend_root()
    if not root:
        return None, None
    for candidate in (os.path.join(root, "bin", "atc"), os.path.join(root, "bin", "atc.sh")):
        if os.path.isfile(candidate):
            return candidate, root
    return None, root


def _normalized_env(ascend_root: str) -> dict:
    """继承父进程环境 + PYTHONPATH 归一化（CANN site-packages 必须在列）。"""
    env = dict(os.environ)
    cann_py = os.path.join(ascend_root, "python", "site-packages")
    pythonpath = env.get("PYTHONPATH", "")
    entries = [p for p in pythonpath.split(":") if p]
    if cann_py not in entries:
        entries.insert(0, cann_py)
    env["PYTHONPATH"] = ":".join(entries)
    return env


def _shape_option(input_shapes: dict) -> str:
    parts = []
    for name, shape in input_shapes.items():
        parts.append(f"{name}:{','.join(str(d) for d in shape)}")
    return ";".join(parts)


def _atc_args_string(input_shapes: dict) -> str:
    """缓存键的参数串（命令模板的稳定投影——入缓存键的参数须与实际命令一致）。"""
    return (
        f"--framework=5 --precision_mode=must_keep_origin_dtype --input_shape={_shape_option(input_shapes)} --log=error"
    )


def _detect_om_path(case_dir: str) -> str:
    """产物路径探测：静态 <output>.om / 动态 <output>_linux_aarch64.om。"""
    exact = os.path.join(case_dir, "model.om")
    if os.path.isfile(exact):
        return exact
    for candidate in sorted(glob.glob(os.path.join(case_dir, "model_*.om"))):
        return candidate
    return ""


def _plugin_fingerprint(ascend_root):
    """atc ONNX 插件目录指纹（name:size:mtime_ns 逐 .so 拼接）。

    缓存键原仅含 onnx 字节 + soc + atc 参数——插件 .so 热替换/重编后同键仍
    命中旧 om（修复态假 PASS 实证）。mtime_ns 捕捉 cp 热替换与重编；
    ``.bak`` 等非 ``.so`` 后缀不参与（备份文件的在场/离场不抖动键）。目录不可
    达/不可读返回常量：不阻塞编译，缓存退化为原内容+soc+参数键语义。
    """
    if not ascend_root:
        return "no-ascend-root"
    plugin_dir = os.path.join(ascend_root, "opp", "built-in", "framework", "onnx")
    try:
        parts = []
        for name in sorted(os.listdir(plugin_dir)):
            if not name.endswith(".so"):
                continue
            stat = os.stat(os.path.join(plugin_dir, name))
            parts.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
        return ";".join(parts) if parts else "no-plugins"
    except OSError:
        return "unreadable-plugin-dir"


def _cache_key(model_data, soc_version, args_string, fingerprint):
    """缓存键 = md5(onnx 字节 + soc + atc 参数 + 插件指纹)——含插件指纹。"""
    return hashlib.md5(  # noqa: S324  # 非安全用途：缓存键校验和
        model_data + b"|" + soc_version.encode() + b"|" + args_string.encode() + b"|" + fingerprint.encode()
    ).hexdigest()


def compile_om(onnx_path: str, input_shapes: dict, soc_version: str, work_dir: str) -> str:
    """onnx → om；返回 om 路径。失败抛 OmniCompileError（stdout+stderr 尾部透传）。

    onnx_path: 输入模型文件（内容参与缓存键，被复制进缓存目录）；
    input_shapes: {"input_0": (2,3), ...}（与模型输入名一致）；
    work_dir: 缓存根目录（如 <root_path>/onnx_cache/<op_type>）。
    缓存键含 atc ONNX 插件目录指纹——.so 热替换/重编后自动失效。
    """
    atc, ascend_root = _find_atc()
    if not atc:
        raise OmniCompileError(
            f"ATC_NOT_FOUND: 未找到 atc（探测根 {ascend_root or '未探测到 Ascend 安装'}）。"
            "请安装 CANN 工具链并在运行前 source set_env.sh（或经 ASCEND_TOOLKIT_HOME 让 TTK 自动探测）"
        )

    with open(onnx_path, "rb") as f:
        model_data = f.read()
    args_string = _atc_args_string(input_shapes)
    key = _cache_key(model_data, soc_version, args_string, _plugin_fingerprint(ascend_root))
    case_dir = os.path.join(work_dir, key)
    os.makedirs(case_dir, exist_ok=True)
    lock_path = os.path.join(work_dir, f"{key}.lock")
    stamp_path = os.path.join(case_dir, "stamp")

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            om_path = _detect_om_path(case_dir)
            cached_stamp = ""
            if os.path.isfile(stamp_path):
                with open(stamp_path) as sf:
                    cached_stamp = sf.read().strip()
            if om_path and cached_stamp == key:
                logging.info("onnx om cached: %s", om_path)
                return om_path
            logging.info("onnx om cache miss (key=%s), compiling via atc", key)

            model_onnx = os.path.join(case_dir, "model.onnx")
            with open(model_onnx, "wb") as f:
                f.write(model_data)
            output_prefix = os.path.join(case_dir, "model")
            cmd = [
                atc,
                "--framework=5",
                f"--model={model_onnx}",
                f"--output={output_prefix}",
                f"--soc_version={soc_version}",
                "--precision_mode=must_keep_origin_dtype",
                "--log=error",
            ]
            if input_shapes:
                cmd.append(f"--input_shape={_shape_option(input_shapes)}")
            logging.info("onnx atc compile: %s", " ".join(cmd))
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=_ATC_TIMEOUT_S,
                    env=_normalized_env(ascend_root),
                    cwd=case_dir,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise OmniCompileError(f"ATC_COMPILE_FAILED: atc 超时（{_ATC_TIMEOUT_S}s）：{exc}") from exc

            om_path = _detect_om_path(case_dir)
            if result.returncode != 0 or not om_path:
                combined = (result.stdout or "") + "\n" + (result.stderr or "")
                tail = combined[-_FAILURE_TAIL_BYTES:].strip()
                raise OmniCompileError(f"ATC_COMPILE_FAILED: atc rc={result.returncode}\n{tail}")

            with open(stamp_path, "w") as sf:
                sf.write(key)
            return om_path
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
