# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""common/logging pin 测试。

钉住当前日志布局契约：
- 主进程/引导（testcase_name=None）：不落文件，仅 stdout
- per-case（testcase_name 非空）：ttk.log（非 COMPARE 全级别）+ ttk-compare.log
  （COMPARE 专属），均为覆写（w）——复跑不累积
- COMPARE 级（debugc）独立成档：compare_log Index/Diff 行过滤的数据源
"""

import logging
import os
import time

import pytest

from ttk.core_modules.common.logging import build_single_log_dir, default_logging_config


@pytest.fixture
def restore_root_logger():
    handlers = list(logging.getLogger().handlers)
    level = logging.getLogger().level
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)
    root.setLevel(level)


def test_main_process_writes_no_file(tmp_path, restore_root_logger):
    """主进程/引导（testcase_name=None）：不产生任何日志文件。"""
    default_logging_config(file_handler=True, log_dir=str(tmp_path))
    logging.getLogger().info("pin-info")
    logging.getLogger().debug("pin-debug")
    assert list(tmp_path.iterdir()) == []


def test_per_case_two_files_no_level_split(tmp_path, restore_root_logger):
    """per-case：恰好两个文件——ttk.log（非 COMPARE 混写）+ ttk-compare.log。"""
    default_logging_config(file_handler=True, testcase_name="pin_case", log_dir=str(tmp_path))
    logging.getLogger().info("pin-info")
    logging.getLogger().debug("pin-debug")
    logging.getLogger().error("pin-error")
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["ttk-compare.log", "ttk.log"]
    content = (tmp_path / "ttk.log").read_text(encoding="utf-8")
    assert "pin-info" in content
    assert "pin-debug" in content  # DEBUG 也进同一文件
    assert "pin-error" in content
    assert "[INFO]" in content
    assert "[DEBUG]" in content


def test_per_case_append_within_run(tmp_path, restore_root_logger):
    """per-case 追加：同运行多任务（编译/执行）重配共写，互不截断。"""
    target = tmp_path / "ttk.log"
    default_logging_config(file_handler=True, testcase_name="pin_case", log_dir=str(tmp_path))
    logging.getLogger().info("compile-line")
    default_logging_config(file_handler=True, testcase_name="pin_case", log_dir=str(tmp_path))
    logging.getLogger().info("exec-line")
    content = target.read_text(encoding="utf-8")
    assert "compile-line" in content, "后续任务重配不得截断先前任务日志"
    assert "exec-line" in content


def test_reset_case_log_rerun_fresh(tmp_path):
    """复跑覆写语义：主进程重置只清 ttk.log；compare 档（跨复跑历史）不动。"""
    from ttk.core_modules.common.logging import reset_case_log

    (tmp_path / "ttk.log").write_text("STALE\n", encoding="utf-8")
    (tmp_path / "ttk-compare.log").write_text("HISTORY\n", encoding="utf-8")
    reset_case_log(str(tmp_path))
    assert (tmp_path / "ttk.log").read_text(encoding="utf-8") == ""
    assert (tmp_path / "ttk-compare.log").read_text(encoding="utf-8") == "HISTORY\n"


def test_compare_level_in_dedicated_file(tmp_path, restore_root_logger):
    """COMPARE 级（debugc）独立进 ttk-compare.log——compare_log Index/Diff 行过滤的数据源。"""
    default_logging_config(file_handler=True, testcase_name="pin_case", log_dir=str(tmp_path))
    logging.debugc("Index: 0, Diff: pin")
    logging.getLogger().info("non-compare-line")
    compare_content = (tmp_path / "ttk-compare.log").read_text(encoding="utf-8")
    main_content = (tmp_path / "ttk.log").read_text(encoding="utf-8")
    assert "Index: 0, Diff: pin" in compare_content
    assert "[COMPARE]" in compare_content
    assert "non-compare-line" not in compare_content  # 非 COMPARE 不进 compare 档
    assert "Index: 0, Diff: pin" not in main_content  # COMPARE 不进主档
    assert "non-compare-line" in main_content


def test_debugc_root_call_with_plain_string(tmp_path, restore_root_logger):
    """模块级 logging.debugc 纯文本调用不丢记录（args 传递回归）。"""
    default_logging_config(file_handler=True, testcase_name="pin_case", log_dir=str(tmp_path))
    logging.debugc("plain-compare-message")
    for h in logging.getLogger().handlers:
        h.flush()
    assert "plain-compare-message" in (tmp_path / "ttk-compare.log").read_text(encoding="utf-8")


def test_compare_log_appends_across_reruns(tmp_path, restore_root_logger):
    """compare 档追加 + run start 标记：复跑保留历史；增量读取只取末段。"""
    from ttk.core_modules.common.comparison.compare_log import read_compare_log_failures

    # 第一次运行
    default_logging_config(file_handler=True, testcase_name="pin_case", log_dir=str(tmp_path))
    logging.debugc("Index: 0, Diff: run1")
    for h in logging.getLogger().handlers:
        h.flush()
    # 复跑（同用例）
    default_logging_config(file_handler=True, testcase_name="pin_case", log_dir=str(tmp_path))
    logging.debugc("Index: 0, Diff: run2")
    for h in logging.getLogger().handlers:
        h.flush()

    content = (tmp_path / "ttk-compare.log").read_text(encoding="utf-8")
    assert "Diff: run1" in content  # 历史保留
    assert "Diff: run2" in content
    assert content.count("===== run start") == 2  # 每次运行一个标记

    lines, _ = read_compare_log_failures(str(tmp_path / "ttk-compare.log"))
    assert len(lines) == 1  # 只取末段（run2）
    assert "Diff: run2" in lines[0]
    assert "Diff: run1" not in lines[0]


def test_cann_stdout_capture_to_case_log(tmp_path, restore_root_logger, monkeypatch, capfd):
    """CANN slog 打屏开启：fd 1 重定向到用例 ttk.log——C 层直写进档，tail 代理打屏。"""
    from ttk.core_modules.common import logging as ttk_logging

    monkeypatch.setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1")
    fd_backup = os.dup(1)
    try:
        ttk_logging.default_logging_config(file_handler=True, testcase_name="pin_case", log_dir=str(tmp_path))
        assert ttk_logging._CAPTURE_STATE["installed"]
        # C 层等价写：直接 os.write fd 1（不经 Python logging，无 TTK 前缀直写进档）
        os.write(1, b"[INFO]SLOGC fake cann line\n")
        content = ""
        for _ in range(50):
            time.sleep(0.02)
            content = (tmp_path / "ttk.log").read_text(encoding="utf-8")
            if "fake cann line" in content:
                break
        assert "fake cann line" in content  # 原样进档（CANN 行自带级别/组件前缀）
        # 打屏由镜像线程代理（真实终端收到同一行）
        seen = ""
        for _ in range(50):
            time.sleep(0.02)
            seen += capfd.readouterr().out
            if "fake cann line" in seen:
                break
        assert "fake cann line" in seen
    finally:
        ttk_logging._reset_stdout_capture()
        os.dup2(fd_backup, 1)
        os.close(fd_backup)


def test_cann_tee_off_by_default(tmp_path, restore_root_logger, monkeypatch):
    """打屏关闭：不装 tee，fd 写入不进日志文件。"""
    from ttk.core_modules.common import logging as ttk_logging

    monkeypatch.setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "0")
    fd_backup = os.dup(1)
    try:
        ttk_logging.default_logging_config(file_handler=True, testcase_name="pin_case", log_dir=str(tmp_path))
        assert not ttk_logging._CAPTURE_STATE["installed"]
        os.write(1, b"[INFO]SLOGC should not be captured\n")
        time.sleep(0.1)
        content = (tmp_path / "ttk.log").read_text(encoding="utf-8")
        assert "should not be captured" not in content
    finally:
        os.dup2(fd_backup, 1)
        os.close(fd_backup)


def test_mirror_screen_no_partial_line_loss(tmp_path, restore_root_logger, monkeypatch, capfd):
    """跨读块的不完整行不丢：一行分两段写，屏幕收到完整行。"""
    from ttk.core_modules.common import logging as ttk_logging

    monkeypatch.setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1")
    fd_backup = os.dup(1)
    try:
        ttk_logging.default_logging_config(file_handler=True, testcase_name="pin_case", log_dir=str(tmp_path))
        os.write(1, b"[INFO]SLOGC part1-")  # 无换行——留在镜像线程 pending
        time.sleep(0.3)  # 跨过至少一个镜像读轮
        os.write(1, b"part2\n")  # 补全同一行
        seen = ""
        for _ in range(50):
            time.sleep(0.02)
            seen += capfd.readouterr().out
            if "part1-part2" in seen:
                break
        assert "part1-part2" in seen, "不完整行前段不得丢失"
    finally:
        ttk_logging._reset_stdout_capture()
        os.dup2(fd_backup, 1)
        os.close(fd_backup)


def test_release_case_stdout_stops_mirror(tmp_path, restore_root_logger, monkeypatch, capfd):
    """任务结束释放：镜像排干旧档后挂起——后续写入进档但不再上屏（防双镜像重复）。"""
    from ttk.core_modules.common import logging as ttk_logging

    monkeypatch.setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1")
    fd_backup = os.dup(1)
    try:
        ttk_logging.default_logging_config(file_handler=True, testcase_name="pin_case", log_dir=str(tmp_path))
        os.write(1, b"[INFO]SLOGC before-release\n")
        seen = ""
        for _ in range(50):
            time.sleep(0.02)
            seen += capfd.readouterr().out
            if "before-release" in seen:
                break
        assert "before-release" in seen

        ttk_logging.release_case_stdout()
        # 释放异步生效（镜像 ≤50ms 注意到并排干旧档）——排干窗口内的写入
        # 仍上屏（任务尾部 flush 兜底），生效后的写入不再转发
        time.sleep(0.2)
        capfd.readouterr()
        os.write(1, b"[INFO]SLOGC after-release\n")  # 仍进档（fd 1 未还原）
        time.sleep(0.3)
        tail = capfd.readouterr().out
        assert "after-release" not in tail, "释放后镜像不得再转发"
        assert "after-release" in (tmp_path / "ttk.log").read_text(encoding="utf-8"), "进档不受释放影响"
    finally:
        ttk_logging._reset_stdout_capture()
        os.dup2(fd_backup, 1)
        os.close(fd_backup)


def test_build_single_log_dir_layout(tmp_path):
    """per-case 目录布局：{root}/log/{mode}/{testcase_name}（镜像 msprof 用例粒度）。"""
    d = build_single_log_dir("kernel", "add_example_01", str(tmp_path))
    assert d == str(tmp_path / "log" / "kernel" / "add_example_01")
    assert (tmp_path / "log" / "kernel" / "add_example_01").is_dir()
    # 模式目录映射（op → kernel / framework-api → e2e）
    assert build_single_log_dir("op", "case_x", str(tmp_path)).endswith("log/kernel/case_x")
    assert build_single_log_dir("framework-api", "case_x", str(tmp_path)).endswith("log/e2e/case_x")
