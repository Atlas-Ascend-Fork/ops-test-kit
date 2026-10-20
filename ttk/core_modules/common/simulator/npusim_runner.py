#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""NPUSim ``record`` / ``report`` invocation via the CANN-bundled ``cannsim``.

TTK runs the CANN-installed ``cannsim`` entry script
(``$ASCEND_TOOLKIT_HOME/bin/cannsim``) for both ``record`` and ``report``.
Its record format matches the installed camodel, so no separate
npu-simulator repository source is needed.
"""

import contextlib
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


def _is_cannsim_script(path: Path) -> bool:
    """A cannsim entry script references the cannsim package (python shebang
    or an import), rejecting an unrelated executable found on PATH."""
    try:
        head = path.read_bytes()[:512]
    except OSError:
        return False
    return b"python" in head.lower() or b"cannsim" in head.lower()


def locate_cannsim_executable() -> Path:
    """Resolve the CANN-bundled ``cannsim`` entry script.

    Prefers ``$ASCEND_TOOLKIT_HOME/bin/cannsim`` (the CANN-installed cannsim
    whose record format matches the installed camodel), then ``cannsim`` on
    PATH. Raises RuntimeError when no CANN cannsim is available — TTK no
    longer falls back to the npu-simulator repository source.
    """
    import shutil

    candidates = []
    asc_home = os.getenv("ASCEND_TOOLKIT_HOME")
    if asc_home:
        candidates.append(Path(asc_home) / "bin" / "cannsim")
    exe = shutil.which("cannsim")
    if exe:
        candidates.append(Path(exe))
    for cand in candidates:
        if cand.is_file() and _is_cannsim_script(cand):
            return cand.resolve()
    raise RuntimeError(
        "CANN cannsim not found. Install a CANN toolkit that ships cannsim "
        "(check $ASCEND_TOOLKIT_HOME/bin/cannsim) or add cannsim to PATH."
    )


def build_sim_env(sw, extra_pythonpaths: Iterable[str] = ()) -> dict:
    """Environment for the cannsim subprocess (inherits TTK's sourced CANN env)."""
    env = os.environ.copy()
    pythonpath_parts = [sw.root_path, *extra_pythonpaths]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    # Skip record's inter-stage sleeps (used by CI/automation).
    env.setdefault("NPUSIM_NO_DELAY", "1")
    env.setdefault("CANNSIM_NO_DELAY", "1")
    return env


def _cannsim_cmd() -> list:
    # ``sys.executable`` keeps numpy/plotly from the same interpreter as TTK and
    # lets the cannsim import resolve to the CANN-installed package.
    return [sys.executable, str(locate_cannsim_executable())]


def _latest_record_dir(record_out: Path, before: Iterable[str] = ()) -> Path:
    """Return the newest cannsim_*/npusim_* archive NOT present in ``before``.

    The archive prefix is set by the installed cannsim package and differs
    across CANN versions (``cannsim_`` on older toolkits, ``npusim_`` on
    9.2.0+), so accept both rather than hard-coding a single prefix.

    ``clear_case_dir`` keeps ``record_out`` across reruns, so filtering out the
    pre-existing archives prevents a crashed record from silently returning a
    stale prior run's directory.
    """
    prev = set(before)
    dirs = []
    for d in record_out.iterdir():
        if not d.is_dir():
            continue
        if d.name in prev:
            continue
        if not (d.name.startswith("cannsim_") or d.name.startswith("npusim_")):
            continue
        dirs.append(d)
    dirs.sort(key=lambda p: p.stat().st_mtime)
    if not dirs:
        raise RuntimeError(f"cannsim record produced no new archive under {record_out}")
    return dirs[-1]


def _terminate_group(proc: subprocess.Popen, grace_s: float = 2.0) -> None:
    """Stop the whole process group of ``proc`` (npusim + wrapper descendants).

    SIGTERM first, then SIGKILL whatever survives the grace window. Safe when
    the group is already gone: ProcessLookupError is ignored (the group exists
    while any member is alive, so the final SIGKILL reaps stragglers).
    """
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=grace_s)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)


class _GroupSignal(Exception):
    """SIGTERM/SIGHUP 转成异常，复用清理路径。"""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def _run_with_cleanup(cmd: list, env: dict, timeout: Optional[float]) -> subprocess.CompletedProcess:
    """subprocess.run replacement that also reaps npusim's descendants.

    ``subprocess.run(timeout=...)`` kills only the direct child: npusim
    spawns the wrapper via ``bash -c``, which would survive the timeout and
    keep simulating / writing result files. Run npusim in its own session so
    the whole tree shares one process group; on timeout or interrupt the group
    gets SIGTERM then SIGKILL, the pipes are drained and the exception is
    re-raised (callers keep their current timeout semantics).

    TTK 收到外部 SIGTERM/SIGHUP（CI 取消）时会直接退出、漏掉本函数清理。
    运行窗口内注册处理器把这些信号转成 ``_GroupSignal``：走同一套整组回收，
    再恢复默认处置重发，保留信号的退出语义。
    """

    def _handler(signum, frame):
        raise _GroupSignal(signum)

    old_handlers = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGHUP):
            old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _handler)
    except ValueError:
        # signal.signal requires the main thread; keep the legacy
        # timeout/KeyboardInterrupt-only cleanup when not available.
        old_handlers.clear()

    signal_death = False
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except BaseException as exc:
            # TimeoutExpired, KeyboardInterrupt and _GroupSignal alike: stop the
            # whole group, drain the pipes (repeated communicate without input
            # is allowed) and re-raise the original exception so callers keep
            # their current semantics.
            #
            # Draining can block forever only if a descendant escaped the group
            # (setsid/daemonize) and still holds the pipe fds — the wrapper runs
            # camodel in-process and stays in the group, so this does not happen.
            _terminate_group(proc)
            proc.communicate()
            if isinstance(exc, _GroupSignal):
                signal_death = True
                signal.signal(exc.signum, signal.SIG_DFL)
                os.kill(os.getpid(), exc.signum)
            raise
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    finally:
        if not signal_death:
            for sig, prev in old_handlers.items():
                signal.signal(sig, prev)


def run_record(sw, wrapper_path: Path, case_dir: Path, extra_argv: Iterable[str] = ()) -> Path:
    """Run ``record <wrapper> -s <soc> -o <case_dir>/record_out [-n ...] [-f ...]``.

    Returns the archive root ``<case_dir>/record_out/{cannsim,npusim}_<ts>_<label>/``.
    Note: record's exit code is unreliable for TTK kernels (camodel teardown
    may SIGSEGV after the user program finishes), so the caller must decide
    PASS/FAIL from the wrapper-written ``result.json``, not the return code.
    """
    record_out = Path(case_dir) / "record_out"
    record_out.mkdir(parents=True, exist_ok=True)
    # Snapshot pre-existing archives so a crashed rerun cannot return a stale one.
    before = {d.name for d in record_out.iterdir() if d.is_dir()}

    cmd = [
        *_cannsim_cmd(),
        "record",
        str(wrapper_path.resolve()),
        "-s",
        sw.sim_soc_version,
        "-o",
        str(record_out),
    ]
    if getattr(sw, "sim_cores", None):
        cmd += ["-n", sw.sim_cores]
    if getattr(sw, "sim_object_file", None):
        cmd += ["-f", sw.sim_object_file]
    cmd += list(extra_argv)

    logging.info("npusim record: %s", " ".join(cmd))
    proc = _run_with_cleanup(cmd, build_sim_env(sw), sw.proc_timeout or None)
    if proc.stdout:
        logging.debug("npusim record stdout tail: %s", proc.stdout[-2000:])
    if proc.returncode != 0:
        # rc != 0 also covers the benign camodel teardown SIGSEGV; print the
        # captured stderr so a real record failure is distinguishable.
        logging.warning(
            "npusim record returned rc=%s (teardown crashes are expected); stderr tail:\n%s",
            proc.returncode,
            (proc.stderr or "")[-2000:],
        )
    return _latest_record_dir(record_out, before)


def _instr_dir_of(export_dir: Path) -> Path:
    """The directory that directly contains instr.bin for ``report -e``.

    The CANN-bundled cannsim record writes instr.bin at the archive root
    (``{cannsim,npusim}_*/instr.bin``); some layouts nest it under ``record/``,
    so fall back to the archive root when the nested one is absent.
    """
    record_dir = export_dir / "record"
    if (record_dir / "instr.bin").is_file():
        return record_dir
    return export_dir


def run_report(sw, export_dir: Path, output_dir: Optional[Path] = None) -> Optional[Path]:
    """Generate the performance report (trace_core*.json + HTML) for one case.

    Returns the report output directory, or None on failure (caller logs only).
    """
    if not export_dir.is_dir():
        logging.warning("report input missing: %s", export_dir)
        return None
    instr_dir = _instr_dir_of(export_dir)
    if not (instr_dir / "instr.bin").is_file():
        logging.warning("no instr.bin found under %s; skipping report", export_dir)
        return None
    out = output_dir or (export_dir / "report")
    out.mkdir(parents=True, exist_ok=True)

    # Same env as record: skip inter-stage sleeps and keep ttk root on
    # PYTHONPATH so the cannsim package resolves identically.
    env = build_sim_env(sw)
    try:
        # _cannsim_cmd() may raise (cannsim not installed); per the docstring
        # report is best-effort, so the whole invocation stays inside the try.
        cmd = [*_cannsim_cmd(), "report", "-e", str(instr_dir), "-o", str(out)]
        if getattr(sw, "sim_cores", None):
            cmd += ["-n", sw.sim_cores]
        logging.info("npusim report: %s", " ".join(cmd))
        proc = _run_with_cleanup(cmd, env, sw.proc_timeout or None)
    except Exception as exc:  # noqa: BLE001 - report is best-effort
        logging.warning("npusim report failed: %s", exc)
        return None
    if proc.returncode != 0:
        logging.warning(
            "npusim report rc=%s; stdout tail:\n%s\nstderr tail:\n%s",
            proc.returncode,
            (proc.stdout or "")[-2000:],
            (proc.stderr or "")[-2000:],
        )
        return None
    return out
