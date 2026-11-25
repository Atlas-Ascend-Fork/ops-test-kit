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
from typing import Iterable, List, Optional, Tuple


def _is_cannsim_script(path: Path) -> bool:
    """A cannsim entry script imports the cannsim/npusim package main module.

    CANN ships the entries as python launchers (``from npusim.main import
    main`` / ``from cannsim.main import main``); matching that import keeps
    unrelated python programs named ``npusim``/``cannsim`` on PATH from
    hijacking the pick — a bare ``python``-in-shebang check used to pass
    any script, so a bogus PATH ``npusim`` won over a valid CANN
    ``cannsim`` and every testcase died with SIM_RECORD_FAILED.
    """
    import re

    try:
        head = path.read_bytes()[:512]
    except OSError:
        return False
    # tolerate CANN-internal module paths under the two package roots
    return re.search(rb"(?:from|import) +(?:npusim|cannsim)[.\w]*", head) is not None


def locate_cannsim_executable() -> Path:
    """Resolve the CANN-bundled ``cannsim``/``npusim`` entry script.

    Prefers ``$ASCEND_TOOLKIT_HOME/bin/npusim`` (the cannsim command was
    renamed to npusim and is deprecated on newer toolkits), then the
    CANN-installed ``cannsim`` whose record format matches the installed
    camodel, then either name on PATH. Raises RuntimeError when no CANN
    cannsim is available. TTK no longer falls back to the npu-simulator
    repository source.
    """
    import shutil

    names = ("npusim", "cannsim")
    candidates = []
    asc_home = os.getenv("ASCEND_TOOLKIT_HOME")
    if asc_home:
        candidates.extend(Path(asc_home) / "bin" / name for name in names)
    for name in names:
        exe = shutil.which(name)
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
    # The helper's per-launch DB relies on intercepting aclrt* launches; TTK's
    # wrapper dlopens the runtime libs directly and misses that. Per-chip DB
    # mode (same as ktest) writes the DB from simulation callbacks instead.
    env.setdefault("CANNSIM_HELPER_PER_CHIP_DB", "1")
    # The helper registers its camodel callbacks 10s late by default; most
    # TTK kernels finish sooner, leaving the DB without instructions (report
    # then dies with "float division by zero"). Shrink the delay; explicit
    # user settings win. Older helpers ignore the variable.
    env.setdefault("CANNSIM_HELPER_REGISTER_DELAY_MS", "100")
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
    # NOTE: sim_object_file is deliberately NOT passed to record -- cannsim
    # record (>= 0.7 / CANN 9.3) rejects unknown "-f"; run_report forwards it to
    # ``report --object-file`` instead.
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


def _find_instr_bin_dir(export_dir: Path) -> Optional[Path]:
    """Legacy instr.bin holder directory (``record/`` when nested), or None."""
    for candidate in (export_dir / "record", export_dir):
        if (candidate / "instr.bin").is_file():
            return candidate
    return None


def _has_flat_contract_db(export_dir: Path) -> bool:
    """True when flat contract DBs exist (root / ``record/`` *.db, or ``cannsim_helper_dbs/``)."""
    record_dir = export_dir / "record"
    roots = [export_dir, record_dir] if record_dir.is_dir() else [export_dir]
    for root in roots:
        if any(root.glob("*.db")):
            return True
        helper_dbs = root / "cannsim_helper_dbs"
        if helper_dbs.is_dir() and any(helper_dbs.glob("*.db")):
            return True
    return False


def _chip_name_sort_key(path: Path) -> tuple:
    """Order chip{N}/ directories numerically (chip0 < chip1 < ... < chip10).

    Plain name sort puts chip10 before chip2, and glob order is
    filesystem-dependent (CI returned chip1 first) — multi-chip report
    runs must have a stable, documented order.
    """
    import re

    match = re.fullmatch(r"chip(\d+)", path.name)
    return (0, int(match.group(1)), "") if match else (1, 0, path.name)


def _chip_db_dirs(export_dir: Path) -> list:
    """Per-chip DB directories (``chip{N}/`` holding *.db), numerically sorted."""
    record_dir = export_dir / "record"
    found = []
    for base in (record_dir, export_dir):
        for chip in base.glob("chip*"):
            if chip.is_dir() and any(chip.glob("*.db")):
                found.append(chip)
    found.sort(key=_chip_name_sort_key)
    return found


def _report_export_inputs(export_dir: Path) -> Optional[List[Tuple[Path, Optional[str]]]]:
    """Pick the ``report -e`` inputs for one record archive.

    Returns a list of ``(report -e directory, output subdir or None)``
    tuples, or None when the archive has no reportable artifact.

    Contract DBs win over ``instr.bin``: reporters on cannsim >= 0.7 no
    longer accept instr.bin, so letting instr.bin short-circuit the search
    would silently drop reports for archives that did collect data
    (e.g. ``record/instr.bin`` next to ``record/chip0/kernel.db`` — a
    flat-DB reporter handed ``record/`` finds zero DBs). The legacy
    instr.bin directory is only used when no DB layout exists at all —
    exactly the old-cannsim archives that report from instr.bin.

    Layout rules (cannsim >= 0.7 helper SQLite contract DBs):
    - flat ``*.db`` (archive root / ``record/`` / ``cannsim_helper_dbs/`` /
      ``record/cannsim_helper_dbs/``): pass the archive root — every
      >= 0.7 reporter resolves them from there.
    - single ``chip{N}/``: pass the chip directory itself — flat-only
      reporters (e.g. 9.3.0-beta.1) cannot route chip dirs from the
      root, while chip-routing reporters also accept a directory that
      directly holds ``*.db``.
    - multiple ``chip{N}/``: one report run per chip with its own output
      subdir — flat-only reporters find zero DBs from the archive root,
      and a single root run would also make the second chip's report
      overwrite the first (run_report loops these inputs instead).
    """
    if _has_flat_contract_db(export_dir):
        return [(export_dir, None)]
    chips = _chip_db_dirs(export_dir)
    if len(chips) == 1:
        return [(chips[0], None)]
    if len(chips) > 1:
        logging.info(
            "multiple per-chip DB dirs found under %s; one report run per chip: %s",
            export_dir,
            ", ".join(c.name for c in chips),
        )
        return [(chip, chip.name) for chip in chips]
    instr_dir = _find_instr_bin_dir(export_dir)
    if instr_dir is not None:
        return [(instr_dir, None)]
    return None


def _report_export_dir(export_dir: Path) -> Optional[Path]:
    """Backward-compatible single-input view over :func:`_report_export_inputs`."""
    inputs = _report_export_inputs(export_dir)
    return inputs[0][0] if inputs else None


def run_report(sw, export_dir: Path, output_dir: Optional[Path] = None) -> Optional[Path]:
    """Generate the performance report (trace_core*.json + HTML) for one case.

    Multi-chip archives run one report per chip, each into its own output
    subdir (``<out>/chip{N}/``) so the second run cannot overwrite the
    first — flat-DB-only reporters find zero DBs from the archive root.

    Returns the report output directory, or None on failure (caller logs only).
    """
    if not export_dir.is_dir():
        logging.warning("report input missing: %s", export_dir)
        return None
    inputs = _report_export_inputs(export_dir)
    if not inputs:
        logging.warning("no instr.bin / helper *.db found under %s; skipping report", export_dir)
        return None
    out = output_dir or (export_dir / "report")
    out.mkdir(parents=True, exist_ok=True)

    # Same env as record: skip inter-stage sleeps and keep ttk root on
    # PYTHONPATH so the cannsim package resolves identically.
    env = build_sim_env(sw)
    failed = False
    try:
        base_cmd = _cannsim_cmd()
        for report_input, out_subdir in inputs:
            per_out = out / out_subdir if out_subdir else out
            if out_subdir:
                per_out.mkdir(parents=True, exist_ok=True)
            # _cannsim_cmd() may raise (cannsim not installed); per the docstring
            # report is best-effort, so the whole invocation stays inside the try.
            cmd = [*base_cmd, "report", "-e", str(report_input), "-o", str(per_out)]
            if getattr(sw, "sim_cores", None):
                cmd += ["-n", sw.sim_cores]
            if getattr(sw, "sim_object_file", None):
                # source/asm mapping for the report; record (>= 0.7) rejects -f,
                # so the object file is routed to the report stage. Resolve to an
                # absolute path so the report subprocess finds it regardless of cwd.
                cmd += ["--object-file", str(Path(sw.sim_object_file).expanduser().resolve())]
            logging.info("npusim report: %s", " ".join(cmd))
            proc = _run_with_cleanup(cmd, env, sw.proc_timeout or None)
            if proc.returncode != 0:
                logging.warning(
                    "npusim report rc=%s (input %s); stdout tail:\n%s\nstderr tail:\n%s",
                    proc.returncode,
                    report_input,
                    (proc.stdout or "")[-2000:],
                    (proc.stderr or "")[-2000:],
                )
                failed = True
    except Exception as exc:  # noqa: BLE001 - report is best-effort
        logging.warning("npusim report failed: %s", exc)
        return None
    return None if failed else out
