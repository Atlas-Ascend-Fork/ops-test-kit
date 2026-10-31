#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""TTK 开发规范守护断言。

检查项：
  1. 依赖方向——ttk/core_modules/common 存在时附加目标规则
  2. 跨目录下划线成员 import——仅限 ttk 内部包目录；
     顶层单文件模块（ttk/_env 等）与外部包不在此列
  3. common 公开面——common 存在时启用：__init__ 须声明 __all__ 且
     根公开面对照 scripts/common_public_surface.json
  4. ruff 复杂度基线（参数≤5 / 实现≤50 行）——PLR0913/C901/PLR0915，
     存量计数进基线，新违例即失败，存量清零后 --update-baseline 缩基线
  5. 注释规范——注释/docstring 只描述代码功能：不引用 issue/PR 编号
     （井号后接两位以上数字）、不含变更叙事词（旧 bug/旧实现）；
     存量登记进基线（人工 reason），新违例即失败

用法：
  python3 scripts/check_conventions.py                   # 检查（新违例退出码 1）
  python3 scripts/check_conventions.py --update-baseline # 刷新基线（保留人工 reason）
"""

import argparse
import ast
import json
import pathlib
import re
import subprocess
import sys
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parent.parent
TTK = REPO / "ttk"
COMMON = TTK / "core_modules" / "common"
BASELINE = REPO / "scripts" / "conventions_baseline.json"
SURFACE = REPO / "scripts" / "common_public_surface.json"
RUFF_CODES = ["PLR0913", "C901", "PLR0915"]

# 通路目录名（通路互 import 规则与 geir→kernel 白名单用）
PATHWAYS = ("kernel", "aclnn", "framework_api", "geir")
# geir → kernel 单向复用白名单（依赖方向规则的唯一例外，按模块枚举）
GEIR_KERNEL_WHITELIST = (
    "ttk.core_modules.kernel.input_generate",  # __gen_input（算子级生成逻辑，Kernel 生态）
    "ttk.core_modules.kernel.golden",  # __gen_output（同上）
    "ttk.core_modules.kernel.info_keeper",  # OpInfoKeeper def.cpp 参数序
    "ttk.core_modules.kernel.testcase",  # TestcaseKernel 算子级用例基类
    "ttk.core_modules.kernel.device_state",  # 轮间 clear_device_state（与 e2e 共用形态）
)
# common 内允许 import ttk.remote 的文件（仅 instance.py 与 xpu/dispatch.py）
COMMON_REMOTE_ALLOW = (
    "ttk/core_modules/common/instance.py",
    "ttk/core_modules/common/xpu/dispatch.py",
)


def _rel(path):
    return str(path.relative_to(REPO))


# ── import 扫描 ─────────────────────────────────────────────────────────────


class Edge:
    __slots__ = ("file", "src", "names", "level", "module")

    def __init__(self, file, src, names, level, module):
        self.file = file  # 相对仓根文件路径
        self.src = src  # 绝对模块（ttk.*）；外部包保持原名
        self.names = names  # from-import 的名字列表（import 语句为别名列表）
        self.level = level
        self.module = module


def scan_imports():
    """AST 扫描 ttk/ 全部 .py，产出 import 边（相对 import 归一为绝对）。"""
    edges = []
    for f in sorted(TTK.rglob("*.py")):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            print(f"[WARN] 语法错误，跳过 {f}: {exc}")
            continue
        parts = list(pathlib.PurePosixPath(_rel(f)).with_suffix("").parts)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                lvl = node.level or 0
                if lvl:
                    base = parts[: len(parts) - lvl]
                    src = ".".join(base + node.module.split(".")) if node.module else ".".join(base)
                else:
                    src = node.module or ""
                edges.append(Edge(_rel(f), src, [a.name for a in node.names], lvl, node.module))
            elif isinstance(node, ast.Import):
                for a in node.names:
                    edges.append(Edge(_rel(f), a.name, [a.asname or a.name.split(".")[-1]], 0, a.name))
    return edges


def _file_pkg(path):
    """文件所在包目录的模块名（ttk/a/b/c.py → ttk.a.b；__init__.py 同目录）。"""
    p = pathlib.Path(path)
    if p.name == "__init__.py":
        p = p.parent
    return ".".join(p.with_suffix("").parts[:-1]) if len(p.parts) > 2 else ""


# ── 规则 1：依赖方向 ────────────────────────────────────────────────────────

DIRECTION_BASELINE_KEY = "import_direction"


def check_direction(edges, phase_common, baseline):
    """返回违例 [(file, src, reason)]。base 阶段 4 条 + common 阶段附加 5 条。"""
    viol = []
    allow = baseline.get(DIRECTION_BASELINE_KEY, {})

    def deny(edge, rule):
        if edge.src in allow.get(edge.file, {}):
            return
        viol.append((edge.file, edge.src, rule))

    for e in edges:
        f, src = e.file, e.src
        fdir = f.split("/")[1] if f.startswith("ttk/") and f.count("/") >= 2 else ""
        # —— base 阶段规则（现状即应成立）——
        if fdir == "utilities" and src.startswith(("ttk.cli", "ttk.core_modules", "ttk.remote")):
            deny(e, "utilities 不 import 上层任何包（基础层）")
        if fdir == "config" and src.startswith(("ttk.cli", "ttk.core_modules", "ttk.remote", "ttk.utilities")):
            deny(e, "config 不 import 上层任何包（基础层）")
        if f.startswith("ttk/core_modules/") and src.startswith("ttk.cli"):
            deny(e, "core_modules 不 import CLI（禁止反向）")
        if f.startswith("ttk/remote/") and src.startswith("ttk.core_modules"):
            deny(e, "remote 不 import core_modules（独立部署单元）")
        if not phase_common:
            continue
        # —— common 阶段附加规则 ——
        in_common = f.startswith("ttk/core_modules/common/")
        pathway_src = len(src.split(".")) >= 3 and src.split(".")[2] in PATHWAYS
        if in_common and pathway_src:
            deny(e, "common 不 import 通路（禁止反向）")
        if in_common and src.startswith("ttk.cli"):
            deny(e, "common 不 import CLI（禁止反向）")
        if pathway_src and f.startswith("ttk/core_modules/") and not in_common:
            own = f.split("/")[2]
            other = src.split(".")[2]
            if own in PATHWAYS and other != own and not (own == "geir" and src.startswith(GEIR_KERNEL_WHITELIST)):
                deny(e, f"通路间禁止互 import：{own} → {other}（唯一例外 geir→kernel）")
            if own in PATHWAYS and src.startswith("ttk.cli"):
                deny(e, "通路不 import CLI（禁止反向）")
        if in_common and src.startswith("ttk.remote") and f not in COMMON_REMOTE_ALLOW:
            deny(e, "common 内仅 instance.py/xpu.py 可依赖 remote")
    return viol


# ── 规则 2：跨目录下划线成员 import ─────────────────────────────────────────

UNDERSCORE_BASELINE_KEY = "underscore"

# 豁免：顶层单文件模块（模块级内部 API，非包目录公开面管辖）
TOP_MODULES = ("ttk._env", "ttk._version", "ttk.__main__")


def check_underscore(edges, baseline):
    viol = []
    allow = baseline.get(UNDERSCORE_BASELINE_KEY, {})
    for e in edges:
        if not e.src.startswith("ttk") or e.src.startswith(TOP_MODULES):
            continue  # 外部包 / 顶层单文件模块不在此列
        src_parts = e.src.split(".")
        if len(src_parts) < 3:
            continue  # ttk 或 ttk.xxx 包根的 __init__，同包消费
        src_dir = ".".join(src_parts[:-1])
        for name in e.names:
            if not name.startswith("_") or name.startswith("__"):
                continue
            key = f"{e.src}:{name}"
            if key in allow.get(e.file, {}):
                continue
            if src_dir == _file_pkg(e.file):
                continue  # 同目录（含本包 __init__）
            viol.append((e.file, e.src, name))
    return viol


# ── 规则 3：common 公开面 ───────────────────────────────────────────────────


def check_common_surface():
    """common 存在时：根 __all__ 对照清单；子包 __init__ 须声明 __all__。"""
    if not COMMON.is_dir():
        return None, "common 目录不存在，跳过公开面检查"
    viol = []
    manifest = {}
    if SURFACE.exists():
        manifest = set(json.loads(SURFACE.read_text(encoding="utf-8")))
    root_init = COMMON / "__init__.py"
    if root_init.exists():
        exported = _parse_all(root_init)
        if exported is None:
            viol.append((_rel(root_init), "", "根 __init__ 未声明 __all__"))
        elif manifest:
            for name in sorted(set(exported) - manifest):
                viol.append((_rel(root_init), name, "导出名不在 scripts/common_public_surface.json"))
            for name in sorted(manifest - set(exported)):
                viol.append((_rel(root_init), name, "清单名未导出（总表与 __all__ 同步维护）"))
        else:
            print("[HINT] common_public_surface.json 为空——请填充公开面总表")
    for init in COMMON.rglob("__init__.py"):
        if init == root_init:
            continue
        if _parse_all(init) is None:
            viol.append((_rel(init), "", "子包 __init__ 未声明 __all__（公开面收敛）"))
    return viol, None


def _parse_all(path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            try:
                return [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
            except AttributeError:
                return None
    return None


# ── 规则 4：ruff 复杂度基线 ──────────────────────────────────────────────────

RUFF_BASELINE_KEY = "ruff"


def check_ruff(baseline):
    """PLR0913/C901/PLR0915 按文件计数对照基线；新增/增长即违例。"""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "ttk", "--select", ",".join(RUFF_CODES), "--output-format", "json"],
            capture_output=True,
            text=True,
            cwd=REPO,
            check=False,
        )
    except FileNotFoundError:
        return None, "ruff 不可用（dev 依赖：pip install ttk[dev]）"
    if proc.returncode not in (0, 1):
        return None, f"ruff 运行失败: {proc.stderr.strip()[:200]}"
    records = json.loads(proc.stdout or "[]")
    counts = Counter((_rel_repo(r["filename"]), r["code"]) for r in records)
    base = baseline.get(RUFF_BASELINE_KEY, {})
    viol = []
    for (fname, code), n in sorted(counts.items()):
        allowed = base.get(fname, {}).get(code, 0)
        if n > allowed:
            viol.append((fname, code, f"{allowed} → {n}"))
    return viol, None


# ── 主流程 ─────────────────────────────────────────────────────────────────


COMMENT_BASELINE_KEY = "comments"
_COMMENT_SCAN_TOPS = ("ttk", "tests", "scripts")
_COMMENT_ISSUE_RE = re.compile(r"#\d{2,}")
_COMMENT_NARRATIVE_RE = re.compile(r"旧\s?bug|旧实现")


def _comment_violations():
    """扫描注释与 docstring：issue/PR 编号引用、变更叙事词。

    返回 [(relpath, lineno, 违例文本)]。编号与修复历史的追踪关系归 commit
    message / issue 平台，注释只描述代码当前行为与契约。
    """
    import ast
    import io
    import tokenize

    viol = []
    for top in _COMMENT_SCAN_TOPS:
        for path in sorted((REPO / top).rglob("*.py")):
            if not path.is_file() or path.name == pathlib.Path(__file__).name:
                continue  # 本脚本自身的规则描述豁免
            rel = path.relative_to(REPO).as_posix()
            src = path.read_text(encoding="utf-8", errors="ignore")
            try:
                for tok in tokenize.generate_tokens(io.StringIO(src).readline):
                    if tok.type != tokenize.COMMENT:
                        continue
                    for kind_re in (_COMMENT_ISSUE_RE, _COMMENT_NARRATIVE_RE):
                        m = kind_re.search(tok.string)
                        if m:
                            viol.append((rel, tok.start[0], m.group()))
                            break
            except tokenize.TokenError:
                pass
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                doc = ast.get_docstring(node)
                if not doc:
                    continue
                for kind_re in (_COMMENT_ISSUE_RE, _COMMENT_NARRATIVE_RE):
                    m = kind_re.search(doc)
                    if m:
                        # ast.Module 无 lineno（函数/类有），统一回退首行
                        viol.append((rel, getattr(node, "lineno", 1), m.group()))
                        break
    return viol


def check_comments(baseline):
    """返回 (存量外新增违例, 存量计数)。基线键 = 文件 → {违例文本: reason}。"""
    allow = baseline.get(COMMENT_BASELINE_KEY, {})
    viol = []
    total = 0
    for rel, lineno, text in _comment_violations():
        total += 1
        if text in allow.get(rel, {}):
            continue
        viol.append((rel, lineno, text))
    return viol, total


def load_baseline():
    if BASELINE.exists():
        return json.loads(BASELINE.read_text(encoding="utf-8"))
    return {}


def save_baseline(viol_direction, viol_underscore, ruff_counts, old, comment_viol):
    """生成基线：direction/underscore 保留既有 reason；ruff 存按文件计数。"""
    doc = {}
    direction = {}
    for fname, src, _rule in viol_direction:
        direction.setdefault(fname, {})[src] = (
            old.get(DIRECTION_BASELINE_KEY, {}).get(fname, {}).get(src, "TODO: 补充理由")
        )
    doc[DIRECTION_BASELINE_KEY] = direction
    underscore = {}
    for fname, src, name in viol_underscore:
        key = f"{src}:{name}"
        underscore.setdefault(fname, {})[key] = (
            old.get(UNDERSCORE_BASELINE_KEY, {}).get(fname, {}).get(key, "TODO: 补充理由")
        )
    doc[UNDERSCORE_BASELINE_KEY] = underscore
    per_file = {}
    for (fname, code), n in sorted(ruff_counts.items()):
        per_file.setdefault(fname, {})[code] = n
    doc[RUFF_BASELINE_KEY] = per_file
    comments = {}
    for rel, _lineno, text in comment_viol:
        existing = old.get(COMMENT_BASELINE_KEY, {}).get(rel, {}).get(text)
        comments.setdefault(rel, {})[text] = existing or "TODO: 补充理由"
    doc[COMMENT_BASELINE_KEY] = comments
    BASELINE.write_text(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"基线已写入 {_rel(BASELINE)}（请为 TODO 条目补充理由）")


def main():
    parser = argparse.ArgumentParser(description="TTK 架构守护断言")
    parser.add_argument("--update-baseline", action="store_true", help="按当前违例刷新基线")
    parser.add_argument(
        "--dump-graph",
        nargs="?",
        const="-",
        help="输出跨目录 import 图（每卡开工前刷新对照；默认 stdout，可给文件路径）",
    )
    args = parser.parse_args()

    baseline = load_baseline()
    edges = scan_imports()

    if args.dump_graph is not None:
        dump_graph(edges, args.dump_graph)
        return 0
    phase_common = COMMON.is_dir()

    viol_direction = check_direction(edges, phase_common, baseline)
    viol_underscore = check_underscore(edges, baseline)
    viol_surface, surface_note = check_common_surface()
    viol_ruff, ruff_note = check_ruff(baseline)
    viol_comments, comments_total = check_comments(baseline)

    if args.update_baseline:
        # 基线更新必须捕获**全量**违例（不过滤）——否则已基线条目会在二次更新时丢失
        all_direction = check_direction(edges, phase_common, {})
        all_underscore = check_underscore(edges, {})
        counts = _current_ruff_counts()
        save_baseline(all_direction, all_underscore, counts, baseline, _comment_violations())
        return 0

    failed = False
    if viol_direction:
        failed = True
        print(f"\n[FAIL] 依赖方向违例 {len(viol_direction)} 处:")
        for fname, src, rule in viol_direction:
            print(f"  {fname} → {src}\n      {rule}")
    if viol_underscore:
        failed = True
        print(f"\n[FAIL] 跨目录下划线 import 违例 {len(viol_underscore)} 处:")
        for fname, src, name in viol_underscore:
            print(f"  {fname} → from {src} import {name}")
    if viol_comments:
        failed = True
        print(f"\n[FAIL] 注释规范违例 {len(viol_comments)} 处（issue/PR 编号、变更叙事词——追踪关系归 commit message）:")
        for rel, lineno, text in viol_comments:
            print(f"  {rel}:{lineno}  {text}")
    if viol_surface:
        failed = True
        print(f"\n[FAIL] common 公开面违例 {len(viol_surface)} 处:")
        for fname, name, rule in viol_surface:
            print(f"  {fname} {name}: {rule}")
    if viol_ruff:
        failed = True
        print(f"\n[FAIL] ruff 复杂度新增/增长 {len(viol_ruff)} 处（基线 {BASELINE.name}）:")
        for fname, code, delta in viol_ruff:
            print(f"  {fname} [{code}] {delta}")

    if ruff_note:
        print(f"\n[WARN] {ruff_note}")
    if surface_note:
        print(f"[INFO] {surface_note}")

    # 存量基线摘要（不失败，供存量清零参考）
    base_ruff = baseline.get(RUFF_BASELINE_KEY, {})
    if base_ruff:
        total = sum(n for f in base_ruff for n in base_ruff[f].values())
        print(
            f"[INFO] 基线存量：注释 {comments_total} 处 / "
            f"ruff 复杂度 {total} 处 / "
            f"依赖方向 {sum(len(v) for v in baseline.get(DIRECTION_BASELINE_KEY, {}).values())} 处 / "
            f"下划线 {sum(len(v) for v in baseline.get(UNDERSCORE_BASELINE_KEY, {}).values())} 处"
        )
    print("\n[PASS] 架构守护断言通过" if not failed else "\n违例含基线外新增项，需修复或登记基线（附 reason）")
    return 1 if failed else 0


def _current_ruff_counts():
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "ttk", "--select", ",".join(RUFF_CODES), "--output-format", "json"],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    if proc.returncode not in (0, 1):
        return Counter()
    records = json.loads(proc.stdout or "[]")
    return Counter((_rel_repo(r["filename"]), r["code"]) for r in records)


def _rel_repo(fname):
    return str(pathlib.Path(fname).resolve().relative_to(REPO)).replace("\\", "/")


def dump_graph(edges, target):
    """跨目录 import 图：目录级汇总 + 逐边明细（ttk 内部，相对 import 已归一）。"""
    lines = []

    def dir_of(path):
        parts = pathlib.PurePosixPath(path).parts
        return "/".join(parts[:3]) if len(parts) >= 4 else "/".join(parts[:2])

    cross = []
    for e in edges:
        if not e.src.startswith("ttk."):
            continue
        src_dir = e.src.replace(".", "/")
        f_dir = dir_of(e.file)
        # 同目录边（src 目录包含文件目录或反之，视为内部）跳过
        if e.file.startswith(src_dir + "/") or src_dir.startswith(f_dir + "/"):
            continue
        cross.append((f_dir, src_dir, e))
    lines.append("# TTK 跨目录 import 图（生成: scripts/check_conventions.py --dump-graph）")
    lines.append(f"# 边数: {len(cross)}\n")
    agg = Counter((a, b) for a, b, _e in cross)
    lines.append("## 目录级汇总（from → to: 条数）")
    for (a, b), n in sorted(agg.items()):
        lines.append(f"  {a} → {b}: {n}")
    lines.append("\n## 逐边明细")
    for e in sorted((e for _a, _b, e in cross), key=lambda x: (x.file, x.src)):
        names = ",".join(e.names[:4]) + ("…" if len(e.names) > 4 else "")
        lines.append(f"  {e.file} → {e.src} import {names}")
    text = "\n".join(lines) + "\n"
    if target == "-":
        print(text)
    else:
        out = pathlib.Path(target)
        out.write_text(text, encoding="utf-8")
        print(f"import 图已写入 {out}（{len(cross)} 边）")


if __name__ == "__main__":
    sys.exit(main())
