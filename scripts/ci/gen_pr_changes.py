#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""
合并 PR 与本地 master 并导出改动文件列表

输入一笔 PR（gitcode MR 编号）与本地 ops-test-kit 仓库，产出：
1. 基于本地 master 新建的合并分支（默认 st），PR 内容已合入——一份合并后的 ops-test-kit 代码
2. 合并结果相对 master 的全部改动文件路径列表（默认写 pr_filelist.txt，每行一个路径）

用法:
    # 合并 PR 561 到本地 master，产出 st 分支 + pr_filelist.txt
    python3 gen_pr_changes.py -p 561 -r /home/s00588525/ops-test-kit

    # 自定义合并分支与输出文件
    python3 gen_pr_changes.py -p 561 -r /path/to/ops-test-kit --branch my-branch -o changes.txt

    # 合并分支已存在时删除重建
    python3 gen_pr_changes.py -p 561 -r /path/to/ops-test-kit --force

说明:
    - PR 从 --remote（默认 upstream，可传远端名或 URL）按 gitcode MR 引用
      refs/merge-requests/<N>/head 拉取到本地分支 pr-<N>（强制更新）
    - 合并基线为本地 --base 分支（默认 master）；合并方式 git merge --no-edit，
      冲突时自动 git merge --abort 恢复干净状态并退出
    - 改动列表 = git diff --name-only <base>..<branch>，覆盖新增/修改/删除的全部路径
    - 输出文件每行一个路径，可直接作为下游变更驱动工具（如 gen_ci_cmd.py -f）的输入
"""

import argparse
import os
import subprocess
import sys
from typing import NoReturn

# ============================================================
# 配置常量
# ============================================================
DEFAULT_REMOTE = "upstream"  # PR 所在远端（cann 主仓）；不存在时回退 origin
DEFAULT_BASE = "master"  # 合并基线分支（本地 master）
DEFAULT_BRANCH = "st"  # 合并产出分支名
DEFAULT_OUTPUT = "pr_filelist.txt"  # 改动文件列表输出文件


def die(msg) -> NoReturn:
    """打印错误信息并以返回码 1 退出"""
    print(f"错误: {msg}", flush=True)
    sys.exit(1)


def run_git(repo, args):
    """在仓库目录执行 git 命令，失败时打印 stderr 并退出

    Args:
        repo: 仓库路径
        args: git 参数列表，如 ["rev-parse", "HEAD"]

    Returns:
        str: stdout（已去除首尾空白）
    """
    result = subprocess.run(["git", "-C", repo] + args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(
            f"错误: git {' '.join(args)} 失败（返回码 {result.returncode}）",
            flush=True,
        )
        if result.stderr.strip():
            print(result.stderr.strip(), flush=True)
        sys.exit(1)
    return result.stdout.strip()


def resolve_remote(repo, remote):
    """解析 PR 所在远端

    Args:
        repo: 仓库路径
        remote: 远端名或 URL

    Returns:
        str: 远端名或 URL（原样传给 git fetch）
    """
    out = run_git(repo, ["remote"])
    names = out.split() if out else []
    # 显式 URL 或已存在的远端名直接使用
    if remote in names or "://" in remote or remote.startswith("git@"):
        return remote
    # 默认远端不存在时回退 origin（常见于直接 clone 主仓的场景）
    if remote == DEFAULT_REMOTE and "origin" in names:
        print(f"提示: 远端 {DEFAULT_REMOTE} 不存在，回退使用 origin", flush=True)
        return "origin"
    die(f"远端不存在: {remote}（现有远端: {' '.join(names) if names else '无'}）")


def ensure_clean_worktree(repo):
    """确保仓库无未提交的跟踪文件改动

    未跟踪文件不影响分支创建，允许存在。
    """
    status = run_git(repo, ["status", "--porcelain", "--untracked-files=no"])
    if status:
        print("错误: 仓库存在未提交的改动，请先提交或暂存:", flush=True)
        print(status, flush=True)
        sys.exit(1)


def fetch_pr(repo, remote, pr):
    """从远端拉取 PR 到本地分支 pr-<N>

    gitcode 的 MR 引用为 refs/merge-requests/<N>/head；
    refspec 前缀 '+' 强制更新本地分支（支持重复执行）。

    Args:
        repo: 仓库路径
        remote: 远端名或 URL
        pr: PR(MR) 编号

    Returns:
        str: 本地分支名 pr-<N>
    """
    pr_branch = f"pr-{pr}"
    current = run_git(repo, ["branch", "--show-current"])
    if current == pr_branch:
        die(f"当前正在 {pr_branch} 分支上，fetch 无法更新当前分支；请先切换到其他分支")
    refspec = f"+refs/merge-requests/{pr}/head:refs/heads/{pr_branch}"
    run_git(repo, ["fetch", remote, refspec])
    return pr_branch


def verify_base(repo, base):
    """确认基线分支存在

    Returns:
        str: 基线提交短 SHA
    """
    return run_git(repo, ["rev-parse", "--short", f"{base}^{{commit}}"])


def prepare_branch(repo, base, branch, force):
    """基于基线分支创建合并分支

    分支已存在时默认报错；--force 时先切回基线分支再删除重建
    （否则删除当前所在分支会被 git 拒绝，幂等重跑不可用）。

    Args:
        repo: 仓库路径
        base: 基线分支名
        branch: 合并产出分支名
        force: 已存在时是否删除重建
    """
    out = run_git(repo, ["branch", "--format", "%(refname:short)"])
    branches = out.split() if out else []
    if branch in branches:
        if not force:
            die(f"分支 {branch} 已存在；换 --branch 指定新名称，或使用 --force 删除重建")
        # 先切回基线再删除：上一次运行结束时 HEAD 停留在本分支，直接 branch -D
        # 会因 "Cannot delete branch ... checked out" 失败，--force 幂等重跑不可用
        run_git(repo, ["checkout", base])
        run_git(repo, ["branch", "-D", branch])
        print(f"已删除旧分支 {branch}", flush=True)
    run_git(repo, ["checkout", "-b", branch, base])


def merge_pr(repo, pr_branch):
    """将 PR 分支合入当前分支

    合并冲突时自动 git merge --abort 恢复干净状态，
    列出冲突文件后以返回码 1 退出。

    Args:
        repo: 仓库路径
        pr_branch: PR 本地分支名
    """
    result = subprocess.run(
        ["git", "-C", repo, "merge", "--no-edit", pr_branch],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        conflicts = subprocess.run(
            ["git", "-C", repo, "diff", "--name-only", "--diff-filter=U"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        subprocess.run(["git", "-C", repo, "merge", "--abort"], capture_output=True, check=False)
        print("错误: 合并 PR 失败，已执行 git merge --abort 恢复干净状态", flush=True)
        if result.stderr.strip():
            print(result.stderr.strip(), flush=True)
        if conflicts:
            print(f"冲突文件:\n{conflicts}", flush=True)
        sys.exit(1)
    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)


def get_changed_files(repo, base, branch):
    """获取合并结果相对基线的全部改动文件路径

    Args:
        repo: 仓库路径
        base: 基线分支名
        branch: 合并产出分支名

    Returns:
        list: 仓库相对路径列表（新增/修改/删除，按 git 输出顺序）
    """
    out = run_git(repo, ["diff", "--name-only", f"{base}..{branch}"])
    return [line.strip() for line in out.splitlines() if line.strip()]


def write_file_list(filepath, files):
    """将改动文件列表写入输出文件（每行一个路径）

    Args:
        filepath: 输出文件路径
        files: 改动文件路径列表
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for path in files:
            f.write(path + "\n")


def main():
    parser = argparse.ArgumentParser(description="合并 PR 与本地 master 并导出改动文件列表")
    parser.add_argument("-p", "--pr", required=True, help="PR(MR) 编号 必选参数")
    parser.add_argument(
        "-r",
        "--repo",
        default=".",
        help="ops-test-kit 本地仓库路径 可选参数 默认当前目录",
    )
    parser.add_argument(
        "--remote",
        default=DEFAULT_REMOTE,
        help=f"PR 所在远端名或 URL 可选参数 默认 {DEFAULT_REMOTE}（不存在时回退 origin）",
    )
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help=f"合并基线分支 可选参数 默认 {DEFAULT_BASE}（本地分支）",
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help=f"合并产出分支名 可选参数 默认 {DEFAULT_BRANCH}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"改动文件列表输出文件 可选参数 默认 {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="合并分支已存在时删除重建 可选参数",
    )
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    if not os.path.isdir(os.path.join(repo, ".git")):
        die(f"不是 git 仓库: {repo}")

    remote = resolve_remote(repo, args.remote)

    ensure_clean_worktree(repo)
    base_sha = verify_base(repo, args.base)

    print(f"[1/4] 拉取 PR {args.pr}（远端 {remote}）...", flush=True)
    pr_branch = fetch_pr(repo, remote, args.pr)

    print(f"[2/4] 基于 {args.base}（{base_sha}）创建分支 {args.branch}...", flush=True)
    prepare_branch(repo, args.base, args.branch, args.force)

    print(f"[3/4] 合入 {pr_branch}...", flush=True)
    merge_pr(repo, pr_branch)

    print("[4/4] 计算改动文件列表...", flush=True)
    files = get_changed_files(repo, args.base, args.branch)
    write_file_list(args.output, files)

    print(f"基线: {args.base}（{base_sha}）  合并分支: {args.branch}", flush=True)
    print(f"改动文件 {len(files)} 个，已写入 {args.output}:", flush=True)
    for path in files:
        print(f"  {path}", flush=True)


if __name__ == "__main__":
    main()
