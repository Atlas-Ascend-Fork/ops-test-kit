import argparse
import os
import sys

from ttk.cli.aclnn import register_aclnn_command
from ttk.cli.e2e import register_e2e_command
from ttk.cli.geir import register_geir_command
from ttk.cli.info import register_info_command
from ttk.cli.kernel import register_kernel_command
from ttk.cli.list_cmd import register_list_command


def main():
    os.environ["TTK_PARENT_PID"] = str(os.getpid())

    from ttk._env import setup_env

    setup_env()

    try:
        _cli_main()
    finally:
        try:
            from ttk.core_modules.common.process import SimpleCommandProcess

            for proc in SimpleCommandProcess.all_processes:
                proc.close()
        except Exception:  # noqa: S110 退出清理尽力而为：进程已死/半关不阻断退出
            pass


def _print_version():
    from ttk._version import __version__, get_build

    print(f"TTK v{__version__}, build {get_build()}")


def _version_callback(value):
    if value:
        _print_version()
        raise SystemExit(0)


_EPILOG = """\
deep-dive help:
  ttk kernel help               grouped help (works for kernel/geir/aclnn/e2e/info/list)
  ttk kernel help devices       parameter deep-dive (syntax & interplay)
  ttk kernel --devices --help   same, --help written after the parameter
  ttk help devices              topic lookup without a mode
"""

# 四测试模式：`<mode> help` 别名 + 裸 `<mode>` 打分组 help + 参数级 --help 后缀。
# 展示序 = 架构分层序（kernel 算子层 → geir GE 图 → aclnn 引擎 API → e2e 框架层）。
_MODE_COMMANDS = ("kernel", "geir", "aclnn", "e2e")
# 全部子命令：`<cmd> help` 别名（list 的 -i required 陷阱同样需要预路由）
_SUBCOMMANDS = ("kernel", "geir", "aclnn", "e2e", "info", "list")


def build_parser():
    """主 parser 装配（help 子命令复用同一构造，避免两份漂移）。"""
    parser = argparse.ArgumentParser(
        prog="ttk",
        description="TTK — Ascend NPU Single Operator Test Framework",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Show version and build info",
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    from .help_cmd import register_help_command

    sub_map = {
        "kernel": register_kernel_command(subparsers),
        "geir": register_geir_command(subparsers),
        "aclnn": register_aclnn_command(subparsers),
        "e2e": register_e2e_command(subparsers),
        "info": register_info_command(subparsers),
        "list": register_list_command(subparsers),
        "help": register_help_command(subparsers),
    }
    return parser, sub_map


def _cli_main():
    parser, sub_map = build_parser()

    argv = sys.argv[1:]
    # 预解析帮助路由（三种写法，匹配逻辑收敛在 help_cmd）：
    #   1. `<cmd> help [PARAM]` 别名     2. `<mode> --<param> --help` 后缀
    #   3. 裸 `ttk <mode>` → 分组 help（git/docker 惯例，不抛 required 报错）
    from .help_cmd import handle_topic_tokens, match_help_alias, match_help_suffix
    from .help_topics import print_topic

    alias_tokens = match_help_alias(argv, _SUBCOMMANDS)
    if alias_tokens is not None:
        handle_topic_tokens(alias_tokens)
        return
    suffix_topic = match_help_suffix(argv, _MODE_COMMANDS)
    if suffix_topic is not None:
        print_topic(suffix_topic)
        return
    if len(argv) == 1 and argv[0] in _MODE_COMMANDS:
        sub_map[argv[0]].print_help()
        return

    args = parser.parse_args()

    if args.version:
        _print_version()
        return

    if not args.command:
        parser.print_help()
        return

    try:
        args.handler(args)
    except (ValueError, RuntimeError) as exc:
        # bridge/handler 的参数组合与配置校验（--dump token、--super-kernel 互斥、
        # --xpu-perf 缺远端配置等）——单行报错替代 traceback；TTK_DEBUG=1 看全栈。
        if os.environ.get("TTK_DEBUG"):
            raise
        print(f"ttk: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
