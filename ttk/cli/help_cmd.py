#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""``ttk help`` — second-level help: grouped command help + parameter deep-dives.

    ttk help                      main help (same as ttk --help)
    ttk help kernel               same as ttk kernel --help (grouped)
    ttk help kernel devices       deep-dive topic for --devices
    ttk help devices              topic lookup without a command

The ``ttk <mode> help [PARAM]`` spelling (e.g. ``ttk kernel help devices``)
is routed to the same logic by ``_cli_main`` before argparse runs — the
per-mode required ``-i`` check would otherwise reject it.
"""

import sys

from .help_topics import HELP_TOPICS, print_topic


def register_help_command(subparsers):
    parser = subparsers.add_parser("help", help="Command help and parameter deep-dive topics")
    parser.add_argument(
        "topic",
        nargs="*",
        metavar="COMMAND [PARAM]",
        help=f"command name and/or parameter topic (topics: {', '.join(sorted(HELP_TOPICS))})",
    )
    parser.set_defaults(handler=_handle_help)
    return parser


def _handle_help(args):
    handle_topic_tokens(args.topic)


def match_help_alias(argv, subcommands):
    """``<cmd> help [PARAM]`` 别名 → topic tokens；不匹配返回 None。

    argparse 的 required(-i) 检查先于未识别 token，此形式必须预解析路由。
    """
    if len(argv) >= 2 and argv[1] == "help" and argv[0] in subcommands:
        return [argv[0], *argv[2:]]
    return None


def match_help_suffix(argv, modes):
    """``<mode> --<param> --help``（--help 位置随意）→ topic 名；不匹配返回 None。

    仅当其余 token 全为 topic 参数时命中；带普通参数（如 ``-i x.csv --help``）
    返回 None，回落 argparse 的模式级 --help。
    """
    if len(argv) < 2 or argv[0] not in modes:
        return None
    rest = argv[1:]
    if not any(a in ("-h", "--help") for a in rest):
        return None
    others = [a for a in rest if a not in ("-h", "--help")]
    names = [a.lstrip("-") for a in others if a.startswith("--")]
    if others and len(names) == len(others) and set(names) <= set(HELP_TOPICS):
        return names[0]
    return None


def handle_topic_tokens(tokens):
    """共享入口：`ttk help [COMMAND] [PARAM]` 与 `ttk <mode> help [PARAM]` 同一语义。"""
    tokens = [t for t in (tokens or []) if t]
    if not tokens:
        from ttk.cli import build_parser

        build_parser()[0].print_help()
        return
    if len(tokens) == 1:
        token = tokens[0]
        if token in HELP_TOPICS:
            print_topic(token)
            return
        sub = _subparser_of(token)
        if sub is not None:
            sub.print_help()
            return
        _unknown(token)
    if len(tokens) == 2:
        command, param = tokens
        if _subparser_of(command) is None:
            _unknown(command)
        if param not in HELP_TOPICS:
            print(
                f"ttk help: unknown topic {param!r} for {command!r}; available: {', '.join(sorted(HELP_TOPICS))}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        print_topic(param)
        return
    print("ttk help: expected 'ttk help [COMMAND] [PARAM]'", file=sys.stderr)
    raise SystemExit(2)


def _subparser_of(command):
    from ttk.cli import build_parser

    _, sub_map = build_parser()
    return sub_map.get(command)


def _unknown(token):
    from ttk.cli import build_parser

    _, sub_map = build_parser()
    commands = [name for name in sub_map if name != "help"]
    print(
        f"ttk help: unknown command or topic {token!r}; "
        f"commands: {', '.join(commands)}; topics: {', '.join(sorted(HELP_TOPICS))}",
        file=sys.stderr,
    )
    raise SystemExit(2)
