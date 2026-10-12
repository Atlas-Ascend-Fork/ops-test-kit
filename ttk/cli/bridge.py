# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import logging
import pathlib
import re

_TYPED_CLEAN_VALUE = re.compile(r"(?P<dtype>[A-Za-z][A-Za-z0-9_]*)\((?P<value>[^()]*)\)$")
_INTEGER_CLEAN_VALUE = re.compile(r"[+-]?(?:0[xX][0-9a-fA-F]+|0[oO][0-7]+|0[bB][01]+|[0-9]+)$")
_FLOAT_CLEAN_VALUE = re.compile(r"[+-]?(?:(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)(?:[eE][+-]?[0-9]+)?)$")


def _apply_io_args(sw, args):
    sw.input_files = [args.input]
    if getattr(args, "sheet", None):
        sw.sheet = args.sheet
    if hasattr(args, "append_file") and args.append_file:
        sw.output_file_name = args.append_file
        sw.append_mode = True
    else:
        sw.output_file_name = args.output
        sw.append_mode = False


def _split_csv(value: str) -> list:
    """逗号串 → 剔空格、滤空段（'a, b' → ['a', 'b']——裸 split 会留 ' b' 导致静默失配）。"""
    return [v.strip() for v in value.split(",") if v.strip()]


_FALSY_FLAG_TOKENS = ("false", "0", "no", "off")


def parse_cli_bool(value: str) -> bool:
    """可选布尔参数取值语义：'false'/'0'/'no'/'off' → False，其余 → True。

    CLI 层唯一布尔解析（--warmup/--task-prof 的 type、-c/-d/--eager/-b 的
    bridge 落位共用）——避免各处 lambda 只认 "false" 导致 --warmup 0 静默变 True。
    """
    return value.lower() not in _FALSY_FLAG_TOKENS


def _apply_case_selection_args(sw, args):
    if hasattr(args, "testcase") and args.testcase:
        sw.selected_testcases = _split_csv(args.testcase)
    if hasattr(args, "testcase_index") and args.testcase_index:
        sw.selected_testcase_indexes = _parse_indexes(args.testcase_index)
    if hasattr(args, "testcase_count") and args.testcase_count is not None:
        sw.selected_testcase_count = args.testcase_count


def _apply_filter_args(sw, args):
    if hasattr(args, "priority") and args.priority:
        sw.priorities = _parse_priorities(args.priority)
    if hasattr(args, "operator") and args.operator:
        sw.selected_operators = tuple(_split_csv(args.operator))
    if hasattr(args, "exclude_operator") and args.exclude_operator:
        sw.excluded_operators = tuple(_split_csv(args.exclude_operator))


def _apply_devices(sw, spec, dev_count=None):
    """--devices ID 表达式解析（正选 '3'/'0,1'/'0-7' + '^' 剔除）。与 --dev N
    组合：N 决定池上限，正选在池内取交集（池外正选无效）；
    '^' 单独用时黑名单直接表达、池上限由 --dev 或探测决定。内部解析为
    device_count + device_blacklist（黑名单 = 全量 - available），
    由 _initialize_device_lock 统一消费。"""
    if not isinstance(spec, str):
        return
    spec = spec.strip()
    picks: set = set()
    excludes: set = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if part.startswith("^"):
            excludes.update(_parse_ids(part[1:].strip()))
        else:
            picks.update(_parse_ids(part))
    if not picks:
        # '^' 单独形式：正选 = 探测全量，黑名单直接表达；device_count 不动（探测默认）
        sw.device_blacklist = tuple(sorted(excludes)) if excludes else ()
        return
    if excludes:
        picks -= excludes
    if not picks:
        raise ValueError(f"--devices {spec!r} resolves to an empty device set")
    count = dev_count if dev_count is not None else max(picks) + 1
    # available 集 → 数量 + 黑名单（全量 - available；_initialize_device_lock 唯一消费面）
    sw.device_count = count
    sw.device_blacklist = tuple(i for i in range(count) if i not in picks)


def _parse_ids(token: str):
    if re.fullmatch(r"\d+", token):
        return {int(token)}
    m = re.fullmatch(r"(\d+)-(\d+)", token)
    if m:
        return set(range(int(m.group(1)), int(m.group(2)) + 1))
    raise ValueError(f"invalid device id expression: {token!r}")


def _apply_compare_dump_args(sw, args):
    if hasattr(args, "random_seed") and args.random_seed is not None:
        sw.random_seed = args.random_seed
    if hasattr(args, "input_dist"):
        sw.input_distribution = args.input_dist
    if hasattr(args, "compare"):
        sw.compare_method = args.compare
    if hasattr(args, "golden_mode"):
        sw.golden_mode = args.golden_mode
    # DumpConfig 由本层注入（utilities 不 import common）
    from ttk.core_modules.common.dump import DumpConfig

    if hasattr(args, "dump") or hasattr(args, "dump_format") or hasattr(args, "dump_on_fail"):
        sw.dump_config = DumpConfig.from_cli(getattr(args, "dump", None))
        if hasattr(args, "dump_format"):
            sw.dump_config.file_format = args.dump_format
        if getattr(args, "dump_on_fail", False):
            sw.dump_config.dump_on_fail = True
    if hasattr(args, "xpu_perf") and args.xpu_perf:
        sw.xpu_perf = True


def _apply_plugin_rerun_args(sw, args):
    if hasattr(args, "plugin") and args.plugin:
        sw.plugin_path = tuple(pathlib.Path(p.strip()).resolve() for p in args.plugin.split(",") if p.strip())
    if hasattr(args, "rerun") and args.rerun:
        sw.rerun_targets = [t.lower() for t in _split_csv(args.rerun)]


def _apply_output_log_args(sw, args):
    if hasattr(args, "proc_no_reuse") and args.proc_no_reuse:
        sw.proc_no_reuse = True
    if hasattr(args, "task_prof") and not args.task_prof:
        sw.task_profiling = False
    if hasattr(args, "progress_output") and args.progress_output:
        sw.progress_output = args.progress_output


def _apply_device_args(sw, args):
    dev_count = getattr(args, "device", None)
    if dev_count is not None:
        sw.device_count = dev_count
    if hasattr(args, "devices") and getattr(args, "devices", None) is not None:
        _apply_devices(sw, args.devices, dev_count)
    if hasattr(args, "process_count") and args.process_count is not None:
        sw.process_per_device = args.process_count
    if hasattr(args, "platform") and args.platform:
        sw.dev_plat = args.platform
    if hasattr(args, "proc_timeout") and args.proc_timeout:
        sw.proc_timeout = args.proc_timeout


def _apply_run_args(sw, args):
    if hasattr(args, "validate_only") and args.validate_only:
        sw.validate_only = True
    if hasattr(args, "no_memory_check") and args.no_memory_check:
        sw.no_memory_check = True
    if hasattr(args, "warmup") and not args.warmup:
        sw.warmup = False
    if hasattr(args, "run") and args.run is not None:
        sw.run_time = args.run
    if hasattr(args, "npu_timeout") and args.npu_timeout:
        sw.run_timeout = args.npu_timeout
    if hasattr(args, "deterministic_level") and args.deterministic_level:
        sw.deterministic_level = args.deterministic_level


def args_to_switches(args):
    from ttk.switches import SWITCHES

    sw = SWITCHES()
    sw.logging_to_file = True
    sw.config_path = getattr(args, "config", None)  # NEW: 经 SWITCHES pickle 传 worker
    sw.provider_filter = getattr(args, "provider", None)  # NEW: --provider CLI 过滤器

    # 无条件加载配置（config_path=None 时走标准路径：default.yaml + ~/.config/ttk + ./ttk.conf.yaml）。
    # 删 get_config lazy fallback 后，这是 parent 侧唯一的 load 入口。
    from ttk.config.loader import load_config

    load_config(sw.config_path)

    _apply_io_args(sw, args)
    _apply_case_selection_args(sw, args)
    _apply_filter_args(sw, args)
    _apply_compare_dump_args(sw, args)
    _apply_plugin_rerun_args(sw, args)
    _apply_output_log_args(sw, args)
    _apply_device_args(sw, args)
    _apply_run_args(sw, args)
    # --core-limit（add_perf_args 共享注册）：统一在此解析（args_to_switches
    # 四通路必经，收口于此）。类型守卫对齐 CLI 契约（str/None；int 供编程式调用），
    # 同时天然跳过 MagicMock 自动属性（测试桩无需逐字段置 None）。
    core_limit = getattr(args, "core_limit", None)
    if isinstance(core_limit, (int, str)):
        sw.core_limit = _parse_core_limit(core_limit)
    # -l/--limit 与 --simt-*（device control 组 hbm_simt 档：kernel/aclnn 共享
    # 注册，消费侧 kernel=tiling/RTS、aclnn=守门/rtLimit）
    limit = getattr(args, "limit", None)
    if isinstance(limit, int):
        sw.hbm_size_limit = limit
    simt_ub = getattr(args, "simt_ub", None)
    if isinstance(simt_ub, int):
        sw.force_simt_ub_size = [simt_ub] * 3
    elif isinstance(simt_ub, (list, tuple)):
        sw.force_simt_ub_size = list(simt_ub[:3])
    if isinstance(getattr(args, "simt_stack_dcu", None), int):
        sw.simt_cfg.dcu_stack = args.simt_stack_dcu
    if isinstance(getattr(args, "simt_stack_dvg", None), int):
        sw.simt_cfg.dvg_stack = args.simt_stack_dvg

    return sw


def _apply_device_state_args(sw, args):
    """--clear-ub/l1/l0 设备状态控制组（四通路消费；aclnn 经 AclInterface→RTSInterface 委托）。"""
    if hasattr(args, "clear_ub") and args.clear_ub is not None:
        sw.force_clear_ub = _parse_clean_val("UB", args.clear_ub)
    if hasattr(args, "clear_l1") and args.clear_l1 is not None:
        sw.force_clear_l1 = _parse_clean_val("L1", args.clear_l1)
    if hasattr(args, "clear_l0") and args.clear_l0 is not None:
        sw.force_clear_l0 = _parse_clean_val("L0", args.clear_l0)


def apply_kernel_args(sw, args):
    if hasattr(args, "dynamic") and args.dynamic is not None:
        val = args.dynamic
        if isinstance(val, str):
            val = parse_cli_bool(val)
        sw.dyn_switches.enabled = val
    if hasattr(args, "const") and args.const is not None:
        val = args.const
        if isinstance(val, str):
            val = parse_cli_bool(val)
        sw.cst_switches.enabled = val
    if hasattr(args, "binary") and args.binary is not None:
        if args.binary == "release":
            sw.bin_switches.enabled = True
            sw.bin_switches.realtime = "release"
        else:
            # 裸 -b（True）启用 binary 模式（默认 realtime）；-b=false 关闭
            val = args.binary
            if isinstance(val, str):
                val = parse_cli_bool(val)
            sw.bin_switches.enabled = val
    if hasattr(args, "compile_only") and args.compile_only:
        sw.compile_only = True
    # --compile-opts key=value (append, direct passthrough)
    if hasattr(args, "compile_opts") and args.compile_opts:
        for pair in args.compile_opts:
            if "=" not in pair:
                raise ValueError(f"--compile-opts requires KEY=VALUE format, got: {pair}")
            key, value = pair.split("=", 1)
            sw.compile_options[key] = value
    if hasattr(args, "impl_mode") and args.impl_mode:
        sw.op_impl_mode = args.impl_mode
    # -l/--limit 与 --simt-* 的解析位于 args_to_switches（kernel/aclnn 共享注册）
    if hasattr(args, "reuse_hbm") and args.reuse_hbm:
        sw.reuse_hbm = True
    if hasattr(args, "reserve_hbm") and args.reserve_hbm is not None:
        sw.reserve_hbm = args.reserve_hbm
    if hasattr(args, "no_prof") and args.no_prof:
        sw.dyn_switches.prof = False
        sw.cst_switches.prof = False
        sw.bin_switches.prof = False
    if hasattr(args, "tiling_run") and args.tiling_run is not None:
        sw.tiling_run_time = args.tiling_run
    if hasattr(args, "cce") and args.cce is not None:
        _apply_cce(sw, args.cce)
    if hasattr(args, "clear_atomic") and args.clear_atomic:
        _apply_clear_atomic(sw, args.clear_atomic)
    if hasattr(args, "force_block_dim") and args.force_block_dim is not None:
        bd = args.force_block_dim
        if isinstance(bd, int):
            sw.force_block_dim = [bd] * 3
        else:
            sw.force_block_dim = list(bd)


def apply_aclnn_args(sw, args):
    pass


def apply_e2e_args(sw, args):
    if hasattr(args, "dynamic") and args.dynamic is not None:
        val = args.dynamic
        if isinstance(val, str):
            val = parse_cli_bool(val)
        sw.dyn_switches.enabled = val
    if hasattr(args, "const") and args.const is not None:
        val = args.const
        if isinstance(val, str):
            val = parse_cli_bool(val)
        sw.cst_switches.enabled = val
    if hasattr(args, "fullgraph"):
        sw.fullgraph = args.fullgraph
    if getattr(args, "aclgraph", False):
        sw.aclgraph_enabled = True
    # -b/--binary：默认二进制（三种模式全生效），-b=false 恢复在线编译；显式启用
    # 且与 --super-kernel 同开时报错（SuperKernel 编译流程要求在线编译）
    if hasattr(args, "binary") and args.binary is not None:
        binary = args.binary
        if isinstance(binary, str):
            binary = parse_cli_bool(binary)
        if binary and getattr(args, "super_kernel", False):
            raise ValueError("--binary cannot be combined with --super-kernel")
        sw.e2e_binary = bool(binary)
    if hasattr(args, "core_limit") and args.core_limit is not None:
        sw.core_limit = _parse_core_limit(args.core_limit)
    if hasattr(args, "eager") and args.eager is not None:
        val = args.eager
        if isinstance(val, str):
            val = parse_cli_bool(val)
        sw.eager_enabled = val
        if not sw.eager_enabled and not (sw.dyn_switches.enabled or sw.cst_switches.enabled or sw.aclgraph_enabled):
            raise ValueError("--eager=false requires a graph mode: -c/-d/--aclgraph")
    # --core-limit 的解析位于 args_to_switches（四通路共享注册，add_perf_args）
    if getattr(args, "super_kernel", False):
        sw.super_kernel_enabled = True
        if getattr(args, "backend", None) == "cpu":
            raise ValueError("--super-kernel cannot be combined with --backend cpu")
        if not (sw.dyn_switches.enabled or sw.cst_switches.enabled or sw.aclgraph_enabled):
            raise ValueError("--super-kernel requires a graph mode: -c/-d/--aclgraph")
        # SuperKernel 要求在线编译：隐式默认二进制时自动回退在线（显式 -b 已在上面报错）
        if getattr(sw, "e2e_binary", False):
            sw.e2e_binary = False
            logging.warning("--super-kernel requires online kernel compilation; forcing online compile (-b=false)")
    # npusim 仿真要求算子在线编译（camodel 编译产物），二进制复用强制关闭
    if getattr(args, "backend", None) == "npusim" and getattr(sw, "e2e_binary", False):
        sw.e2e_binary = False
        logging.warning("--binary is ignored under --backend npusim; online kernel compilation is required")


def _parse_core_limit(value):
    """Parse --core-limit into (ai_core_limit, vector_core_limit), either side optional."""
    if isinstance(value, int):
        return (value, None)
    raw = str(value).strip()
    if not raw:
        raise ValueError("--core-limit requires at least one of 'a,v', got empty value")
    ai_part, sep, vec_part = raw.partition(",")
    if not sep and "|" in raw:
        raise ValueError("--core-limit uses ',' as separator, e.g. 2,32")

    def _to_int(part):
        part = part.strip()
        if not part:
            return None
        try:
            parsed = int(part)
        except ValueError as exc:
            raise ValueError(f"--core-limit values must be integers, got {part!r}") from exc
        if parsed < 1:
            raise ValueError(f"--core-limit values must be >= 1, got {part!r}")
        return parsed

    ai_limit = _to_int(ai_part)
    vec_limit = _to_int(vec_part)
    if ai_limit is None and vec_limit is None:
        raise ValueError(f"--core-limit requires at least one of 'a,v', got {value!r}")
    return (ai_limit, vec_limit)


def _configure_manual_data_dump(sw, command, directories):
    """--no-prof 落盘方向：显式目录或默认目录，仅保留写侧成立的硬约束。

    落什么由 DumpConfig 门控自然决定（不上板就没有 output 可落）；不做
    "必须/禁止 out" 类人工约束。Kernel 例外：output 分配依赖 Golden，
    落盘必须 in+golden 成对。
    """
    cfg = sw.dump_config
    kernel_pair = cfg.dump_input and cfg.dump_golden
    if command == "kernel" and not directories and not kernel_pair:
        return  # kernel legacy dry-run：--no-prof 纯关执行，无落盘意图
    if command == "kernel" and sw.compile_only:
        raise ValueError("--compile-only cannot be combined with Kernel manual-data output")
    if command == "kernel" and directories and not kernel_pair:
        raise ValueError("Kernel manual-data output requires exactly --no-prof --dump in,golden")
    if command == "geir" and sw.golden_mode == "XPU":
        raise ValueError(
            "--no-prof writes host-generated input/golden; "
            "--golden-mode XPU (golden from third-party execution) has nothing to write"
        )
    if not cfg.dump_input:
        if directories:
            raise ValueError("--no-prof manual-data output requires --dump in")
        raise ValueError("--no-prof requires --dump in (manual-data output)")
    if cfg.file_format not in ("bin", "pt", "npy"):
        raise ValueError(
            f"--dump-format {cfg.file_format!r} is not restorable for manual-data output; use bin, pt, or npy"
        )
    if cfg.dump_golden and sw.golden_mode == "Disable":
        raise ValueError("--no-prof golden output requires CPU golden generation; --golden-mode Disable is invalid")
    if len(directories) > 1:
        raise ValueError("--no-prof writes one dataset; specify at most one --manual-data-dirs path")
    sw.manual_data_dirs = directories or (_default_manual_data_dir(sw),)


def configure_manual_data(sw, args, command):
    """Resolve --manual-data-dirs and --no-prof into switches (no mode inference).

    full-flow + 目录集 → load 方向（优先恢复已存用例）；--no-prof → 不上板，
    dump 门控开启时向（显式或默认）目录落盘。上不上板与数据目录互不触发。
    """
    sw.no_prof = getattr(args, "no_prof", False) is True
    raw_dirs = getattr(args, "manual_data_dirs", None)
    if isinstance(raw_dirs, str):
        raw_dirs = (raw_dirs,)
    elif not isinstance(raw_dirs, (tuple, list)):
        raw_dirs = ()
    directories = tuple(
        str(pathlib.Path(item).expanduser().resolve()) for item in raw_dirs if isinstance(item, str) and item
    )

    if not sw.no_prof:
        if not directories:
            return
        if sw.golden_mode != "Enable":
            raise ValueError("--manual-data-dirs loading uses an existing golden; use --golden-mode Enable")
        if sw.validate_only:
            raise ValueError("--validate cannot be combined with --manual-data-dirs")
        if command == "kernel" and sw.compile_only:
            raise ValueError("--compile-only cannot be combined with Kernel --manual-data-dirs")
        if command == "e2e" and sw.force_cpu:
            raise ValueError("--manual-data-dirs loading is the device stage and cannot use --backend cpu")
        sw.manual_data_dirs = directories
        return

    _configure_manual_data_dump(sw, command, directories)


def _default_manual_data_dir(sw):
    if sw.plugin_path:
        if len(sw.plugin_path) > 1:
            raise ValueError(
                "--no-prof with multiple --plugin paths require one explicit --manual-data-dirs output directory"
            )
        plugin = pathlib.Path(sw.plugin_path[0])
        plugin_root = plugin.parent if plugin.is_file() or plugin.suffix == ".py" else plugin
        return str((plugin_root / "manual_data").resolve())
    return str((pathlib.Path.cwd() / "manual_data").resolve())


def _log_manual_data_configuration(sw):
    default_dir = str((pathlib.Path.cwd() / "manual_data").resolve())
    if sw.no_prof and not sw.plugin_path and sw.manual_data_dirs == (default_dir,):
        logging.info(
            "No --plugin was provided; using current-directory manual-data output: %s",
            default_dir,
        )


def _detect_framework_from_csv(input_files, sheet=None):
    """Peek at the first table to detect framework from api_name column.

    Reads the table header (CSV or XLSX via the unified table reader) to
    find the api_name column, then checks the first data row's api_name
    value. Returns 'tf' if it starts with 'tf.' or 'tensorflow.',
    otherwise 'torch'.

    One table must contain only one framework's APIs: torch_npu and
    npu_device each initialize the NPU runtime exclusively, so mixing
    frameworks in a single run causes runtime conflicts. This function
    detects the framework from the first data row and validates that all
    subsequent rows are consistent.
    """
    if not input_files:
        return "torch"

    from ttk.core_modules.common.backend import detect_framework
    from ttk.core_modules.common.testcase import read_table

    try:
        header, rows = read_table(input_files[0], sheet)
    except Exception as e:
        logging.warning(f"Failed to detect framework from input, defaulting to torch: {e}")
        return "torch"

    try:
        api_idx = header.index("api_name")
    except ValueError:
        return "torch"

    first_api = None
    for row in rows:
        if api_idx < len(row) and row[api_idx]:
            first_api = row[api_idx]
            break
    if not first_api:
        return "torch"

    first_framework = detect_framework(first_api)
    for row in rows:
        if api_idx < len(row):
            row_framework = detect_framework(row[api_idx])
            if row_framework != first_framework:
                raise ValueError(
                    f"Mixed frameworks in one table is not supported: "
                    f"first row is {first_framework} (api_name='{first_api}'), "
                    f"but found {row_framework} (api_name='{row[api_idx]}') in a later row. "
                    f"Please split into separate files per framework."
                )
    return first_framework


def run_with_switches(sw):
    from ttk.core_modules.common.logging import default_logging_config

    sw.activate()
    default_logging_config()  # 主进程 stdout only——文件仅 per-case
    _log_manual_data_configuration(sw)

    from ttk.utilities import set_process_name, set_thread_name

    set_process_name()
    set_thread_name()

    logging.info(f"Command: ttk {sw.test_mode} -i {sw.input_files[0] if sw.input_files else ''}")

    # 注册表分发（无 if-else）。注册是 import 副作用：按 test_mode
    # import 通路 Instance 模块触发 @register_instance（CLI 层向下依赖合法）
    if sw.test_mode in ("op", "kernel"):
        from ttk.core_modules.kernel import instance  # noqa: F401
    elif sw.test_mode == "aclnn":
        from ttk.core_modules.aclnn import instance  # noqa: F401
    elif sw.test_mode == "geir":
        import ttk.core_modules.geir  # noqa: F401  # Instance 宿主为包根
    elif sw.test_mode == "framework-api":
        sw.framework = _detect_framework_from_csv(sw.input_files, getattr(sw, "sheet", None))
        from ttk.core_modules.framework_api import instance  # noqa: F401
    from ttk.core_modules.common.instance import get_instance_by_name

    ins = get_instance_by_name(sw.test_mode)
    ins.run()
    if ins.fail_count:
        raise SystemExit(1)


def _parse_indexes(spec):
    selected = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            selected.extend(range(int(lo), int(hi) + 1))
        else:
            selected.append(int(part))
    return tuple(selected)


def _parse_priorities(spec):
    priorities = []
    for p in (raw.strip() for raw in spec.split(",")):
        if not p:
            continue
        if "-" in p:
            lo, hi = p.split("-", 1)
            lo, hi = lo.strip(), hi.strip()
            priorities.append((int(lo) if lo else 0, int(hi) if hi else float("inf")))
        else:
            priorities.append((int(p), int(p)))
    return tuple(priorities)


def _apply_cce(sw, value):
    if value is True or value is None:
        sw.dyn_switches.realtime = False
        sw.cst_switches.realtime = False
        if sw.bin_switches.realtime != "release":
            sw.bin_switches.realtime = False
    else:
        for mode in value.lower().split(","):
            if mode in ("d", "dyn", "dynamic"):
                sw.dyn_switches.realtime = False
            elif mode in ("c", "cst", "const"):
                sw.cst_switches.realtime = False
            elif mode in ("b", "bin", "binary") and sw.bin_switches.realtime != "release":
                sw.bin_switches.realtime = False


def _apply_clear_atomic(sw, value):
    if value is True:
        sw.force_clear_atomic = [True] * 3
    else:
        for mode in value.lower().split(","):
            if mode in ("d", "dyn"):
                sw.force_clear_atomic[0] = True
            elif mode in ("c", "cst"):
                sw.force_clear_atomic[1] = True
            elif mode in ("b", "bin"):
                sw.force_clear_atomic[2] = True


def _parse_clean_val(name, value):
    """Parse a numeric clear value without evaluating caller-provided code."""
    import numpy

    if value is None or not str(value).strip():
        return numpy.int32(0)

    raw_value = str(value).strip().lower()
    match = _TYPED_CLEAN_VALUE.fullmatch(raw_value)
    if match is None:
        dtype_name = None
        literal = raw_value
    else:
        dtype_name = match.group("dtype")
        literal = match.group("value").strip()

    if len(literal) >= 2 and literal[0] == literal[-1] and literal[0] in "\"'":
        literal = literal[1:-1]
    if literal in ("inf", "+inf", "-inf", "nan", "+nan", "-nan"):
        parsed_value = float(literal)
        is_float = True
    elif _INTEGER_CLEAN_VALUE.fullmatch(literal):
        parsed_value = int(literal, 0)
        is_float = False
    elif _FLOAT_CLEAN_VALUE.fullmatch(literal):
        parsed_value = float(literal)
        is_float = True
    else:
        raise ValueError(
            f"Cannot parse {name} clean value {value!r}; use a numeric literal or "
            "a NumPy numeric dtype constructor such as float32(1.0)"
        )

    dtype_name = dtype_name or ("float32" if is_float else "int32")
    try:
        dtype = numpy.dtype(dtype_name)
    except TypeError as exc:
        raise ValueError(f"Unsupported {name} clean value dtype {dtype_name!r}") from exc
    if dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"Unsupported {name} clean value dtype {dtype_name!r}")
    constructor = getattr(numpy, dtype_name, None)
    if constructor is None or not callable(constructor):
        raise ValueError(f"Unsupported {name} clean value dtype {dtype_name!r}")
    try:
        return constructor(parsed_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Cannot parse {name} clean value {value!r}: {exc}") from exc
