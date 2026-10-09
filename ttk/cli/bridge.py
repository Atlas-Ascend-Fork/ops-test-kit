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


def _apply_case_selection_args(sw, args):
    if hasattr(args, "testcase") and args.testcase:
        sw.selected_testcases = [item.strip() for item in args.testcase.split(",") if item.strip()]
    if hasattr(args, "testcase_index") and args.testcase_index:
        sw.selected_testcase_indexes = _parse_indexes(args.testcase_index)
    if hasattr(args, "testcase_count") and args.testcase_count is not None:
        sw.selected_testcase_count = args.testcase_count


def _apply_filter_args(sw, args):
    if hasattr(args, "priority") and args.priority:
        sw.priorities = _parse_priorities(args.priority)
    if hasattr(args, "operator") and args.operator:
        sw.selected_operators = tuple(args.operator.split(","))
    if hasattr(args, "exclude_operator") and args.exclude_operator:
        sw.excluded_operators = tuple(args.exclude_operator.split(","))


def _apply_compare_dump_args(sw, args):
    if hasattr(args, "random_seed") and args.random_seed is not None:
        sw.random_seed = args.random_seed
    if hasattr(args, "input_dist"):
        sw.input_distribution = args.input_dist
    if hasattr(args, "compare"):
        sw.compare_method = args.compare
    if hasattr(args, "golden_mode"):
        sw.golden_mode = args.golden_mode
    if hasattr(args, "dump") and args.dump is not None:
        _apply_dump_config(sw.dump_config, args.dump)
    if hasattr(args, "dump_format"):
        sw.dump_config.file_format = args.dump_format
    if hasattr(args, "dump_on_fail") and args.dump_on_fail:
        sw.dump_config.dump_on_fail = True
    if hasattr(args, "xpu_perf") and args.xpu_perf:
        sw.xpu_perf = True
    if getattr(args, "xpu_zero_upload", False):
        sw.xpu_zero_upload = True
    if getattr(args, "geir_serve", False):
        sw.geir_serve = True


def _apply_plugin_rerun_args(sw, args):
    if hasattr(args, "plugin") and args.plugin:
        sw.plugin_path = tuple(pathlib.Path(p.strip()).resolve() for p in args.plugin.split(",") if p.strip())
    if hasattr(args, "rerun") and args.rerun:
        sw.rerun_targets = [item.strip().lower() for item in args.rerun.split(",") if item.strip()]


def _apply_output_log_args(sw, args):
    if hasattr(args, "title") and args.title:
        sw.custom_columns = args.title.split(",")
    if hasattr(args, "csv_preserve") and args.csv_preserve:
        sw.preserve_original_csv = True
    if hasattr(args, "single_log") and args.single_log:
        sw.single_testcase_log_mode = True
    if hasattr(args, "summary_print") and not args.summary_print:
        sw.summary_print = False
    if hasattr(args, "proc_no_reuse") and args.proc_no_reuse:
        sw.proc_no_reuse = True
    if hasattr(args, "no_memory_check") and args.no_memory_check:
        sw.no_memory_check = True
    if hasattr(args, "task_prof") and not args.task_prof:
        sw.TASK_PROFILING = False
    if hasattr(args, "progress_output") and args.progress_output:
        sw.progress_output = args.progress_output


def _apply_device_args(sw, args):
    if hasattr(args, "device") and args.device is not None:
        sw.device_count = args.device
    if hasattr(args, "device_blacklist") and args.device_blacklist:
        sw.device_blacklist = tuple(int(x) for x in args.device_blacklist.split(","))
    if hasattr(args, "device_whitelist") and args.device_whitelist:
        sw.device_whitelist = tuple(int(x) for x in args.device_whitelist.split(","))
    if hasattr(args, "process_count") and args.process_count is not None:
        sw.process_per_device = args.process_count
    if hasattr(args, "platform") and args.platform:
        sw.dev_plat = args.platform
    if hasattr(args, "proc_timeout") and args.proc_timeout:
        sw.proc_timeout = args.proc_timeout


def _apply_run_args(sw, args):
    if hasattr(args, "validate_only") and args.validate_only:
        sw.validate_only = True
    if hasattr(args, "warmup") and not args.warmup:
        sw.warmup = False
    if hasattr(args, "run") and args.run is not None:
        sw.run_time = args.run
    if hasattr(args, "npu_timeout") and args.npu_timeout:
        sw.run_timeout = args.npu_timeout
    if hasattr(args, "deterministic_level") and args.deterministic_level:
        sw.deterministic_level = args.deterministic_level


def args_to_switches(args):
    from ttk.utilities.classes import SWITCHES

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

    return sw


def apply_kernel_args(sw, args):
    if hasattr(args, "dynamic") and args.dynamic is not None:
        val = args.dynamic
        if isinstance(val, str):
            val = val.lower() not in ("false", "0", "no", "off")
        sw.dyn_switches.enabled = val
    if hasattr(args, "const") and args.const is not None:
        val = args.const
        if isinstance(val, str):
            val = val.lower() not in ("false", "0", "no", "off")
        sw.cst_switches.enabled = val
    if hasattr(args, "binary") and args.binary is not None:
        if args.binary == "release":
            sw.bin_switches.enabled = True
            sw.bin_switches.realtime = "release"
        elif args.binary is not True:
            val = args.binary
            if isinstance(val, str):
                val = val.lower() not in ("false", "0", "no", "off")
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
    if hasattr(args, "limit") and args.limit is not None:
        sw.DAVINCI_HBM_SIZE_LIMIT = args.limit
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
    if hasattr(args, "clear_ub") and args.clear_ub is not None:
        sw.force_clear_ub = _parse_clean_val("UB", args.clear_ub)
    if hasattr(args, "clear_l1") and args.clear_l1 is not None:
        sw.force_clear_l1 = _parse_clean_val("L1", args.clear_l1)
    if hasattr(args, "clear_l0") and args.clear_l0 is not None:
        sw.force_clear_l0 = _parse_clean_val("L0", args.clear_l0)
    if hasattr(args, "force_block_dim") and args.force_block_dim is not None:
        bd = args.force_block_dim
        if isinstance(bd, int):
            sw.force_block_dim = [bd] * 3
        else:
            sw.force_block_dim = list(bd)
    if hasattr(args, "core_limit") and args.core_limit is not None:
        sw.core_limit = _parse_core_limit(args.core_limit)
    if hasattr(args, "simt_ub") and args.simt_ub is not None:
        v = args.simt_ub
        if isinstance(v, int):
            sw.force_simt_ub_size = [v] * 3
        else:
            sw.force_simt_ub_size = list(v[:3])
    if hasattr(args, "simt_stack_dcu") and args.simt_stack_dcu is not None:
        sw.simt_cfg.dcu_stack = args.simt_stack_dcu
    if hasattr(args, "simt_stack_dvg") and args.simt_stack_dvg is not None:
        sw.simt_cfg.dvg_stack = args.simt_stack_dvg


def apply_aclnn_args(sw, args):
    pass


def apply_e2e_args(sw, args):
    if hasattr(args, "dynamic") and args.dynamic is not None:
        val = args.dynamic
        if isinstance(val, str):
            val = val.lower() not in ("false", "0", "no", "off")
        sw.dyn_switches.enabled = val
    if hasattr(args, "const") and args.const is not None:
        val = args.const
        if isinstance(val, str):
            val = val.lower() not in ("false", "0", "no", "off")
        sw.cst_switches.enabled = val
    if getattr(args, "cpu", False):
        sw.force_cpu = True
    if hasattr(args, "fullgraph"):
        sw.fullgraph = args.fullgraph
    if getattr(args, "aclgraph", False):
        sw.aclgraph_enabled = True
    if getattr(args, "binary", None) is not None:
        binary = args.binary
        if isinstance(binary, str):
            binary = binary.lower() not in ("false", "0", "no", "off")
        if binary:
            if not sw.aclgraph_enabled:
                raise ValueError("--binary requires --aclgraph in e2e mode")
            if getattr(args, "super_kernel", False):
                raise ValueError("--binary cannot be combined with --super-kernel")
        sw.aclgraph_online_compile = not binary
    if hasattr(args, "core_limit") and args.core_limit is not None:
        sw.core_limit = _parse_core_limit(args.core_limit)
    if getattr(args, "super_kernel", False):
        sw.super_kernel_enabled = True
        if sw.force_cpu:
            raise ValueError("--super-kernel cannot be combined with --cpu")
        if not (sw.dyn_switches.enabled or sw.cst_switches.enabled or sw.aclgraph_enabled):
            raise ValueError("--super-kernel requires a graph mode: -c/-d/--aclgraph")


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


def _configure_manual_data_prepare(sw, command, directories, complete_prepare, is_prepare_dump):
    # Kernel's legacy --no-prof dry run remains valid until a complete prepare dump is requested.
    if command == "kernel" and not complete_prepare:
        if directories:
            raise ValueError("Kernel manual-data preparation requires exactly --no-prof --dump in,golden")
        return
    if not is_prepare_dump:
        raise ValueError(
            "--no-prof requires exactly --dump in,golden or --dump in; "
            "output/full dump cannot be produced before device execution"
        )
    if sw.dump_config.file_format not in ("bin", "pt", "npy"):
        raise ValueError(
            f"--dump-format {sw.dump_config.file_format!r} is not restorable with --no-prof; use bin, pt, or npy"
        )
    if sw.dump_config.dump_on_fail:
        raise ValueError("--dump-on-fail requires comparison and cannot be used with --no-prof")
    if sw.golden_mode == "Disable" and complete_prepare:
        raise ValueError("--no-prof requires CPU golden generation; --golden-mode Disable is invalid")
    if sw.validate_only:
        raise ValueError("--validate cannot be combined with --no-prof data preparation")
    if command == "kernel" and sw.compile_only:
        raise ValueError("--compile-only cannot be combined with Kernel manual-data preparation")
    graph_enabled = any((sw.cst_switches.enabled, sw.dyn_switches.enabled, sw.fullgraph))
    if command == "e2e" and graph_enabled:
        raise ValueError("graph execution options cannot be combined with --no-prof data preparation")
    if len(directories) > 1:
        raise ValueError("--no-prof writes one dataset; specify at most one --manual-data-dirs path")
    sw.manual_data_mode = "prepare"
    sw.manual_data_dirs = directories or (_default_manual_data_dir(sw),)


def configure_manual_data(sw, args, command):
    """Validate and resolve the E2E/ACLNN/Kernel two-stage manual-data mode."""
    from ttk.utilities.classes import DumpLevel

    raw_dirs = getattr(args, "manual_data_dirs", None)
    if isinstance(raw_dirs, str):
        raw_dirs = (raw_dirs,)
    elif not isinstance(raw_dirs, (tuple, list)):
        raw_dirs = ()
    directories = tuple(
        str(pathlib.Path(item).expanduser().resolve()) for item in raw_dirs if isinstance(item, str) and item
    )

    no_prof = getattr(args, "no_prof", False) is True
    input_dump = DumpLevel.INPUT.value
    complete_dump = input_dump | DumpLevel.GOLDEN.value
    input_only = sw.dump_config.mode == input_dump
    complete_prepare = sw.dump_config.mode == complete_dump
    is_prepare_dump = no_prof and (input_only or complete_prepare)
    if not no_prof and not directories:
        return

    if not no_prof:
        if sw.golden_mode != "Enable":
            raise ValueError("--manual-data-dirs replay loads an existing golden; use --golden-mode Enable")
        if sw.validate_only:
            raise ValueError("--validate cannot be combined with --manual-data-dirs replay")
        if command == "kernel" and sw.compile_only:
            raise ValueError("--compile-only cannot be combined with Kernel manual-data replay")
        if command == "e2e" and sw.force_cpu:
            raise ValueError("--manual-data-dirs replay is the device stage and cannot use --cpu")
        sw.manual_data_mode = "replay"
        sw.manual_data_dirs = directories
        return

    _configure_manual_data_prepare(sw, command, directories, complete_prepare, is_prepare_dump)


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
    if (
        getattr(sw, "manual_data_mode", None) == "prepare"
        and not sw.plugin_path
        and sw.manual_data_dirs == (default_dir,)
    ):
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

    from ttk.core_modules.framework_api.framework_detector import detect_framework
    from ttk.utilities.table_reader import read_table

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
    from ttk.core_modules.tbe_logging import default_logging_config
    from ttk.utilities import set_global_storage

    set_global_storage(sw)
    default_logging_config(file_handler=sw.logging_to_file)
    _log_manual_data_configuration(sw)

    from ttk.utilities import set_process_name, set_thread_name

    set_process_name()
    set_thread_name()

    logging.info(f"Command: ttk {sw.test_mode} -i {sw.input_files[0] if sw.input_files else ''}")

    if sw.test_mode == "framework-api":
        from ttk.core_modules.framework_api.instance import FrameworkApiInstance

        sw.framework = _detect_framework_from_csv(sw.input_files, getattr(sw, "sheet", None))
        ins = FrameworkApiInstance()
    elif sw.test_mode == "geir":
        from ttk.core_modules.geir.instance import GeirInstance

        ins = GeirInstance()
    else:
        from ttk.core_modules.npu.instance_refactor import NpuInstance

        ins = NpuInstance()

    ins.profile()
    if ins.fail_count:
        raise SystemExit(1)


def _parse_indexes(spec):
    selected = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            selected.extend(range(int(lo), int(hi) + 1))
        else:
            selected.append(int(part))
    return tuple(selected)


def _parse_priorities(spec):
    priorities = []
    for p in spec.split(","):
        if "-" in p:
            lo, hi = p.split("-", 1)
            priorities.append((int(lo) if lo else 0, int(hi) if hi else float("inf")))
        else:
            priorities.append((int(p), int(p)))
    return tuple(priorities)


def _apply_dump_config(dump_config, value):
    if value == "full":
        dump_config.enable_all()
        return
    modes = [raw_mode.strip() for raw_mode in value.lower().split(",") if raw_mode.strip()]
    valid_modes = {"in", "out", "golden", "xpu", "full"}
    unknown_modes = [mode for mode in modes if mode not in valid_modes]
    if unknown_modes:
        raise ValueError(
            "Unsupported --dump mode(s): " + ", ".join(unknown_modes) + "; use full, in, out, golden, or xpu"
        )
    for mode in modes:
        if mode == "in":
            dump_config.enable_input()
        elif mode == "out":
            dump_config.enable_output()
        elif mode == "golden":
            dump_config.enable_golden()
        elif mode == "xpu":
            dump_config.enable_xpu()
        elif mode == "full":
            dump_config.enable_all()


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
