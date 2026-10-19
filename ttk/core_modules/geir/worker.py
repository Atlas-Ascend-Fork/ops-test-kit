#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""geir/worker.py — GEIR worker 入口 run + 模式展开 + 逐模式 _geir_run 编排。

本文件只保留编排：图执行段见 execute.py（输入 bin / 轮循环 /
pipe 协议 / dl=1 MD5 / 轮间清），比对与 dump 见 compare.py。
阶段序：输入 → 判据解析 → golden → 三方采集 → 图编译 →
图执行 → 比对 → 结果打包。manual-data 两方向（--no-prof 落盘 /
--manual-data-dirs 装载复跑）在 run 层裁决：数据集（输入+golden）与图模式
无关，no-prof 单遍落盘；复跑时存储数据逐模式恢复、golden 装载一次全程复用。
"""

__all__ = ["run"]

import gc
import logging
import os

from ttk.core_modules.kernel.golden import gen_output
from ttk.core_modules.kernel.input_generate import gen_input
from ttk.switches import get_global_storage
from ttk.utilities import is_4bit_dtype, render_table

from ..common.comparison.batch_consistency import assemble_batch_summary
from ..common.comparison.resolve import needs_promote_golden, resolve_tolerance
from ..common.dump import (
    dump_manual_dataset,
    load_replay_goldens,
    manual_data_dump_store,
    resolve_manual_case,
    snapshot_manual_values,
)
from ..common.logging import build_single_log_dir, default_logging_config
from ..common.plugin import SpecResolver
from ..common.process import DeviceLock, gate_host_memory, process_context
from ..common.result import format_xpu_metrics
from ..common.testcase.batch import resolve_deterministic_level
from ..common.testcase.input_data import reseed_per_testcase
from ..common.testcase.tensor import values_match_slots
from ..common.xpu import (
    MANUAL_XPU_PROVIDER,
    XpuDispatch,
    collect_third_party,
    dump_manual_xpu_archive,
    dump_xpu_slot_files,
    load_manual_third_parties,
    manual_xpu_results,
    overlap_deadline_s,
    snapshot_xpu_inputs,
    xpu_join_timeout,
    xpu_mode,
    xpu_overlap_enabled,
)
from .compare import CompareInputs, compare_case
from .compiler import GeirCompiler
from .execute import GraphExecPlan, execute_graph
from .graph_builder import GeirGraphBuilder
from .proto_loader import ProtoLoader
from .result import GeirReturnStructure
from .testcase import GeirTestcase


def run(
    testcase: GeirTestcase,
    device_grant_events: dict,
    device_granted_indices: dict,
    dev_id: int,
) -> GeirReturnStructure:
    """GEIR worker 入口（原 geir_profile_process 改名；Task.func 同批切换）。

    模式展开：cst/dyn 开关 × geir_binary → {const, dynamic, const_binary,
    dynamic_binary}，逐模式 _geir_run 结果写同一 GeirReturnStructure 的
    cst_/dyn_ 槽。
    """
    switches = get_global_storage()
    process_ctx = process_context()
    process_ctx.change_name(testcase.testcase_name)

    _log_dir = build_single_log_dir(switches.test_mode, testcase.testcase_name, switches.root_path)
    default_logging_config(
        file_handler=switches.logging_to_file, testcase_name=testcase.testcase_name, log_dir=_log_dir
    )

    if not testcase.is_valid:
        result = GeirReturnStructure()
        result.precision_status = "INVALID"
        result.precision = testcase.fail_reason or "INVALID"
        result.passed = False
        return result

    # --seed 用例级播种先于一切数据生成（逐模式输入/golden），与派发解耦
    reseed_per_testcase(testcase, switches)

    # host 内存闸门：入口位（校验早退之后的首个重负载前）——manual 落盘加载、
    # 输入生成、golden、图执行全在闸门之后
    gate_host_memory(switches.no_memory_check, process_ctx)

    try:
        with DeviceLock(
            process_ctx,
            dev_id,
            use_device=True,
            grant_event=device_grant_events.get(dev_id),
            granted_idx=device_granted_indices.get(dev_id),
        ):
            manual_case, stored_goldens, dump_store, early = _manual_data_stage(testcase, switches, process_ctx)
            if early is not None:
                result = early
            elif switches.no_prof:
                result = _no_prof_dump(testcase, switches, process_ctx, dump_store)
            else:
                result = None
                for mode in _expand_modes(switches):
                    mode_result = _geir_run(
                        testcase, dev_id, switches, process_ctx, mode, manual_case, stored_goldens, dump_store
                    )
                    result = _merge_mode_result(result, mode_result, mode)
                    dump_store = None  # 首跑数据集由首个模式写一次，后续模式不再落
    except Exception as e:
        logging.exception("GEIR profiling exception")
        message = str(e)
        if message.startswith(("MANUAL_DATA_PREPARE_FAILURE:", "MANUAL_DATA_READ_FAILURE:")):
            # manual 阶段失败经 raise 通道上抛：对齐 kernel early_end 的失败行
            # （precision 承载完整 "STATUS: exc"，不再落 EXEC_FAILURE/ERROR）
            status = message.split(":", 1)[0]
            result = _manual_data_failure(status, RuntimeError(message.split(":", 1)[1]))
        else:
            result = GeirReturnStructure()
            result.precision_status = f"ERROR: {str(e)[:200]}"
            result.precision = "EXEC_FAILURE"
            result.passed = False

    process_ctx.notify_status("OnReturning")
    _mark_off_modes(result, switches)
    _geir_profiling_end_print(result)
    gc.collect()
    return result


def _mark_off_modes(result, switches):
    """未启用模式槽位补 OFF 标记（对齐 kernel 屏显/CSV 的 xxx_OFF 口径）。

    以 -c/-d 开关为准——数据面 None/空串区分不了"未启用"与"执行异常"
    （异常路径 RAN 模式的槽位同为 None/空，其失败经 STATUS 行暴露，
    perf 列保持 None 屏显）。标记后 pick_data 与屏显直接读槽位即可，
    CSV perf 列的 None→CST_OFF 兜底随之退化为双保险。
    """
    if not switches.cst_switches.enabled:
        result.cst_perf_us = "CST_OFF"
        result.cst_precision = "CST_OFF"
    if not switches.dyn_switches.enabled:
        result.dyn_perf_us = "DYN_OFF"
        result.dyn_precision = "DYN_OFF"


def _expand_modes(switches) -> list:
    """模式展开（复用 Kernel 的 -c/-d 开关 + --geir-binary）。"""
    modes = []
    for base in ("const", "dynamic"):
        base_enabled = switches.cst_switches.enabled if base == "const" else switches.dyn_switches.enabled
        if not base_enabled:
            continue
        modes.append(f"{base}_binary" if switches.geir_binary else base)
    return modes


def _merge_mode_result(result, mode_result, mode):
    """逐模式结果合并进同一 GeirReturnStructure 的 cst_/dyn_ 槽。"""
    if mode in ("const", "const_binary"):
        mode_result.cst_precision = mode_result.precision
    else:
        mode_result.dyn_precision = mode_result.precision
    if result is None:
        return mode_result
    _absorb_fields(result, mode_result)
    return result


def _absorb_fields(result, mode_result):
    """后继模式结果吸收进累计结果（perf/metrics 覆盖，passed 语义取首个通过模式）。"""
    if mode_result.cst_precision:
        result.cst_precision = mode_result.cst_precision
    if mode_result.dyn_precision:
        result.dyn_precision = mode_result.dyn_precision
    if mode_result.cst_perf_us is not None:
        result.cst_perf_us = mode_result.cst_perf_us
    if mode_result.dyn_perf_us is not None:
        result.dyn_perf_us = mode_result.dyn_perf_us
    if mode_result.xpu_metrics:
        result.xpu_metrics.update(mode_result.xpu_metrics)
    if mode_result.precision_metrics:
        result.precision_metrics.update(mode_result.precision_metrics)
    if mode_result.passed and not result.passed:
        result.precision = mode_result.precision
        result.passed = True
        result.precision_status = mode_result.precision_status
        result.log = mode_result.log


def _geir_profiling_end_print(result):
    """结果后屏显：kernel 同款 render_table（MODE × Performance/Comparison）。"""
    switches = get_global_storage()
    mode_suffix = "_BIN" if getattr(switches, "geir_binary", False) else ""
    status = (
        result.precision_status
        if result.precision_status in ("PASS", "FAIL", "SKIP")
        else ("PASS" if result.passed else "FAIL")
    )
    lines = [
        ("MODE", "Performance", "Comparison"),
        (f"CST{mode_suffix}", str(result.cst_perf_us), str(result.cst_precision)),
        (f"DYN{mode_suffix}", str(result.dyn_perf_us), str(result.dyn_precision)),
        ("STATUS", status, str(result.precision_status)),
    ]
    logging.info("\n" + render_table(lines))


def _geir_param_order(testcase) -> list:
    """def.cpp param order (inputs then attrs) for server-side pool merge."""
    from ttk.core_modules.kernel.info_keeper import OpInfoKeeper

    op_info = OpInfoKeeper().info_of(testcase.op_name)
    if not op_info:
        return []
    return [inp["name"] for inp in op_info["inputs"]] + [attr["name"] for attr in op_info["attr"]]


def _manual_data_failure(status: str, exc: Exception) -> GeirReturnStructure:
    """manual-data 阶段失败结果（开仓/装载/落盘异常，整例 FAIL）。

    对齐 kernel early_end：precision 直接承载 "STATUS: exc" 全串（屏显/CSV
    直读失败原因），STATUS 列记 FAIL。
    """
    result = GeirReturnStructure()
    result.precision = f"{status}: {str(exc)[:150]}"
    result.precision_status = "FAIL"
    result.passed = False
    return result


def _manual_data_stage(testcase, switches, process_ctx):
    """manual-data 来源裁决（与 kernel/aclnn/e2e worker 同构）。

    返回 (manual_case, stored_goldens, dump_store, early_result)；
    early_result 非 None = 提前结束（开仓/装载失败）。

    - dump 方向（--no-prof）：开单目录 store（旧同名用例先失效）；
    - load 方向（--manual-data-dirs 全流程）：命中已存用例则装载输入与
      golden（golden 缺席时回退现算——no-prof 落盘可仅 --dump in）；
      目录全空且 --dump in 开启 → 首跑落盘 store（生成后写入）。
    """
    if switches.no_prof:
        try:
            return None, None, manual_data_dump_store(testcase, switches), None
        except Exception as exc:
            logging.exception("Manual data dump store failure")
            return None, None, None, _manual_data_failure("MANUAL_DATA_PREPARE_FAILURE", exc)
    if not switches.manual_data_dirs:
        return None, None, None, None
    try:
        manual_case = resolve_manual_case(
            testcase,
            "geir",
            switches,
            before_load=lambda: process_ctx.notify_status("OnLoadManualData"),
            require_goldens=False,
        )
    except Exception as exc:
        logging.exception("Manual data loading failure")
        return None, None, None, _manual_data_failure("MANUAL_DATA_READ_FAILURE", exc)
    if manual_case is None and switches.dump_config.dump_input:
        # load 方向目录全空的首跑：正常生成并落盘（后续运行即可加载）
        try:
            return None, None, manual_data_dump_store(testcase, switches), None
        except Exception as exc:
            logging.exception("Manual data dump store failure")
            return None, None, None, _manual_data_failure("MANUAL_DATA_PREPARE_FAILURE", exc)
    stored_goldens = None
    if manual_case is not None and manual_case.has_goldens:
        try:
            process_ctx.notify_status("OnLoadManualGolden")
            # 槽位声明经 testcase 协议（manual_data_golden_shapes/dtypes——
            # 含 shape-unknown 元数据槽，GeirTestcase 继承 TestcaseKernel）
            stored_goldens = load_replay_goldens(
                manual_case,
                shapes=testcase.manual_data_golden_shapes,
                dtypes=testcase.manual_data_golden_dtypes,
            )
        except Exception as exc:
            logging.exception("Manual golden loading failure")
            return None, None, None, _manual_data_failure("MANUAL_DATA_READ_FAILURE", exc)
    return manual_case, stored_goldens, None, None


def _no_prof_dump(testcase, switches, process_ctx, dump_store) -> GeirReturnStructure:
    """--no-prof：输入 → 快照 → 判据/golden（Promote 遵从）→ store 落盘。

    数据集（输入+golden）与图模式（const/dynamic × binary）无关，单遍落盘；
    三方采集 / 图编译 / 图执行全部跳过（geir 的主执行 = 图编译 + 图执行）。
    """
    try:
        process_ctx.notify_status("OnGenerateInput")
        gen_input(testcase)
        _reconcile_inputs(testcase)
        input_snapshot = snapshot_manual_values(list(_iter_flat_values(testcase.input_arrays or ())), "input")
        _, _, golden_is_xpu, _ = _resolve_stage(testcase, switches)
        write_goldens = switches.dump_config.dump_golden
        if golden_is_xpu:
            # 对齐 kernel：三方 golden 无可落盘——占位后由 dump_manual_dataset
            # 的占位串过滤仅落 input（no-prof + XPU 合法组合）
            testcase.golden_arrays = ["GOLDEN_FAILURE"]
        elif write_goldens:
            process_ctx.notify_status("OnGenerateGolden")
            _golden_stage(testcase, golden_is_xpu)
        if dump_store is None:
            raise ValueError("no manual-data output directory configured")
        process_ctx.notify_status("OnWriteManualData")
        case_dir = dump_manual_dataset(
            dump_store,
            testcase,
            "geir",
            input_snapshot,
            testcase.golden_arrays if write_goldens else (),
            switches,
        )
    except Exception as exc:
        logging.exception("GEIR manual-data dump failure")
        return _manual_data_failure("MANUAL_DATA_PREPARE_FAILURE", exc)
    logging.info("[%s] manual GEIR data dumped: %s", testcase.testcase_name, case_dir)
    result = GeirReturnStructure()
    result.precision = "MANUAL_DATA_PREPARED"
    result.precision_status = "MANUAL_DATA_PREPARED"
    result.passed = True
    return result


def _write_first_run_dump(testcase, switches, process_ctx, dump_store, input_snapshot) -> None:
    """load 方向首跑落盘：首模式输入与 golden 生成后写入数据集（输入快照 + golden）。"""
    try:
        process_ctx.notify_status("OnWriteManualData")
        case_dir = dump_manual_dataset(dump_store, testcase, "geir", input_snapshot, testcase.golden_arrays, switches)
    except Exception as exc:
        raise RuntimeError(f"MANUAL_DATA_PREPARE_FAILURE: {exc}") from exc
    logging.info("[%s] manual GEIR data dumped (first run): %s", testcase.testcase_name, case_dir)


def _geir_run(
    testcase, dev_id, switches, process_ctx, mode="const", manual_case=None, stored_goldens=None, dump_store=None
):
    """单模式编排（输入/golden/判据/三方/图编译/图执行/比对/打包，图编译嵌在图执行前）。

    manual_case/stored_goldens：load 方向复跑装载（存储输入逐模式恢复，
    golden 装载一次全程复用）；dump_store：首跑空目录落盘（仅首模式生效）。
    """
    process_ctx.notify_status("OnGenerateInput")
    testcase.device_id = dev_id

    if manual_case is not None:
        gen_input(testcase, stored_inputs=manual_case.inputs)  # 输入段：存储输入恢复
    else:
        gen_input(testcase)  # 输入段：输入生成（同 kernel 模式）
    _reconcile_inputs(testcase)  # 输入段后对账（结构 fail-fast + 字节 advisory）

    input_snapshot = None
    if dump_store is not None and manual_case is None:
        # 生成态数据将写入 store：先快照（图执行可能原地改动 backing storage）
        input_snapshot = snapshot_manual_values(list(_iter_flat_values(testcase.input_arrays or ())), "input")

    process_ctx.notify_status("OnResolveTolerance")
    standards, need_3party, golden_is_xpu, pre_compare = _resolve_stage(testcase, switches)  # 判据解析

    process_ctx.notify_status("OnGenerateGolden")
    _golden_stage(testcase, golden_is_xpu, stored_goldens)  # golden 生成

    if dump_store is not None:
        if golden_is_xpu:
            # 对齐 kernel full-flow + XPU：三方 golden 无可落盘，warning 跳过
            logging.warning(
                "[%s] golden-mode XPU: skip manual-data output (golden produced by third party)",
                testcase.testcase_name,
            )
        else:
            _write_first_run_dump(testcase, switches, process_ctx, dump_store, input_snapshot)

    xpu = _dispatch_xpu(testcase, switches, process_ctx, manual_case, need_3party)  # 三方 dispatch
    if golden_is_xpu:
        # golden-XPU：三方即 golden，设备前收割（_xpu_golden_missing 提前失败省编译执行）
        third_parties, _results, _provider = xpu.collect()
        early = _xpu_golden_missing(testcase, third_parties)
        if early is not None:
            xpu.discard()
            return early
        testcase.golden_arrays = list(third_parties)

    process_ctx.notify_status("OnGeirCompile")
    compiler, binary, config_path = _compile_stage(testcase, switches, mode)  # 图编译

    process_ctx.notify_status("OnGeirExecute")
    input_prefix = os.path.join(compiler.build_dir, f"{testcase.testcase_name}_input")
    plan = GraphExecPlan(
        binary=binary,
        config_path=config_path,
        build_dir=compiler.build_dir,
        prof_path=_prof_path(switches, testcase, mode),
        input_prefix=input_prefix,
    )
    exec_result = execute_graph(testcase, dev_id, switches, plan, pre_compare)  # 图执行（XPU 并行中）

    third_parties, xpu_results = _collect_xpu(testcase, switches, xpu)

    process_ctx.notify_status("OnGeirCompare")
    inputs = CompareInputs(
        tuple(exec_result.outputs), tuple(testcase.golden_arrays), standards, third_parties, pre_compare
    )
    # store 已落输入/golden 时跳过常规数据 dump（输出仍落——来自执行，store 表达不了）
    compared = compare_case(
        testcase, mode.replace("_binary", ""), inputs, skip_stored_data=dump_store is not None
    )  # 比对 + dump

    # dl=3 batch 一致性：比对后切片哈希——按 batch_axis 切
    # output 0 逐样本 MD5 → BatchSummary（打包时随 ReturnStructure 回传，主进程
    # cleanup 按 id 分组比对）；不支持切片时记录原因不阻断。
    batch_summary = None
    if resolve_deterministic_level(switches, testcase) == 3:
        batch_summary = assemble_batch_summary(list(exec_result.outputs), testcase, testcase.batch_consistency_id)

    result = _pack_result(mode, exec_result, compared, xpu_results, testcase)  # 结果打包
    result.batch_summary = batch_summary
    return result


def _resolve_stage(testcase, switches) -> tuple:
    """判据解析（提前到 golden 前；SpecResolver 直用，无吞错包装）。

    唯一目的：判据要求升精度时设 golden_mode_override 让 golden 走 Promote
    高精度真值（与 kernel/worker.py 的 tolerance 前置形态对齐）。
    pre_compare 一并 resolve（双消费点：比对段一次 + dl≥1 逐轮规范化）。
    """
    resolver = SpecResolver(testcase.op_name, "geir", getattr(switches, "plugin_path", None))
    standards = resolve_tolerance(
        resolver.resolve_tolerance(),
        testcase.flat_precision_tolerances,
        testcase.flat_absolute_precision,
        testcase.flat_output_dtypes,
        switches.compare_method,
        input_dtypes=testcase.flat_input_dtypes,
    )
    need_3party_outputs = any(s.token == "cross_check" for s in standards)  # noqa: S105  # token 为比对标准名，非口令
    # golden-mode XPU：三方输出升格为 golden 位——与 Promote 互斥，
    # golden 函数旁路不生成，三方执行后回填；失败整例 FAIL。
    golden_is_xpu = switches.golden_mode == "XPU"
    # cross_check 三方 / mixed·mix_tolerance 单标杆均升精度；Disable 是显式
    # 豁免（golden 不生成），升精度无从谈起，不得覆盖（与 kernel/worker.py 对齐）
    if needs_promote_golden(standards) and not golden_is_xpu and switches.golden_mode != "Disable":
        testcase.golden_mode_override = "Promote"
    return standards, need_3party_outputs, golden_is_xpu, resolver.resolve_pre_compare()


def _reconcile_inputs(testcase) -> None:
    """输入段后对账。

    - 结构（fail-fast）：values_match_slots——生成 input_arrays 与声明槽表逐位
      一致（None 占位 / TensorList 嵌套）。      optional 错位/槽型不匹配若漏到执行段，
      表现为 bin 落盘错位→C++ 读到错位数据→神秘比对失败，此处提前显式暴露。
    - 字节（advisory）：逐槽声明 nbytes vs 实际数组 nbytes——dtype/shape 漂移
      预警（pre_inputs 契约为原地改数据；漂移即契约违例信号，先预警观察）。
    对账基础设施自身不得破坏通路：槽物化失败仅 debugc 记录后跳过。
    """
    try:
        slots = testcase.input_tensor_slots
    except Exception as exc:
        logging.debugc("[GEIR/%s] input_tensor_slots unavailable, skip reconcile: %s", testcase.op_name, exc)
        return
    if not slots:
        return
    if not values_match_slots(slots, testcase.input_arrays):
        raise RuntimeError(
            f"[{testcase.testcase_name}] input structure reconciliation failed: generated "
            f"input_arrays mismatch declared tensor slots (nesting / None placement)."
        )
    _reconcile_input_bytes(testcase, slots)


def _reconcile_input_bytes(testcase, slots) -> None:
    """逐槽声明字节 vs 实际字节（advisory 预警，不阻断）。

    跳过：None 占位（槽/值任一）、非具体维声明（动态 -1）、4bit（unpacked
    存储，nbytes 声明口径差半）、未知 dtype 宽度（nbytes 会抛 RuntimeError）。
    """
    drift = []
    for slot, value in zip(_iter_flat_slots(slots), _iter_flat_values(testcase.input_arrays)):
        if slot is None or value is None:
            continue
        if any(isinstance(d, int) and d < 0 for d in slot.shape):
            continue
        try:
            expected = slot.nbytes
        except RuntimeError:
            continue
        if slot.dtype and is_4bit_dtype(slot.dtype):
            continue
        actual = getattr(value, "nbytes", None)
        if expected and actual is not None and actual != expected:
            drift.append(f"slot#{slot.name or '?'}: declared {expected}B vs actual {actual}B")
    if drift:
        logging.warning(
            "[%s] input byte drift (dtype/shape drift warning, non-blocking): %s",
            testcase.testcase_name,
            "; ".join(drift[:4]),
        )


def _iter_flat_slots(slots):
    """槽表深展平（与 values 结构逐位对齐，None 叶子保留——flat_slots 语义一致）。"""
    for slot in slots:
        if isinstance(slot, (list, tuple)):
            yield from _iter_flat_slots(slot)
        else:
            yield slot


def _iter_flat_values(values):
    """生成值深展平（None 叶子保留；与槽对齐性已由结构校验保证）。"""
    for value in values:
        if isinstance(value, (list, tuple)):
            yield from _iter_flat_values(value)
        else:
            yield value


def _golden_stage(testcase, golden_is_xpu, stored_goldens=None) -> None:
    """golden 生成（复用 kernel/golden.py；GEIR 逐模式重生成——已扩展过的输出
    元数据不重复增长，_output_meta_count 判定，66c6491）。

    stored_goldens（manual-data 复跑装载）先切齐声明输出槽数再传入：存储集
    含 shape-unknown 元数据槽，而元数据 golden 由 gen_output 依各 data golden
    的 shape 重算追加——首模式（flat 未扩展）与后续模式（flat 已扩展）切齐后
    两侧槽数一致，不因逐模式重入而重复追加。"""
    if golden_is_xpu:
        # golden 由三方采集产出（本侧参照不生成）；回填在 collect 之后
        testcase.golden_arrays = ["GOLDEN_FAILURE"]
    else:
        if stored_goldens is not None:
            declared = len(testcase.flat_output_shapes) - testcase.shape_unknown_slot_count
            stored_goldens = list(stored_goldens[:declared])
        try:
            gen_output(testcase, stored_goldens=stored_goldens)
        finally:
            if hasattr(testcase, "golden_mode_override"):
                del testcase.golden_mode_override
    testcase.output_arrays = ()


def _dispatch_xpu(testcase, switches, process_ctx, manual_case, need_3party) -> XpuDispatch:
    """geir 三方采集统一入口（dispatch/collect 两段式）。

    分流：manual 源 / golden-XPU（升格 golden + 提前失败）/ 总闸关闭 →
    ready（内联同步完成）；否则 cross_check / PERF / dump_xpu →
    background（NPU 执行期重叠）。collect 统一返回
    (third_parties, xpu_results, provider)。
    """
    golden_is_xpu = switches.golden_mode == "XPU"
    need_data = need_3party or golden_is_xpu
    if need_data:
        manual = load_manual_third_parties(testcase, manual_case)
        if manual is not None:
            process_ctx.notify_status("OnLoadManualXpu")
            dump_manual_xpu_archive(testcase, manual)
            return XpuDispatch.ready((manual, manual_xpu_results(), MANUAL_XPU_PROVIDER))
    if not xpu_mode(switches, need_data):
        return XpuDispatch.ready((None, None, None))
    if golden_is_xpu or not xpu_overlap_enabled():
        process_ctx.notify_status("OnXpuProfiling")
        return XpuDispatch.ready(_collect_xpu_remote(testcase, switches, need_data))
    # 重叠：主线程派生+快照（防图执行原地改写 backing storage vs 线程序列化读竞争）
    inputs = snapshot_xpu_inputs(getattr(testcase, "original_input_arrays", None) or testcase.input_arrays)
    budget = overlap_deadline_s(switches)
    process_ctx.notify_status("OnXpuDispatch")
    return XpuDispatch.background(
        _collect_xpu_remote, testcase, switches, need_data, inputs, budget, join_timeout=xpu_join_timeout(budget)
    )


def _collect_xpu(testcase, switches, xpu) -> tuple:
    """NPU 执行完毕的三方收割（ready 即取/幂等；重叠形态 join）+ case 目录槽位落盘。"""
    process_ctx = process_context()
    if xpu.pending:
        process_ctx.notify_status("OnXpuCollect")
    third_parties, xpu_results, xpu_provider = xpu.collect()
    dump_xpu_slot_files(testcase, switches, third_parties, xpu_provider)
    return third_parties, xpu_results


def _collect_xpu_remote(testcase, switches, need_data, inputs=None, deadline_s=None) -> tuple:
    """geir 远端采集（同步形态 = 现状语义；重叠形态由 _dispatch_xpu 以快照
    inputs/deadline 调起）。经 common.xpu 门面，GEIR 的 input_names 来自
    ProtoLoader，op_type 无概念传 None（服务端靠 op_name 推导）。"""
    if inputs is None:
        inputs = getattr(testcase, "original_input_arrays", None) or testcase.input_arrays
    priority, third_parties, xpu_results = collect_third_party(
        op_name=testcase.op_name,
        inputs=inputs,
        input_names=_proto_input_names(testcase),
        op_type=None,
        attributes=getattr(testcase, "attributes", {}) or {},
        testcase_name=testcase.testcase_name,
        switches=switches,
        need_data=need_data,
        param_order=_geir_param_order(testcase),
        input_formats=getattr(testcase, "input_formats", None),
        input_dtypes=getattr(testcase, "input_dtypes", None),
        deadline_s=deadline_s,
    )
    return third_parties, xpu_results, (priority or "xpu")


def _proto_input_names(testcase) -> list:
    try:
        proto_info = ProtoLoader().get_op_info(testcase.op_name)
        return proto_info.inputs[:] if proto_info else []
    except Exception as exc:
        logging.debugc("[GEIR/%s] proto input_names unavailable, fallback to []: %s", testcase.op_name, exc)
        return []


def _xpu_golden_missing(testcase, third_parties):
    """golden-mode XPU：三方输出缺失 = golden 位真值源缺失，整例 FAIL——在
    compile/execute 前显式落败（编译执行链没必要跑）。非缺失返回 None。"""
    if third_parties is not None:
        return None
    logging.error(
        "[%s] golden-mode XPU: no third_party output (no third_party declaration / XPU down); case FAIL",
        testcase.testcase_name,
    )
    result = GeirReturnStructure()
    result.precision = "GOLDEN_FAILURE"
    result.passed = False
    result.precision_status = "FAIL"
    return result


def _compile_stage(testcase, switches, mode) -> tuple:
    """图编译：proto 渲染 C++ 源码（op 级缓存）→ per-case 配置 → g++ 编译（stamp 缓存）。"""
    builder = GeirGraphBuilder(switches)
    source_path = builder.generate_op_source(testcase.op_name, mode=mode)
    if source_path is None:
        raise RuntimeError(
            f"GEIR op source generation failed for operator '{testcase.op_name}': "
            f"its IR (REG_OP) was not found in the op_proto headers, or source rendering failed. "
            f"See the log for the searched proto paths."
        )
    config_path = builder.write_case_config(testcase, mode=mode)
    if config_path is None:
        raise RuntimeError("GEIR case config generation failed")
    compiler = GeirCompiler(switches, build_dir=builder.work_dir)
    binary = compiler.compile_op(source_path, testcase.op_name, op_dir=builder.op_dir)
    if binary is None:
        raise RuntimeError("GEIR compilation failed")
    return compiler, binary, config_path


def _prof_path(switches, testcase, mode) -> str:
    if not getattr(switches, "task_profiling", False):
        return ""
    root = getattr(switches, "root_path", os.getcwd())
    prof_path = os.path.join(root, "msprof", "geir", testcase.testcase_name, mode)
    os.makedirs(prof_path, exist_ok=True)
    return prof_path


def _pack_result(mode, exec_result, compared, xpu_results, testcase):
    """打包：mode → cst_/dyn_ 槽位 + det_status + xpu_metrics + bcid（passed 三态透传）。"""
    precision, log_str, passed, metrics = compared
    xpu_metrics = format_xpu_metrics(xpu_results) if xpu_results else {}
    result = GeirReturnStructure()
    result.precision = precision
    result.passed = passed
    if passed is None:
        result.precision_status = "SKIP"
    else:
        result.precision_status = "PASS" if passed else "FAIL"
    result.log = log_str
    result.xpu_metrics = xpu_metrics
    result.deterministic_status = exec_result.det_status
    # batch 一致性 id 回传（在 TestcaseBase 派生；对齐 framework_api 形态，
    # dl=3 时 CSV 出非空 batch_consistency_id 列）
    result.batch_consistency_id = getattr(testcase, "batch_consistency_id", None)
    if mode in ("const", "const_binary"):
        result.cst_perf_us = exec_result.device_perf_us
        result.precision_metrics = {"cst": metrics or {}}
    elif mode in ("dynamic", "dynamic_binary"):
        result.dyn_perf_us = exec_result.device_perf_us
        result.precision_metrics = {"dyn": metrics or {}}
    return result
