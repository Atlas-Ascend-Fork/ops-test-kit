#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TF 参数域模块（tf/ 域）。

签名解析 —— inspect.signature 直解（TF ops are standard Python functions,
unlike torch which needs aten schema / pyi stub parsing）。

参数级可变性（ParamInfo.is_mutable / is_resource，解析期落位）—— 判定依据
是 OpDef 注册表的 input_arg 元数据（is_ref / DT_RESOURCE），而非算子名或
参数名：参数名叫 ref 的输入不全是 mutable（部分 ref 实为 resource handle）；
mutable 输入也不全叫 ref（存在 var/accum/handle 等命名）。snake_case 函数名
做 CamelCase 归一化后查 OpDef —— 注意这是代理查询：元数据描述概念算子
（ScatterDiv），运行期 wrapper 实际分发 Resource 变体（ResourceScatterDiv），
槽位对齐依赖 Resource* 从 TF1 算子机械派生（输入序保持、ref→resource 改名）
的约定。tf.raw_ops.* 的 is_ref 输入要求 TF1 ref-dtype 可变张量，TF2 无法构造
（即使在 tf.function 图内，resource 变量也会被 op_def_library 以 "be a mutable
tensor" 拒绝），CSV 的 api_name 应直接写 TF2 可调用接口（tf.compat.v1 对应的
snake_case 封装）。tf.raw_ops.Resource* 系的 DT_RESOURCE 位要求 resource 句柄：
tf.Variable 在 op 边界会被自动解引用为值，须显式传 .handle（snake_case 封装
能工作正因其内部自行取 handle 分发，raw op 直调则由本模块的
convert_variables_to_handles 代劳）。

模型读取 —— resource_param_names / mutable_flat_indexes 经 FrameworkApiInfoKeeper
单源读取签名模型（运行期钩子：handle 转换 / 图模式槽位排除与 L1 变量包装），
非 TF 前缀短路（torch 进程不触达 TF C 扩展，torch_npu 与 TF 同进程段错误）。
参数槽位/谓词类读取（is_mutable_api / mutable_slots）已上提 FrameworkAdapter
基类（模型驱动，经 keeper 单源）。

变量放置 —— np_to_tf_inputs 自身不做放置决策（随调用方作用域）：输入生成/
golden 阶段在 CPU 作用域物化（input_generate._cpu_placement_scope），执行
侧在 open 后随默认设备 NPU-born 重建，设备侧不直接消费 CPU 版。
NPU 放置的前提是 CANN 环境变量初始化可编译 —— cann-9.2.0 中
op_host/lib/linux/x86_64/ 下 libpad_infershape.so 与 libophost_custom.so
缺 TbeLoadSoAndSaveToRegistry 导出接口, 会使变量初始化编译失败(E40021),
需先移出这两个 so。
"""

import functools
import inspect
import logging
from typing import Optional

from ttk.core_modules.common.invoke import resolve_callable_str

from ..api_model import APIParamInfo, OverloadInfo, ParamInfo

_TF_SCALAR_TYPES = {"int", "float", "bool", "str", "Number", "Scalar"}

_TF_NON_TENSOR_PARAM_NAMES = frozenset(
    {
        "name",
        "name_",
    }
)


def _is_tensor_param(p: inspect.Parameter, api_name: str) -> bool:
    """Determine if a TF op parameter is a tensor parameter.

    Strategy:
    1. If annotation is tf.Tensor/tf.Variable → True.
    2. If param name is a known non-tensor name (name, axis, dims, etc.) → False.
    3. If annotation is a scalar type (int, float, bool, str) → False.
    4. tf.raw_ops.* convention: all params except 'name' are tensors.
    5. If annotation is Annotated/Any/empty → treat positional params as tensors.
    """

    import tensorflow as tf

    ann = p.annotation
    if ann is not inspect.Parameter.empty:
        try:
            if isinstance(ann, type) and issubclass(ann, (tf.Tensor, tf.Variable)):
                return True
        except TypeError:
            pass
        ann_name = getattr(ann, "__name__", str(ann))
        if ann_name in ("int", "float", "bool", "str", "dtype"):
            return False
        if p.name in _TF_NON_TENSOR_PARAM_NAMES:
            return False
        if api_name.startswith(("tf.raw_ops.", "tensorflow.raw_ops.")):
            return True
        return p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
    if p.name in _TF_NON_TENSOR_PARAM_NAMES:
        return False
    if api_name.startswith(("tf.raw_ops.", "tensorflow.raw_ops.")):
        return True
    return p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)


def _infer_param_type(p: inspect.Parameter, api_name: str) -> str:
    """Infer type string for a TF op parameter."""
    if _is_tensor_param(p, api_name):
        return "Tensor"
    ann = p.annotation
    if ann is inspect.Parameter.empty:
        return "Number"
    ann_name = getattr(ann, "__name__", str(ann))
    if ann_name in ("int", "float", "bool", "str"):
        return ann_name
    if ann_name in ("list", "tuple"):
        return "Number"
    return ann_name if ann_name in _TF_SCALAR_TYPES else "Number"


def extract_tf_params(api_name: str) -> Optional[APIParamInfo]:
    """Parse a TF op's signature and return APIParamInfo.

    Uses inspect.signature to extract parameters. TF ops (tf.raw_ops.*,
    tf.nn.*, tf.math.*) are standard Python callables.

    Args:
        api_name: e.g. 'tf.raw_ops.Add', 'tf.nn.relu'

    Returns:
        APIParamInfo with a single overload, or None on failure.
    """
    try:
        func = resolve_callable_str(api_name)
    except Exception as e:
        logging.warning(f"Cannot resolve TF api {api_name}: {e}")
        return None

    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError) as e:
        logging.warning(f"Cannot get signature for {api_name}: {e}")
        return None

    params = []
    is_raw_ops = api_name.startswith(("tf.raw_ops.", "tensorflow.raw_ops."))
    for name, p in sig.parameters.items():
        is_kw_only = p.kind == inspect.Parameter.KEYWORD_ONLY
        is_var_pos = p.kind == inspect.Parameter.VAR_POSITIONAL
        has_default = p.default is not inspect.Parameter.empty
        is_tensor = _is_tensor_param(p, api_name)

        # tf.raw_ops ops require keyword args even when inspect shows POSITIONAL_OR_KEYWORD
        if is_raw_ops and name != "name":
            is_kw_only = True

        # The 'name' parameter in TF ops is always a string, not a tensor
        if name == "name" and not is_tensor:
            is_kw_only = True

        pi = ParamInfo(
            name=name,
            type=_infer_param_type(p, api_name),
            default=p.default if has_default else None,
            is_optional=has_default or is_var_pos,
            is_keyword_only=is_kw_only,
            is_var_positional=is_var_pos,
        )
        params.append(pi)

    # 参数级可变性落位（OpDef 按名匹配；python 形参名与 input_arg 名一致时生效，
    # 改名/未收录算子保守跳过——与旧 OpDef 下标通道在真实 mutable API 上等价）
    mutable_names, resource_names = mutable_input_flags(api_name)
    if mutable_names or resource_names:
        for pi in params:
            if pi.name in mutable_names or pi.name in resource_names:
                pi.is_mutable = True
            if pi.name in resource_names:
                pi.is_resource = True

    overload = OverloadInfo(params=params)
    info = APIParamInfo(
        api_name=api_name,
        params=params,
        source="tf_inspect",
        overloads=[overload],
    )
    logging.debug(f"Parsed {api_name}: {len(params)} params from tf_inspect")
    return info


# ── OpDef 走查（填充侧：extract_tf_params 消费）─────────────────────────

# tensorflow dtypes.DT_RESOURCE 的枚举值(OpDef input_arg.type 序列化值)。
# type=0(DT_INVALID) 表示由 type_attr 描述, 显式 20 仅出现在 Resource* 系
# raw op 的 handle 输入上, 与 is_ref 互斥。
_DT_RESOURCE = 20


@functools.lru_cache(maxsize=None)
def _lookup_op_def(api_name):
    """按 API 名查 OpDef; python 函数名(scatter_div)归一化为算子名(ScatterDiv)。

    仅 TF 系 api_name 才 import tensorflow: torch 进程先加载 torch_npu 后再
    加载 TF 的 C 扩展会段错误(两大框架运行时冲突), 故非 TF 前缀直接返回。
    """
    if not isinstance(api_name, str) or not api_name.startswith(("tf.", "tensorflow.")):
        return None
    try:
        from tensorflow.python.framework import op_def_registry

        op_name = api_name.rsplit(".", 1)[-1]
        op_def = op_def_registry.get(op_name)
        if op_def is None and op_name[:1].islower():
            op_def = op_def_registry.get("".join(p.capitalize() for p in op_name.split("_")))
    except Exception:
        return None
    return op_def


def _is_mutable_input_arg(arg):
    """is_ref(TF1 mutable-ref)或 DT_RESOURCE(Resource* raw op 的 handle 位)。"""
    return arg.is_ref or arg.type == _DT_RESOURCE


def mutable_input_flags(api_name):
    """OpDef 可变性元数据 → (mutable 名字集, resource 名字集)。

    供 extract_tf_params 按参数名落 ParamInfo.is_mutable / is_resource:
    is_ref 或 DT_RESOURCE 记 mutable; DT_RESOURCE 额外记 resource(handle
    转换的驱动位)。无 OpDef / 解析失败返回空集(普通 API 无 mutable 输入)。
    """
    op_def = _lookup_op_def(api_name)
    if op_def is None:
        return frozenset(), frozenset()
    mutable = frozenset(arg.name for arg in op_def.input_arg if _is_mutable_input_arg(arg))
    resource = frozenset(arg.name for arg in op_def.input_arg if arg.type == _DT_RESOURCE)
    return mutable, resource


# ── 模型读取（运行期钩子消费：keeper → ParamInfo 单源）─────────────────


def _api_info(api_name):
    """签名模型读取（FrameworkApiInfoKeeper 单源）。

    非 TF 前缀短路返回 None —— torch 名不触达 keeper（防 torch 进程内
    TF C 扩展加载），与 _lookup_op_def 的前缀守卫同语义。
    """
    if not isinstance(api_name, str) or not api_name.startswith(("tf.", "tensorflow.")):
        return None
    from ..info_keeper import FrameworkApiInfoKeeper

    return FrameworkApiInfoKeeper().get(api_name)


def resource_param_names(api_name):
    """DT_RESOURCE 参数名集（.handle 转换的驱动位，模型读取）。"""
    info = _api_info(api_name)
    if not info:
        return frozenset()
    return frozenset(p.name for p in info.params if p.is_resource)


def mutable_flat_indexes(api_name):
    """mutable 槽的 flat 张量下标（张量序参数中的位次）。

    L1 构造（tf.Variable 包装）与图模式 input_signature 槽位排除消费：
    TF 惯例 mutable 输入居签名前部（input_args 先于 attrs），张量序位次与
    flat 输入对位。TensorList 在场时调用方自行短路（mutable-ref 算子无
    TensorList 参数，防御性约定，见 tensor_binding/graph_runner）。
    """
    info = _api_info(api_name)
    if not info:
        return ()
    return tuple(j for j, p in enumerate(info.tensors) if p.is_mutable)


# ── 运行期槽位值操作（调用期 handle 替换 / 调用后状态回读）──────────────


def convert_variables_to_handles(api_name, args=None, kwargs=None):
    """把 DT_RESOURCE 参数位的 tf.Variable 就地换成 .handle。

    Resource* raw op 的 handle 位要求 resource 张量; tf.Variable 在 op 边界
    会被自动解引用为值(dtype 不符直接报错), 必须显式传 .handle。args 按参数
    表位次对位, kwargs 按参数名对位。非 Variable 值和非 resource 位不动。
    """
    args = args if args is not None else []
    kwargs = kwargs if kwargs is not None else {}
    info = _api_info(api_name)
    if not info:
        return
    resource_positions = {i for i, p in enumerate(info.params) if p.is_resource and not p.is_keyword_only}
    resource_names = {p.name for p in info.params if p.is_resource}
    if not resource_positions and not resource_names:
        return
    import tensorflow as tf

    for i in resource_positions:
        if i < len(args) and isinstance(args[i], tf.Variable):
            args[i] = args[i].handle
    for name in resource_names & set(kwargs):
        if isinstance(kwargs[name], tf.Variable):
            kwargs[name] = kwargs[name].handle


def collect_mutable_variables(api_name, args=None, kwargs=None):
    """收集 mutable 参数位(is_ref/DT_RESOURCE)的 tf.Variable, 按参数位次升序。

    无输出有状态算子的输出语义支撑: 结果体现在变量原地更新里, 调用后回读
    这些位取值(对齐 torch inplace 的"原地位即输出位")。args 按位次、kwargs
    按名对位(位置优先, 与旧 OpDef 通道同规则); 非 Variable 位跳过。
    """
    args = args if args is not None else ()
    kwargs = kwargs if kwargs is not None else {}
    info = _api_info(api_name)
    if not info:
        return []
    import tensorflow as tf

    variables = []
    for i, p in enumerate(info.params):
        if not p.is_mutable:
            continue
        v = args[i] if i < len(args) else kwargs.get(p.name)
        if isinstance(v, tf.Variable):
            variables.append(v)
    return variables


def read_mutable_values(variables):
    """读取变量列表的当前值(原地更新后的结果), 返回 numpy 数组列表。"""
    return [v.read_value().numpy() for v in variables]


def reset_ref_variables(dev_tensors, raw_inputs):
    """把 dev_tensors 中的 tf.Variable 按位对位复位回 raw_inputs 初值。

    L2 绑定(bind_tensors, 各模式入口)与 golden 重算前(TfAdapter.reset_mutable_inputs)
    共用: 变量只创建一次(testcase.tensors), 前一模式/上一轮的原地更新须先还原。
    raw_inputs 是原始 numpy 输入, 不被任何执行阶段改写。
    """
    for i, t in enumerate(dev_tensors or ()):
        if is_ref_variable(t) and i < len(raw_inputs) and raw_inputs[i] is not None:
            t.assign(raw_inputs[i])


def is_ref_variable(value):
    """value 是否为 tf.Variable(to_device/clone 需 bypass 普通张量处理)。"""
    import tensorflow as tf

    return isinstance(value, tf.Variable)
