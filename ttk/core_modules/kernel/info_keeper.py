#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
op info keeper.
"""

__all__ = ["OpInfoKeeper", "restore_keeper_snapshot", "snapshot_keeper"]


# Standard Packages
import glob
import json
import logging
import os
from collections import defaultdict
from configparser import ConfigParser
from typing import Any, Callable, Dict, List, Optional, Union

# Third-Party Packages
from ...utilities import DTYPE_MAP, Singleton, camel_to_snake, is_main_process
from ..common.device import get_impl_base_paths, get_npu_hw_info, get_op_info_paths

_ATTR_DTYPE_MAP = {"int": int, "float": float, "bool": bool, "str": str}


class OpInfoKeeper(metaclass=Singleton):
    """
    Singleton Class to keep ops info
    """

    def __init__(self):
        self._ops_info = None
        self._alias_index = None
        self._op_func_cache = {}
        self._kernel_info_cache = {}
        self._binary_static_cache = {}
        self._binary_info_cache = {}

    @property
    def ops_info(self) -> dict:
        if self._ops_info is None:
            self._ops_info = self._load_op_info_cfg()
        return self._ops_info

    @property
    def short_soc_version(self):
        full_soc = os.environ.get("TTK_FULL_SOC_VERSION")
        if not full_soc:
            return None
        return get_npu_hw_info(full_soc).get("short_soc_version")

    @property
    def alias_index(self) -> dict:
        """Underscore-stripped lowercase alias -> canonical key.

        Fallback lookup aid for info_of: alternative spellings of an
        operator name (CamelCase op_type or snake names with different
        underscore placements) resolve to the canonical registry key.
        """
        if self._alias_index is None:
            self._alias_index = {k.replace("_", "").lower(): k for k in self.ops_info}
        return self._alias_index

    def info_of(self, op_name: str) -> dict:
        """Return full op_info dict, or None if not found."""
        info = self.ops_info.get(op_name)
        if info is None and op_name:
            key = self.alias_index.get(op_name.replace("_", "").lower())
            if key is not None:
                info = self.ops_info.get(key)
        return info

    def op_type_of(self, op_name: str) -> Optional[str]:
        """Return CamelCase op_type (e.g. 'GeGluV2') for a snake_case op_name (e.g. 'ge_glu_v2')."""
        info = self.info_of(op_name)
        return info["op_type"] if info else None

    def op_output_defined(self, op_name: str) -> bool:
        """Check if the operator has output definitions in op_info config."""
        info = self.info_of(op_name)
        return info is None or bool(info["outputs"])

    def _resolve_path(self, relative_path: str, ki: dict) -> str:
        """Resolve a relative binary path to absolute, inserting ops_group automatically."""
        kernel_root = ki["kernel_root"]
        ops_group = ki["ops_group"]
        if ops_group and f"/{ops_group}/" not in relative_path:
            parts = relative_path.split("/", 1)
            if len(parts) == 2:
                relative_path = parts[0] + "/" + ops_group + "/" + parts[1]
        return os.path.join(kernel_root, relative_path)

    def _resolve_bin_paths(self, bin_list: list, ki: dict):
        """Resolve jsonPath/binPath/jsonFilePath in bin_list entries to absolute paths."""
        if not bin_list:
            return
        for item in bin_list:
            if not isinstance(item, dict):
                continue
            for key in ("jsonPath", "binPath"):
                if key in item:
                    item[key] = self._resolve_path(item[key], ki)
            bin_info = item.get("binInfo")
            if isinstance(bin_info, dict) and "jsonFilePath" in bin_info:
                bin_info["jsonFilePath"] = self._resolve_path(bin_info["jsonFilePath"], ki)

    def kernel_info_of(self, op_name: str) -> Optional[dict]:
        """Return kernel path info for the operator's source. Cached per op_name."""
        if op_name in self._kernel_info_cache:
            return self._kernel_info_cache[op_name]
        info = self.info_of(op_name)
        if not info:
            return None
        impl_base = info["_impl_base_path"]
        if not impl_base:
            return None
        ops_group = info["_ops_group"]
        short_soc = (self.short_soc_version or "").lower()
        kernel_root = os.path.join(impl_base, "kernel")
        binary_base = os.path.join(kernel_root, short_soc)
        if ops_group:
            binary_base = os.path.join(binary_base, ops_group)
        config_dir = os.path.join(kernel_root, "config", short_soc)
        if ops_group:
            config_dir = os.path.join(config_dir, ops_group)
        result = {
            "kernel_root": kernel_root,
            "binary_config_dir": config_dir,
            "binary_base": binary_base,
            "ops_group": ops_group,
        }
        self._kernel_info_cache[op_name] = result
        logging.info(
            "[op source] %s: binary_base=%s (source=%s)",
            op_name,
            binary_base,
            info.get("_impl_source", "builtin"),
        )
        return result

    def get_operator_function(self, op_name: str) -> Optional[Callable]:
        """Load operator implementation function with cache."""
        if op_name in self._op_func_cache:
            return self._op_func_cache[op_name]
        info = self.info_of(op_name)
        if not info:
            return None
        impl_prefix = info["_impl_prefix"]
        ops_group = info["_ops_group"]
        op_file = info["opFile.value"]
        parts = [f"{impl_prefix}impl"]
        if ops_group:
            parts.append(ops_group)
        parts.extend(["dynamic", op_file])
        module_path = ".".join(parts)
        try:
            module = __import__(module_path, fromlist=[""])
            func_name = info["opInterface.value"] or op_file
            func = getattr(module, func_name, None)
            self._op_func_cache[op_name] = func
            info["_impl_type"] = "asc" if getattr(module, "__version__", "1.0.0") == "2.0.0" else "tbe"
            logging.info(
                "[op source] %s: impl=%s (source=%s, vendor=%s)",
                op_name,
                module_path,
                info.get("_impl_source", "builtin"),
                info.get("_impl_vendor") or "-",
            )
            return func
        except (ImportError, AttributeError) as e:
            logging.error(f"Failed to load operator function for {op_name}: {e}")
            self._op_func_cache[op_name] = None
            return None

    def impl_source_of(self, op_name: str) -> Optional[dict]:
        """逐算子实现来源描述（source=builtin/vendor/custom + 厂商名 + 实现根路径）。"""
        info = self.info_of(op_name)
        if info is None:
            return None
        return {
            "source": info.get("_impl_source", "builtin"),
            "vendor": info.get("_impl_vendor", ""),
            "impl_prefix": info.get("_impl_prefix", ""),
            "impl_base_path": info.get("_impl_base_path", ""),
        }

    def impl_type_of(self, op_name: str) -> str:
        """Return impl type ('asc' or 'tbe'), lazy-resolved on first call."""
        info = self.info_of(op_name)
        if not info:
            return "tbe"
        if "_impl_type" not in info:
            func = self.get_operator_function(op_name)
            if func is None:
                return "tbe"
            # get_operator_function sets info["_impl_type"] on success
        return info.get("_impl_type", "tbe")

    def binary_static_key_config_of(self, op_name: str) -> Optional[dict]:
        """Lazy-load per-operator staticKey binary config with cache."""
        if op_name in self._binary_static_cache:
            return self._binary_static_cache[op_name]

        ki = self.kernel_info_of(op_name)
        if not ki:
            self._binary_static_cache[op_name] = None
            return None

        config_dir = ki["binary_config_dir"]
        op_info = self.info_of(op_name)  # None has been checked in `kernel_info_of``
        op_type = op_info["op_type"]
        op_type_snake = camel_to_snake(op_type)
        op_file = op_info.get("opFile.value", "")

        for search_name in (op_type_snake, op_file):
            if not search_name:
                continue
            candidates = glob.glob(os.path.join(config_dir, "**", f"{search_name}.json"), recursive=True)
            if not candidates:
                continue
            sorted_candidates = sorted(candidates, key=lambda x: 0 if "legacy" in x.lower() else 1)
            full_path = sorted_candidates[-1]
            with open(full_path, encoding="UTF-8") as f:
                data = json.load(f)
            if data and "binList" in data:
                self._resolve_bin_paths(data.get("binList", []), ki)
                self._binary_static_cache[op_name] = data
                return data

        self._binary_static_cache[op_name] = None
        return None

    def binary_info_config_of(self, op_name: str) -> Optional[dict]:
        """Lazy-load per-operator simplifiedKey binary info config with cache."""
        if op_name in self._binary_info_cache:
            return self._binary_info_cache[op_name]

        ki = self.kernel_info_of(op_name)
        if not ki:
            self._binary_info_cache[op_name] = None
            return None

        config_dir = ki["binary_config_dir"]
        op_type = self.op_type_of(op_name)

        config_files = sorted(glob.glob(os.path.join(config_dir, "**", "binary_info_config.json"), recursive=True))
        if not config_files:
            self._binary_info_cache[op_name] = None
            return None

        for cfg_path in config_files:
            with open(cfg_path, encoding="UTF-8") as f:
                content = json.load(f)
            if op_type in content:
                entry = content[op_type]
                if isinstance(entry, dict):
                    self._resolve_bin_paths(entry.get("binaryList", []), ki)
                self._binary_info_cache[op_name] = entry
                return entry

        self._binary_info_cache[op_name] = None
        return None

    def _load_op_info_cfg(self) -> dict:
        keys = (
            "coreType.value",
            "opInterface.value",
            "opFile.value",
            "op.pattern",
            "dynamicShapeSupport.flag",
            "dynamicCompileStatic.flag",
            "dynamicRankSupport.flag",
            "needCheckSupport.flag",
            "dynamicFormat.flag",
        )
        full_soc = os.environ.get("TTK_FULL_SOC_VERSION")
        hw_info = get_npu_hw_info(full_soc)
        soc_lower = (hw_info.get("short_soc_version") or "").lower()
        cv_split = hw_info.get("cv_split", False)

        # Collect all op info config files: built-in → vendors(reversed) → custom(reversed)
        # Later loading overrides earlier = custom highest priority
        # Each entry is (file_path, source_type, vendor_name)
        all_op_info_paths = []

        # 1. built-in op info
        builtin_op_info_path = self._find_builtin_op_info_paths(soc_lower)
        if builtin_op_info_path:
            all_op_info_paths.extend([(f, "builtin", "") for f in builtin_op_info_path])

        # 2. vendor op info (reversed: config.ini lists highest priority first)
        vendor_op_info_paths = self._find_ext_op_info_paths("vendor", soc_lower)
        for f in reversed(vendor_op_info_paths):
            all_op_info_paths.append((f, "vendor", self._extract_vendor_name(f)))

        # 3. custom op info from ASCEND_CUSTOM_OPP_PATH (reversed, highest priority)
        # ASCEND_CUSTOM_OPP_PATH supports colon-separated multi-path; each path
        # may carry its own vendor (e.g. .../vendors/<vendor>/...), extracted per-file.
        custom_op_info_paths = self._find_ext_op_info_paths("custom", soc_lower)
        all_op_info_paths.extend([(f, "custom", self._extract_vendor_name(f)) for f in reversed(custom_op_info_paths)])

        if not all_op_info_paths:
            logging.warning("No op info config found (built-in or custom).")
            return {}

        op_info_snake: Dict[str, dict] = {}

        for f, source_type, vendor_name in all_op_info_paths:
            if f.endswith(".ini"):
                op_info = _get_op_info_from_ini_file(f, keys)
            else:
                op_info = _get_op_info_from_json_file(f, keys)

            source_info = self._derive_op_source(f, source_type, vendor_name)

            for op in op_info:
                op_interface = (
                    op_info[op]["opInterface.value"] if op_info[op]["opInterface.value"] else camel_to_snake(op)
                )
                if not op_info[op]["coreType.value"]:
                    op_info[op]["coreType.value"] = "VectorCore" if cv_split else "AiCore"
                elif len(op_info[op]["coreType.value"].split(",")) > 1:
                    #  AiCore,VectorCore or AiCore,AIV
                    op_info[op]["coreType.value"] = "AiCore"  # do not know exactly, consider it as AiCore
                if op_interface.replace("_", "").lower() != op.lower():
                    # Add & AddV2 use add as interface...
                    continue
                if op_info[op].get("opFile.value") is None:
                    op_info[op]["opFile.value"] = camel_to_snake(op)
                op_info_snake[op_interface] = op_info[op]
                op_info_snake[op_interface].update({"op_type": op})
                op_info_snake[op_interface].update(source_info)
                self._eval_attr_str(op_info_snake[op_interface]["attr"], op)
        return op_info_snake

    def _find_builtin_op_info_paths(self, soc_lower: str) -> List[str]:
        """Find built-in op info config files."""
        base = get_impl_base_paths("builtin")[0]
        # ini
        op_info_path = os.path.abspath(
            os.path.join(base, "op_info_cfg", "ai_core", soc_lower, "aic-" + soc_lower + "-ops-info.ini")
        )
        if os.path.exists(op_info_path):
            return [op_info_path]
        # json (single file or all in config dir)
        config_dir = get_op_info_paths("builtin", soc_lower)[0]
        single_json = os.path.abspath(os.path.join(config_dir, "aic-" + soc_lower + "-ops-info.json"))
        if os.path.exists(single_json):
            return [single_json]
        if os.path.isdir(config_dir):
            paths = [
                os.path.join(config_dir, f)
                for f in os.listdir(config_dir)
                if f.endswith(".json") and os.path.isfile(os.path.join(config_dir, f))
            ]
            if paths:
                return sorted(paths, key=lambda x: 0 if "legacy" in x.lower() else 1)
        logging.warning(f"Built-in op info config path does not exist for soc: {soc_lower}")
        return []

    @staticmethod
    def _find_ext_op_info_paths(source: str, soc_lower: str) -> List[str]:
        """Find op info config files from vendor/custom directories."""
        result = []
        for config_dir in get_op_info_paths(source, soc_lower):
            if not os.path.isdir(config_dir):
                continue
            for f in os.listdir(config_dir):
                full = os.path.join(config_dir, f)
                if f.endswith(".json") and os.path.isfile(full):
                    logging.info(f"Loading {source} op info from: {full}")
                    result.append(full)
        return result

    @staticmethod
    def _extract_vendor_name(path: str) -> str:
        """Extract vendor name from a path containing .../vendors/<vendor>/..."""
        vendors_marker = os.sep + "vendors" + os.sep
        if vendors_marker in path:
            return path.split(vendors_marker)[1].split(os.sep)[0]
        return ""

    @staticmethod
    def _derive_op_source(config_file_path: str, source_type: str, vendor_name: str) -> dict:
        """Derive impl_prefix, ops_group, _impl_base_path from config file path."""
        impl_prefix = ""
        if source_type in ("vendor", "custom") and vendor_name:
            impl_prefix = vendor_name + "_"

        config_filename = os.path.basename(config_file_path)
        ops_group = ""
        if config_filename.startswith("aic-") and "-ops-info-" in config_filename:
            category = config_filename.split("-ops-info-")[1].replace(".json", "")
            if category:
                ops_group = "ops_" + category

        config_dir = os.path.dirname(config_file_path)
        config_parent = os.path.dirname(config_dir)
        impl_base_path = os.path.dirname(config_parent)

        return {
            "_impl_prefix": impl_prefix,
            "_ops_group": ops_group,
            "_impl_base_path": impl_base_path,
            # 逐算子来源信息（builtin/vendor/custom + 厂商名），供日志与
            # --expect-impl-source 校验消费——自定义包未生效静默回落内置时可见。
            "_impl_source": source_type,
            "_impl_vendor": vendor_name if source_type in ("vendor", "custom") else "",
        }

    def _eval_attr_str(self, attrs: List[dict], op_name: str):
        for attr in attrs:
            attr_type: str = attr["type"]
            attr_default: Any = attr["defaultValue"]
            if attr_default is None:
                continue
            try:
                if attr_type.startswith("list"):
                    if attr_type.endswith("Bool"):
                        attr_default = attr_default.replace("true", "True").replace("false", "False")
                    attr_default = eval(attr_default)  # noqa: S307 受控数据（op_info cfg），含 list/int(...) 形态非 literal_eval 可覆盖
                elif attr_type == "bool":
                    attr_default = attr_default.lower() == "true"
                else:
                    typ = self._attr_dtype_str_to_type(attr_type)
                    attr_default = typ(attr_default)
            except Exception as e:
                # some operator may configure wrong value,
                # like  attr_spatial_scale  in RoiExtractor. SHIT...
                if is_main_process():
                    logging.warning(
                        f"Try to parse [defaultValue: {attr_default}] "
                        f"of attribute [{attr['name']}] "
                        f"for operator [{op_name}] failed: {e}"
                    )
            finally:
                attr["defaultValue"] = attr_default

    @staticmethod
    def _attr_dtype_str_to_type(s):
        return list if s.startswith("list") else eval(s)  # noqa: S307 受控数据（op_info cfg 类型名）


def _get_op_info_from_ini_file(file_path: str, keys: Union[list, tuple] = ("coreType.value",)):
    def _get_val(_cfp, _op, key, default=None):
        if _cfp.has_section(_op) and key.lower() in _cfp.options(_op):
            return _cfp.get(_op, key.lower())
        return default

    cfp = ConfigParser()
    cfp.read(file_path, encoding="UTF-8")
    ops = cfp.sections()
    return _extract_op_info(cfp, ops, keys, _get_val)


def _get_op_info_from_json_file(file_path: str, keys: Union[list, tuple] = ("coreType.value",)):
    def _get_val(full_json, _op, key, default=None):
        val = full_json
        key_section = key.split(".")
        key_section.insert(0, _op)
        for ks in key_section:
            if ks in val:
                val = val[ks]
            else:
                val = default
                break
        return val

    with open(file_path, encoding="UTF-8") as f:
        json_data: dict = json.load(f)
    ops = json_data.keys()
    return _extract_op_info(json_data, ops, keys, _get_val)


def _pre_normalize_dtype(xput: dict):
    if xput["dtype"]:
        xput["dtype"] = [DTYPE_MAP.get(d, d) for d in xput["dtype"].split(",")]


def _pre_normalize_format(xpt: dict):
    if not xpt["unknownshape_format"]:
        xpt["unknownshape_format"] = xpt["format"]
    if xpt["unknownshape_format"]:
        xpt["unknownshape_format"] = xpt["unknownshape_format"].split(",")
    else:
        xpt["unknownshape_format"] = []
    if xpt["format"]:
        xpt["format"] = xpt["format"].split(",")
    else:
        xpt["format"] = []


def _extract_op_info(all_info, ops, keys: Union[list, tuple], get_val_func: Callable) -> dict:
    op_info: Dict[str, dict] = defaultdict(dict)
    xput_key_defaults = {
        "paramType": "required",
        "valueDepend": "ignore",
        "dtype": None,
        "unknownshape_format": None,
        "format": None,
    }
    attr_key_defaults = {"type": None, "value": None, "defaultValue": None}
    for op in ops:
        for k in keys:
            op_info[op][k] = get_val_func(all_info, op, k)
        # load inputs & outputs
        for ele in ("input", "output"):
            idx = 0
            op_info[op][f"{ele}s"] = []
            while True:
                xput_name_key = f"{ele}{idx}.name"
                xput_name = get_val_func(all_info, op, xput_name_key)
                if not xput_name:
                    break
                xput: dict = {"name": xput_name}
                for k, v in xput_key_defaults.items():
                    fk = f"{ele}{idx}.{k}"
                    xput[k] = get_val_func(all_info, op, fk, v)
                _pre_normalize_dtype(xput)
                _pre_normalize_format(xput)
                op_info[op][f"{ele}s"].append(xput)
                idx = idx + 1
        # load attrs
        attrs = get_val_func(all_info, op, "attr.list")
        op_info[op]["attr"] = []
        if attrs:
            attr_list = attrs.split(",")
            for attr in attr_list:
                attr_dict: dict = {"name": attr}
                for k, v in attr_key_defaults.items():
                    fk = f"attr_{attr}.{k}"
                    attr_dict[k] = get_val_func(all_info, op, fk, v)
                op_info[op]["attr"].append(attr_dict)
    return op_info


def snapshot_keeper() -> dict:
    """Picklable cache snapshot, taken after testcase parsing filled the cache.

    只携带全量扫描产物 ops_info（纯 JSON 派生数据——forkserver worker 不继承
    主进程内存，全量 op_info 扫描结果随引导载荷传递，worker 查询全命中）。
    逐算子缓存（op_func / kernel_info / binary_*）不随行：op_func 是进程内
    加载的可调用对象（不可 pickle 也不应跨进程），其余按需惰性重建（廉价）。
    """
    keeper = OpInfoKeeper()
    if keeper._ops_info is None:
        keeper.ops_info  # noqa: B018  # property access triggers the lazy parse
    return {"ops_info": keeper._ops_info}


def restore_keeper_snapshot(snapshot: dict) -> None:
    """Restore a snapshot into this process' keeper cache.

    Injects the instance into the Singleton registry without running
    ``__init__``; alias_index 与逐算子缓存留空按需重建。恢复失败由调用方
    降级为惰性重解析兜底（快照是性能优化不是正确性前提）。
    """
    if not snapshot or "ops_info" not in snapshot:
        return
    key = (OpInfoKeeper, (), ())
    keeper = Singleton._instances.get(key)
    if keeper is None:
        keeper = OpInfoKeeper.__new__(OpInfoKeeper)
        keeper._ops_info = None
        keeper._alias_index = None
        keeper._op_func_cache = {}
        keeper._kernel_info_cache = {}
        keeper._binary_static_cache = {}
        keeper._binary_info_cache = {}
        Singleton._instances[key] = keeper
    keeper._ops_info = snapshot["ops_info"]
    keeper._alias_index = None  # 从恢复的 ops_info 一行推导，不必随行
    logging.debug(f"OpInfoKeeper snapshot restored ({len(keeper._ops_info)} entries)")
