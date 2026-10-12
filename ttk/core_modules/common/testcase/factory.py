#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Universal testcase manager for csv support
"""

__all__ = ["UniversalTestcaseFactory"]


# Standard Packages
import collections
import logging
from typing import Any, Dict, List, Optional

from ....utilities import set_process_name, set_thread_name

# Third-Party Packages
from .base import TestcaseBase


class PLACEHOLDER:
    """
    Simple Placeholder
    """


# 载入阶段被丢弃的用例统计（喂入 − 载入）。解析与最终汇总都在主进程, 不跨进程共享。
# 为什么要有它: 用例被跳过只按 reason 各自 debug 一行, 默认日志级看不见, 而 Summary 的
# Total 是**载入条数**不是**喂入条数** —— 喂 200 跑 181 也会报 PassRate 98.34%,
# 分母悄悄变小且毫无提示（如 is_enabled=FALSE 的行被静默跳过）。
LOAD_STATS: "collections.Counter" = collections.Counter()

_SKIP_REASON_DESC = {
    "disabled": "is_enabled=FALSE",
    "soc": "current soc not in soc_series",
    "name_filter": "not in --testcase-name selection",
    "index_filter": "not in --testcase-index selection",
    "op_filter": "not in --op-name selection",
    "priority_filter": "not in --priority selection",
    "rerun_filter": "not in rerun selection",
}


class UniversalTestcaseFactory:
    """Table → 用例对象列表。

    纯函数化：无 IO（reader 的事）、无 switches 依赖（筛选归聚合层 instance）、
    无模式推断（testcase_class 参数化——测试模式由 CLI 子命令显式指定）。
    筛选与重名处理不在此类（instance 聚合层）。
    """

    __slots__ = [
        "raw_data",
        "header",
        "real_header_indexes",
        "testcase_instance",
        "testcases",
        "skip_stats",
        "skip_names",
    ]

    def __init__(self, table, testcase_class: type, skip_validate: bool = False):
        """table: reader.Table（或 (header, rows) 二元组）；testcase_class:
        TestcaseBase 子类（通路绑定——Instance.testcase_class）。"""
        if hasattr(table, "header"):
            header, rows = table.header, table.rows
        else:
            header, rows = table
        self.header: List[str] = list(header)
        self.raw_data: List[List[str]] = [list(row) for row in rows]
        self.real_header_indexes: Dict[str, int] = {}
        self.testcase_instance: TestcaseBase = testcase_class()
        self.testcases: List[TestcaseBase] = []
        set_process_name("TestcaseManager")
        set_thread_name("Initialization")
        self._testcase_hdr_check()
        self._parse_testcase(skip_validate=skip_validate)
        set_process_name()
        set_thread_name()

    def get(self) -> List[TestcaseBase]:
        """
        Get all testcases
        :return:
        """
        return self.testcases

    def _testcase_hdr_check(self):
        set_thread_name("HeaderCheckTestcaseName")
        # Testcase name generation
        if "testcase_name" not in self.header:
            logging.warning(
                "Testcase name not found! It is important to add a testcase_name in order to identify your testcases"
            )
            self.header.append("testcase_name")
            for idx, row in enumerate(self.raw_data):
                row.append(f"auto_testcase_name_{idx + 1}")

        set_thread_name("HeaderCheckUnidentifiedHeaders")
        ignored_headers = []
        for actual_header in self.header:
            if not self.testcase_instance.is_legit_header(actual_header):
                ignored_headers.append(actual_header)
        if ignored_headers:
            logging.warning(f"Detected unidentified headers and will be ignored: {ignored_headers}")

        set_thread_name("HeaderCheckDuplicateHeaders")
        # Check duplicates
        header_check_set = set()
        for actual_header in self.header:
            if actual_header in header_check_set and actual_header not in ignored_headers:
                logging.error(f"Detected duplicate header: {actual_header}")
                raise RuntimeError(f"Detected duplicate header: {actual_header}")
            header_check_set.add(actual_header)

    def _parse_testcase(self, skip_validate: bool = False):
        set_thread_name("HeaderParseMapping")
        # Get idx of required testcase header in real header
        for header_name in self.testcase_instance.get_all_legit_headers():
            header_idx = self._get_idx_of_header(header_name)
            self.real_header_indexes[header_name] = header_idx

        set_thread_name("TestcasePreParsing")
        raw_testcases = []

        # Process each line
        for idx, line in enumerate(self.raw_data):
            try:
                sub_result = self._process(line, idx)
            except:
                logging.exception(f"Failed to process row index [{idx}]. Whole row is: {','.join(line)}")
                raise
            raw_testcases.append(tuple(sub_result.values()))
        logging.debug(f"Processed {len(raw_testcases)} raw testcases")
        set_thread_name("FinalTestcaseStructureFormation")
        self._parse(raw_testcases, skip_validate=skip_validate)

    def _get_idx_of_header(self, header_name, searched_tag=None) -> Optional[int]:
        """
        Get header position in actual header sequence
        :param header_name: string name of the header
        :param searched_tag: DO NOT USE
        :return: index of the header or its equivalent
        """
        result = None
        if header_name in self.header:
            result = self.header.index(header_name)
        else:
            equivalents = self.testcase_instance.get_equivalent_headers(header_name)
            # Header not found, search for equivalent
            if not equivalents:
                return result
            # Initialize recursion variable
            if searched_tag is None:
                searched_tag = []
            for equivalent in equivalents:
                if equivalent not in searched_tag:
                    searched_tag.append(header_name)
                    result = self._get_idx_of_header(equivalent, searched_tag)
                    if result is not None:
                        break
        return result

    def _process(self, row: list, row_index: int) -> Dict[str, Any]:
        # Initialize full testcase fields
        result = dict(
            zip(
                self.testcase_instance.get_all_legit_headers(),
                [PLACEHOLDER() for _ in self.testcase_instance.get_all_legit_headers()],
            )
        )
        changed, queue = self._process_over_result(result, row, row_index)
        while changed:
            changed, queue = self._process_over_result(result, row, row_index, queue)
        if queue:
            changed, queue = self._process_over_result(result, row, row_index, queue, True)
        while changed:
            changed, queue = self._process_over_result(result, row, row_index, queue)
        if queue:
            raise RuntimeError(f"Could not determine value of missing fields of row {row_index}: {queue}")
        return result

    def _process_over_result(self, result, row, row_index, result_keys=None, apply_default=False):
        changed = False
        placeholder_queue = []
        keys = result_keys if result_keys else result
        for current_header_name in keys:
            column_value = result[current_header_name]
            # Parsed line
            if not isinstance(column_value, PLACEHOLDER):
                continue
            header_raw_idx = self.real_header_indexes[current_header_name]
            # Header not exist or value is empty, check for equivalent and default value
            if header_raw_idx is None or header_raw_idx >= len(row) or not row[header_raw_idx]:
                if self.testcase_instance.has_equivalent_header(current_header_name) and not apply_default:
                    # Check for equivalent
                    resolved = False
                    value = None
                    for equivalent_header in self.testcase_instance.get_equivalent_headers(current_header_name):
                        equivalent_header_value = result[equivalent_header]
                        if isinstance(equivalent_header_value, PLACEHOLDER):
                            # Equivalent not exist either, check for next equivalent
                            continue
                        value = equivalent_header_value
                        resolved = True
                        break
                    if not resolved:
                        placeholder_queue.append(current_header_name)
                        continue
                elif self.testcase_instance.has_default_value(current_header_name):
                    # Check for default value
                    changed = True
                    default_value = str(self.testcase_instance.get_default_value(current_header_name))
                    result[current_header_name] = self.testcase_instance.get_header_func(current_header_name)(
                        default_value
                    )
                    continue
                else:
                    placeholder_queue.append(current_header_name)
                    continue
            else:
                value = row[header_raw_idx]
            try:
                changed = True
                if isinstance(value, str):
                    result[current_header_name] = self.testcase_instance.get_header_func(current_header_name)(value)
                else:
                    result[current_header_name] = value
            except Exception:
                logging.exception(
                    f"Failed to process header [{current_header_name}] of row [{row_index}] with value [{value}]"
                )
                raise
        return changed, placeholder_queue

    def _parse(self, testcases: list, skip_validate: bool = False):
        for testcase_idx, testcase in enumerate(testcases):
            # Construct Testcase Structure
            testcase_struct = type(self.testcase_instance)()
            testcase_struct.original_line = self.raw_data[testcase_idx]
            testcase_struct.original_dict = dict(zip(self.header, self.raw_data[testcase_idx]))
            testcase_struct._csv_row_index = testcase_idx
            headers = testcase_struct.get_all_legit_headers()
            unidentified_headers = []
            for header_idx, header in enumerate(headers):
                if hasattr(testcase_struct, header):
                    setattr(testcase_struct, header, testcase[header_idx])
                else:
                    unidentified_headers.append(header)
            if unidentified_headers:
                raise KeyError(f"TestcaseManager header not match. Report Bug to us: {unidentified_headers}")
            if not skip_validate:
                set_thread_name(testcase_struct.testcase_name)
                testcase_struct.validate()
            self.testcases.append(testcase_struct)
