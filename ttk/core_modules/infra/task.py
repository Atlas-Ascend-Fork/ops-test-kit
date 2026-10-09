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
Profiling Task
"""

__all__ = ["TaskType", "TaskA", "TaskKeeper"]


# Standard Packages
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

# Third-Party Packages
from ..testcase_manager import TestcaseBase


class TaskType(Enum):
    COMPILE = auto()
    PROFILE = auto()


@dataclass
class TaskA:
    testcase: TestcaseBase
    func: Callable
    params: tuple
    type: TaskType = TaskType.PROFILE
    sub_type: str = ""
    proc_options: dict = field(default_factory=dict)
    device_ids: Optional[list] = None

    def is_multi_device(self) -> bool:
        return self.device_ids is not None and len(self.device_ids) > 1


class TaskKeeper:
    def __init__(self):
        self._tasks: Dict[TaskType, List[TaskA]] = defaultdict(list)

    def append(self, tasks: Union[TaskA, List[TaskA], Tuple[TaskA]]):
        if not isinstance(tasks, (list, tuple)):
            tasks = [tasks]
        for t in tasks:
            self._tasks[t.type].append(t)

    def insert(self, tasks: Union[TaskA, List[TaskA], Tuple[TaskA]]):
        if not isinstance(tasks, (list, tuple)):
            tasks = [tasks]
        for t in reversed(tasks):
            self._tasks[t.type].insert(0, t)

    def pop(self, typ: Optional[TaskType] = None) -> Optional[TaskA]:
        if typ is None:
            typ = (TaskType.COMPILE, TaskType.PROFILE)
        elif not isinstance(typ, (list, tuple)):
            typ = (typ,)
        for t in typ:
            if self._tasks[t]:
                return self._tasks[t].pop(0)
        return None

    def empty(self) -> bool:
        if not self._tasks:
            return True
        return all(not self._tasks[t] for t in self._tasks)
