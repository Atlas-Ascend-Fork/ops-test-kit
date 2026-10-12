#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""调度任务单元与任务队列。

- ``task_type: str``：类型串由通路自定义（"compile_dyn" / "profile" / ...），
  新增 task 类型不改 common；编译类 task 的模式经 ``compile_mode`` 属性
  还原（"compile_dyn"→"Dyn"），消费方按 ``startswith("compile")`` 分支
- ``pool: str``：进程池路由键——TaskKeeper.pop(pool=...) 按池过滤，
  池键不匹配则取不到，编译/执行 task 由此物理隔离
- ``depends_on``：依赖的 task_id 列表，全部完成后本 task 才可弹出
- ``is_final``：本 task 完成 = 用例完成（进度/摘要计数依据）
- ``device_ids``：多卡自描述（None=调度自选；len>1=多卡；len==1=指定单卡）

``Task.testcase`` 为鸭子类型——本包不 import 具体用例类，数据经参数流动。
"""

__all__ = ["Task", "TaskKeeper"]

# Standard Packages
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Set, Union


@dataclass
class Task:
    """调度任务单元。testcase/func/params 必填，其余字段带默认值即可跑单池通路。"""

    testcase: Any  # "TestcaseBase"——鸭子类型（worker 内全部数据来源）
    func: Callable  # 模块级 worker 函数（可 pickle，forkserver 约束）
    params: tuple  # func 的位置参数（不含 dev_id——由 ProcessGroup.push 注入 kwargs）

    task_type: str = "profile"  # 通路自定义类型串（"profile" / "compile_dyn" / ...）
    pool: str = "profile"  # 进程池路由键（执行池统一 "profile"；Kernel 双池另加 "compile"）
    depends_on: List[str] = field(default_factory=list)  # 依赖的 task_id
    task_id: str = ""  # 自动生成（uuid4[:8]，append 时若空则补）
    is_final: bool = True  # True: 本 task 完成 = 用例完成（进度/摘要计数依据）
    device_ids: Optional[list] = None  # None=调度自选；len>1=多卡；len==1=指定单卡
    proc_options: dict = field(default_factory=dict)  # 扩展位（超时/env 等）

    def __post_init__(self):
        if not self.task_id:
            self.task_id = uuid.uuid4().hex[:8]

    @property
    def is_multi_device(self) -> bool:
        return self.device_ids is not None and len(self.device_ids) > 1

    @property
    def compile_mode(self) -> str:
        """compile 类 task 的编译模式（"compile_dyn"→"Dyn"，与 Kernel 构造
        task_type=f"compile_{mode.lower()}" 对偶还原）。仅对 startswith("compile")
        的 task_type 有意义——调用方已按此前缀分支。"""
        prefix = "compile_"
        body = self.task_type[len(prefix) :] if self.task_type.startswith(prefix) else self.task_type
        return body.capitalize()

    @property
    def label(self) -> str:
        name = getattr(self.testcase, "testcase_name", "?")
        return f"{self.task_type}:{name}"


class TaskKeeper:
    """任务队列：依赖解析 + 池过滤分发。仅主进程使用。

    不变式：
    1. pop() 返回的 task 的 depends_on 已全部 done
    2. task_id 全局唯一（uuid4[:8] 自动生成）
    3. empty() 为真 ⇔ 队列内无未弹出 task（done 集仅供依赖判定）
    """

    def __init__(self):
        self._tasks: List[Task] = []
        self._done: Set[str] = set()

    def append(self, tasks: Union[Task, List[Task], tuple]):
        """入队（task_id 为空时自动补）。"""
        if not isinstance(tasks, (list, tuple)):
            tasks = [tasks]
        for t in tasks:
            if not t.task_id:
                t.task_id = uuid.uuid4().hex[:8]
            self._tasks.append(t)

    def insert(self, tasks: Union[Task, List[Task], tuple]):
        """队首插入（多卡 task 无空闲执行池时的放回；亦用于编译 task 优先插队）。"""
        if not isinstance(tasks, (list, tuple)):
            tasks = [tasks]
        for t in reversed(tasks):
            if not t.task_id:
                t.task_id = uuid.uuid4().hex[:8]
            self._tasks.insert(0, t)

    def pop(self, pool: Optional[str] = None) -> Optional[Task]:
        """弹出队首「depends_on 全部完成 且 pool 匹配」的 task；无则 None。

        pool=None 不过滤（全部池）。依赖未就绪的 task 留队，不阻塞后续。"""
        for idx, t in enumerate(self._tasks):
            if pool is not None and t.pool != pool:
                continue
            if any(dep not in self._done for dep in t.depends_on):
                continue
            return self._tasks.pop(idx)
        return None

    def done(self, task_id: str) -> None:
        """标记完成，解锁 depends_on 它的 task（幂等）。"""
        self._done.add(task_id)

    def count(self, pool: Optional[str] = None) -> int:
        if pool is None:
            return len(self._tasks)
        return sum(1 for t in self._tasks if t.pool == pool)

    def empty(self) -> bool:
        return not self._tasks
