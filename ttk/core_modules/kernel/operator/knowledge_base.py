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
Knowledge Base Sequence for Universal testcases
"""

# Standard Packages
import logging
import os
import time
from contextlib import contextmanager

# Third-party Packages
from ....utilities import set_thread_name
from ...common.device import get_npu_hw_info
from ...common.process import process_context


class KnowledgeBaseInterface:
    """
    Class Interface for Knowledge Base
    """

    def __init__(self):
        try:
            self.interface = __import__("tbe.common.repository_manager", fromlist=["interface"]).interface
            logging.info("KnowledgeBaseInterface init success.")
        except ModuleNotFoundError as e:
            logging.info(f"Import `tbe.common.repository_manager` failed. Maybe it has been removed: {e}")
        except BaseException as e:
            logging.critical(f"KnowledgeBaseInterface init failed: {e}")
            raise e

    @contextmanager
    def knowledge_base(self):
        if hasattr(self, "interface"):
            try:
                full_soc = os.environ.get("TTK_FULL_SOC_VERSION", "")
                hw_info = get_npu_hw_info(full_soc) if full_soc else {}
                ret = self.interface.cann_kb_init(
                    {"core_num": hw_info.get("ai_core_cnt"), "soc_version": full_soc}, {}, {}
                )
                logging.debug(f"Knowledge Base initialize [cann_kb_init] return: {ret}")
            except BaseException as e:
                logging.critical(f"Knowledge Base initialize exception: {e}. The Knowledge Base process will exit.")
                raise e
            yield
            self.interface.cann_kb_finalize()
        else:
            yield


def knowledge_base_sequence():
    set_thread_name("MT")
    interface = KnowledgeBaseInterface()
    with interface.knowledge_base():
        while process_context().get_data("switch"):
            time.sleep(1)
