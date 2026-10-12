#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Small, mode-neutral deterministic and batch-relation contracts."""

from collections.abc import Mapping
from numbers import Integral

BATCH_RELATION_FIELDS = ("batch_axis", "batch_slice_info", "batch_seed")


def has_complete_batch_relation(testcase):
    """Return whether all fields needed to construct a batch relation exist."""
    return all(getattr(testcase, name, None) is not None for name in BATCH_RELATION_FIELDS)


def batch_relation_kwargs(testcase):
    """Return all batch fields together, or no fields for a partial relation."""
    if not has_complete_batch_relation(testcase):
        return {}
    return {name: getattr(testcase, name) for name in BATCH_RELATION_FIELDS}


def resolve_deterministic_level(switches, testcase):
    """Resolve a case level while keeping a nonzero CLI level authoritative.

    ``batch_deterministic_level`` is an assets-private CSV attribute.  TTK only
    consumes it as an execution option when the command line did not select a
    nonzero level, so one input CSV can mix ordinary and deterministic cases.
    """
    cli_level = int(getattr(switches, "deterministic_level", 0) or 0)
    if cli_level:
        return cli_level

    attributes = getattr(testcase, "attributes", None)
    if not isinstance(attributes, Mapping):
        attributes = {}
    value = attributes.get("batch_deterministic_level")
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(
            f"{getattr(testcase, 'testcase_name', '<unknown>')}: "
            "attributes.batch_deterministic_level must be an integer from 0 through 3"
        )
    level = int(value)
    if level not in (0, 1, 2, 3):
        raise ValueError(
            f"{getattr(testcase, 'testcase_name', '<unknown>')}: "
            "attributes.batch_deterministic_level must be from 0 through 3"
        )
    return level
