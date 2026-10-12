#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Docker sandbox entrypoint: receive kwargs as JSON via argv[1], run executor, print envelope.

Reads a JSON file path from argv[1] containing the execute_request kwargs dict.
Runs execute_request, prints the envelope JSON to stdout. Exit code 0 on success.

Invoked as ``python -m ttk.remote.server.executor_main <kwargs.json>`` inside
the executor image (WORKDIR=/executor，即 ttk 包根).
"""

import json
import sys

from ttk.remote.server.executor import execute_request


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "usage: executor_main.py <kwargs.json>"}))
        sys.exit(1)

    with open(sys.argv[1]) as f:
        kwargs = json.load(f)

    envelope = execute_request(**kwargs)
    print(json.dumps(envelope))
    # Exit 0 whenever a valid envelope was printed (incl. ok:False / 424 / 400) —
    # the http_status field carries the result. Exit 1 only on infra failure
    # (handled above). Otherwise container.py treats 424 as exit!=0 → 500 and
    # the client's 424→sync→retry loop never fires.
    sys.exit(0)


if __name__ == "__main__":
    main()
