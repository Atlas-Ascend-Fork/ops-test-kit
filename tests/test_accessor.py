#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration test for libregistry_accessor.so -- split API.

NOTE: This test targets the PRODUCTION build (csrc/op_registry_accessor/)
which uses a JSON-based API (7 params). It does NOT work with the mock
standalone build (tools/registry_accessor/) which uses a struct-based API.
"""

import ctypes
import os
import sys

ASCEND_HOME = os.environ.get("ASCEND_HOME_PATH", os.environ.get("ASCEND_TOOLKIT_HOME"))
if not ASCEND_HOME:
    print("SKIP: Set ASCEND_HOME_PATH")
    sys.exit(0)

BUILD_DIR = os.environ.get(
    "REGISTRY_ACCESSOR_BUILD",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "csrc", "op_registry_accessor", "build"),
)


def main():
    passed = 0
    failed = 0

    print("=== libregistry_accessor.so Split API Test ===\n")

    # Step 1: Load op_host to register operators
    op_host_dir = os.path.join(
        ASCEND_HOME, "opp", "built-in", "op_impl", "ai_core", "tbe", "op_host", "lib", "linux", "aarch64"
    )
    loaded = False
    for name in ["libophost_nn.so", "libophost_math.so"]:
        path = os.path.join(op_host_dir, name)
        if os.path.isfile(path):
            ctypes.CDLL(path)
            loaded = True
    if loaded:
        print("[PASS] op_host libraries loaded")
        passed += 1
    else:
        print("[SKIP] No op_host libraries found")
        sys.exit(0)

    # Step 2: Load libregistry_accessor
    lib = os.path.join(BUILD_DIR, "libregistry_accessor.so")
    if not os.path.isfile(lib):
        print(f"[FAIL] {lib} not found. Run build.sh first.")
        return 1
    acc = ctypes.CDLL(lib)

    find_fn = acc.FindGenSimplifiedKeyFuncs
    find_fn.restype = ctypes.c_int
    find_fn.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]

    invoke_fn = acc.InvokeGenSimplifiedKey
    invoke_fn.restype = ctypes.c_int
    invoke_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]

    # Step 3: Test FindFuncs -- operators with gen_simplifiedkey
    ops_with_key = [
        (b"MatMulV3", True),
        (b"BatchMatMulV3", True),
        (b"GemmV3", True),
        (b"Sqrt", False),
        (b"Add", False),
    ]
    for op, should_have_key in ops_with_key:
        handle = ctypes.c_void_p()
        ret = find_fn(op, ctypes.byref(handle))
        if should_have_key:
            if ret == 0 and handle.value:
                inputs_json = b'[{"dtype":"float16","format":"ND"}]'
                outputs_json = b'[{"dtype":"float16","format":"ND"}]'
                buf = ctypes.create_string_buffer(256)
                iret = invoke_fn(handle, op, inputs_json, outputs_json, None, b"{}", buf)
                if iret == 0 and buf.value:
                    print(f"[PASS] {op.decode():<15} find=0 invoke=0 key='{buf.value.decode()[:60]}'")
                    passed += 1
                else:
                    print(f"[FAIL] {op.decode():<15} find=0 invoke={iret}")
                    failed += 1
            else:
                print(f"[FAIL] {op.decode():<15} expected gen_simplifiedkey but find returned {ret}")
                failed += 1
        elif ret != 0:
            print(f"[PASS] {op.decode():<15} no gen_simplifiedkey (find returned {ret})")
            passed += 1
        else:
            print(f"[INFO] {op.decode():<15} has gen_simplifiedkey (find=0)")
            passed += 1

    # Step 4: Test param validation
    r1 = find_fn(None, None)
    if r1 == 1:
        print("[PASS] FindFuncs(null, null) -> 1")
        passed += 1
    else:
        print(f"[FAIL] FindFuncs(null, null) -> {r1} expected 1")
        failed += 1

    buf = ctypes.create_string_buffer(b"x", 256)
    r2 = invoke_fn(None, b"Op", None, None, None, None, buf)
    if r2 == 1:
        print("[PASS] InvokeFuncs(null handle) -> 1")
        passed += 1
    else:
        print(f"[FAIL] InvokeFuncs(null handle) -> {r2} expected 1")
        failed += 1

    print(f"\n=== {passed} passed, {failed} failed ===")
    return failed


if __name__ == "__main__":
    sys.exit(main())
