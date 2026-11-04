#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compile rendered profiling lifecycle blocks with host-only C API doubles.

This checks cleanup and failure-frame ordering without linking GE or touching an
NPU. Native GEIR fail-then-success execution remains a device qualification.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

_TEMPLATE_DIR = Path(__file__).resolve().parents[3] / "ttk/core_modules/geir/templates"


def _block(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end] + ";"


_CPP = r"""
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
using std::string;
struct aclgrphProfConfig {};
int stops = 0, destroys = 0, finalizes = 0;
bool active = false, wrote_while_active = false;
int aclgrphProfStop(aclgrphProfConfig*) { ++stops; return 0; }
int aclgrphProfDestroyConfig(aclgrphProfConfig*) { ++destroys; return 0; }
int aclgrphProfFinalize() { ++finalizes; active = false; return 0; }
size_t observed_write(const void* p, size_t size, size_t count, FILE* file) {
    wrote_while_active = wrote_while_active || active;
    return std::fwrite(p, size, count, file);
}
#define fwrite observed_write
int main(int argc, char** argv) {
    assert(argc == 2);
    const int scenario = std::atoi(argv[1]);
    FILE* data_out = std::tmpfile();
    assert(data_out != nullptr);
    // PROFILE_DECLARATIONS
    // TEARDOWN
    // FAILURE_FRAME
    // SCOPE_TYPE
    // RUN_SCENARIO
    std::fclose(data_out);
}
"""


@pytest.fixture(scope="module", params=[True, False], ids=["profiler", "no-profiler"])
def lifecycle_binary(request, tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is required for the host lifecycle harness")
    profiling = request.param
    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)  # noqa: S701 C++ template
    source = env.get_template("geir_op_template.cpp.j2").render(
        ge_prof_ok=profiling,
        input_names=[],
        output_names=[],
        attr_entries=[],
        dtype_map=[],
        format_map=[],
        dtype_size_cases=[],
        packed_float4_enums=[],
        op_class="Dummy",
        proto_file="dummy.h",
    )
    teardown = _block(source, "auto prof_teardown =")
    failure = _block(source, "auto write_fail_frame =")
    # Also characterize the pre-fix template, which had no iteration guard.
    scope_type = _block(source, "struct ProfScope") if "struct ProfScope" in source else ""
    scope_instance = "ProfScope prof_scope{prof_teardown};" if scope_type else ""
    if profiling:
        declarations = "bool prof_init_done = false, prof_started = false; aclgrphProfConfig* prof_cfg = nullptr;"
        run = r"""
    aclgrphProfConfig cfg;
    auto begin = [&](bool start, bool config) {
        assert(!active);
        prof_init_done = true; active = true;
        prof_cfg = config ? &cfg : nullptr; prof_started = start;
    };
    if (scenario == 0 || scenario == 1 || scenario == 2) {
        begin(scenario == 0, scenario != 2);
        write_fail_frame("original RunGraph error");
        assert(!active && !wrote_while_active);
        assert(finalizes == 1 && destroys == (scenario != 2) && stops == (scenario == 0));
        std::rewind(data_out);
        int64_t tag = 0, size = 0;
        assert(std::fread(&tag, 8, 1, data_out) == 1 && tag == -1);
        assert(std::fread(&size, 8, 1, data_out) == 1 && size == 23);
        char text[24] = {};
        assert(std::fread(text, 1, size, data_out) == static_cast<size_t>(size));
        assert(string(text) == "original RunGraph error");
    } else if (scenario == 3) {
        for (int i = 0; i < 2; ++i) {
            // SCOPE_INSTANCE
            begin(true, true);
            continue; // Configuration/graph failures must not leak into the next case.
        }
        assert(finalizes == 2 && stops == 2 && destroys == 2);
    } else if (scenario == 4) {
        for (;;) {
            // SCOPE_INSTANCE
            begin(true, true);
            break; // EOF / loop exit.
        }
        assert(!active && finalizes == 1);
    } else if (scenario == 5) {
        { // Failed Init leaves all flags clear; cleanup must be a no-op.
            // SCOPE_INSTANCE
            write_fail_frame("original RunGraph error");
        }
        assert(stops == 0 && destroys == 0 && finalizes == 0);
    } else {
        { // Explicit success cleanup and destructor backstop are idempotent.
            // SCOPE_INSTANCE
            begin(true, true); prof_teardown();
        }
        assert(finalizes == 1 && stops == 1 && destroys == 1);
    }
"""
    else:
        declarations = ""
        run = "{ " + scope_instance + ' write_fail_frame("original error"); } assert(finalizes == 0 && stops == 0);'
    code = _CPP.replace("// PROFILE_DECLARATIONS", declarations).replace("// TEARDOWN", teardown)
    code = code.replace("// FAILURE_FRAME", failure).replace("// SCOPE_TYPE", scope_type)
    code = code.replace("// RUN_SCENARIO", run.replace("// SCOPE_INSTANCE", scope_instance))
    directory = tmp_path_factory.mktemp("geir-prof-host")
    cpp = directory / "lifecycle.cpp"
    binary = directory / "lifecycle"
    cpp.write_text(code, encoding="utf-8")
    subprocess.run([compiler, "-std=c++11", str(cpp), "-o", str(binary)], check=True, capture_output=True, text=True)
    return binary


@pytest.mark.parametrize(
    "scenario",
    range(7),
    ids=["run-fail", "start-fail", "init-only", "continue", "break", "init-fail", "success-idempotent"],
)
def test_rendered_profiling_cleanup(lifecycle_binary, scenario):
    result = subprocess.run([str(lifecycle_binary), str(scenario)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
