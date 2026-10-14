#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Second-level help topics: parameter deep-dives beyond the one-line --help.

Usage: ``ttk help <command> <param>`` or ``ttk help <param>``. Each topic
covers syntax, parameter interplay (联动) and examples for the parameters
whose semantics do not fit a single help line.
"""

HELP_TOPICS = {
    "devices": {
        "title": "--devices / --device-whitelist — device selection expression",
        "body": """\
applies to: kernel / geir / aclnn / e2e

syntax:
  --devices 3              pick device 3
  --devices 0,1,2          pick 0, 1 and 2
  --devices 0-7            pick 0 through 7
  --devices 0-7,^3         pick 0-7 excluding 3 ('^' applies to positive picks)
  --devices ^4             pick all detected devices except 4
  --device-whitelist 0,1   legacy spelling of --devices (same argument)

interplay:
  --dev N caps the pool: picks intersect inside the first N devices;
  without --dev the pool size is max(pick)+1 (or full detection for '^').
  CSV `device_ids` intersects the pool: '0,1,2' shrinks to the available
  subset (>=2 devices left -> run multi-device on them; <=1 -> case fails
  with a clear message); 'all' expands to every pool device; '3' pins the
  case to device 3.
  Internal result: device_count + device_blacklist (the old
  --device-blacklist is gone — use '^' exclusion).""",
    },
    "backend": {
        "title": "--backend — execution environment",
        "body": """\
applies to: kernel / geir (npu only) / aclnn / e2e

values:
  npu       real Ascend device (default)
  npusim    NPUSim SoC-level simulation (kernel/aclnn/e2e; forces
            warmup/task-prof/dl off, run_time=1)
  cpu       framework APIs on CPU (e2e only; kernel/aclnn main execution
            needs a device or npusim — CPU reference data is the golden
            system's job)

third-party collection (--golden-mode XPU / --xpu-perf / cross_check) is
remote-only: it requires remote endpoints (ttk.conf.yaml remote.endpoints
or --config); there is no local-execution backend.

interplay:
  remote endpoints configured (ttk.conf.yaml/--config) -> XPU collection
  goes remote; --golden-mode XPU uses the same routing (GPU output as golden).
  --super-kernel cannot be combined with --backend cpu.""",
    },
    "dump": {
        "title": "--dump — data dump slots",
        "body": """\
applies to: kernel / geir / aclnn / e2e

syntax:
  --dump              same as --dump full
  --dump in,out       comma-separated slots (spaces tolerated)
  --dump full         in + out + golden + xpu (auto-skips xpu with a warning
                      when no remote is configured)
  --dump xpu          third-party outputs (needs remote config; auto-skips
                      with a warning when no remote is configured)
  --dump-format       bin (default) / npy / pt / print
  --dump-on-fail      dump everything when the precision check fails

interplay:
  with --no-prof, dump direction becomes manual-data output: --dump in
  (kernel additionally requires golden, i.e. --dump in,golden) writes a
  restorable dataset; only one --manual-data-dirs directory is allowed.
  --dump in,golden requires CPU golden generation (--golden-mode Enable).
  Unknown tokens fail fast at startup.""",
    },
    "golden-mode": {
        "title": "--golden-mode — who computes the reference",
        "body": """\
applies to: kernel / geir / aclnn / e2e

values:
  Enable    CPU golden via plugin/builtin (default)
  Disable   no golden generation, nothing is compared; the case
            precision verdict is SKIP (unverified, counted in the
            summary SKIP/NA bucket, never as PASS)
  Promote   CPU golden computed at promoted input precision
            (fp16/bf16->fp32, fp32->fp64) — for cross_check where the
            golden must be more accurate than both sides
  XPU       third-party (GPU) output AS the golden; the 8c collection
            runs once and feeds both the golden slot and third_party

interplay:
  XPU and Promote are mutually exclusive (single choice).
  cross_check / mixed / mix_tolerance criteria auto-enable Promote
  (per-case override, takes precedence over the global flag).
  golden_api CSV column "disable" suppresses per-case (e2e; also SKIP).""",
    },
    "core-limit": {
        "title": "--core-limit — cap cores visible to compilation",
        "body": """\
applies to: kernel / geir / e2e (same flag, different layer; not aclnn)

syntax:
  --core-limit 2        cap AI cores at 2
  --core-limit 2,32     cap AI cores at 2 AND vector cores at 32
  --core-limit ,32      vector cores only (either side optional)
  (',' is the separator; '|' is rejected with a hint)

semantics:
  kernel: caps the core count visible to op tiling (tiling re-splits
          block_dim accordingly)
  geir:   caps GE graph compilation (ge.aicoreNum / ge.vectorcoreNum)
  e2e:    caps GE graph compilation (ge.aicoreNum); graph modes only
          (-c/-d/--aclgraph), no effect in eager mode

interplay:
  validated against physical core counts at startup
  (validate_core_limit, DSMI).""",
    },
    "super-kernel": {
        "title": "--super-kernel — SuperKernel compilation (e2e graph modes)",
        "body": """\
applies to: e2e only

requires a graph mode: -c (static) / -d (dynamic) / --aclgraph.

behavior:
  GE graph mode    wraps ops with super_kernel scope marking
  aclgraph mode    passes super_kernel_optimize option to torch.compile
                   (debug_per_op_max_core_num=1 by default)

interplay:
  cannot be combined with --backend cpu;
  requires online kernel compilation — explicit -b errors out, the
  implicit binary default auto-falls back to online compile with a
  warning;
  no effect in eager mode or for TF APIs.""",
    },
    "binary": {
        "title": "-b / --binary — binary kernel reuse (e2e)",
        "body": """\
applies to: e2e (kernel / geir have their own -b with different
semantics: precompiled release kernels / ge.jit_compile session option)

behavior (default: on) — one switch for all three e2e legs:
  eager     torch.npu.set_compile_mode(jit_compile=False): reuse the
            prebuilt aclnn host-side code instead of online compiling
  -c/-d     ge.jit_compile=0 injected into the torchair GE session
            options: GE graph compile reuses prebuilt kernels
  aclgraph  static_kernel_compile=False: reuse captured kernels,
            graph capture & replay kept

syntax:
  -b / -b=release / --binary    explicit binary (same as the default)
  -b=false                      restore online kernel compilation

interplay:
  explicit -b + --super-kernel errors (SuperKernel needs online
  compile); the implicit default auto-falls back to online with a
  warning;
  forced off under --backend npusim (simulation compiles online);
  no effect under --backend cpu or for TF APIs.""",
    },
    "manual-data-dirs": {
        "title": "--manual-data-dirs — manual data load & dump directories",
        "body": """\
applies to: kernel / geir / aclnn / e2e

load direction (default, running on device):
  directories holding dumped datasets; matching testcase data is
  restored instead of being regenerated. Requires --golden-mode Enable
  (reuses the stored golden); incompatible with --validate.

dump direction (with --no-prof):
  --no-prof stops before device execution and writes a restorable
  dataset to the given directory (at most one). Requires --dump in
  (kernel: exactly --dump in,golden). Default directory: manual_data/
  under the plugin root (or cwd without --plugin).

interplay:
  --no-prof cannot be combined with --backend npusim (the simulator
  backend prepares input/golden itself).""",
    },
    "no-prof": {
        "title": "--no-prof — data preparation without device execution",
        "body": """\
applies to: kernel / geir / aclnn / e2e

suppresses the main execution (kernel: dynamic/const/binary modes;
geir: the GE graph compile + execute; aclnn: the aclnn call; e2e: the
framework API) while input/golden generation and dumping still run.

typical pairing:
  --no-prof --dump in,golden        write a restorable manual dataset
  --manual-data-dirs DIR            ... to an explicit directory

interplay:
  incompatible with --backend npusim;
  with --dump in (kernel: in,golden) this becomes the manual-data
  output workflow; --dump-format must be restorable (bin/npy/pt).""",
    },
    "xpu-perf": {
        "title": "--xpu-perf — third-party performance collection",
        "body": """\
applies to: kernel / geir / aclnn / e2e

collects per-case third-party (XPU) device_us only — no data transfer,
no influence on the precision comparison. Results land in the
xpu_metrics column; the NPU side keeps its normal golden comparison.

interplay:
  requires remote XPU config (ttk.conf.yaml remote.endpoints or
  --config) — a missing config is a hard error (unlike --dump xpu,
  which degrades to a warning). With --backend xpu and no remote
  endpoints, collection runs locally via the frameworks.<fw>.xpu
  segment. --provider narrows which providers to collect.""",
    },
    "deterministic-level": {
        "title": "--deterministic-level / --dl — determinism control",
        "body": """\
applies to: kernel / geir / aclnn / e2e

levels:
  0  off (default)
  1  deterministic compute — output MD5 consistent across NPU runs
  2  strong consistency
  3  deterministic execution — additionally enables phase-two batch
     slice compare when the testcase carries complete batch fields
     (batch_axis / batch_slice_info / batch_seed)

interplay:
  a nonzero CLI level is authoritative; per-case
  attributes.batch_deterministic_level applies only when the CLI level
  is 0 (one CSV can mix ordinary and deterministic cases).
  Forced off under --backend npusim.""",
    },
}


def print_topic(name):
    """Print one topic; unknown names list the available ones."""
    topic = HELP_TOPICS.get(name)
    if topic is None:
        import sys

        print(
            f"ttk help: unknown topic {name!r}; available: {', '.join(sorted(HELP_TOPICS))}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    width = 100
    print("=" * width)
    print(topic["title"])
    print("=" * width)
    for line in topic["body"].splitlines():
        print(line)
    print("=" * width)
