"""
TTK 远端执行 —— 跨机器 XPU 算子执行。

Public API:
    get_tenant_id() — 获取当前 TTK 实例的租户 ID
    is_remote_configured() — 检查是否配置了远端执行
    dispatch_to_remote() — 发送输入到远端 xpu_server，返回输出
"""

import os
import uuid
from dataclasses import dataclass
from typing import Optional

# --- X-Mode bitmask. Shared protocol between client (dispatcher)
# and server (xpu_server / executor). Keep these bit-stable across versions.
DATA = 0b01  # return remote xpu outputs
PERF = 0b10  # collect performance


def xpu_mode(switches, need_data: bool) -> int:
    """三方采集模式位：xpu_perf→PERF，need_data/--dump xpu→DATA。

    正本在协议层（client/dispatcher 与 xpu_mode 消费方共享）；
    ttk.core_modules.common.xpu re-export 供 worker 侧统一导入。
    """
    mode = 0
    if getattr(switches, "xpu_perf", False):
        mode |= PERF
    dump_config = getattr(switches, "dump_config", None)
    if need_data or (dump_config is not None and getattr(dump_config, "dump_xpu", False)):
        mode |= DATA
    return mode


def has_data(mode: int) -> bool:
    return bool(mode & DATA)


def has_perf(mode: int) -> bool:
    return bool(mode & PERF)


# 实例级 tenant_id：必须跨进程一致。原注释假设 fork（子进程继承父进程本模块的 _TENANT_ID），
# 但 TTK 用 forkserver——forkserver 独立重新 import 本模块、另生成一个 uuid，其 worker 继承的是
# forkserver 的 id 而非主进程的，导致【心跳(主进程)的租户 ≠ 干活/同步 golden 的 worker 租户】，
# 干活租户拿不到心跳 → 600s 超时被 server 清掉 golden → missing-golden。改为经环境变量透传：
# 首个 import 本模块的进程生成并写入 env，forkserver/worker 继承 env 得到同一 id。
_TENANT_ID: str = os.environ.get("TTK_TENANT_ID") or uuid.uuid4().hex[:12]
os.environ["TTK_TENANT_ID"] = _TENANT_ID


def get_tenant_id() -> str:
    return _TENANT_ID


def is_remote_configured() -> bool:
    """Check if remote execution is configured (has endpoints in yaml).

    This answers "is it configured", not "is the server reachable".
    删了 TTK_XPU_ENDPOINTS env 检查——endpoints 现从 yaml + get_remote_config() 来。
    """
    try:
        from ttk.remote.config import get_remote_config

        config = get_remote_config()
        return config is not None and bool(config.endpoints)
    except RuntimeError:
        # 防御性兜底：config 未加载时返回 False（而非崩）
        return False


@dataclass
class ExecutionSpec:
    """Per-provider remote execution specification.

    type == 'api':      call ``api`` (a dotted callable string) on the server.
    type == 'spec':  sync ``spec_file`` and invoke ``spec_class``
                        (whose ``third_party[provider]`` is the impl) on the server.

    Context data fields (default None — remote flow untouched):
    inputs/testcase_name/attrs carry the caller's context alongside the
    execution description (ExecutionSpec = 执行描述+数据一体)。
    """

    provider: str
    type: str = "api"
    api: Optional[str] = None
    spec_module: Optional[str] = None
    spec_file: Optional[str] = None
    spec_class: Optional[str] = None
    inputs: Optional[list] = None
    attrs: Optional[dict] = None
    testcase_name: Optional[str] = None


def _derive_provider_from_api(api: Optional[str], fallback: str) -> str:
    """Derive provider from API module prefix ('torch.add' → 'torch').

    Known frameworks (torch/tf/tensorflow/numpy/np) are normalized:
    tensorflow→tf, np→numpy, npu→torch (torch_npu 生态走 torch provider
    设备分支)。Unknown prefixes pass through verbatim —
    resolve_providers filters what the server doesn't detect.
    """
    if not api or "." not in api:
        return fallback
    prefix = api.split(".", 1)[0]
    return {"tensorflow": "tf", "np": "numpy", "npu": "torch"}.get(prefix, prefix)
