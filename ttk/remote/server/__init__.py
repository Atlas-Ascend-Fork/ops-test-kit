"""xpu-server — remote XPU execution server (整仓部署).

Deployment boundary:
  * Server deploys WITH the whole ttk repo (git clone / pip install / 拷贝整仓)
    and starts via ``python -m ttk.remote.server.xpu_server``. It consumes
    same-source contract code and the profiler engine from
    ``ttk.core_modules.common`` (bind_params / UnknownParamError /
    resolve_activities / summarize_events) — single copy, no mirror drift.
  * Copying just this subdirectory and running it standalone
    (``python -m <pkg>.xpu_server``) is NOT supported — the contract code
    lives outside this package.
  * bfloat16/complex32/float8 are handled with numpy bit views plus torch —
    never ml_dtypes.
  * Runtime deps: Python stdlib + numpy + torch/tensorflow (lazy-imported).
"""
