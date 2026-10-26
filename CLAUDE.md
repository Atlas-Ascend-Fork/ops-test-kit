# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

TTK（ops Test Tool Kit）是昇腾 NPU 单算子测试框架：以 CSV 描述用例，一条命令批量完成编译、执行、精度比对。支持四种测试模式——Kernel（TBE/AscendC 算子内核）、GEIR（GE 图编译+执行）、ACLNN（`aclnn*` C API）、E2E（torch/torch_npu 等框架 API）。

## 常用命令

所有命令都必须在仓库根目录运行（`python3 -m ttk` 依赖仓库内相对路径与 C 扩展源码）。

```shell
# 运行测试（四种模式）
python3 -m ttk kernel -i examples/case_store/kernel/cases.csv --plugin examples/case_store/kernel/golden.py
python3 -m ttk geir   -i examples/case_store/kernel/cases.csv --plugin examples/case_store/kernel/golden.py
python3 -m ttk aclnn  -i examples/case_store/aclnn/cases.csv --plugin examples/case_store/aclnn/golden.py
python3 -m ttk e2e    -i examples/case_store/e2e/torch_cases.csv  # --backend cpu 强制 CPU 后端

# NPUSim 仿真后端（kernel/aclnn/e2e 均支持，无真机即可跑）
python3 -m ttk kernel -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast --backend npusim --sim-report

# 只列出用例名（不跑设备、不做字段校验；校验用对应模式 `--validate`）
python3 -m ttk list -i cases.csv
# 查看设备/环境
python3 -m ttk info
```

用例筛选与常用选项（定义见 `ttk/cli/common.py`）：
- 筛选：`-t/--testcase add_01,add_02`、`--ti/--testcase-index 0-9`（从 0 起）、`--op`、`--priority`、`--tc`（随机取 N 条；注意 `--seed` 只固定输入数据生成（`numpy.random.seed`），**不固定** `--tc` 的用例抽取（`random.sample` 未播种））
- 精度：`--compare close|stat_rel_err|mixed|cosine|binary|requant|cross_check`（默认按 Spec.tolerance 路由，无插件时 mixed；Spec.tolerance 标准名为 `mix_tolerance`）
- 排障/数据：`--rerun precision_status` 按上次结果 CSV 的列重跑失败用例（**`-i` 须指向含该列的结果 CSV**，不能是原始用例 CSV）；`--dump in,golden` 备数据、`--dump-on-fail` 失败自动 dump、`--dump-format npy|pt|bin|print`
- 自定义 golden/输入：`--plugin <path>`，指向包含 TestSpec 的目录或文件
- 手工数据两阶段：`--no-prof --dump in,golden` 准备数据，`--manual-data-dirs <dir>` 回放
- XPU 结果落盘：`--dump xpu`，可与 `in,out,golden` 组合，按 `<testcase_name>_xpu_golden_<index>.<bin|npy>` 保存第三方输出（多 provider 时按 provider 分子目录）
- NPUSim 仿真（参数定义见 `ttk/cli/sim_args.py`）：`--backend npusim` 切到仿真后端；`--sim-soc`（默认 `Ascend950`）、`--sim-output`（默认 `<root>/sim_output`）、`--sim-report`（生成 trace_core*.json 与 HTML 性能报告）、`--sim-cores`、`--sim-obj`。与 `--no-prof`、`--backend cpu` 互斥

### 开发/测试

```shell
pip install -e ".[dev]"          # pytest/ruff/mypy 等开发依赖
pytest                           # 全部单测（tests/，需 NPU 的自动跳过）
pytest tests/core_modules/common/plugin/test_plugin.py -k golden   # 单文件 / 关键字筛选
ruff check ttk/                  # lint（pre-commit 见 .pre-commit-config.yaml）
pre-commit run --all-files       # ruff + codespell + 基础检查（v4，非 pre-commit 2.x）
```

- pytest 配置在 `pyproject.toml` 的 `[tool.pytest.ini_options]`：marker 有 `slow`/`device`/`e2e`，`xfail_strict=true`。
- 需真机 NPU 的测试用 skipif 自动跳过（如 `tests/core_modules/framework_api/test_hccl_apis.py`，无 CANN 环境跳过；其余套件按 torch/tf 可用性跳过）。
- `tests/core_modules/framework_api/test_extractor_chain_matrix.py` 的 28 条解析链胜者分布用例按基线 `torch==2.9.0` + `torch_npu==2.9.0.post7` 钉死：其他组合（如 README 推荐的 `torch==2.12.0`）下自动降级为编排层结构校验并 skip 钉死断言（属预期，非故障）；按当期版本重冻结基线的方法见该文件 docstring。
- `tests/conftest.py` 会清掉 `ASCEND_*` 环境变量并预加载默认 config，防止测试意外吃到真机配置。

## 架构

### 入口与分发（`ttk/cli/` → `ttk/core_modules/`）

`python3 -m ttk` 走 `ttk/cli/__init__.py`：
1. `ttk/_env.py::setup_env()` 最先执行——source CANN `setenv.bash`、设 `ASCEND_OPP_PATH`、把 TBE op 实现路径前置进 PYTHONPATH、调 ulimit、预加载 libgomp、清理仓库根 `ttk-*.log`。
2. `ttk/cli/bridge.py::args_to_switches()` 把 argparse 结果灌进 `SWITCHES` 对象（`ttk/switches.py`），CLI 层不碰业务逻辑。
3. `run_with_switches()` 按 `sw.test_mode` 经注册表（`@register_instance`）分发：`framework-api` → `FrameworkApiInstance`（E2E），`geir` → `GeirInstance`，`op`/`kernel` → `KernelInstance`（`ttk/core_modules/kernel/instance.py`），`aclnn` → `AclnnInstance`（`ttk/core_modules/aclnn/instance.py`）。

### CSV → 用例（`ttk/core_modules/common/testcase/`）

`field_parser.py` 把 CSV 中的 Python 字面量字符串解析成 shape/dtype/format 结构；`factory.py::UniversalTestcaseFactory` 按子命令指定的模式构建用例（用例类在各通路：`kernel/testcase.py`、`geir/testcase.py`、`aclnn/testcase.py`、`framework_api/testcase.py`）。测试模式由 CLI 子命令唯一决定，CSV 表头与该模式不匹配会解析失败。

### TestSpec 插件体系（`ttk/core_modules/common/plugin/`）

算子测试规范：`golden`（真值）、`third_party`（三方标杆）、`compare`/`pre_compare`（自定义比对）、`customize_inputs`（自定义输入）、`tolerance`（精度标准）。这是独立于框架的纯规范模块，被 `--plugin` 目录加载。`spec_loader.py` 先用 `ast.parse` 静态索引模块级 `__spec__` 映射，再对命中的文件惰性 exec（未命中文件不执行、无副作用）。规范细节见 `ttk/core_modules/common/plugin/README.md`——注意各类模式 golden 的入参类型不同：Kernel/GEIR 收 numpy.ndarray，ACLNN/E2E 收 CPU（host）侧 torch.Tensor（先于设备执行生成）。

### 比对、E2E Backend 与远程执行

- 比对方法集中在 `ttk/core_modules/common/comparison/`；`cross_check` 三方交叉校验需配合 `third_party`。
- E2E 的硬件中立 Backend 抽象与 api_resolver 在 `ttk/core_modules/common/backend/`（torch/npu/cpu/xpu backend），profiler 在 `ttk/core_modules/common/profiler/`；eager/graph 执行在 `ttk/core_modules/framework_api/torch/`（eager_runner、graph_runner）。
- 远端 XPU 执行（mTLS、server/client、heartbeat）在 `ttk/remote/`，客户端配置见 `ttk/config/default.yaml` 的 `remote:` 段；server 侧框架/硬件配置见 `ttk/remote/server/xpu_server.yaml`（运行时探测）。

### NPUSim 仿真后端（`ttk/core_modules/common/simulator/`）

`--backend npusim` 不引入新 Instance 类型：kernel/aclnn/e2e 仍走各自 Instance，只是把真机执行替换为 NPUSim 仿真。CLI 参数与归一化在 `ttk/cli/sim_args.py`（设置 `sw.backend="npusim"`、把 `dev_plat` 归一为 platform ini 名）；执行在 `simulator/npusim_runner.py`（定位 cannsim、`run_record`、`run_report`）与 `simulator/wrapper.py`（Python user_app，由 NPUSim `record` 启动）；配套 `case_writer.py`、`report.py`、`sim_profiling.py`。SoC 注册表在 `simulator/config.py::SIM_PLATFORM_BY_SOC`，当前仅 `Ascend950`/`Ascend950DT`。仿真结果写 `output_*.bin` 后走既有比对管线。指南见 `docs/NPUSim/npusim_usage.md`，设计文档见 `docs/NPUSim/npusim_design.md`。生效的 NPUSim 实现是 `common/simulator/`。

### C++ 扩展（`csrc/`）

`libttk_op_registry_accessor.so`（二进制匹配时调用算子注册的 `gen_simplifiedkey`）与 `libttk_error_manager_cleaner.so`。由 `ttk/core_modules/kernel/compile/cext_loader.py` 在首次使用时**按需 cmake 编译**到 `csrc/<sub>/build/`（带锁与 done 标记，并发安全）。改 C++ 后删 `csrc/*/build/` 即可强制重编。

## 配置链

`ttk/config/loader.py::load_config()` 按 `default.yaml` → `~/.config/ttk` → `./ttk.conf.yaml` 逐层加载，`--config` 可整体覆盖。**不要直接改 `ttk/config/default.yaml`**——复制一份再改。

## 关键约定

- 执行 ttk 必须在本仓库根目录；每次运行会清理根目录 `ttk-*.log`（`_env.py`）。
- 仓库根目录的 `kernel_meta/`、`sim_output/`、`ttk-*.log` 是运行时产物（已 gitignore），不是源码。
- 版本号以 `ttk/_version.py` 的 `__version__` 为准（当前 3.0.0），`pyproject.toml` 可能滞后。
- **Commit 规范：commit message 不要添加 `Co-Authored-By` 行。**

## Agent Skills

详见 `AGENTS.md`（TTK 使用指南）与 `.claude/skills/`：
- `ttk-how-run-test` — 运行测试/构造命令/查看设备
- `ttk-how-write-case` — 编写 Kernel/ACLNN/E2E 的 CSV 用例
- `ttk-how-diagnose` — 诊断测试失败/精度/编译错误
- `ttk-how-write-plugin` — 编写 golden/TestSpec 插件
- `ttk-prof2case` — profiling 报告转测试用例
