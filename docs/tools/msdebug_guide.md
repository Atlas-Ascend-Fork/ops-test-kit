# msdebug 使用指南

## 使用须知

[msdebug](https://gitcode.com/Ascend/msdebug) 是 MindStudio 提供的算子源码级调试工具（基于 lldb 定制），支持算子核函数的源码级断点 / 单步 / 变量查看，断点命中在 AiCore 上。

- **可源码断点**：kernel 产物（`.o` / fatbin）需带 DWARF；msdebug 下 TTK 自动 `-O0 -g` 自编译。
- **需要 root**：`/proc/debug_switch` 默认关闭。

文档导览：

1. **[环境准备](#一环境准备)**
2. **[基本用法](#二基本用法)**
3. **[调试流程](#三调试流程)**
4. **[命令速查](#四命令速查)**
5. **[常见问题](#五常见问题)**
6. **[验证清单](#六验证清单)**

## 一、环境准备

- Python 3.8+
- 完成 CANN 包安装（含 firmware / driver 适配）
- 完成 MindStudio 算子工具（msdebug）安装

确认可用：

```bash
ls -l /usr/local/Ascend/cann-9.1.0/tools/msdebug/bin/msdebug  # 示例路径
<msdebug路径>/msdebug --version                               # 应见 MindStudio banner + 版本号
```

> [!CAUTION]
> **msdebug 需要 root 权限。** `/proc/debug_switch` 默认关闭，开启需要 root；不具备特权环境需联系系统管理员处理。

```bash
cat /proc/debug_switch          # 期望 debug_switch_status = 1
echo 1 > /proc/debug_switch     # 不为 1 时用 root 开启
ls -l /dev/drv_debug            # 设备调试节点应存在
```

> [!WARNING]
> 若开关无法置为 1，则断点功能不可用。容器/云环境中 `/proc` 常被隔离，容器内看到 1 可能是虚假状态，需在宿主机真实开启。

## 二、基本用法

msdebug 包裹被调试命令启动：

```shell
msdebug -- python3 -m ttk kernel \
    -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast \
    -d=true -c=false -b=false
```

启动后进入 `(msdebug)` 提示符，输入 `run` 启动 TTK。

| 模式 | 命令 | 适用 |
|------|------|------|
| 交互式 REPL | `msdebug -- cmd` | 人工单步 / 断点（推荐） |
| 机器接口（MI） | `msdebug-mi` | 供 MI 客户端（如 Eclipse 插件） |

> [!CAUTION]
> MI driver **仅支持远程目标调试**，不能以 `msdebug-mi -- <本地程序及参数>` 启动本地程序；本地调试用 `msdebug -- <程序及参数>`。

## 三、调试流程

以 TTK Kernel 动态编译（`-d`）为例。`run` 后需等 kernel 拉起（约 60~100s，含 `-O0 -g` 编译）。

### 1. 启动 msdebug + TTK

```shell
cd <TTK项目根目录>
msdebug -- python3 -m ttk kernel \
    -i examples/case_store/kernel/cases.csv \
    -t add_fp16_broadcast \
    -d=true -c=false -b=false
```

| 参数 | 作用 |
|------|------|
| `-d=true` | 动态 kernel 编译 + 执行（自编译，带调试信息） |
| `-c=false` | 关闭 const 场景（减少干扰） |
| `-b=false` | **必须**（`=release` 无 DWARF，无法断点） |
| `-t=<用例名>` | 只跑指定用例 |

### 2. 启动与观测

```text
(msdebug) run
```

启动后应看到 stub 日志与 `Launch of Kernel`。

```text
[INFO]GetStubFuncPtr funcName=rtGetDevice
[INFO]rtGetDevice done.
[Launch of Kernel dyn_op_add_fp16_broadcast_8 ...]
```

> [!TIP]
> 完整 stub 日志：`export DEBUGGER_RT_STUB_LOG=1`。看不到输出 = 劫持未生效，见"常见问题"。

### 3. 设源码断点

等 kernel 跑起来后设断点：

```text
(msdebug) b add.cpp:45
(msdebug) process continue
```

期望命中（`stop reason = breakpoint`、`frame #0` 显示源码行）：

```text
Process 533359 stopped
[Switching to focus on Kernel dyn_op_add_fp16_broadcast_8, CoreId 60, Type aiv]
* thread #1, name = 'python3', stop reason = breakpoint 1.1
    frame #0: device_debugdata_1`void add_0_tilingkey<1ul>(...) (.vector) at add.cpp:45:38
-> 45          BroadcastSch<schMode, OpDag> sch(tiling);
```

> [!CAUTION]
> 断点设得太早（kernel 未拉起）报 `breakpoint 1: no locations (pending ...)` 属正常，等 kernel launch 后再 `b`，或 `process interrupt` 停下再设。若 `/proc/debug_switch` 是容器/云里的虚假状态，`run` 可能报 `error: 'A' packet returned an error: 8`，需在宿主机以 root 真正开启。

> [!NOTE]
> 断点命中后 msdebug 自动聚焦断点所在核；AIV 多核场景每核都停在断点，`continue` 一次放一个核。

### 4. 单步 / 查看

```text
(msdebug) bt                    # 调用栈
(msdebug) frame variable x1     # 查看变量
(msdebug) var                   # 查看全部变量
(msdebug) next                  # 单步跳过
(msdebug) step                  # 单步进入
(msdebug) memory read <addr>    # 读设备内存
(msdebug) register read -a      # 读全部寄存器
```

### 5. 查询 AiCore 运行信息

```text
(msdebug) ascend info devices        # Device
(msdebug) ascend info cores          # 各核断点/PC/stop reason
(msdebug) ascend info tasks          # 当前 task
(msdebug) ascend info stream         # stream 类型
(msdebug) ascend info blocks         # block 列表
```

### 6. 继续 / 结束

```text
(msdebug) process continue      # 继续执行
(msdebug) q                     # 退出
```

> [!NOTE]
> 不放行 kernel，用例停在 `OnDynProfiling` 属正常（kernel 等 continue），不是卡死。

## 四、命令速查

| 命令 | 说明 |
|------|------|
| `run` / `process launch` | 启动调试 |
| `process interrupt` | 强制中断 |
| `process continue` | 继续 |
| `b <文件>:<行号>` / `breakpoint set -f <文件> -l <行号>` | 源码行断点 |
| `breakpoint set -n <函数名>` | 函数名断点 |
| `breakpoint set -a <地址>` | 地址断点 |
| `breakpoint list` / `breakpoint delete N` | 查看 / 删除断点 |
| `next` / `step` / `thread step-out` | 单步跳过 / 进入 / 跳出 |
| `bt` | 调用栈 |
| `frame variable` / `var` | 查看变量 / 全部 |
| `memory read <addr>` | 读设备内存 |
| `reg read` / `register read -a` | 读寄存器 |
| `ascend info devices/cores/tasks/stream/blocks` | 查询运行上下文 |
| `q` + `y` | 退出调试 |

## 五、常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| stub 日志不打印、收不到 kernel 信息 | 劫持未生效 | 排查环境/补丁 |
| 有 `Launch of Kernel` 但断点无法解析（`frame #0: 0x0`） | kernel 无 `.debug_info`（`-b=release` 内置包） | `-d=true -b=false` 自编译 |
| 断点报 `pending on future shared library load` | 断点设得太早（kernel 未拉起） | 等 kernel launch 后再设，或 `process interrupt` 停下再设 |
| `error: 0x20200`（OPEN_KO_ERR） | 设备被占用 | 清理 msdebug/lldb-server 进程 |
| 用例停在 `OnDynProfiling` 一直 RUNNING | kernel 等 continue（正常） | continue 或删断点 |
| `run` 后立即退出 | stdin 提前 EOF | 保持交互 TTY / PTY 驱动 |
| `step` 报警 `invalid thread index` | 命令误解析 | 用完整命令 |

## 六、验证清单

```bash
# 1) msdebug 可用
<msdebug路径>/msdebug --version          # 见 MindStudio banner + 版本号

# 2) 劫持生效
export DEBUGGER_RT_STUB_LOG=1
msdebug -- python3 -m ttk kernel -i ... -d=true -b=false ...
#   期望见 [Launch of Kernel ...] / [INFO]GetStubFuncPtr ...

# 3) kernel 带源码级调试信息
readelf -S kernel_meta/dyn_op_*.o | grep -E "debug_info|debug_line"   # 应存在

# 4) 断点命中源码
(msdebug) breakpoint set -f add.cpp -l 45
(msdebug) process continue
#   期望见 frame #0: ... at add.cpp:45:38
```
