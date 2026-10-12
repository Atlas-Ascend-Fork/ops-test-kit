# msdebug 使用指南

[toc]

---

# 概述

[msdebug](https://gitcode.com/Ascend/msdebug) 是 MindStudio 提供的算子**源码级调试工具**（底层基于 lldb 定制，Ascend 版本 26.0.0 / lldb 19.1.7）。它用于对算子核函数（kernel）做**源码级断点 / 单步 / 变量查看**，定位行为错误、数据异常等逻辑问题。

msdebug 通过注入一个 `libruntime_stub.so` 劫持库，拦截运行时（`libruntime.so`）里的 kernel 启动调用，从而在 **AiCore 上**（而非宿主机 CPU 上）对核函数源码设置断点。因此：

- **可源码级断点的前提**：待调试 kernel 的产物（`.o` / fatbin）必须带 DWARF（`.debug_info` / `.debug_line`）。
- TTK 使用 msdebug 调试 kernel 时，必须让 kernel **自编译并带调试信息**（见下文"交互调试流程"）。

---

# 环境准备

## 基本环境配置

- 建议 Python 3.8+
- 完成 CANN 包安装（含 firmware / driver 适配）
  - CANN 包下载地址：<https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/>
- 完成 MindStudio 算子工具（msdebug）安装

## 确认 msdebug 可用

```bash
# msdebug 可执行文件在 CANN 工具包内，示例路径：
ls -l /usr/local/Ascend/cann-9.1.0/tools/msdebug/bin/msdebug

# 查看版本（应看到 MindStudio banner + 版本号）
<msdebug路径>/msdebug --version
```

期望输出：

```text
=================================================================
                   >>>>>   MindStudio   <<<<<
    THE END-TO-END TOOLCHAIN TO UNLEASH HUAWEI ASCEND COMPUTE
=================================================================

msdebug version 26.0.0-453f94e
lldb version 19.1.7, revision: ...
```

> msdebug 是一个 bash 包装脚本（设置 `LD_LIBRARY_PATH` / `TERMINFO` 后再调用 `msdebug.bin`）,
> 因此一般直接执行 `msdebug` 即可，无需手动设置工具自身的库路径。

## 内核调试开关（强制前置）

> [!CAUTION]
> **msDebug 需要 root 权限。** `/proc/debug_switch` 默认关闭，开启需要 root；
> 在不具备特权的环境（共享开发机、普通容器）中可能无法开启，需联系系统管理员处理。

```bash
# 查询调试开关状态（期望: debug_switch_status = 1）
cat /proc/debug_switch

# 若不为 1，用 root 开启：
echo 1 > /proc/debug_switch

# 设备调试节点应存在
ls -l /dev/drv_debug
```

> 若开关无法置为 1，则 msDebug 断点功能不可用。注意容器/云环境中 `/proc` 常被隔离，
> 容器内看到 1 可能是虚假状态，需在宿主机真实开启。

---

# 调用 msdebug

## 基本用法

msdebug 以**包裹**被调试命令的形式启动：

```shell
msdebug -- <被调试程序及其参数>
```

对 TTK，即：

```shell
msdebug -- python3 -m ttk kernel \
    -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast \
    -d=true -c=false -b=false
```

启动后 msdebug 进入 **`(msdebug)` 交互提示符**（对应 lldb 的 `(lldb)`）：

```text
(msdebug) target create "python3"
Current executable set to '.../python3' (x86_64).
(msdebug) settings set -- target.run-args "-m" "ttk" ...
(msdebug)
```

此时输入 `run`（或 `process launch`）真正启动 TTK：

```text
(msdebug) run
```

## 两种使用模式

| 模式 | 命令 | 适用 |
|------|------|------|
| 交互式 REPL | `msdebug -- cmd` | 人工单步/打断点调试（推荐）；本地调试 TTK 的 wrapper 方式 |
| 机器接口（MI） | `msdebug-mi` | 供 MI 客户端（如 Eclipse 插件）远程调试，输出结构化 MI 协议 |

> [!CAUTION]
> MI driver 当前**仅支持远程目标调试，本地调试尚未实现**（`msdebug-mi --help` 明确标注 `Local debugging has not been implemented`，`executable` 亦为 `NOT IMPLEMENTED`）。它**不能**以 `msdebug-mi -- <本地程序及参数>` 的 wrapper 方式启动本地程序——该写法只会把程序当作目标文件加载符号（输出 `-file-exec-and-symbols` 后停在 MI 提示符），不会执行。本地调试 TTK 请使用 `msdebug -- <程序及参数>`。

---

# 交互调试流程（推荐）

以 TTK Kernel 动态编译（`-d`）为例。源码级断点能否命中，**时序**很关键：TTK 先起主进程，再派 profiling 子进程编译并拉起 kernel。`run` 后需等待子进程把 kernel 拉起（约 60~100s，含 `-O0 -g` 编译时间）。

## 1. 启动 msdebug + TTK

```shell
cd <TTK项目根目录>

msdebug -- python3 -m ttk kernel \
    -i examples/case_store/kernel/cases.csv \
    -t add_fp16_broadcast \
    -d=true -c=false -b=false
```

关键参数：

| 参数 | 作用 |
|------|------|
| `-d=true` | 动态 kernel 编译 + 执行（自编译，可带调试信息） |
| `-c=false` | 关闭 const 场景（减少干扰） |
| `-b=false` | **必须**。若 `=release` 复用内置包（无 DWARF，无法源码断点） |
| `-t=<用例名>` | 只跑指定用例 |

> **调试信息自动注入**：msdebug 下 TTK 自动以 `-O0 -g` 自编译 kernel 使其带 DWARF，
> 无需手动加编译选项。

## 2. 启动目标进程

```text
(msdebug) run
```

## 3. 观测劫持与 kernel launch

启动后应用 stub 日志确认劫持生效、kernel 已拉起：

```text
[INFO]GetStubFuncPtr funcName=rtGetDevice
[INFO]rtGetDevice done.
[INFO]GetStubFuncPtr funcName=rtStreamSynchronize
[INFO]Native device id: 0, pid: 533747, send message: $kernel_name:dyn_op_add_fp16_broadcast_8;stream_id:60;kernel_hash:...;pc_base_addr:120050000000
[Launch of Kernel dyn_op_add_fp16_broadcast_8 ...]
```

> 若要看完整的 stub 日志，可先 `export DEBUGGER_RT_STUB_LOG=1`。
> 若看不到上述输出：劫持未生效，先排查环境/补丁（见"常见问题"）。

## 4. 设源码断点

TTK 用例在 `OnDynProfiling` 阶段启动 kernel，等 kernel 跑起来后设断点（`b <文件>:<行号>` 为 `breakpoint set` 的简写）：

```text
(msdebug) b add.cpp:45
(msdebug) process continue
```

期望命中（关键标志：`stop reason = breakpoint`，`frame #0` 显示源码行）：

```text
Process 533359 stopped
[Switching to focus on Kernel dyn_op_add_fp16_broadcast_8, CoreId 60, Type aiv]
* thread #1, name = 'python3', stop reason = breakpoint 1.1
    frame #0: device_debugdata_1`void add_0_tilingkey<1ul>(...) (.vector) at add.cpp:45:38
-> 45          BroadcastSch<schMode, OpDag> sch(tiling);
```

> [!CAUTION]
> 若断点设得太早、kernel 尚未拉起，会报 `breakpoint 1: no locations (pending on future shared library load)`。
> 这是正常现象，待 kernel launch 后再重新 `b` 即可；或用 `process interrupt` 强制停下再设。
>
> 另外，若 `/proc/debug_switch` 是容器/云环境里的虚假状态（底层未真正开启调试能力），
> `run` 时可能报 `error: 'A' packet returned an error: 8`。此时需在宿主机以 root 正确开启调试开关，否则无法断点。

> - 断点命中后 msdebug 自动 `[Switching to focus on ... CoreId X]`，跟随断点所在 AiCore 核；
> - AIV 多核场景下，每个核都会停在断点，`continue` 一次放一个核。

## 5. 单步 / 查看

```text
(msdebug) bt                    # 调用栈
(msdebug) frame variable x1     # 查看某个核函数形参/局部变量
(msdebug) var                   # 简写：显示当前作用域内所有局部变量
(msdebug) next                  # 单步跳过（在源码行停）
(msdebug) step                  # 单步进入（可步入模板/库函数）
(msdebug) memory read <addr>    # 读设备内存
(msdebug) reg read              # 读寄存器
(msdebug) register read -a      # 读全部寄存器（PC/COND/CTRL/GPR* 等）
```

查看全部局部变量示例（`var`）：

```text
(msdebug) var
(KernelAdd *__stack__) this = 0x00000000001d78a8
(uint8_t *__gm__) x = 0x000012c0c0013000 ""
(uint8_t *__gm__) y = 0x000012c0c001c000 ""
(uint8_t *__gm__) z = 0x000012c0c0025000 ""
(uint32_t) totalLength = 16384
(uint32_t) tileNum = 8
```

## 6. 查询 AiCore 运行信息（ascend info）

msdebug 提供 `ascend info` 系列子命令，查询 Device / AiCore / Task / Stream / Block 等运行上下文，便于定位核上有哪些 wave 被断点命中：

```text
(msdebug) ascend info devices        # 查询 Device（Aic/Aiv 数量与掩码）
(msdebug) ascend info cores          # 查询各 AiCore 的断点/PC/stop reason
(msdebug) ascend info tasks          # 查询当前 task（算子 task 级信息）
(msdebug) ascend info stream         # 查询 stream 类型
(msdebug) ascend info blocks         # 查询 block 列表
```

`ascend info cores` 示例（多核各 wave 的断点状态）：

```text
(msdebug) ascend info cores
  CoreId Type Device Stream Task Block               PC    stop reason Filename Line
*      0  aiv      3    47    0     4  0x12c041200920  breakpoint 1.1       NA   NA
       1  aiv      3    47    0     5  0x12c041200920  breakpoint 1.1       NA   NA
       2  aiv      3    47    0     6  0x12c041200920  breakpoint 1.1       NA   NA
```

## 7. 继续 / 结束

```text
(msdebug) process continue      # 继续执行，让用例跑完
(msdebug) q                     # 退出调试会话（简写）
(msdebug) y                     # 退出时若提示确认，输入 y
```

> 不放行 kernel，TTK 用例会一直停在 `OnDynProfiling`（kernel 停在断点等人 `continue`），
> **这是正常现象**，不是卡死。跟进并 `continue` 即可，或删除断点后放行。

---

# 命令速查

| 命令 | 说明 |
|------|------|
| `run` / `process launch` | 启动被调试目标 |
| `process interrupt` | 在 kernel 运行中强制停下，便于设断点 |
| `process continue` | 继续执行 |
| `b <文件>:<行号>` / `breakpoint set -f <文件> -l <行号>` | 源码行断点（需 DWARF，`b` 为简写） |
| `breakpoint set -n <函数名>` | 函数名断点 |
| `breakpoint set -a <地址>` | 地址断点（无 DWARF 时的退路，汇编级） |
| `breakpoint list` / `breakpoint delete N` | 查看 / 删除断点 |
| `next` / `thread step-over` | 单步跳过 |
| `step` / `thread step-in` | 单步进入 |
| `thread step-out` | 跳出当前函数 |
| `bt` | 调用栈 |
| `frame variable` / `var` | 查看局部变量/形参（`var` 看全部） |
| `memory read <addr>` | 读设备内存 |
| `reg read` / `register read -a` | 读寄存器 / 读全部寄存器 |
| `ascend info devices/cores/tasks/stream/blocks` | 查询 Device / AiCore / Task / Stream / Block 运行信息 |
| `q` + `y` | 退出调试会话 |
| `quit` | 退出 |

---

# 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| stub 日志不打印、收不到 kernel 信息 | 劫持未生效（ctypes 局部加载盖过 `LD_PRELOAD`） | 确认已合入 `rts_interface.py` 的 `CDLL(None)` 适配补丁 |
| 有 `Launch of Kernel` 但 `frame #0: 0x0`、断点 `Unable to resolve ... any actual locations` | kernel 无 `.debug_info`（多为 `-b=release` 内置包） | 改为 `-d=true -b=false` 自编译，使 `-O0 -g` 生效 |
| kernel 拉起前断点报 `pending on future shared library load` | 断点设得太早（kernel 未加载） | 等 kernel launch 后再设断点；或用 `process interrupt` 强制停下再设 |
| `error: 0x20200`（OPEN_KO_ERR） | 设备被另一 msdebug/lldb-server 占用 | `ps -ef \| grep -iE 'msdebug\|lldb-server'` 清理后再跑 |
| TTK 用例停在 `OnDynProfiling` 一直 RUNNING | kernel 停在断点等人 `continue`（正常） | 跟进 `continue` 放行或删断点 |
| `run` 后立即退出 | 交互驱动 stdin 提前 EOF | 保持在交互 TTY 中，或用 PTY 驱动（MI driver 不支持本地调试，不能用于自动化驱动本地 TTK） |
| `step` 报警 `invalid thread index` | 命令拼写被误解析 | 用 `next`/`step` 完整命令，勿用简化写法 |

---

# 验证清单

```bash
# 1) msdebug 可用
<msdebug路径>/msdebug --version          # 期望见 MindStudio banner + 版本号

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
