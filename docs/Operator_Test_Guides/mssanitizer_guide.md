# mssanitizer适配方法指南


---

# 概述

[mssanitizer](https://gitcode.com/Ascend/mssanitizer/blob/master/docs/zh/user_guide/mssanitizer_user_guide.md)（MindStudio Sanitizer）是 MindStudio 提供的算子级异常检测工具，包含内存检测（memcheck）、竞争检测（racecheck）、未初始化检测（initcheck）和同步检测（synccheck）四个子功能，可检测算子执行过程中的非法读写、多核踩踏、非对齐访问、内存泄漏、数据竞争、未初始化读取及同步异常等问题。

TTK 内部自带预热（warmup）与 Profiling 采集流程。使用 mssanitizer 包裹 TTK 执行时，预热会额外编译并执行一个 warmup kernel 被 mssanitizer 插桩检测，Profiling 采集会增加额外开销并与检测产生干扰。因此建议通过参数关闭 TTK 内部的预热与 Profiling 采集：

```
--warmup=false --task-prof=false
```

同时，mssanitizer 命令格式为 `mssanitizer [<options>] [--] <user_program> [<user_options>]`。由于 TTK 命令带有自身的参数（如 `-i`、`--warmup` 等），需使用 `--` 分隔 mssanitizer 选项与 TTK 命令，避免参数解析冲突。

# 环境准备

## 基本环境配置

- 建议Python 3.8+
- 完成CANN包安装

## 安装 MindStudio Sanitizer 工具

mssanitizer 随 CANN toolkit 自带安装，安装 CANN 后即可使用。确认工具可用：

```shell
mssanitizer --version
```

设置环境变量：

```shell
source /usr/local/Ascend/cann/set_env.sh
```

> 注：mssanitizer 工具还需要 firmware 和 driver 包的适配，请关注这两个包是否安装正确。

# 调用 mssanitizer 命令

## 基本用法

使用 `mssanitizer` 包裹 `ttk kernel` / `ttk aclnn` 命令，并建议添加 `--warmup=false --task-prof=false` 关闭 TTK 内部预热与 Profiling：

```shell
# Kernel 模式（动态 shape 编译）
mssanitizer --tool=memcheck -- python3 -m ttk kernel -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast --warmup=false --task-prof=false

# Kernel 模式（二进制）
mssanitizer --tool=memcheck -- python3 -m ttk kernel -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast -b=release --warmup=false --task-prof=false

# ACLNN 模式
mssanitizer --tool=memcheck -- python3 -m ttk aclnn -i examples/case_store/aclnn/cases.csv -t add_fp16_broadcast --warmup=false --task-prof=false
```

> 注：`--tool=memcheck` 为默认值，可省略。上例中显式写出便于理解。

## 结合 TTK 筛选参数

mssanitizer 包裹的 TTK 命令可以使用 TTK 自带的用例筛选与编译参数：

```shell
# 按索引范围运行
mssanitizer --tool=memcheck -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false --ti=0-9

# 按算子名筛选
mssanitizer --tool=memcheck -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false --op add,mat_mul_v3

# 按优先级筛选
mssanitizer --tool=memcheck -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false --priority=1-3

# 随机选取10个用例
mssanitizer --tool=memcheck -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false --tc=10

# 运行Add算子（动态 shape 编译）
mssanitizer --tool=memcheck -- python3 -m ttk kernel -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast -d --warmup=false --task-prof=false

# 运行Add算子（二进制）
mssanitizer --tool=memcheck -- python3 -m ttk kernel -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast --binary --warmup=false --task-prof=false
```

## 切换检测子工具

mssanitizer 提供四种检测子工具，通过 `--tool`（简写 `-t`）参数指定：

| 子工具 | 检测内容 | 命令示例 |
|--------|---------|---------|
| `memcheck`（默认） | 非法读写、多核踩踏、非对齐访问、内存泄漏、非法释放 | `mssanitizer -t memcheck -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false` |
| `racecheck` | 数据竞争（WAW / WAR / RAW） | `mssanitizer -t racecheck -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false` |
| `initcheck` | 读取未初始化内存导致的脏数据 | `mssanitizer -t initcheck -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false` |
| `synccheck` | 未正确使用同步指令导致的同步失败 | `mssanitizer -t synccheck -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false` |

多种检测功能可通过多次指定 `--tool` 参数同时开启：

```shell
# 同时开启内存检测和竞争检测
mssanitizer -t memcheck -t racecheck -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false
```

> 建议先运行 memcheck 确认算子无内存异常后，再按需运行 racecheck / initcheck / synccheck。

## 附加 mssanitizer 参数

可在 `mssanitizer` 与 `--` 之间添加 mssanitizer 自身参数：

```shell
# 开启内存泄漏检测
mssanitizer --tool=memcheck --leak-check=yes -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false

# 开启分配内存未使用检测
mssanitizer --tool=memcheck --check-unused-memory=yes -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false

# 只检测指定名称的算子（支持模糊匹配）
mssanitizer --tool=memcheck --kernel-name="add" -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false

# 只检测指定 block（单 block 调试模式）
mssanitizer --tool=memcheck --block-id=0 -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false

# 显示 AscendC API 内的完整调用栈
mssanitizer --tool=memcheck --full-backtrace=yes -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false

# 将报告输出到文件
mssanitizer --tool=memcheck --log-file=result.log -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false
```

更多 mssanitizer 参数详见：[mssanitizer 用户指南](https://gitcode.com/Ascend/mssanitizer/blob/master/docs/zh/user_guide/mssanitizer_user_guide.md)

# 异常报告解读

执行后，mssanitizer 会在控制台输出异常报告，并在当前目录生成工具日志文件。

## 报告级别

异常报告具有以下级别：

| 级别 | 说明 |
|------|------|
| `ERROR` | 最高严重级别的异常，涉及针对内存操作的确定性错误，如非法读写、内存泄漏、非对齐访问、内存未初始化、竞争异常等。强烈建议检查此级别的异常。 |
| `WARNING` | 不确定性的风险，可能出现的异常现象由实际情况决定，如多核踩踏、内存分配未使用等。 |

## 常见异常类型

| 子工具 | 异常类型 | 说明 |
|--------|---------|------|
| memcheck | 非法读写 | 访问了未分配的内存导致的异常 |
| memcheck | 多核踩踏 | AI Core 核心访问了重叠的内存导致的踩踏问题 |
| memcheck | 非对齐访问 | DMA 搬运的地址与最小访问粒度未对齐 |
| memcheck | 内存泄漏 | 申请内存使用后未释放（需 `--leak-check=yes`） |
| memcheck | 非法释放 | 对未分配或已释放的地址进行释放 |
| memcheck | 分配内存未使用 | 申请了内存但未使用（需 `--check-unused-memory=yes`） |
| racecheck | WAW / WAR / RAW | 两个内存事件尝试访问同一块内存时的数据竞争 |
| initcheck | 未初始化读取 | 读取了已申请但未初始化的内存 |
| synccheck | 未配对 SetFlag | 算子中存在未配对的 SetFlag 同步指令 |
| synccheck | 算子卡死 | 错误使用同步指令导致算子卡死 |

## 输出文件

| 文件名 | 说明 |
|--------|------|
| `mssanitizer_{TIMESTAMP}_{PID}.log` | 工具运行日志，位于 `mindstudio_sanitizer_log` 目录下，TIMESTAMP 为时间戳，PID 为工具进程号 |
| `kernel.{PID}.o` | 算子缓存文件，用于解析异常调用栈。工具正常退出时自动清理，异常退出（如 Ctrl+C 中止）时保留，建议及时删除 |
| `tmp_{PID}_{TIMESTAMP}` | 临时文件夹，用于生成算子 Kernel 二进制。工具正常退出时自动清理，异常退出时保留，建议及时删除 |

> 注：不修改编译选项时（快速定界），异常报告不显示调用栈信息；如需全量检测并显示调用栈，请参考 [mssanitizer 用户指南](https://gitcode.com/Ascend/mssanitizer/blob/master/docs/zh/user_guide/mssanitizer_user_guide.md) 中的"算子编译选项配置"章节修改编译选项后重新编译算子。
