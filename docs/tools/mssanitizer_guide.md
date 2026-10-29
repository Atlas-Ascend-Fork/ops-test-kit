# mssanitizer 适配方法

## 使用须知

[mssanitizer](https://gitcode.com/Ascend/mssanitizer/blob/master/docs/zh/user_guide/mssanitizer_user_guide.md)（MindStudio Sanitizer）提供内存（memcheck）、竞争（racecheck）、未初始化（initcheck）、同步（synccheck）四种检测。包裹 `ttk kernel` / `ttk aclnn` 使用，需关闭 TTK 内部预热与 Profiling，并用 `--` 分隔双方参数。

文档导览：

1. **[环境准备](#一环境准备)**
2. **[基本用法](#二基本用法)**
3. **[工具与参数](#三工具与参数)**
4. **[报告解读](#四报告解读)**

## 一、环境准备

- Python 3.8+
- mssanitizer 随 CANN 自带，完成 CANN 安装后确认可用：

```shell
source /usr/local/Ascend/cann/set_env.sh
mssanitizer --version
```

> [!NOTE]
> 工具还需 firmware 和 driver 包适配，请确认这两个包安装正确。

## 二、基本用法

> [!IMPORTANT]
> 包裹 TTK 时：1) 用 `--warmup=false --task-prof=false` 关闭 TTK 内部预热与 Profiling；2) 命令格式为 `mssanitizer [选项] [--] <程序> [参数]`，TTK 参数前需 `--` 分隔。

### 1. 包裹 TTK 命令

```shell
# Kernel（动态 shape 编译）
mssanitizer --tool=memcheck -- python3 -m ttk kernel -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast --warmup=false --task-prof=false

# Kernel（二进制）
mssanitizer --tool=memcheck -- python3 -m ttk kernel -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast -b=release --warmup=false --task-prof=false

# ACLNN
mssanitizer --tool=memcheck -- python3 -m ttk aclnn -i examples/case_store/aclnn/cases.csv -t add_fp16_broadcast --warmup=false --task-prof=false
```

> [!TIP]
> `--tool=memcheck` 为默认值，可省略。

### 2. 结合 TTK 参数

TTK 自带的筛选与编译参数照常可用：

```shell
mssanitizer --tool=memcheck -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false --ti=0-9
mssanitizer --tool=memcheck -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false --op add,mat_mul_v3
mssanitizer --tool=memcheck -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false -d
```

## 三、工具与参数

### 1. 检测子工具（--tool）

| 子工具 | 检测内容 |
|--------|---------|
| `memcheck`（默认） | 非法读写、多核踩踏、非对齐访问、内存泄漏、非法释放 |
| `racecheck` | 数据竞争（WAW / WAR / RAW） |
| `initcheck` | 未初始化读取 |
| `synccheck` | 同步失败（未配对 SetFlag 等） |

多个 `--tool` 可同时开启：

```shell
mssanitizer -t memcheck -t racecheck -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false
```

> [!TIP]
> 建议先跑 memcheck 确认无内存异常，再按需跑 racecheck / initcheck / synccheck。

### 2. 常用附加参数

```shell
mssanitizer --tool=memcheck --leak-check=yes -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false
mssanitizer --tool=memcheck --kernel-name="add" -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false   # 只检测指定算子（模糊匹配）
mssanitizer --tool=memcheck --log-file=result.log -- python3 -m ttk kernel -i cases.csv --warmup=false --task-prof=false # 报告输出到文件
```

更多参数详见 [mssanitizer 用户指南](https://gitcode.com/Ascend/mssanitizer/blob/master/docs/zh/user_guide/mssanitizer_user_guide.md)。

## 四、报告解读

执行后在控制台输出异常报告，当前目录生成工具日志。

### 1. 报告级别

| 级别 | 说明 |
|------|------|
| `ERROR` | 确定性内存错误（非法读写、内存泄漏、非对齐访问等） |
| `WARNING` | 不确定风险（多核踩踏、内存分配未使用等） |

### 2. 常见异常类型

| 子工具 | 异常类型 | 说明 |
|--------|---------|------|
| memcheck | 非法读写 | 访问未分配内存 |
| memcheck | 多核踩踏 | AI Core 访问重叠内存 |
| memcheck | 非对齐访问 | DMA 地址未对齐 |
| memcheck | 内存泄漏 | 申请后未释放（需 `--leak-check=yes`） |
| memcheck | 非法释放 | 释放未分配或已释放的地址 |
| memcheck | 分配内存未使用 | 申请后未使用（需 `--check-unused-memory=yes`） |
| racecheck | WAW / WAR / RAW | 数据竞争 |
| initcheck | 未初始化读取 | 读取未初始化内存 |
| synccheck | 未配对 SetFlag | SetFlag 未配对 |
| synccheck | 算子卡死 | 同步指令使用错误 |

### 3. 输出文件

| 文件名 | 说明 |
|--------|------|
| `mssanitizer_{TIMESTAMP}_{PID}.log` | 工具运行日志（`mindstudio_sanitizer_log` 目录） |
| `kernel.{PID}.o` | 算子缓存文件；工具正常退出自动清理 |
| `tmp_{PID}_{TIMESTAMP}` | 临时文件夹；工具正常退出自动清理 |

> [!NOTE]
> 不修改编译选项时（快速定位），异常报告不含调用栈；需全量检测与调用栈时参考 [mssanitizer 用户指南](https://gitcode.com/Ascend/mssanitizer/blob/master/docs/zh/user_guide/mssanitizer_user_guide.md) "算子编译选项配置"。
