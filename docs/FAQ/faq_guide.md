# FAQ一本通


---

# 问题定位自查

1. TTK版本是否最新，可执行 `python3 -m ttk -v` 查看版本号
2. 自查命令是否有误，可执行 `python3 -m ttk kernel --help` 查看参数说明
3. CANN包是否已安装并source环境变量
4. 如果出现算子内部报错，请检查算子名称书写是否有误
5. 如果使用自定义插件，检查注册器名称是否与CSV中的算子名匹配

# 环境问题

## 运行报错 "ASCEND_HOME_PATH not found"

TTK依赖CANN工具包。请先安装CANN并source环境变量：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## CANN分包环境变量设置

如使用CANN分包安装，需额外设置环境变量：

```shell
source /usr/local/Ascend/latest/bin/setenv.bash
export ASCEND_HOME_PATH=/usr/local/Ascend/latest
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/latest
```

常见报错（未source环境变量）：
- `ImportError: libhccl.so`
- `AttributeError: 'NoneType' object has no attribute 'acl_init'`

## Python版本问题

建议Python 3.8+，可通过以下命令确认版本：

```shell
python3 --version
```

# 用例编写问题

## CSV中数据类型怎么写？

支持的数据类型字符串：`float16`/`fp16`、`float32`/`fp32`、`float64`/`fp64`、`bfloat16`/`bf16`、`int8`、`int16`、`int32`、`int64`、`uint8`、`bool`、`complex64`、`complex128` 等。

## CSV中包含特殊字符的字段需要用双引号包裹吗？

是的。CSV中包含括号、逗号、引号的字段（如shape、dtype元组、字典等）需要用双引号包裹：

```csv
"((128, 1024), (1, 1024))"     # 正确
((128, 1024), (1, 1024))       # 错误，逗号会被解析为CSV分隔符
```

## 输出Shape依赖输入Tensor中的值，无法从输入shape推导（如NonZero）怎么写CSV？

`output_shapes` 填写一个占位shape（非空），同时在 `output_shape_unknown_indexes` 中标记未知输出的索引：

```csv
output_shapes,"((8, 1716),)"
output_shape_unknown_indexes,"(0,)"
```

框架会在运行时确定实际输出shape。

## TensorList算子怎么写CSV？

通过 `input_shapes` 字段的嵌套结构标记TensorList。多个tensor放在同一层元组中表示它们组成一个TensorList：

```csv
input_shapes,"(((3,3),(3,2),(3,4)),(3,5))"
```

表示前3个输入tensor组成一个TensorList，第4个是单独张量。

## 如何查看CSV中定义了哪些用例？

使用 `list` 命令预览：

```shell
python3 -m ttk list -i cases.csv
python3 -m ttk list -i cases.csv --op add
```

> 注：`ttk list` 仅列出用例名，并要求 `testcase_name` 列存在且非空（否则直接报错），不做其他字段校验；完整校验请用对应模式 `--validate`。

# 执行问题

## 算子太大执行超时或超出内存

- 设置单用例超时时间：`--proc-timeout=300`
- 设置HBM内存限制：`-l 10`（10GB，仅 Kernel/ACLNN 通路注册）
- 预留HBM内存：`--reserve-hbm=512`（仅 Kernel 通路注册）

## 多卡并行时部分用例失败

- 检查设备是否正常：`python3 -m ttk info`
- 排除问题设备：`--devices='^2,^3'`（每个被剔除 ID 均需带 `^` 前缀，等价于全选后剔除，也可写 `'0-7,^2,^3'`）
- 减少每张卡进程数：`--pc=1`

## 如何复现某次测试结果？

固定随机种子：

```shell
python3 -m ttk kernel -i cases.csv --seed 42
```

## 如何快速验证环境是否正常？

```shell
# 查看设备信息
python3 -m ttk info

# 运行内置示例
python3 -m ttk kernel -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast
```

# 精度问题

## 精度比对失败如何排查？

1. 使用 `--dump-on-fail` 在失败时自动Dump数据
2. 使用 `--dump in,out,golden --dump-format npy` 保存输入、输出和Golden到npy文件
3. 用 `-t case_name` 锁定到单个用例（单用例执行时框架自动产出该用例的独立日志目录）
4. 检查CSV中 `precision_tolerances` 和 `absolute_precision` 设置是否合理（`precision_tolerances` 由 `--compare close`（rtol/ptol）/`cosine`（rtol）/`requant`（ptol）及 TestSpec 声明的 `quant` 判据（ptol）读取、`absolute_precision` 仅 `close` 生效；默认 `mixed` 需在 TestSpec `tolerance` 中调整）

```shell
# 失败时自动Dump数据
python3 -m ttk kernel -i cases.csv --dump-on-fail

# Dump输入和Golden到npy文件
python3 -m ttk kernel -i cases.csv --dump in,golden --dump-format npy
```

## 精度比对方法选择建议

| 场景 | 推荐方法 | 参数 |
|------|---------|------|
| 浮点运算常规测试（默认） | 混合容差（生态算子开源精度标准） | `--compare mixed`（默认） |
| 统计误差均值/最大值 | 统计相对误差（社区标准） | `--compare stat_rel_err` |
| 逐点 isclose | 数值近似 | `--compare close` |
| 大规模向量整体趋势 | 余弦相似度 | `--compare cosine` |
| 整型运算/需要精确结果 | 二进制精确 | `--compare binary` |
| hifloat8类型 | 重量化 | `--compare requant`（自动） |
| 三方交叉校验 | 三方交叉校验 | `--compare cross_check`（需 `third_party`） |

## 如何调整精度容差？

在CSV中通过以下字段设置：

```csv
precision_tolerances,"((0.001, 0.001),)"
absolute_precision,1e-8
```

- `precision_tolerances`：每个输出的 (rtol, ptol) 对
- `absolute_precision`：全局绝对精度容差

> 注意：这两个 CSV 字段是 legacy 字段，`precision_tolerances` 由 `--compare close`（rtol/ptol）/`cosine`（rtol）/`requant`（ptol）及 TestSpec 声明的 `quant` 判据（ptol）读取、`absolute_precision` 仅 `close` 读取。默认的 `mixed` 不读 CSV 容差字段，需在 TestSpec 的 `tolerance` 中覆盖（需 `--plugin`）：
>
> ```python
> tolerance = {"float32": {"standard": "mix_tolerance", "rtol": 0.002, "atol": 1e-5}}
> ```
>
> 可覆盖项：`rtol` / `atol` / `required_matched_ratio` / `max_abs_error_limit`（配置后将完全替代 ULP 动态上限，见[精度比对方法](../Precision_Comparison.md)）。

# 插件问题

## 自定义Golden函数不生效

1. 确认已通过 `--plugin` 参数传入插件文件路径
2. 确认 TestSpec 类名/`__spec__` 注册名与CSV中的 `op_name` 一致（类名遵循 `PascalCase+TestSpec`，如 `AbsTestSpec`）
3. 检查插件文件是否有语法错误

## 多个插件有同名算子注册

TestSpec（`__spec__`）注册同名时先加载的搜索路径优先（first match wins）；旧 `__golden__`/`__input__` 注册同名时后扫描的插件会覆盖先扫描的。两种情况均不会打印警告，请确保注册名称唯一。
