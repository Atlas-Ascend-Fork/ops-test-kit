# GEIR 模式用例编写

子命令：`python3 -m ttk geir`
结构类：`GeirTestcase`（继承 `TestcaseKernel`；标准字段 31 个 = Kernel 30 个 + GEIR 专属 `dyn_input_shapes`，另有 `attributes1`～`attributes9` 扩展列，全量 40 个）

## 与 Kernel 的关系

GEIR **直接复用 Kernel 的 CSV 用例与 golden 资产**，无需单独编写：`examples/case_store/kernel/cases.csv` 原样即可跑 `ttk geir`。Kernel 的必填字段、完整字段表与常见模式（TensorList、可选输入 None 占位、ValueDepend 属性定值等）见 `kernel-case.md`，本节只讲 GEIR 多出来的部分。

> [!TIP]
> 直接用 Kernel 的 CSV 跑 GEIR 无需改动；仅当需要精确控制某输入维度的动态性时，在末尾追加 `dyn_input_shapes` 列即可。

## GEIR 专属字段

| 字段 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `dyn_input_shapes` | 否 | `None` | 动态编译时输入张量的**图编译 desc_shape**。`-1` 表示未知维度，`-2` 表示未知秩（rank）。未指定时框架自动将 `input_shapes` 的**全部维度**替换为 `-1`（与 Kernel 仅替换正数维度不同）。如 `"((128,-1),(1,1024))"` 表示第 1 个输入第 0 维固定 128、第 1 维动态，第 2 个输入固定；单输入未知秩写 `"((-2,),)"` |

## input_shapes 与 dyn_input_shapes 的关系

| 字段 | 作用 | 示例 |
|------|------|------|
| `input_shapes` | 实际执行时的数据 shape | `"((128,1024),(1,1024))"` |
| `dyn_input_shapes` | 图编译时的 desc_shape，`-1` 未知维度、`-2` 未知秩 | `"((128,-1),(1,1024))"` |

## 常见模式

### 缺省（全动态）

不写 `dyn_input_shapes` 列，动态模式（`-d`）下等价于将 `input_shapes` 全部维度替换为 `-1`。

### 尾维固定 / 逐维混合

```csv
testcase_name,op_name,input_shapes,dyn_input_shapes
add_dyn_dim0,add,"((128,1024),(1,1024))","((128,-1),(1,1024))"
```

第 1 个输入第 0 维固定 128、第 1 维动态，第 2 个输入固定。

### `-2` 未知秩

单输入且秩未知时写 `"((-2,),)"`。

### TensorList 嵌套 / 可选输入 None 槽

`dyn_input_shapes` 与 `input_shapes` 同构：TensorList 逐元素嵌套声明；可选输入（`None` 占位）槽同样写 `None`。

## 编译模式差异

| 参数 | 模式 | GEIR 默认 | Kernel 默认 |
|------|------|-----------|-------------|
| `-c` | 静态 shape 编译 | **开启** | 关闭 |
| `-d` | 动态 shape 编译 | 关闭 | **开启** |

> [!IMPORTANT]
> GEIR 默认开启静态编译（`-c`），与 Kernel 默认开启动态编译（`-d`）相反。`dyn_input_shapes` 只在 `-d` 下生效，需显式指定 `-d`。

## 示例

```csv
testcase_name,op_name,input_shapes,input_dtypes,dyn_input_shapes
geir_add_01,add,"((128,1024),(1,1024))","('float32','float32')","((128,-1),(1,1024))"
```

更多示例见 `examples/case_store/geir/`（`dyn_input_shapes` 全语义：缺省全 `-1` / 尾维固定 / 逐维混合 / `-2` 未知秩 / 嵌套 TensorList / 可选输入 None，配套 `golden.py` 插件）。

## 验证

```shell
python3 -m ttk geir -i cases.csv --validate
python3 -m ttk list -i cases.csv
```
