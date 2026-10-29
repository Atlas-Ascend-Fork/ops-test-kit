# 三方交叉校验与性能采集

## 使用须知

通过远端 XPU（GPU/MLU/CPU 等非 NPU 加速器）运行同一算子，与 NPU 输出做交叉比对，或只采集 XPU 侧性能。

> [!NOTE]
> XPU 采集为远端方式：三方硬件在本机时同样在本机启动 xpu-server，endpoint 指向 `127.0.0.1` 即可。

## 文档导览

1. **[前置准备](#一前置准备)**
2. **[交叉校验](#二交叉校验)**
3. **[性能采集](#三性能采集)**
4. **[参数约束](#四参数约束)**

## 一、前置准备

### 1. 启动 xpu-server（server 端）

在 XPU 机器（server 端）上启动：

```bash
python -m ttk.remote.server.xpu_server --port 9090
```

跨机 / CI 部署见 [xpu-server 部署指南](../reference/xpu_server.md)。

### 2. 配置 worker 端

```yaml
# ttk.conf.yaml
remote:
  endpoints:
    - host: 127.0.0.1      # XPU 机器地址
      port: 9090
```

跨机加密时补 mTLS 三件套（tls_ca / tls_cert / tls_key），本机可省略。

### 3. 声明 third_party（cross_check 必需）

在算子 TestSpec 里声明三方参考：

```python
class AddTestSpec:
    third_party = {"torch": "torch.add"}
```

未声明时框架按 `op_name`/算子名自动匹配，匹配失败则 `GOLDEN_FAILURE`。

## 二、交叉校验

### 1. 命令

```bash
python3 -m ttk kernel -i cases.csv \
  --plugin /path/to/assets.py \
  --config ttk.conf.yaml \
  --compare cross_check
```

`cross_check` 计算 NPU 输出与三方输出的误差比值（mare/mere/rmse），按 level 预设容差判定，详见 [精度比对方法](../reference/precision_comparison.md)。

### 2. provider 过滤

多个算子配了不同 provider 时，用 `--provider torch` 缩小范围（不改变 spec 里的声明）。

### 3. 离线执行（可选）

三方比对也可离线：先在 XPU 环境生成三方输出（见 [离线数据准备与导入](./offline_data_prepare_guide.md)「XPU 三方输出」），再回放：

- **CSV 路径**：`manual_xpu_binaries` 写文件路径，与 `manual_input_binaries` / `manual_golden_binaries` 成套提供、数量与张量一致；跳过远端采集，不需要 `--config` / `--provider`
- **目录回放**：`--manual-data-dirs` 指向含 `xpu_{provider}_{idx}_{dtype}` 槽位文件的目录（`--dump xpu` 产物）

远端采集失败时的表现：xpu-server 不可达 → `GOLDEN_FAILURE`；三方执行报错 → 该 provider 标 `FAIL`，不参与比对。

## 三、性能采集

```bash
python3 -m ttk kernel -i cases.csv --config ttk.conf.yaml --xpu-perf
```

| 维度 | `--xpu-perf` | `--compare cross_check` |
|------|-------------|------------------------|
| 输出数据 | 不取 | 取回参与比对 |
| 精度比对 | 不影响（仍按默认比对） | 替换为三方比对 |
| 结果落列 | `xpu_metrics` | `precision_status` + `xpu_metrics` |

`xpu_metrics` 含各 provider 的 `status`、`api`、`device_us`、`peak_memory_mb`。

## 四、参数约束

三种用法（cross_check / `--xpu-perf` / `--dump xpu`）的开关要求：

- `--config`：都要。只有 cross_check 缺它不报错，结果为 `GOLDEN_FAILURE`
- `--plugin`：建议。未声明 `third_party` 时框架自动匹配
- `--provider`：可选，缩小范围
- `--no-prof` / `--validate`：这三种用法都不触发采集

> [!TIP]
> E2E 且 `op_name` 为点分路径（如 `torch.add`）时可不声明 `third_party`，框架直接按路径解析。
