# 离线数据准备与导入

## 使用须知

适合两类场景：

- **无卡生成、有卡执行**：在没有 NPU 卡的机器上生成输入/Golden，拷到有卡机器上执行并比对
- **固定输入复用**：只制备输入，多次执行共用同一份数据

三步：dump 落盘 → 把数据目录拷到目标机 → load 回放。两次用同一份 CSV，按 `testcase_name` 匹配。

## 文档导览

1. **[数据落盘](#一数据落盘)**
2. **[数据转移](#二数据转移)**
3. **[数据回放](#三数据回放)**
4. **[CSV 路径方式](#四csv路径方式)**
5. **[注意事项](#五注意事项)**

## 一、数据落盘

```bash
python3 -m ttk e2e -i cases.csv --plugin /path/to/assets \
  --no-prof --dump in,golden --dump-format bin \
  --manual-data-dirs /data/manual -o prepare_result.csv
```

落盘位置：

- 指定 `--manual-data-dirs /data/manual` 时写到该目录；缺省落到 plugin 目录或当前目录的 `manual_data/`
- 产物为 `/data/manual/<case>/` 目录，内含 `input_*` / `scalar_*` / `golden_*` 文件（命名见 [Dump 数据调试](../dump_debug_guide.md)）

选项：只准备输入用 `--dump in`；`--dump-format` 支持 bin / npy / pt，默认 bin。

### XPU 三方输出

参与三方比对的 XPU 输出也可提前备好：在 XPU 机器上跑 TTK，让输入、Golden 与三方输出一起落盘。

支持范围：

- 目前仅 e2e 通路的 torch / tf
- 需在 XPU 机器上先起 xpu-server（三方硬件在本机时 endpoint 指向 `127.0.0.1` 即可，启动方式见 [三方交叉校验](./cross_check_guide.md)；跨机部署见 [xpu-server 部署指南](../reference/xpu_server.md)）

XPU 机器上的配置：

```yaml
# ttk.conf.yaml（XPU 机器）
remote:
  endpoints:
    - host: 127.0.0.1
      port: 9090
frameworks:
  torch:
    xpu:
      torch_lib: cuda          # GPU；MLU 用 mlu
```

运行（`--dump full` 会把输入、Golden 与三方输出一起落盘）：

```bash
python3 -m ttk e2e -i cases.csv --config ttk.conf.yaml --dump full
```

产物拷回 NPU 环境，作 `manual_xpu_binaries` 回放参与三方比对（见 [三方交叉校验](./cross_check_guide.md)）。

## 二、数据转移

```bash
scp -r /data/manual user@目标机:/data/manual
```

目录内文件不要改名或增删；改了 API、attributes、shape 或插件后需重新 dump。

## 三、数据回放

```bash
python3 -m ttk e2e -i cases.csv --plugin /path/to/assets \
  --manual-data-dirs /data/manual -o replay_result.csv
```

- 可给多个目录（按顺序搜），也可直接指向调试 dump 目录（规则见 [Dump 数据调试](../dump_debug_guide.md)）
- 目录里有该用例的数据就恢复执行；某个用例没有则报错；空目录才按普通方式生成并落盘
- 只有输入的数据：静态用例先补算 Golden 再执行，动态 shape 用例执行后补
- 约束：回放时不能带 `--no-prof`、`--validate`、`--backend cpu`（E2E）、`--co`（Kernel）；`--seed` 不生效
- 精度比对、插件、custom compare 与普通执行一样生效

## 四、CSV 路径方式

不用上面的目录方式，直接在 CSV 里写文件路径：

| 字段 | kernel | geir | aclnn | e2e |
| --- | --- | --- | --- | --- |
| `manual_input_binaries` | ✓ | ✓ | — | — |
| `manual_tensor_binaries`（历史名） | — | — | ✓ | — |
| `manual_golden_binaries` | ✓ | ✓ | ✓ | — |
| `manual_xpu_binaries` | ✓ | ✓ | ✓ | ✓ |

优先级：`--manual-data-dirs` 目录集 > CSV 手工路径 > 正常生成。

三方输出（`manual_xpu_binaries`）也按此回放，用于三方比对；输出文件如何在 XPU 环境生成见 [三方交叉校验](./cross_check_guide.md)。

## 五、注意事项

- 数据目录不记录"这份数据是哪个 CSV、哪版插件生成的"，文件被改名或替换不会被发现
- 改了 shape 但元素总数没变时，旧数据可能被当成新数据用
- 不同通路生成的同名目录不要放在一起

常见报错与检查项见 [FAQ 一本通](../faq/faq_guide.md)。
