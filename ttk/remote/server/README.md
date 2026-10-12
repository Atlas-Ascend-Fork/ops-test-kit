# xpu-server — 部署指南

> TTK 远端 XPU 执行服务器。接收 TTK worker 端的算子执行请求，在多种硬件加速器上跑 PyTorch/TensorFlow 算子，回传结果。
>
> 本文档出现的硬件厂商名、设备路径、镜像名仅作示例；TTK 通过配置驱动支持任意符合接口的硬件加速器。

---

## 快速开始

```bash
# 1. 在 XPU 机器上准备整仓（git clone / pip install / 拷贝整个仓库皆可）
git clone <ttk-repo> ~/ttk && cd ~/ttk

# 2. 启动（唯一对外入口）
python -m ttk.remote.server.xpu_server --port 9090
```

TTK worker 端通过 `ttk.conf.yaml`（或 `--config`）配置 `remote.endpoints` 连过来。到此就通了。

> **部署形态说明**：server 与 TTK 同仓同源（整仓部署）——executor 消费
> `ttk.core_modules.common` 的单份契约代码（bind_params）与 profiler 共享引擎，
> 因此不再支持"只拷 `ttk/remote/server/` 子目录独立运行"的旧方式
> （旧命令 `python -m xpu_server.xpu_server` 已废止）。

---

## 两个概念

- **provider**：用什么框架/库执行算子。用户通过 `--provider=torch` 指定，server 通过 `/v1/heartbeat` 声明自己能提供哪些。自动探测的常见值：`torch`、`tf`（其余可经 `provider_framework` 映射为别名，如 `flash_attn`→torch）
- **hardware**：底层硬件类型，决定设备字符串格式（如 torch 的 `mlu:0`、tf 的 `/device:MLU:0`）。启动时 /dev 自动探测，也可手动覆盖

---

## 两个正交维度

部署场景 = **传输** × **执行隔离**，两两组合：

| | per-Process 执行 | per-Container 执行 |
|---|---|---|
| **HTTP 明文** | 场景 1（本机）/ 场景 2（内部） | 场景 2b（可选的容器隔离） |
| **mTLS 加密** | 场景 3（内部跨机） | 场景 4（CI 对外） |

- **传输**：HTTP 用于同信任域，mTLS 用于跨机/不可信网络
- **执行隔离**：per-Process（fork 子进程）用于可信代码，per-Container（Docker）用于不可信代码

---

## 部署配置

所有配置在一个 YAML 文件里（默认 `ttk/remote/server/xpu_server.yaml`）：

```yaml
server:
  bind: "127.0.0.1"       # 监听地址
  port: 9090
  max_concurrent: 16       # 并发上限
  run_deadline_s: 300      # 单请求超时(秒)

execution:
  sandbox: none            # none（per-Process）| docker（per-Container）

# providers 可选，不配则框架探测到什么就暴露什么
providers:
  - torch
  # - tf                  # 注释掉 = 不暴露（即使 tf 已安装）
  # - flash_attn          # 只暴露 flash_attn，不暴露 raw torch
#
# 注:providers 列表只表达「能力集合」,顺序不代表优先级。客户端优先级由 specs 顺序决定(应用层),
# 不依赖 detect 返回顺序。endpoint providers 默认由 /v1/heartbeat 动态发现,也可在 yaml 显式配置(override)。

# hardware 可选，不配则启动时 /dev 自动探测（实体段配置见 xpu_server.yaml）
```

优先级：**命令行参数 > YAML > 内置默认**

---

## 场景 1：本机测试 — HTTP + per-Process

TTK 和 xpu-server 在同一台机器，零配置，最快。

```bash
python -m ttk.remote.server.xpu_server --port 9090
# TTK worker 端
ttk kernel -i case.csv --config ttk.conf.yaml
```

- 绑定 `127.0.0.1`，无网络暴露
- 每请求一个子进程（`multiprocessing.Process`），跑完即毁
- 无 TLS、无 Docker

---

## 场景 2：内部测试 — HTTP + 灵活部署

同信任域内，代码可信。Server 可按需选择部署位置和执行方式：

### 2a：Server 在 provider 容器中（per-Process）

利用用户已有的官方框架镜像，xpu-server 部署在容器内，端口映射对外：

```bash
# 加速器机器：用户已有官方镜像
# 用 -dit（不是 -d）：厂商镜像默认 CMD=bash，纯 -d 无 tty 会立即 EOF 退出 → 容器 Exited
docker run -dit --device /dev/cambricon0 --name tf-server -p 9090:9090 \
    cambricon/tensorflow:latest

# 拷贝整个 ttk 仓进容器（server 消费 ttk.core_modules.common 同源契约，不能只拷 server 子目录）
scp -r <ttk-repo-root>/ttk user@xpu-box:/tmp/ttk
docker cp /tmp/ttk tf-server:/opt/ttk
# 容器内安装依赖
docker exec tf-server pip install numpy pyyaml

# 写 server 配置（只暴露 tf），拷进容器
# （不能 docker exec ... --config <(echo ...)：进程替换的 /dev/fd/N 在宿主机，容器读不到）
echo 'providers: [tf]' > /tmp/xpu_server.yaml
docker cp /tmp/xpu_server.yaml tf-server:/opt/xpu_server.yaml

# 容器内启动（PYTHONPATH 指向 ttk 包根的父目录）
docker exec -d -e PYTHONPATH=/opt tf-server \
    python -m ttk.remote.server.xpu_server --port 9090 --bind 0.0.0.0 --config /opt/xpu_server.yaml
```

同理，torch 容器用 `cambricon/pytorch:latest` 镜像，配 `providers: [torch]`。镜像天然隔开了框架环境——tf 镜像里没有 torch，反之亦然。

### 2b：Server 在 Host 上（per-Process 或 per-Container）

Server 直接跑在 Host，有完整的 framework 探测能力：

```bash
# per-Process（默认，最快）
python -m ttk.remote.server.xpu_server --port 9090

# 或 per-Container（需要 Docker，~300ms 额外开销）
python -m ttk.remote.server.xpu_server --port 9090 \
    --config <(sed 's/sandbox: none/sandbox: docker/' ttk/remote/server/xpu_server.yaml)
```

**注意**：per-Container 模式下 server 必须在 Host 上运行，不能在容器内再起子容器（Docker-in-Docker 需要 `--privileged`，安全上不推荐）。

---

## 场景 3：内部跨机 — mTLS + per-Process

TTK 和 XPU 在不同机器，团队内部代码可信，传输加密但执行从简。

```bash
# 1. 生成证书（三步：CA 一次；server 证书每个 XPU 一次；client 证书每个租户按需）
./scripts/gen_tls_certs.sh init-ca /opt/ttk-certs
./scripts/gen_tls_certs.sh server /opt/ttk-certs IP:<xpu-box 的 IP>   # 或 DNS:<主机名>
./scripts/gen_tls_certs.sh client /opt/ttk-certs <tenant-name>

# 2. 分发
scp /opt/ttk-certs/ca.crt /opt/ttk-certs/server.* user@xpu-box:/opt/ttk-certs/
# ca.crt + <tenant-name>.client.crt/.key 发给 TTK worker 机器；ca.key 只留在 CA 机器

# 3. XPU 机器配置 + 启动（cwd = ttk 仓根，或 PYTHONPATH 指向仓根）
cat > /opt/xpu_server/xpu_server.yaml << 'EOF'
server:
  bind: "0.0.0.0"
  port: 9090
execution:
  sandbox: none              # per-Process（代码可信）
tls:
  enabled: true
  ca_cert: "/opt/ttk-certs/ca.crt"
  server_cert: "/opt/ttk-certs/server.crt"
  server_key: "/opt/ttk-certs/server.key"
EOF
cd <ttk-repo-root> && python -m ttk.remote.server.xpu_server --port 9090 --config /opt/xpu_server/xpu_server.yaml

# 4. TTK worker 端：default.yaml 配 TLS 证书路径
```

---

## 场景 4：CI 对外 — mTLS + per-Container

不可信的用户代码。Server 部署在 Host 上，每个请求一个隔离容器。

```bash
# 1. 构建 executor 镜像（注意：构建上下文 = 仓库根，Dockerfile 里的 COPY ttk 依赖它；
#    executor 镜像与 server 需同仓同版本——旧镜像与新版 server 的入口/布局不兼容）
docker build -f ttk/remote/server/Dockerfile.torch -t xpu-executor-torch:latest .
docker build -f ttk/remote/server/Dockerfile.tf   -t xpu-executor-tf:latest   .

# 2. 配置
cat > /opt/xpu_server/xpu_server.yaml << 'EOF'
server:
  bind: "0.0.0.0"
  port: 9090
execution:
  sandbox: docker              # per-Container（不可信代码隔离）
tls:
  enabled: true
  ca_cert: "/opt/ttk-certs/ca.crt"
  server_cert: "/opt/ttk-certs/server.crt"
  server_key: "/opt/ttk-certs/server.key"
docker:
  images:
    torch: "xpu-executor-torch:latest"
    tf:    "xpu-executor-tf:latest"
  memory: "8g"
  network: "none"              # 用户代码无网络
EOF

# 3. 启动（必须在 Host 上，不能在容器内）
cd <ttk-repo-root> && python -m ttk.remote.server.xpu_server --port 9090 --config /opt/xpu_server/xpu_server.yaml
```

容器防护矩阵：

| 恶意行为 | 容器阻断方式 |
|----------|-------------|
| 删文件 | `--read-only` 根文件系统 |
| 连外网 | `--network none` |
| 提权 | `--cap-drop ALL` |
| 持久化 | `--rm`（跑完即毁） |
| 资源耗尽 | `--memory` + deadline timeout |

---

## 部署模式对照

| | per-Process | per-Container |
|---|---|---|
| 配置 | `sandbox: none` | `sandbox: docker` |
| Server 部署位置 | Host 或容器内均可 | **必须 Host**（不能 DinD） |
| 隔离级别 | 进程级（`sys.modules` 隔离） | 完整 OS 级 |
| 启动开销 | ~10ms（fork） | ~300-600ms（容器） |
| 需要 Docker | 否 | 是 |
| 适用 | 可信代码、内部团队 | 不可信代码、CI |
| 传输加密 | 可选 mTLS | mTLS 必须 |

---

## 硬件自动探测

启动时不配 `hardware` 且不传 `--devices` 则 /dev 自动探测：

```
按 yaml `hardware:` 各段（shipped：gpu/nvidia、mlu/cambricon）匹配 /dev 设备文件
（毫秒级,不依赖框架；如需 NPU 需自行加段：dev_prefix: davinci + visible_env: ASCEND_RT_VISIBLE_DEVICES）
Fallback: "cpu"
```

`/dev` 检测提取**实际 device id**（regex `^prefix(\d+)$`），不假设从 0 连续。例如 `/dev/cambricon2,cambricon5` → `device_ids=[2,5]`。控制设备（无数字尾）自动排除。

手动指定（覆盖探测，直接给 device id 列表）：

```bash
python -m ttk.remote.server.xpu_server --devices 0,1
```

---

## 命令行参考

```
python -m ttk.remote.server.xpu_server \
    --port 9090          # 监听端口（默认：配置或 9090）
    --bind 127.0.0.1     # 绑定地址（默认：127.0.0.1）
    --config ./xpu_server.yaml  # 配置文件路径
    --devices 0,1        # 设备 ID 列表（不传则 /dev 自动探测）
    --dry-run            # 空跑模式（返回随机数据）
```

---

## 健康检查 / 服务发现

```bash
# /v1/heartbeat 合并了探活 + 服务发现 + tenant 注册（旧 /health、/v1/detect、/heartbeat 已并入此端点）
curl http://127.0.0.1:9090/v1/heartbeat
# → {"status":"ok","hardware":"gpu","device_count":2,"providers":["torch","tf"],"features":["keepalive"]}
# device_count = 实际检测到的 device 数（多卡时 >1）
```

---

## 多卡并发

多卡 server 自动启用 per-device 并发执行：

- 不同 device 上的请求**并发**跑（吞吐 × 卡数）
- 同一 device 上的请求**串行**（保护 PERF 测量精度：`reset_peak_memory_stats` + Event timing 不被其他请求干扰）
- 分配策略：RR 起点 + try-lock 遍历（找空闲 device）+ fallback 阻塞起点（全占时等）
- `max_concurrent`（yaml 配置）建议 ≥ device 数以用满并发；小于则 data_gate 限制总并发，部分 device 空闲（启动 warning 提示，不阻止启动）

**注意**：单卡 server 下，所有请求（含 DATA 模式）都串行——这是为了保证 PERF 测量精度。多卡场景下 DATA 仍跨卡并发。

---

## X-API 响应头

`/v1/run` 成功和错误的响应都带 `X-API` header，回传 server 实际执行的 API 标识：

```
X-API: torch.add          # 推导模式（client 没配 api，server 从 op_name 推导）
X-API: torch.add          # ACLNN 算子（aclnnAdd → strip → add → torch.add，回传实际执行的 API）
X-API: AddSpec            # spec 模式（回传 spec_class 名）
```

客户端读此 header 显示真实 API（不再显示 "custom"）。旧 server 不发此 header，客户端 fallback 到 `spec.api` 或 "custom"。

---

## 文件结构

```
ttk/remote/server/          ← server 本体（整仓部署：依赖 ttk.core_modules.common
│                              的 bind_params 契约与 profiler 共享引擎，单份同源）
├── README.md               本文件
├── xpu_server.py           主进程（HTTP + 子进程/容器派发）——对外唯一入口
├── xpu_server.yaml         服务端配置
├── executor.py             子进程入口（算子执行 + perf 双 pass）
├── _framework_preload.py   forkserver 预载 shim（RTLD_DEEPBIND，torch/tf 共存）
├── xpu_profiler.py         XPU 采集层（torch/tf pass，引擎与 common/profiler 同源）
├── executor_main.py        Docker 沙箱入口（python -m ttk.remote.server.executor_main）
├── container.py            Docker 后端（per-Container 派发）
├── config.py               配置加载器
├── Dockerfile.torch        Torch executor 镜像（示例，构建上下文 = 仓库根）
└── Dockerfile.tf           TF executor 镜像（示例，构建上下文 = 仓库根）
```
