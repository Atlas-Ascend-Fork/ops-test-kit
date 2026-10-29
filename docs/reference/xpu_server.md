# xpu-server 部署指南

## 使用须知

xpu-server 是 TTK 的远端 XPU 执行服务：接收 TTK worker 的算子执行请求，在 GPU/MLU/CPU 等硬件上运行算子并回传结果。server 与 TTK 同仓同源，**必须整仓部署**，不支持只拷贝 `ttk/remote/server/` 子目录。

部署场景由两个维度组合：**传输**（HTTP 明文 / mTLS 加密）× **执行隔离**（per-Process / per-Container）。

> 本文档出现的硬件厂商名、设备路径、镜像名仅作示例。

## 文档导览

1. **[快速开始](#一快速开始)**
2. **[配置](#二配置)**
3. **[部署场景](#三部署场景)**
4. **[设备探测](#四设备探测)**
5. **[命令行参考](#五命令行参考)**
6. **[健康检查](#六健康检查)**
7. **[部署模式对照](#七部署模式对照)**

## 一、快速开始

```bash
git clone <ttk-repo> ~/ttk && cd ~/ttk
python -m ttk.remote.server.xpu_server --port 9090
```

TTK worker 端在 `ttk.conf.yaml`（或 `--config`）里配置 `remote.endpoints` 连过来即可。

## 二、配置

所有配置在 `ttk/remote/server/xpu_server.yaml`（可用 `--config` 覆盖）：

```yaml
server:
  bind: "127.0.0.1"       # 监听地址
  port: 9090
  max_concurrent: 16      # 并发上限
  run_deadline_s: 300     # 单请求超时（秒）

execution:
  sandbox: none           # none（per-Process）| docker（per-Container）

providers:                # 可选：暴露哪些框架
  - torch
  # - tf

tls:                      # 跨机加密时启用
  enabled: true
  ca_cert: "/opt/ttk-certs/ca.crt"
  server_cert: "/opt/ttk-certs/server.crt"
  server_key: "/opt/ttk-certs/server.key"
```

优先级：命令行参数 > YAML > 内置默认。

## 三、部署场景

### 1. 本机测试（HTTP + per-Process）

```bash
python -m ttk.remote.server.xpu_server --port 9090
```

绑定 `127.0.0.1`，无网络暴露、无 TLS、无 Docker。

### 2. 内部测试（HTTP + 灵活部署）

Server 可放在 provider 容器内（利用已有官方框架镜像，如 `cambricon/tensorflow:latest`）：

```bash
docker run -dit --device /dev/cambricon0 --name tf-server -p 9090:9090 cambricon/tensorflow:latest
docker cp <ttk-repo> tf-server:/opt/ttk
docker exec tf-server pip install numpy pyyaml
docker exec -d -e PYTHONPATH=/opt tf-server \
    python -m ttk.remote.server.xpu_server --port 9090 --bind 0.0.0.0
```

也可直接跑在 Host（per-Process 默认；切换 per-Container 时把 `sandbox` 改为 `docker`，且 server 必须在 Host 上）。

### 3. 内部跨机（mTLS + per-Process）

```bash
# 1. 生成证书
./scripts/gen_tls_certs.sh init-ca /opt/ttk-certs
./scripts/gen_tls_certs.sh server /opt/ttk-certs IP:<xpu-box 的 IP>
./scripts/gen_tls_certs.sh client /opt/ttk-certs <tenant-name>

# 2. 分发：ca.crt + server.* 给 XPU 机器；ca.crt + client 证书给 worker；ca.key 只留 CA 机器
# 3. XPU 机器配置 tls 段（见上）并启动
cd <ttk-repo> && python -m ttk.remote.server.xpu_server --port 9090 --config /opt/xpu_server.yaml
```

### 4. CI 对外（mTLS + per-Container）

不可信用户代码：server 在 Host，每个请求一个隔离容器。

```bash
# 1. 构建 executor 镜像（构建上下文 = 仓库根）
docker build -f ttk/remote/server/Dockerfile.torch -t xpu-executor-torch:latest .
docker build -f ttk/remote/server/Dockerfile.tf   -t xpu-executor-tf:latest   .

# 2. 配置 sandbox: docker + tls + docker.images（torch/tf 镜像名、memory、network: none）
# 3. 在 Host 启动（不能在容器内再起容器）
cd <ttk-repo> && python -m ttk.remote.server.xpu_server --port 9090 --config /opt/xpu_server.yaml
```

容器以 `--read-only` 根文件系统、`--network none`、`--cap-drop ALL`、`--rm`、内存与超时上限运行，防止删文件、连外网、提权、持久化和资源耗尽。

## 四、设备探测

不配 `hardware`、不传 `--devices` 时按 `/dev` 下的设备节点自动探测（如 gpu/nvidia、mlu/cambricon），提取实际设备 ID（不一定从 0 连续），控制设备自动排除。手动指定：

```bash
python -m ttk.remote.server.xpu_server --devices 0,1
```

多卡时不同设备上的请求并发执行，同一设备上串行（保证性能测量精度）。

## 五、命令行参考

```
python -m ttk.remote.server.xpu_server \
    --port 9090              # 监听端口（默认 9090）
    --bind 127.0.0.1         # 绑定地址
    --config <yaml>          # 配置文件路径
    --devices 0,1            # 设备 ID（不传则自动探测）
    --dry-run                # 空跑模式（返回随机数据）
```

## 六、健康检查

```bash
curl http://127.0.0.1:9090/v1/heartbeat
# → {"status":"ok","hardware":"gpu","device_count":2,"providers":["torch","tf"]}
```

worker 按此端点发现服务与可用 provider；多卡时 `device_count > 1`。

## 七、部署模式对照

| | per-Process | per-Container |
|---|---|---|
| 配置 | `sandbox: none` | `sandbox: docker` |
| Server 位置 | Host 或容器内 | 必须 Host |
| 隔离 | 进程级 | 系统级 |
| 需要 Docker | 否 | 是 |
| 适用 | 可信代码、内部团队 | 不可信代码、CI |
| 加密 | 可选 mTLS | 必须 mTLS |
