# P1.1 K8s 上手 — 学习笔记

**完成日期**：2026-05-03
**目标**：搭建生产级 K8s 集群 + GPU 接入 + vLLM Pod 部署，对应路线图 P1.1

## 集群信息

- 节点：`ai-infra-lab` (192.168.1.101)，单节点 8 卡 A30
- 发行版：kubeadm 上游 K8s `v1.31.14`（不用 k3s，对齐大厂技术栈）
- CNI：Calico v3.x（VXLANCrossSubnet，Pod CIDR `10.244.0.0/16` 避开 LAN 段）
- 容器运行时：containerd v2.2.1（K8s 1.24+ 不再用 dockershim）

## 装机流程关键步骤

1. **系统准备**：关 swap、加载 `br_netfilter` + `overlay` 内核模块、改 sysctl `net.bridge.bridge-nf-call-iptables=1` / `net.ipv4.ip_forward=1`
2. **containerd**：已随 docker.io 装，需通过 `nvidia-ctk runtime configure --runtime=containerd --set-as-default` 配置 nvidia runtime
3. **kube 三件套**：kubeadm / kubelet / kubectl 装好
4. **kubeadm init**：拉控制面镜像 + 生成证书 + 起 static Pod 形式的控制面组件 + 输出 join 命令
5. **CNI**：装 Calico operator（默认 Pod CIDR 192.168.0.0/16 撞 LAN，已改 10.244.0.0/16）
6. **解禁单节点调度**：移除 control-plane taint（`kubectl taint nodes ai-infra-lab node-role.kubernetes.io/control-plane:NoSchedule-`）

## GPU 接入

### NVIDIA Device Plugin 部署

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
```

### 关键踩坑：限制 device plugin 视野

**问题**：device plugin 默认报告所有 8 张 GPU 给 K8s，但 GPU 0/1/3 已被现有 Docker 容器（vllm-qwen-awq / vllm-qwen-bf16-tmp / sglang-qwen-no-radix）占用。如果 K8s Pod 调度时分到这些卡会 OOM。

**解决**：通过 DaemonSet 环境变量 `NVIDIA_VISIBLE_DEVICES=4,5,7` 限制 device plugin 只暴露空闲 GPU：

```bash
kubectl -n kube-system set env daemonset/nvidia-device-plugin-daemonset \
    NVIDIA_VISIBLE_DEVICES=4,5,7
```

效果：节点 `nvidia.com/gpu` 容量从 8 → 3，Pod 调度只会从 GPU 4/5/7 中选。

### 容器内 GPU 索引重映射

NVIDIA Container Runtime 给容器挂载 GPU 时会重映射索引：
- 主机 GPU 5 → 容器内显示为 GPU 0
- 容器只看见自己分配的卡，索引从 0 开始

## 学到的 K8s 概念（按学习顺序）

### 1. YAML / declarative 范式
- 命令式（docker run）vs 声明式（YAML + 控制器持续收敛）
- 跟 Terraform 是同一个范式（IaC），但 K8s 是**持续收敛**，TF 是 **apply 时收敛**
- YAML 四段固定结构：apiVersion / kind / metadata / spec

### 2. K8s 控制面 + 数据面
- 控制面（脑子）：apiserver / etcd / scheduler / controller-manager
- 数据面（手脚）：每个节点的 kubelet / kube-proxy / containerd
- 所有跨组件通信走 apiserver；etcd 只允许 apiserver 访问

### 3. Pod
- K8s 最小调度单位，含 1+ 个紧密耦合容器，共享网络和存储
- 每个 Pod 一个 IP，Pod 内多容器共享这个 IP
- sidecar 模式（日志收集 / 监控 / service mesh）才需要多容器
- Pod IP 不稳定（重启换 IP）→ 引出 Service

### 4. Service
- 给一组 Pod 提供稳定 ClusterIP + DNS 名（`<svc>.<ns>.svc.cluster.local`）
- 通过 label selector 关联 Pod，Endpoints 控制器自动跟踪
- 三种类型：ClusterIP（内部）/ NodePort（节点端口）/ LoadBalancer（云）
- 背后是 kube-proxy 改 iptables 实现转发（br_netfilter 模块的意义所在）

### 5. Deployment
- Deployment → ReplicaSet → Pod 三层抽象
- 解决：副本维持、自愈（Pod 挂了自动起）、滚动升级、一键回滚
- 滚动升级本质是"两个 ReplicaSet 此消彼长"，回滚就是反向操作
- 生产里几乎不直接写 Pod，都是 Deployment 包一层

### 6. GPU 资源调度
- `resources.limits.nvidia.com/gpu: 1` 向 device plugin 申请 GPU
- K8s 不能指定具体哪张卡（哲学：用户声明数量，调度器选）
- 要限制具体 GPU 集合，通过 device plugin 的 `NVIDIA_VISIBLE_DEVICES` 配置

## vLLM Pod YAML 关键字段

```yaml
spec:
  containers:
    - resources:
        limits:
          nvidia.com/gpu: 1                    # 申请 1 张 GPU
      volumeMounts:
        - name: models                         # 模型挂载
          mountPath: /models
        - name: dshm                           # 共享内存
          mountPath: /dev/shm
  volumes:
    - name: models
      hostPath:                                # 宿主机目录直挂
        path: /home/liuguangli/models
    - name: dshm
      emptyDir:                                # 共享内存替代 docker 的 --shm-size
        medium: Memory
        sizeLimit: 8Gi
```

## 端到端验证链路

```
外部 curl 192.168.1.101:30800
    ↓ (kube-proxy iptables 转发)
Service vllm-svc:8000 (ClusterIP)
    ↓
Pod vllm-qwen-k8s:8000 (Pod IP 10.244.89.22)
    ↓
vLLM 进程 (容器内 GPU 0 = 主机 GPU 5)
    ↓
Qwen2.5-7B-AWQ 推理
    ↓
返回中文回答 ✅
```

## 简历可写表达

> 基于 kubeadm 在 8 卡 A30 服务器搭建生产级单节点 K8s 集群（v1.31.14 + Calico CNI），完成 NVIDIA Device Plugin 部署与 GPU 调度配置。通过 `NVIDIA_VISIBLE_DEVICES` 限制 device plugin 视野隔离 K8s 与 Docker 的 GPU 资源占用（避免分配到已占用 GPU 导致 OOM）。使用 Pod / Service / Deployment 完成 vLLM 推理服务部署，NodePort Service 对外暴露，端到端验证 HTTP API 推理链路。

## 后续可扩展（路线图待办）

- [ ] vLLM Pod 改成 Deployment（自愈 + 扩缩容）
- [ ] HPA 按 token 速率自动扩缩容（对应 P3.2）
- [ ] 多卡 TP=2/4 实验（Pod 申请多张 GPU + vLLM `--tensor-parallel-size`，对应 P1.2）
- [ ] 接入 Prometheus + Grafana 监控（对应 P1.3）
