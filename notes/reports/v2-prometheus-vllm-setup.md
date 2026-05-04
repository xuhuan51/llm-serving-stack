# V2: Prometheus + vLLM Observability Setup on Kubernetes

End-to-end metrics pipeline for a vLLM inference Pod running on a single-node Kubernetes cluster, scraped by an in-cluster Prometheus deployment.

## Architecture

```
vLLM Pod  (annotation: prometheus.io/scrape=true)
   │ exposes /metrics (vllm:* time series)
   ▼
Prometheus Pod (kubernetes_sd_configs role: pod)
   │ ① ServiceAccount + ClusterRole → list pods via K8s API
   │ ② relabel_configs filter on annotation, rewrite address, inject labels
   │ ③ scrape every 15s, store in local TSDB
   ▼ Service NodePort 30900
External access (browser / port-forward)
```

## Manifests

- [`deploy/vllm/vllm-pod.yaml`](../../deploy/vllm/vllm-pod.yaml) — vLLM Pod with Prometheus scrape annotations.
- [`deploy/monitoring/prometheus.yaml`](../../deploy/monitoring/prometheus.yaml) — full Prometheus stack (Namespace, RBAC, ConfigMap, Deployment, Service).

## Key Configuration: `relabel_configs`

Why these 4 rules matter — `kubernetes_sd_configs` returns *every* Pod in the cluster as a candidate target. The relabel pipeline filters and reshapes that candidate list into actual scrape targets:

```yaml
relabel_configs:
  # 1) Keep only Pods with annotation prometheus.io/scrape=true
  - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
    action: keep
    regex: true

  # 2) Override metrics path from annotation
  - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
    action: replace
    target_label: __metrics_path__
    regex: (.+)

  # 3) Rewrite scrape port using annotation prometheus.io/port
  - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
    action: replace
    regex: ([^:]+)(?::\d+)?;(\d+)
    replacement: $1:$2
    target_label: __address__

  # 4) Promote K8s metadata into permanent labels for queries
  - source_labels: [__meta_kubernetes_namespace]
    target_label: namespace
  - source_labels: [__meta_kubernetes_pod_name]
    target_label: pod
  - source_labels: [__meta_kubernetes_pod_node_name]
    target_label: node
```

After relabeling, the `__meta_*` labels are stripped (any label starting with `__` is internal-only). The remaining labels — `instance`, `job`, `namespace`, `pod`, `node` — appear on every time series and enable per-Pod aggregation in PromQL.

## Validation

After applying both manifests:

```bash
# Confirm Prometheus discovered the vLLM Pod
curl -s http://<node-ip>:30900/api/v1/targets \
  | python3 -c "import json,sys; d=json.load(sys.stdin); \
    print([t['labels'] for t in d['data']['activeTargets'] if t['labels']['job']=='kubernetes-pods'])"

# Generate load
for i in $(seq 1 50); do
  kubectl exec vllm-qwen-k8s -- curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"/models/Qwen2.5-7B-Instruct-AWQ","prompt":"Tell a story","max_tokens":200}' > /dev/null &
done

# Observe in Prometheus UI
# Query: vllm:num_requests_running   → spikes to 50, decays as requests complete
```

## PromQL Examples

```promql
# Concurrency snapshot (per Pod)
vllm:num_requests_running

# Cumulative tokens served (per Pod)
vllm:prompt_tokens_total

# TTFT P95 over the last 5 minutes (across all replicas)
histogram_quantile(0.95,
  sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le)
)

# KV cache utilization
vllm:kv_cache_usage_perc
```

## Known Limitations (Tracked for V2.x)

- **Probes**: vLLM Pod has no `readinessProbe` / `startupProbe`; Service routes traffic during the ~60s torch.compile warmup, causing transient `connection refused` for early callers. Should be addressed before any HPA experiment.
- **Storage**: Prometheus uses `emptyDir` for TSDB — Pod restart loses history. A PVC-backed volume is the production path; not in scope for the lab.
- **Self-monitoring**: Prometheus does not yet scrape itself via the SD path (no `prometheus.io/scrape` annotation on its own Pod). Trivial fix, deferred.

## Next Steps (V2.3 – V2.5)

- V2.3 — Grafana deployment with Prometheus datasource.
- V2.4 — PromQL queries for an SLO-style view: TTFT/TPOT P95, KV cache utilization, queue length, request throughput.
- V2.5 — SLO Dashboard with 6 panels covering the latency lifecycle (queue → prefill → decode → e2e), backend resource (KV cache, GPU memory), and traffic (QPS, prompt/completion tokens).
