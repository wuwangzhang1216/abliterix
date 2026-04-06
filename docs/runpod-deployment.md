# RunPod Deployment Guide

## Pod selection

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | 4× H100 80GB | 4× H100 SXM 80GB (NVLink) |
| Disk | 512 GB | 1 TB local NVMe |
| RAM | 256 GB | 512 GB+ |

**Critical: prefer pods with local NVMe storage** (`/dev/md0`, `/dev/nvme*`).
Network-mounted storage (`mfs#...runpod.net`) makes model loading 10× slower
(~15s/shard vs ~1s/shard for a 125-shard model).

Check storage type after connecting:
```bash
df -h /workspace
# Local NVMe:    /dev/md0  1.0T  ...
# Network (BAD): mfs#us-mo-1.runpod.net:9421  378T  ...
```

## Setup steps

```bash
# 1. Upload project (use tar, not raw scp — much faster for many files)
# On local machine:
tar czf /tmp/abliterix.tar.gz --exclude=.git --exclude=.venv --exclude=__pycache__ abliterix/
scp -P <PORT> /tmp/abliterix.tar.gz root@<HOST>:/workspace/

# On pod:
cd /workspace && tar --no-same-owner -xzf abliterix.tar.gz && rm abliterix.tar.gz

# 2. Run the deploy script
bash deploy_minimax_m25_vllm.sh
```

## Common pitfalls

### 1. NEVER upgrade PyTorch

RunPod pods come with a specific PyTorch + CUDA + torchvision combination that
is tested together. Upgrading PyTorch (e.g., `pip install torch --upgrade`)
**will break**:

- `torchvision` (ABI mismatch → `operator torchvision::nms does not exist`)
- `flash-attn` (undefined symbol errors)
- `transformers` + `peft` (if torch upgrade pulls in transformers 5.x)

**Instead**: find a flash-attn wheel that matches the existing PyTorch version.

### 2. flash-attn installation

flash-attn must be compiled against the exact PyTorch version. Pre-built wheels:

| Source | URL | Coverage |
|--------|-----|----------|
| mjun0812 | github.com/mjun0812/flash-attention-prebuild-wheels | torch 2.4–2.11, cu124/126/128/130, py3.10–3.14 |
| lesj0610 | github.com/lesj0610/flash-attention | torch 2.10–2.11, cu128, py3.10–3.13 |
| Official | pypi.org/project/flash-attn | torch 2.4–2.9 |

Find the right wheel:
```bash
TORCH_VER=2.10  # from: python -c "import torch; print(torch.__version__)"
PY_VER=cp311    # from: python --version
curl -s "https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases" \
  | grep -oP "https://[^\"]+" \
  | grep "torch${TORCH_VER}.*${PY_VER}.*linux_x86_64.whl"
```

Install with `--no-deps --force-reinstall` to avoid pulling incompatible deps.

### 3. MiniMax-M2.5 requires transformers 4.x

MiniMax-M2.5's remote modeling code uses `OutputRecorder` from
`transformers.utils.generic`, which was removed in transformers 5.0.
Pin: `pip install 'transformers>=4.48,<5'`

### 4. GPU memory cleanup after killing processes

After `kill <PID>`, GPU memory may not be freed immediately. Verify:
```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
# Should show empty if process was killed
```
If memory is still held, use `kill -9 <PID>` and wait a few seconds.

### 5. eval prompt count affects Phase 1 duration

Phase 1 runs HF `generate()` on eval prompts at ~4 tok/s (pipeline parallelism).
100 eval prompts × 50 tokens ≈ 20+ minutes. Reduce to 20 prompts for faster
iteration:
```toml
# In config TOML:
[benign_eval_prompts]
split = "train[400:420]"
[target_eval_prompts]
split = "train[400:420]"
```

## Performance optimizations (applied)

| Optimization | Effect | Config |
|---|---|---|
| Expert Parallelism (EP) | Better MoE throughput vs TP-only | `enable_expert_parallel=True` in vllm_backend.py |
| FP8 KV cache | 2x KV cache capacity | `kv_cache_dtype="fp8_e4m3"` |
| CUTLASS grouped GEMM for MoE | +57% expert throughput on H100 | `VLLM_MOE_USE_DEEP_GEMM=0` |
| Projection cache OOM fix | No more dequant cache accumulation | `del W` after each projection |
| speculators fast path | 10x faster hidden state extraction | Auto-enabled when `speculators` installed |
| Skip MoE profiling (vLLM) | Save ~12 min Phase 1 time | Auto for `backend="vllm"` |

## Architecture: Phase 1 (HF) → Phase 2 (vLLM)

- **Phase 1** uses HuggingFace with `device_map="auto"` (pipeline parallelism).
  Only one GPU is active at a time → slow (~4 tok/s). This phase extracts
  hidden states and computes steering vectors. Cannot use vLLM because vLLM
  doesn't expose intermediate hidden states.

- **Phase 2** uses vLLM with tensor parallelism (TP=4). All 4 GPUs compute
  simultaneously → fast (70+ tok/s). Runs the 50-trial optimization loop.
