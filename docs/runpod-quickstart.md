# RunPod 快速部署指南 — MiniMax-M2.5 (4×H100)

## 前置条件

- RunPod 账号 + 足够 credit
- 本地有 `abliterix` 仓库
- SSH key: `~/.ssh/id_ed25519`

## 一、选 Pod

### 选 Pod 要求
- **GPU**: 4× H100 80GB（或 H200）
- **Region**: 优先 US-TX-3、US-KS-2（离 HF CDN 近，网速快）
- **Disk**: 500GB+（模型 230GB + cache）

### 第一步：测网速（5 秒内决定去留）

```bash
ssh -o StrictHostKeyChecking=no root@HOST -p PORT -i ~/.ssh/id_ed25519 \
  "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && \
   curl -sL -H 'Authorization: Bearer hf_YOUR_TOKEN' -o /dev/null -w '%{speed_download}' \
   --max-time 15 'https://huggingface.co/MiniMaxAI/MiniMax-M2.5/resolve/main/model-00000-of-00126.safetensors'" \
  | awk 'END{printf "HF Speed: %.0f MB/s\n", $1/1048576}'
```

| 速度 | 230GB 下载时间 | 建议 |
|------|---------------|------|
| > 200 MB/s | ~19 min | 直接用 |
| 100-200 MB/s | 19-38 min | 可以接受 |
| 50-100 MB/s | 38-77 min | 勉强 |
| < 50 MB/s | > 77 min | **换 pod** |

## 二、一键部署（测速 OK 后执行）

### 第二步：上传代码

```bash
scp -r -P PORT -i ~/.ssh/id_ed25519 ~/abliterix root@HOST:/workspace/
```

### 第三步：一键安装 + 启动

SSH 进入后粘贴以下全部内容：

```bash
cd /workspace/abliterix

# === .env ===
cat > .env << 'EOF'
HF_TOKEN=hf_YOUR_TOKEN
OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY
EOF

# === 安装依赖（注意：不装 flash-attn，vLLM 自带 FA3）===
pip install -U "transformers>=4.57.1,<5.0" accelerate safetensors sentencepiece -q
pip install "vllm>=0.8" "speculators>=0.1.9" -q
pip install -e . --no-deps -q
pip install optuna peft datasets bitsandbytes pydantic-settings questionary hf-transfer psutil kernels rich -q
pip uninstall flash-attn -y 2>/dev/null; true

# === Patch config 用公开数据集 ===
sed -i 's|dataset = "datasets/good_500"|dataset = "mlabonne/harmless_alpaca"|g' configs/minimax_m2.5_vllm.toml
sed -i 's|dataset = "datasets/harmful_500"|dataset = "mlabonne/harmful_behaviors"|g' configs/minimax_m2.5_vllm.toml
sed -i 's|column = "prompt"|column = "text"|g' configs/minimax_m2.5_vllm.toml
sed -i 's|split = "train\[400:\]"|split = "test[:100]"|g' configs/minimax_m2.5_vllm.toml

# === 验证 ===
python3 -c "import vllm, speculators, peft, optuna, transformers; print(f'OK: transformers={transformers.__version__} vllm={vllm.__version__}')"

# === 启动 ===
set -a && . .env && set +a
export HF_HOME=/workspace/hf_cache HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_MOE_USE_DEEP_GEMM=0
mkdir -p $HF_HOME

nohup bash -c 'AX_CONFIG=configs/minimax_m2.5_vllm.toml abliterix 2>&1 | tee /workspace/run.log' &>/dev/null &
echo "Started PID: $!"
```

## 三、监控进度

```bash
# 模型下载进度
du -sh /workspace/hf_cache/hub/models--MiniMaxAI--MiniMax-M2.5/

# 日志最新输出
tail -20 /workspace/run.log

# GPU 使用率（Phase 2 运行时应 >50%）
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

## 四、预期时间线

| 阶段 | 时间 (134 MB/s 网速) |
|------|---------------------|
| 依赖安装 | ~5 min |
| 模型下载 (230GB) | ~29 min |
| Phase 1: speculators hidden states | ~2 min |
| Phase 1: projection cache (safetensors) | ~1 min |
| Phase 2: vLLM 加载 | ~3 min |
| Phase 2: baseline capture | ~1 min |
| Phase 2: 50 trials | ~25 min |
| **总计** | **~66 min**（首次含下载）|
| **再次运行（模型已缓存）** | **~32 min** |

## 五、踩坑记录

| 坑 | 原因 | 解决 |
|----|------|------|
| `undefined symbol` flash-attn 崩溃 | flash-attn pip 包和预装 PyTorch 2.4 ABI 不兼容 | **不要装 flash-attn**，vLLM 自带 FA3 |
| `DatasetNotFoundError datasets/good_500` | 私有数据集不在 pod 上 | 用 sed patch 成公开数据集 |
| FP8 dequant 极慢 (~4 tok/s) | transformers 4.57 < 5.2，自动检测走 bf16 dequant | config 里显式 `skip_fp8_dequant = true` |
| deploy 脚本卡在 flash-attn wheel 搜索 | curl GitHub API 限速/超时 | 跳过，不需要 flash-attn |
| Pod 网速 < 50 MB/s | RunPod 不同 region 网速差异巨大 | 先测速再决定用不用 |
| rsync 不可用 | RunPod 镜像没装 rsync | 用 scp -r |

## 六、进阶：Network Volume 预下载（省 credit）

如果网速慢或需要频繁开关 pod：

1. **创建 Network Volume**: RunPod Dashboard → Storage → 500GB，选和 GPU 同 region
2. **开便宜 pod**: CPU Only / 1×4090，挂载 volume
3. **下载模型**:
   ```bash
   pip install huggingface-hub hf-transfer
   HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download MiniMaxAI/MiniMax-M2.5 \
     --local-dir /workspace/models/MiniMax-M2.5 --token hf_YOUR_TOKEN
   ```
4. **关掉便宜 pod，开 4×H100 pod 挂载同一 volume**
5. 运行时指向本地路径: `model_id = "/workspace/models/MiniMax-M2.5"`
