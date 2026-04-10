[← back to README](../README.md)

# MoE Support

Mixture-of-Experts steering in Abliterix uses four independent mechanisms — most other tools only implement one.

Four independent steering mechanisms for Mixture-of-Experts models:

1. **Expert-Granular Abliteration (EGA)** *(new)* — norm-preserving orthogonal projection applied to **all** expert `down_proj` slices in every MoE layer. Unlike top-N approaches that only modify a few "safety experts", EGA recognizes that refusal signal is distributed across all experts. Critical for models like Gemma 4 26B-A4B where dense-only abliteration leaves ~30% of refusals routed through untouched experts.
2. **Expert Profiling** — hooks router modules to compute per-expert "risk scores" from activation patterns on harmful vs. harmless prompts
3. **Router Weight Suppression** — applies learned negative bias to routing weights of safety-critical experts
4. **Fused Expert Abliteration** — direct rank-1 modification of top-N expert `down_proj` matrices (complementary to EGA)

Supported MoE architectures: Gemma 4 26B-A4B, Qwen3/3.5 MoE, Mixtral, DeepSeek MoE, Granite MoE Hybrid, MiniMax-M2.5, LiquidAI LFM2, GLM-4 MoE, Phi-3.5-MoE, DBRX, Llama-4 Scout/Maverick. See [`configs/`](../configs/) for model-specific examples.
