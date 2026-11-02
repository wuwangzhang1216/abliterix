# Model compatibility boundaries

PR #114 addresses generation-config export and custom Python encoder compatibility.
It does **not** establish full DeepSeek-V4 or GLM-5.3-Flash support, and does not
resolve [issue #19](https://github.com/wuwangzhang1216/abliterix/issues/19).

## Generation config export

Export preserves the selected decoding mode, including `do_sample=False` and
unset `do_sample`. When strict validation rejects inactive sampling parameters,
export resets those fields to the installed Transformers defaults on a copy.
Active sampling settings remain unchanged. The repair is applied only if the
result validates and retains the original generation mode; unrelated errors
remain visible. In particular, it must not turn contrastive search into greedy
decoding. See the official
[Transformers 5.7.0 generation-config implementation](https://github.com/huggingface/transformers/blob/v5.7.0/src/transformers/generation/configuration_utils.py).

## Custom Python encoders

An encoder exposes `encode_messages(messages, ...) -> str`. Required options
belong in `model.custom_encoder_kwargs`. The adapter validates that signature
before use and passes `add_generation_prompt` only if the function accepts it,
including positional-only parameters and `**kwargs`. A runtime flag overrides
a same-named configured default. Encoders without that parameter retain their
own behavior for assistant prefixes; the adapter cannot synthesize a different
prefix policy for them.

The official
[DeepSeek-V4 encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/9e165c30e2704aec5d9d593cce3eebd58bbef1cb/encoding/encoding_dsv4.py)
accepts `thinking_mode` and has no `add_generation_prompt` argument. Encoders own
their special tokens, so tokenization uses `add_special_tokens=False` to avoid
duplicating a BOS token. This also applies to the engine's render-then-tokenize
paths for prompt batches, length sorting, and interactive chat; ordinary
tokenizers keep their existing defaults.

## Dependencies and multi-stream states

The committed dependency lock remains unchanged. Transformers 5.7.0 lacks the
native `deepseek_v4`, `glm5_next`, and `glm5_next_text` registrations. The loader
checks actual registration availability using the model metadata it already
reads and reports an actionable error before tokenizer or weight loading. A
checkpoint-provided `AutoConfig` retains the existing remote-code trust flow.
Known native architectures explicitly declaring multiple residual streams also
stop during this preflight, even when Transformers recognizes them. This avoids
loading a large checkpoint only to reject its unsupported representation later.
Transformers 5.20.2 includes official model code; testing that release in an
isolated environment is separate from validating Abliterix architecture support.

Hidden-state extraction requires `(batch, sequence, hidden)` tensors. It rejects
additional stream dimensions explicitly. In the official
[DeepSeek-V4 implementation at 5.20.2](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py),
decoder states have shape `(batch, sequence, streams, hidden)`. The residual
mixing matrix is approximately doubly stochastic, but the block input and
output routing use separate learned weights. Preserving a stream sum through
the residual mixer does not make a stream mean equivalent to the block input.
DeepSeek's final `HyperHead` also uses learned, input-dependent weights.

The official
[GLM-5.3-Flash implementation at 5.20.2](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/glm5_next/modeling_glm5_next.py)
**does** use an unweighted mean at its final hyper-head. Its intermediate layers
still use learned pre/post routing. That final reduction does not validate
averaging every intermediate layer as a shared representation.

Independent evidence is consistent with keeping this distinction explicit:
[Zhao et al., September 4, 2026](https://arxiv.org/abs/2609.05309v1) report distinct
streams and depth-dependent routing in DeepSeek-V4-Flash.
[HyperDFlash, June 29, 2026 revision](https://arxiv.org/abs/2606.26744v2) studies
architecture-aligned reduction for speculative decoding. Neither study
validates this project's proposed intermediate-layer averaging or establishes
quality preservation after model changes.

## Validation scope

Regression tests cover real `GenerationConfig` save/reload, a tiny random Llama
checkpoint with identical greedy tokens after reload, both PEFT export paths
(the quantized path uses a stand-in for full-precision base loading), encoder
signatures, and early dependency diagnostics. Optional tests on Transformers
5.20.2 run a tiny random DeepSeek-V4 causal LM and a GLM-5.3-Flash text backbone,
confirm finite forward outputs and the multi-stream shapes, and verify the
explicit extraction error. They download no checkpoints and skip when the
installed Transformers does not provide the architectures.

The isolated upstream API checks used Transformers 5.20.2, huggingface-hub 2.4.0,
tokenizers 0.23.3, PEFT 0.18.0, and PyTorch 2.11.0 on CPU. This overlay is **not** a
validated project dependency resolution: hub 2.4.0 is outside the project's
declared `huggingface-hub~=1.6` range. The normal full-suite check used the existing
Transformers 5.7.0 environment. Neither the project lock nor its dependency
constraints were upgraded for these upstream contract tests.

These CPU checks do not cover full checkpoints, GPU kernels, quantized weights,
GLM vision inputs, all attention variants, or model quality. Issue #19 remains
open until its architecture and quality requirements have direct evidence.
