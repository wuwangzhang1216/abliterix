# Model compatibility boundaries

Native DeepSeek-V4 and GLM-5.3-Flash now have an architecture-specific readout
path for the HF engine. The implementation and its local validation are described
below. Full released-checkpoint and model-quality acceptance remain tracked in
[issue #19](https://github.com/wuwangzhang1216/abliterix/issues/19).

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

## Native multi-stream readouts

Use the separate [Transformers 5.20.2 runtime profile](model-runtime-mhc.md).
Its dependencies satisfy the project's declared ranges. The default lock remains
on Transformers 5.7.0, which lacks these native architectures. The loader reports
missing registrations before tokenizer or weight loading. Checkpoint-provided
configuration classes retain the existing remote-code trust flow.

An mHC decoder carries `(batch, sequence, streams, hidden)` states. Each attention
and FFN sublayer learns its own input read, output write weights, and residual
mixer. A doubly stochastic residual mixer does not establish that equal stream
averaging reproduces the input read. DeepSeek also uses a learned final
`HyperHead`. [Native DeepSeek implementation](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py)

GLM's final hyper-head uses an unweighted mean, while its intermediate attention
and FFN reads use learned routing. The final head's reduction therefore applies
at that specific site. [Native GLM implementation](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/glm5_next/modeling_glm5_next.py)

The implementation observes the existing forward pass. It returns FP32 tensors
of shape `(batch, layers+1, hidden)` with this explicit indexing convention.

| Slot | Observable |
| --- | --- |
| `0` through `L-1` | Block `i`'s attention read, after native mHC routing and before `input_layernorm` |
| `L` | Final native hyper-head reduction followed by the model norm, as consumed by the LM head |

Each temporary observer copies only the requested token to the input device.
Extraction disables the KV cache and the full hidden-state tuple, leaving stream
routing to the installed model implementation. Observers are removed after
success or failure, and missing, reordered, or incorrectly shaped reads fail
explicitly. PEFT and GLM's conditional-generation wrapper are unwrapped only to
locate these observation sites. Forward execution still uses the loaded wrapper.
Unknown model classes with four-dimensional states remain unsupported.

This is a deliberately defined sequence of observables. It does not retain all
streams, substitute the FFN's separate read, or prove an optimal direction for
model editing. The existing per-layer editing convention uses slot `i+1` for
block `i`, which is the next block's attention read or the final LM-facing state.
The configured global-vector path continues to interpolate those slots. A
full-model quality study is needed to compare this choice with FFN-specific
readouts or representations retaining every stream.

### Research basis

The original mHC paper separates the residual mixing constraint from the read
and write mappings. This supports retaining each native site's semantics.
[Xie et al., January 2026 revision](https://arxiv.org/abs/2512.24880v2)

Measurements on trained DeepSeek-V4-Flash show distinct streams and routing that
changes with depth. This is evidence against assuming identical intermediate
streams. [Zhao et al., September 2026](https://arxiv.org/abs/2609.05309v1)

HyperDFlash uses pre-collapse states and a reducer inherited from DeepSeek's final
head for speculative decoding. It supports investigating architecture-specific
reductions, but it does not evaluate Abliterix's intermediate readouts or editing
quality. Our attention-read choice is an engineering inference from the native
forward interfaces, with the validation below.
[Lin et al., June 2026 revision](https://arxiv.org/abs/2606.26744v2)

### Execution scope

The native path supports eager HF execution with direct or LoRA edits. The loader
rejects compiled execution, RDO, runtime block-output steering modes, and cliff-head
ablation for known native configs because those features require additional
stream or head mappings. Keep `model.text_only=false` for GLM, including text
prompts, so the native conditional-generation class supplies its language-model
head. The current chat frontend accepts text messages.

Released quantized checkpoints, multi-device/offload execution, fused kernels,
long-context behavior, MTP, and full image/video workflows need separate
acceptance. These are not covered by the small-model tests. In particular, a
matching family name is not a certification of every checkpoint or remote
implementation using that name.

## Validation scope

The offline integration suite saves local random checkpoints and loads them
through the normal `SteeringEngine` constructor. It compares captured states
against native hyper-connection outputs and the LM-head input, checks logits and
greedy token parity, verifies different-length batches and sorted row order, and
exercises LoRA/direct modification, restoration, selected-trial replay, and
merged export/reload. DeepSeek covers compressed sparse and heavily compressed
attention, plus sliding attention in the lifecycle test. GLM covers linear and
indexed attention, dense and sparse MLPs, and a tiny image tensor through the
conditional wrapper. The image test is a model tensor-interface check.
Additional component tests exercise native router profiling, dense/shared MLP
edits, selected and all-expert fused weight edits, exact restoration/replay,
and exported expert weights in both direct and LoRA modes.

CI runs the whole test suite in the pinned CPU profile and fails if the CPU
native integration tests skip. The default dependency environment continues to
run its existing regression suite. CUDA-specific tests run separately where a
compatible GPU is available.

The local [CUDA record](research/mhc-cuda-validation.json) covers eight cases on
an RTX 4080 Laptop GPU with PyTorch 2.11.0+cu130. Both architectures, FP32/BF16,
and LoRA/direct modes pass. Captured reads match the native sites exactly in
these runs. Exported greedy token IDs agree, and the largest measured BF16
LoRA export logit difference is `0.0009765625`, reflecting adapter merge rounding.
These tests use the eager attention implementation and unquantized tiny models.

The optional [checkpoint probe](../scripts/validate_mhc_readouts.py) retrieves
bounded HTTP ranges from pinned DeepSeek-V4-Flash and GLM-5.3-Flash revisions.
The checked-in [result](research/mhc-readout-probe.json) records 5,646,797 response
bytes, hashes, tensor offsets, package versions, and comparisons against an
independent FP64 formula. Both sampled attention gates and RMSNorm computations
match within `atol=2e-6, rtol=2e-5` for both models. First-layer inputs use three
published embedding rows broadcast across streams. Midpoint inputs are seeded
synthetic streams, so those results establish local arithmetic rather than
full-model activation behavior. This probe downloads no full model and is
separate from offline CI.

These checks establish observable correctness and integration behavior at small
scale. Full 284B/320B checkpoint loading, capability retention, and quality after
editing still require direct measurement. Issue #19 remains open for those
acceptance results.

## Qwen3.5 artifact validation

`tests/test_qwen35_compatibility.py` loads a tiny local native Qwen3.5 multimodal
checkpoint through the normal engine. Three cases cover live adapters in both
linear and full attention, decoder-only target paths, native hidden states and
generation parity, baseline restoration, selected-trial replay, and adapter and
merged-model reload. They pass with Transformers 5.7.0 and 5.20.2. The newer
runtime CI requires these cases to execute as well.

The fixtures use random FP32 CPU weights, neutral strings, and synthetic test
directions. They verify model connectivity and artifact contracts, without
reproducing the 9B run in [issue #110](https://github.com/wuwangzhang1216/abliterix/issues/110).
That report still needs the selected export's pinned checkpoint, manifest,
weight hashes, and matched in-memory/reloaded evaluation to distinguish a
search-quality result from an artifact problem. A successful tiny-model test
does not establish a behavioral improvement or capability retention at 9B.
