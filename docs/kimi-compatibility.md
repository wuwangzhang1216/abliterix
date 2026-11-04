# Native Kimi compatibility

The repository now tests the native Kimi wrapper through the normal HF engine in the separate [Transformers 5.20.2 runtime](model-runtime-mhc.md). The default 5.7 lock does not provide this native architecture. This is an integration baseline for [issue #16](https://github.com/wuwangzhang1216/abliterix/issues/16), with full released-checkpoint acceptance still pending.

The published [K2.5 configuration](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/config.json) and [K2.6 configuration](https://huggingface.co/moonshotai/Kimi-K2.6/blob/2755962d07cb42aa2d988a35bcb65cd4a9c2de82/config.json) identify the wrapper as `kimi_k25` and include nested text and vision configuration. Sharing that identifier does not establish identical checkpoint behavior.

For the native class, keep `model.trust_remote_code=false` and `model.text_only=false`. Transformers exposes `Kimi_K25ForConditionalGeneration` through `AutoModelForImageTextToText`, including text-only prompts. Its decoder layers live under `model.language_model.layers`. These interface choices come from the [pinned native implementation](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/kimi_k25/modeling_kimi_k25.py).

The older checkpoint-provided implementation advertises different AutoModel mappings and a different text-wrapper layout. It has not been executed by these tests. Enabling remote code changes the implementation under test and is outside the validated native route.

## What is exercised

`tests/test_kimi_compatibility.py` creates a local, tiny random checkpoint with dense and routed-expert text layers, a vision tower, and a small local tokenizer. In both LoRA and direct mode it verifies:

- Normal engine construction and text-layer discovery.
- Hidden states, logits, and greedy tokens against the native model.
- Attention/MLP changes, baseline restoration, selected-trial replay, and merged export/reload.
- Image-tensor forward parity before editing and after export/reload.

These tests download no model weights. CI requires both cases to execute in the compatible runtime. The image check uses the model tensor API; the application's `ChatMessage` frontend accepts text.

## Acceptance still needed

The small checkpoint uses FP32 CPU eager execution. It does not validate the released packed weights, published tokenizer/processor, remote implementation, GPU or distributed execution, long contexts, video, or quality after editing. The upstream configuration and random-model tests therefore do not constitute a tested full-model deployment recipe. Issue #16 remains open until those results are available.
