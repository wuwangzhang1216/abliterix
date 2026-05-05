# Abliterix — vLLM MoE router suppression via collective_rpc
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""In-place MoE router weight suppression for vLLM-loaded models.

This module bridges the gap between abliterix's HF-path ``_apply_moe_steering``
(which scales down router gate weights for top-N safety experts) and the vLLM
backend where the model is sharded across tensor-parallel workers.

We use ``LLM.llm_engine.collective_rpc`` (vLLM v0.8+, PR #12151) to dispatch
a worker-side callable that locates each layer's router module and scales the
rows corresponding to safety experts in place. Since the router
(``ReplicatedLinear`` for gpt-oss, ``Linear`` for Qwen3 MoE / Mixtral) is
fully replicated on every TP rank, identical edits on every rank keep the
replicated weights coherent without any cross-rank coordination.

Backup-and-restore semantics:

* First ``apply(...)`` call on a given ``(layer, expert)`` pair snapshots the
  original weight row into the worker's ``_abliterix_router_backup`` dict.
* Subsequent ``apply(...)`` calls (new trial) compute the new scaled weight
  from the backup (not the currently-modified weight), so suppression never
  compounds across trials.
* ``restore()`` writes the backup back into every touched row.

MXFP4 / quantization note: router weights are NOT in the MoE expert quant
block. They live in ``modules_to_not_convert`` for gpt-oss (see its
``config.json``) and are stored as BF16. This is why router suppression is
safe to apply on a native MXFP4 model without any dequant step — only the
fused expert weights (``w13_weight``, ``w2_weight``) are packed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..util import print

if TYPE_CHECKING:
    import torch
    from vllm import LLM


# Router module paths to probe, per architecture.  Each entry is a
# dotted attribute path from the transformer layer down to the router
# module (which has a ``.weight`` tensor of shape ``(num_experts, hidden)``).
# Order matters — first match wins.
_ROUTER_PATHS: tuple[str, ...] = (
    "mlp.router",  # gpt-oss (ReplicatedLinear)
    "mlp.gate",  # Qwen3 MoE, DeepSeek MoE
    "block_sparse_moe.gate",  # Mixtral, Phi-3.5-MoE
    "moe.gate",  # some variants
    "mixer.gate",  # LiquidAI / hybrid
)


# ---------------------------------------------------------------------------
# Worker-side functions (run under collective_rpc; must be module-level so
# they can be pickled / imported by Ray workers).
# ---------------------------------------------------------------------------


def _worker_resolve_model(worker: Any):
    """Return the ``nn.Module`` holding ``.layers[i]`` on this worker.

    vLLM v1 layout: ``worker.model_runner.model`` is the top-level model
    (e.g. ``GptOssForCausalLM``), and ``.model`` underneath is the
    decoder (``GptOssModel``) with ``.layers``.
    """
    top = worker.model_runner.model
    # BFS up to depth 3 — handles hybrid VLM/MoE wrappers where the decoder
    # sits behind `language_model` or `model.language_model` (Qwen3.5-397B,
    # Llama-4, etc.). Also handles the simple case where `top.model.layers`
    # is the decoder (GptOss, Qwen3, Llama).
    import collections as _c

    q: _c.deque = _c.deque([(top, 0)])
    seen = {id(top)}
    while q:
        node, depth = q.popleft()
        layers = getattr(node, "layers", None)
        if layers is not None:
            try:
                _ = layers[0]
                return node
            except Exception:
                pass
        if depth >= 3:
            continue
        for _, child in node.named_children():
            if id(child) in seen:
                continue
            seen.add(id(child))
            q.append((child, depth + 1))
    raise RuntimeError(
        f"Cannot locate decoder with .layers on worker model (top={type(top).__name__})"
    )


def _worker_locate_router(layer_module: Any):
    """Return (router_module, path_used) or (None, None) if not found."""
    for path in _ROUTER_PATHS:
        obj = layer_module
        ok = True
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok and hasattr(obj, "weight"):
            return obj, path
    return None, None


def _worker_install_persistent_suppression(worker: Any) -> int:
    """Install MoE router suppression hooks ONCE per worker lifetime.

    Why "once": PyTorch's Dynamo / ``@support_torch_compile`` decorator on
    ``Qwen3MoeModel`` / ``GptOssModel`` silently skips forward hooks that
    were registered AFTER the first forward triggered compilation
    (pytorch/pytorch#117758). Re-registering hooks per trial therefore
    produces no effect on KL — this was the root cause of the
    gpt-oss-120b + Qwen3.5-397B "KL stuck at ~0.0027" failure.

    The install-once-mutate-state pattern works even post-compile because
    the hook closure references ``worker._abliterix_plan`` by identity,
    not value — updating the dict per trial propagates to whatever layer
    the hook eventually fires on. Combined with ``enforce_eager=True``
    (also required — see ``vllm_cudagraph_hook_weight_invisible`` memo),
    this reliably steers per-trial without re-registration.

    The hook subtracts a per-expert penalty from the router logits:
    ``logits[..., eid] -= penalty``. A penalty of ~20 typically suffices
    to push an expert out of top-4 on BF16 routers; 50+ for FP8 routers.
    The penalty is computed from a scale in ``[0, 1]`` via
    ``-log(max(scale, 1e-6)) * 10`` so ``scale=1`` is a no-op and
    ``scale=0`` completely excludes the expert.

    Idempotent: repeated calls return the existing install's hook count
    without re-registering.
    """
    import torch

    if getattr(worker, "_abliterix_persistent_installed", False):
        return len(getattr(worker, "_abliterix_persistent_handles", []))

    decoder = _worker_resolve_model(worker)
    layers = decoder.layers

    # Per-layer mutable plan — hook reads this on every forward.
    # Shape: {layer_idx: (eids_tensor_long, pens_tensor_float32)}.
    worker._abliterix_plan = {}
    worker._abliterix_persistent_handles = []

    def _make_hook(layer_idx: int):
        def hook(module, inp, out):  # noqa: ARG001
            # Fast path: no plan → no-op.
            state = worker._abliterix_plan.get(layer_idx)
            if state is None:
                return out
            eids_t, pens_t = state
            if eids_t.numel() == 0:
                return out

            logits = out[0] if isinstance(out, tuple) else out
            if not isinstance(logits, torch.Tensor) or logits.dim() < 1:
                return out

            # Device-match lazily (first time we see this layer's logits).
            if eids_t.device != logits.device:
                eids_t = eids_t.to(logits.device)
                pens_t = pens_t.to(logits.device, dtype=logits.dtype)
                worker._abliterix_plan[layer_idx] = (eids_t, pens_t)
            elif pens_t.dtype != logits.dtype:
                pens_t = pens_t.to(dtype=logits.dtype)
                worker._abliterix_plan[layer_idx] = (eids_t, pens_t)

            with torch.no_grad():
                logits[..., eids_t] -= pens_t
            if isinstance(out, tuple):
                return (logits,) + tuple(out[1:])
            return logits

        return hook

    n_installed = 0
    for idx, layer in enumerate(layers):
        router, _ = _worker_locate_router(layer)
        if router is None:
            continue
        h = router.register_forward_hook(_make_hook(idx))
        worker._abliterix_persistent_handles.append(h)
        n_installed += 1

    worker._abliterix_persistent_installed = True
    return n_installed


def _worker_set_suppression_plan(
    worker: Any, plan_by_layer: dict[int, tuple[list[int], list[float]]]
) -> int:
    """Update the per-layer suppression plan for this trial.

    ``plan_by_layer`` maps ``layer_idx`` → ``(expert_ids, penalties)`` where
    both are equal-length Python lists. Penalties are positive floats the
    hook subtracts from the router logits (zero = no effect, ~20 = drops
    BF16 expert out of top-4, 138 = effectively zero).

    Returns the number of layers with a non-empty plan entry.
    """
    import torch

    if not getattr(worker, "_abliterix_persistent_installed", False):
        return 0

    new_plan: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_idx, (eids, pens) in plan_by_layer.items():
        if not eids:
            continue
        new_plan[int(layer_idx)] = (
            torch.tensor(list(eids), dtype=torch.long),
            torch.tensor(list(pens), dtype=torch.float32),
        )
    worker._abliterix_plan = new_plan
    return len(new_plan)


def _worker_clear_suppression_plan(worker: Any) -> int:
    """Reset the suppression plan to empty (restore baseline behaviour).

    The persistent hooks stay in place — only the data they read is
    cleared. Returns 1 if the plan existed, 0 otherwise.
    """
    plan = getattr(worker, "_abliterix_plan", None)
    if plan is None:
        return 0
    worker._abliterix_plan = {}
    return 1


# -----------------------------------------------------------------------------
# Legacy per-trial hook path (kept for back-compat with callers that have not
# migrated to install_persistent_suppression). New code MUST use the install-
# once pattern above — this function re-registers hooks each call, which is
# silently skipped by Dynamo on models wrapped with @support_torch_compile
# once the first forward has compiled.
# -----------------------------------------------------------------------------


def _worker_apply_router_scale(worker: Any, plan: list[tuple[int, int, float]]) -> int:
    """DEPRECATED. Use ``_worker_install_persistent_suppression`` +
    ``_worker_set_suppression_plan`` instead. See that docstring for why."""
    import math

    # Reuse the new plumbing: install if needed, then set plan.
    _worker_install_persistent_suppression(worker)

    per_layer: dict[int, dict[int, float]] = {}
    for layer_idx, eid, scale in plan:
        per_layer.setdefault(layer_idx, {})[eid] = float(scale)

    plan_by_layer: dict[int, tuple[list[int], list[float]]] = {}
    for layer_idx, eid_to_scale in per_layer.items():
        eids: list[int] = []
        penalties: list[float] = []
        for eid, scale in eid_to_scale.items():
            safe_scale = max(scale, 1e-6)
            penalties.append(-math.log(safe_scale) * 10.0)
            eids.append(int(eid))
        plan_by_layer[layer_idx] = (eids, penalties)

    return _worker_set_suppression_plan(worker, plan_by_layer)


def _worker_restore_routers(worker: Any) -> int:
    """DEPRECATED. Use ``_worker_clear_suppression_plan`` instead."""
    return _worker_clear_suppression_plan(worker)


def _worker_probe_routers(worker: Any) -> dict[str, Any]:
    """One-time diagnostic: report which layers expose a router, at what path."""
    decoder = _worker_resolve_model(worker)
    layers = decoder.layers
    per_layer: list[tuple[int, str | None, tuple[int, ...] | None]] = []
    for idx, layer in enumerate(layers):
        router, path = _worker_locate_router(layer)
        shape = tuple(router.weight.shape) if router is not None else None
        per_layer.append((idx, path, shape))
    return {
        "n_layers": len(layers),
        "per_layer": per_layer,
    }


def _worker_get_router_weights(worker: Any) -> dict[int, Any]:
    """Return router weight matrices (CPU fp32) for every layer that exposes
    a router.  Used by the weight-based safety-expert heuristic when forward
    hooks don't fire — this happens when vLLM's fused MoE kernel (TRITON
    MxFP4 backend for gpt-oss) reads ``router.weight`` directly and never
    calls the router nn.Module's forward.
    """
    import torch as _torch

    decoder = _worker_resolve_model(worker)
    layers = decoder.layers
    out: dict[int, _torch.Tensor] = {}
    for idx, layer in enumerate(layers):
        router, _ = _worker_locate_router(layer)
        if router is None:
            continue
        w = router.weight
        # Router weight shape: (num_experts, hidden_dim) for gpt-oss.
        out[idx] = w.detach().to(dtype=_torch.float32, device="cpu").clone()
    return out


# ---------------------------------------------------------------------------
# Safety-expert profiling via vLLM's enable_return_routed_experts
# (vLLM equivalent of HF's SteeringEngine.identify_safety_experts).
# Used when the vLLM-native fast extraction path skips HF loading entirely
# — we still need per-layer expert-risk rankings to drive router
# suppression per trial.
#
# Issue #22 / PR #24: this used to be ~150 LoC of collective_rpc + worker
# forward hooks (``_worker_install_router_hooks`` and friends, deleted).
# vLLM 0.20.x's ``RequestOutput.outputs[0].routed_experts`` exposes the
# same per-token routing IDs as a numpy array of shape
# ``(prompt_tokens, n_layers, top_k)`` — no rpc needed, no hooks needed,
# no insecure serialization needed.
#
# Verified on DeepSeek-V2-Lite (issue #22 GPU smoke B7): with
# ``enable_return_routed_experts=True`` and ``max_tokens=1``, the array
# covers all prompt tokens (vllm/v1/core/sched/scheduler.py:1612 sets
# ``num_tokens = request.num_tokens - 1``). Dense layers carry an
# all-zero placeholder slice; MoE layers carry expert ids in
# ``[0, num_experts)``.
# ---------------------------------------------------------------------------


def profile_safety_experts_vllm(
    llm: "LLM",
    benign_msgs: list,
    target_msgs: list,
    tokenizer,
    top_k: int = 4,  # noqa: ARG001 — kept for API stability; top_k is now baked in by vLLM
) -> dict[int, list[tuple[int, float]]]:
    """Compute per-layer expert risk scores by reading vLLM's per-token
    routed-expert IDs directly off the ``RequestOutput``.

    Requires the LLM to have been constructed with
    ``enable_return_routed_experts=True`` (abliterix's
    ``[model].vllm_return_routed_experts`` defaults to True; flip it off
    to skip this path). The function runs ``llm.generate`` once over the
    benign prompt set and once over the target prompt set with
    ``max_tokens=1`` (prefill-only, gives full prompt-token routing),
    aggregates per-(layer, expert) counts driver-side, and computes risk
    as ``P(expert | target) - P(expert | benign)``.

    Returns ``{layer_idx: [(expert_id, risk_score), ...]}`` sorted by
    risk descending. Layer indices are skipped when both benign and
    target slices are all-zero placeholders (dense layers in DeepSeek-V2
    style architectures emit zero rows in the routed_experts array).

    Parameters
    ----------
    top_k : int
        Kept for API compatibility with the legacy hook-based profiler
        but unused: vLLM's routed_experts array is already top-k'd at
        the model's native ``num_experts_per_tok``. Logged when supplied
        but not asserted.
    """
    import numpy as np

    from vllm import SamplingParams

    # Chat-template formatting — mirror VLLMGenerator._format_prompt so the
    # profiling prompts match what trial generation will see.
    def _fmt(msgs):
        out = []
        for m in msgs:
            chat = []
            if getattr(m, "system", None):
                chat.append({"role": "system", "content": m.system})
            chat.append({"role": "user", "content": m.user})
            try:
                out.append(
                    tokenizer.apply_chat_template(
                        chat,
                        enable_thinking=False,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                )
            except TypeError:
                out.append(
                    tokenizer.apply_chat_template(
                        chat,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                )
        return out

    params = SamplingParams(temperature=0.0, max_tokens=1)

    def _accumulate(
        prompts: list[str],
    ) -> tuple[dict[int, dict[int, int]], dict[int, int]]:
        """Return (per_layer_expert_counts, per_layer_token_counts).

        Aggregates the routed_experts arrays for ``llm.generate(prompts)``.
        Skips layers whose slice is all zeros across every token of every
        prompt (those are the dense placeholders).
        """
        counts: dict[int, dict[int, int]] = {}
        tokens: dict[int, int] = {}
        outs = llm.generate(prompts, params, use_tqdm=False)
        for out in outs:
            arr = getattr(out.outputs[0], "routed_experts", None)
            if arr is None or not isinstance(arr, np.ndarray):
                # enable_return_routed_experts is off, or the model has no
                # MoE layers, or vLLM dropped the field on this output.
                continue
            if arr.ndim != 3:
                # Defensive: shape unexpectedly different.
                continue
            n_tokens, n_layers, _k = arr.shape
            for layer in range(n_layers):
                slab = arr[:, layer, :]
                # Dense placeholder detection: every entry is zero AND
                # at least one MoE layer in this same array has a
                # non-zero entry. We treat it as "skip" via the
                # all-zero check; the caller's safety dict will not
                # contain dense-only layers.
                if int(slab.max()) == 0 and int(slab.min()) == 0:
                    continue
                tokens[layer] = tokens.get(layer, 0) + n_tokens
                # Histogram in pure numpy then merge into the running dict.
                vals, vc = np.unique(slab, return_counts=True)
                bucket = counts.setdefault(layer, {})
                for v, c in zip(vals.tolist(), vc.tolist()):
                    bucket[int(v)] = bucket.get(int(v), 0) + int(c)
        return counts, tokens

    print("  Profiling benign prompts via vLLM routed_experts...")
    benign_counts, benign_tokens = _accumulate(_fmt(benign_msgs))

    print("  Profiling target prompts via vLLM routed_experts...")
    target_counts, target_tokens = _accumulate(_fmt(target_msgs))

    if not benign_counts and not target_counts:
        print(
            "  [yellow]profile_safety_experts_vllm: routed_experts was empty "
            "for every output. Confirm vllm_return_routed_experts=true in "
            "the [model] config and the model is actually MoE.[/]"
        )
        return {}

    safety: dict[int, list[tuple[int, float]]] = {}
    all_layers = sorted(set(benign_counts) | set(target_counts))
    for layer in all_layers:
        bc = benign_counts.get(layer, {})
        tc = target_counts.get(layer, {})
        all_experts = set(bc.keys()) | set(tc.keys())
        bt = max(benign_tokens.get(layer, 1), 1)
        tt = max(target_tokens.get(layer, 1), 1)
        scores = [
            (eid, tc.get(eid, 0) / tt - bc.get(eid, 0) / bt)
            for eid in sorted(all_experts)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        safety[layer] = scores

    if safety:
        top_scores = [safety[i][0][1] for i in sorted(safety) if safety[i]]
        avg = sum(top_scores) / len(top_scores) if top_scores else 0.0
        print(
            f"  Profiled {len(safety)} MoE layers via vLLM routed_experts, "
            f"avg top risk diff: {avg:.4f}"
        )
    return safety


def profile_safety_experts_by_weight(
    llm: "LLM",
    steering_vectors: dict[int, Any],
) -> dict[int, list[tuple[int, float]]]:
    """Weight-based fallback: rank experts by alignment between router
    weight rows and the refusal direction.

    For each MoE layer with router weight ``W ∈ ℝ^(E × d)`` and refusal
    direction ``v ∈ ℝ^d`` (the steering vector at that layer), we compute
    per-expert score ``s_e = (W_e · v) / ||v||₂``.  Experts with the
    highest score have gates that fire strongest when the input lies on
    the refusal direction — i.e. they are the "safety experts".

    Use this when forward-hook profiling yields empty counts (vLLM's fused
    TRITON MxFP4 MoE kernel bypasses the router nn.Module's forward and
    reads ``router.weight`` directly, so hooks never fire).

    Returns ``{layer_idx: [(expert_id, score), ...]}`` sorted by score
    descending — same shape as the hook-based profiler.
    """
    import torch as _torch

    engine = llm.llm_engine
    results = engine.collective_rpc(_worker_get_router_weights)
    if not results or not results[0]:
        print(
            "  [yellow]profile_safety_experts_by_weight: no router weights "
            "returned — nothing to rank[/]"
        )
        return {}

    weights = results[0]  # {layer_idx: Tensor (num_experts, hidden_dim)}

    # `steering_vectors` is either a dict[int, Tensor] (per-layer direction)
    # or a stacked Tensor of shape (num_layers, hidden_dim) — the output of
    # ``compute_steering_vectors``.  Normalise to per-layer lookup.
    def _lookup(idx: int):
        if isinstance(steering_vectors, dict):
            return steering_vectors.get(idx)
        if hasattr(steering_vectors, "shape"):
            if idx < 0 or idx >= steering_vectors.shape[0]:
                return None
            return steering_vectors[idx]
        return None

    safety: dict[int, list[tuple[int, float]]] = {}
    for layer_idx in sorted(weights.keys()):
        v = _lookup(layer_idx)
        if v is None:
            continue
        v_t = v.detach().to(dtype=_torch.float32, device="cpu").reshape(-1)
        v_norm = v_t.norm().item()
        if v_norm < 1e-9:
            continue
        W = weights[layer_idx]  # (num_experts, hidden_dim)
        if W.shape[-1] != v_t.shape[0]:
            continue
        scores = (W @ v_t) / v_norm  # (num_experts,)
        ranked = sorted(
            enumerate(scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        safety[layer_idx] = [(int(e), float(s)) for e, s in ranked]

    if safety:
        top_scores = [safety[i][0][1] for i in sorted(safety) if safety[i]]
        avg = sum(top_scores) / len(top_scores) if top_scores else 0.0
        print(
            f"  Profiled {len(safety)} MoE layers by router-weight alignment, "
            f"avg top score: {avg:.4f}"
        )
    return safety


class VLLMMoEEditor:
    """Driver-side facade for router suppression across TP workers.

    Usage::

        editor = VLLMMoEEditor(vllm_gen.llm, safety_experts)
        editor.probe()  # optional: print which layers have routers
        editor.apply(n_suppress=8, bias_value=-3.0)
        # ... generate via vllm ...
        editor.restore()
    """

    def __init__(
        self,
        llm: "LLM",
        safety_experts: dict[int, list[tuple[int, float]]],
    ):
        self.llm = llm
        self.safety_experts = safety_experts
        self._applied = False
        self._probed = False
        self._router_layers: set[int] = set()
        # Persistent-hook install state (see apply() / _ensure_installed).
        self._installed = False
        self._handle_count = 0

    # -- collective_rpc shim ------------------------------------------------

    def _rpc(self, fn, args: tuple = ()):
        """Dispatch ``fn`` to every TP worker and return list of results."""
        engine = self.llm.llm_engine
        # vLLM v1 unified API.  Callable-form collective_rpc shipped in
        # PR #12151 (Jan 2025, vllm >= 0.8).
        return engine.collective_rpc(fn, args=args)

    # -- public methods -----------------------------------------------------

    def probe(self) -> None:
        """Query each worker for router locations (one-time, logs summary)."""
        if self._probed:
            return
        results = self._rpc(_worker_probe_routers)
        if not results:
            return
        # All workers see identical replicated routers for gpt-oss, so the
        # first result is sufficient.
        info = results[0]
        n_layers = info["n_layers"]
        found = [(i, path, shape) for (i, path, shape) in info["per_layer"] if path]
        self._router_layers = {i for (i, _, _) in found}
        if not found:
            print(
                "  [yellow]VLLMMoEEditor.probe: no router modules found on any layer "
                "— router suppression will be a no-op.[/]"
            )
        else:
            paths = sorted({p for (_, p, _) in found})
            shapes = sorted({tuple(s) for (_, _, s) in found if s})
            print(
                f"  [dim]VLLMMoEEditor.probe: {len(found)}/{n_layers} layers expose "
                f"router at paths={paths}, weight shape(s)={shapes}[/]"
            )
        self._probed = True

    def _ensure_installed(self) -> int:
        """Install persistent suppression hooks on all workers. Idempotent."""
        if self._installed:
            return self._handle_count
        results = self._rpc(_worker_install_persistent_suppression)
        self._handle_count = results[0] if results else 0
        self._installed = True
        return self._handle_count

    def apply(self, n_suppress: int, bias_value: float) -> int:
        """Apply router suppression for this trial.

        Parameters
        ----------
        n_suppress : int
            Number of top-risk experts to suppress per layer.
        bias_value : float
            Suppression strength in ``[-10, 0]``. Matches HF
            ``_apply_moe_steering``: ``scale = max(0.0, 1.0 + bias_value/10.0)``.
            ``bias_value = -5``  → scale 0.5 (halve).
            ``bias_value = -10`` → scale 0 (kill).
            ``bias_value = 0``   → scale 1.0 (no-op).

        Returns
        -------
        int
            Number of layers with an active suppression plan (reported by
            the first TP worker; identical across replicated routers).
        """
        import math

        if not self._probed:
            self.probe()

        if n_suppress <= 0 or bias_value >= 0 or not self._router_layers:
            return 0

        scale = max(0.0, 1.0 + bias_value / 10.0)
        safe_scale = max(scale, 1e-6)
        penalty = -math.log(safe_scale) * 10.0  # 20 ≈ out-of-top-4 at BF16

        plan_by_layer: dict[int, tuple[list[int], list[float]]] = {}
        for layer_idx, ranking in self.safety_experts.items():
            if layer_idx not in self._router_layers:
                continue
            eids = [int(eid) for eid, _score in ranking[:n_suppress]]
            if not eids:
                continue
            plan_by_layer[layer_idx] = (eids, [penalty] * len(eids))

        if not plan_by_layer:
            return 0

        # Install hooks on first call (cheap, idempotent).
        self._ensure_installed()

        results = self._rpc(_worker_set_suppression_plan, args=(plan_by_layer,))
        self._applied = True
        return results[0] if results else 0

    def restore(self) -> int:
        """Clear the per-trial suppression plan. Hooks stay installed
        (cheap — they no-op on empty plan)."""
        if not self._applied:
            return 0
        results = self._rpc(_worker_clear_suppression_plan)
        self._applied = False
        return results[0] if results else 0


# ---------------------------------------------------------------------------
# Expert-Granular Abliteration (EGA) on vLLM — in-place orthogonal projection
# of the refusal direction from every expert's fused down_proj (w2_weight).
#
# Unlike router suppression (which edits the small replicated router gate),
# EGA edits the large fused expert tensor. Under vLLM TP this tensor is
# SHARDED along the intermediate dim, so every worker holds a slice
# (num_experts, hidden, intermediate/TP). The steering vector lives in the
# `hidden` dim which stays intact across ranks — each worker applies the
# projection locally with zero cross-rank sync required.
#
# Why this works under CUDA graphs (verified in vllm_backend.py already):
#   * tensor .data.copy_(new_values) preserves the storage pointer → CUDA
#     graphs captured before the edit continue to read the new values.
#   * UnquantizedFusedMoEMethod's forward reads `layer.w2_weight` per call
#     (unquantized_fused_moe_method.py:281), not a cached pointer.
#   * For TRITON backend `process_weights_after_loading` is effectively
#     identity — no repack to invalidate.
#
# ONLY runs on TRITON / BATCHED_TRITON backend (BF16 unquantized MoE).
# For FLASHINFER_TRTLLM backend the kernel reads a repacked tensor;
# we detect that via a shape probe and abort with a clear error.
# ---------------------------------------------------------------------------


def _worker_locate_moe_experts(layer_module: Any):
    """Return (moe_experts_module, path_used) or (None, None).

    Probes common MoE module paths. ``moe_experts_module`` is the
    ``FusedMoE`` instance holding ``w2_weight`` (fused expert down_proj).
    """
    _MOE_PATHS = (
        "mlp.experts",  # gpt-oss (FusedMoE under MLPBlock)
        "mlp.experts.experts",  # some wrappers
        "block_sparse_moe.experts",  # Mixtral variant
        "moe.experts",  # generic
    )
    for path in _MOE_PATHS:
        obj = layer_module
        ok = True
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok and hasattr(obj, "w2_weight"):
            return obj, path
    return None, None


def _worker_probe_experts(worker: Any) -> dict[str, Any]:
    """One-time diagnostic: report expert module locations + w2 shapes."""
    decoder = _worker_resolve_model(worker)
    layers = decoder.layers
    per_layer: list[tuple[int, str | None, tuple[int, ...] | None, str | None]] = []
    for idx, layer in enumerate(layers):
        moe, path = _worker_locate_moe_experts(layer)
        if moe is None:
            per_layer.append((idx, None, None, None))
            continue
        w2 = moe.w2_weight
        per_layer.append((idx, path, tuple(w2.shape), str(w2.dtype)))
    return {"n_layers": len(layers), "per_layer": per_layer}


def _worker_backup_experts(worker: Any, layer_indices: list[int]) -> int:
    """Snapshot original w2_weight for the listed layers into CPU pinned RAM.

    Idempotent: skips layers already backed up. Called once before the
    first trial so restore() can reset to pristine between trials without
    re-downloading the model.
    """
    import torch

    decoder = _worker_resolve_model(worker)
    backup = getattr(worker, "_abliterix_w2_backup", None)
    if backup is None:
        backup = {}
        worker._abliterix_w2_backup = backup

    n_new = 0
    for idx in layer_indices:
        if idx in backup:
            continue
        layer = decoder.layers[idx]
        moe, _ = _worker_locate_moe_experts(layer)
        if moe is None:
            continue
        w2 = moe.w2_weight
        # CPU buffer for fast restore. Pin memory on CUDA for faster H2D
        # at restore time; on non-CUDA (CPU tests, MPS) fall back silently.
        cpu = w2.data.detach().to(device="cpu", dtype=w2.dtype, copy=True)
        if torch.cuda.is_available():
            try:
                cpu = cpu.pin_memory()
            except RuntimeError:
                pass
        backup[idx] = cpu
        n_new += 1
    return n_new


def _worker_restore_experts(worker: Any) -> int:
    """Copy CPU backup back into every w2_weight. Returns layer count restored."""
    import torch  # noqa: F401

    decoder = _worker_resolve_model(worker)
    backup = getattr(worker, "_abliterix_w2_backup", None)
    if not backup:
        return 0
    n = 0
    for idx, cpu_w in backup.items():
        layer = decoder.layers[idx]
        moe, _ = _worker_locate_moe_experts(layer)
        if moe is None:
            continue
        moe.w2_weight.data.copy_(cpu_w.to(moe.w2_weight.device, non_blocking=True))
        n += 1
    return n


def _worker_apply_ega_batch(
    worker: Any,
    plan: list[dict[str, Any]],
    norm_preserve: bool,
) -> dict[str, Any]:
    """Apply EGA projection to a batch of layers in one RPC.

    Each entry in ``plan`` is a dict with:
      * ``layer_idx`` (int)
      * ``v`` (bytes, torch.save of a 1-D float tensor in hidden dim)
      * ``strength`` (float)
      * ``hidden_dim`` (int) — used to disambiguate transposed vs standard layout

    Returns ``{"applied": n, "errors": [...], "per_layer": [(idx, axis, d_edit)]}``
    where ``axis`` tells whether projection ran on axis=1 (standard) or
    axis=2 (transposed-gpt-oss).
    """
    import io
    import torch

    decoder = _worker_resolve_model(worker)
    applied = 0
    errors: list[str] = []
    per_layer: list[tuple[int, int, int]] = []

    for entry in plan:
        idx = int(entry["layer_idx"])
        strength = float(entry["strength"])
        hidden_dim = int(entry["hidden_dim"])
        try:
            v_bytes = entry["v"]
            v = torch.load(io.BytesIO(v_bytes), map_location="cpu", weights_only=True)
        except Exception as e:
            errors.append(f"layer {idx}: deserialize v failed: {e}")
            continue

        layer = decoder.layers[idx]
        moe, _ = _worker_locate_moe_experts(layer)
        if moe is None:
            continue

        w2 = moe.w2_weight  # shape (E, d0, d1)
        if w2.dim() != 3:
            errors.append(f"layer {idx}: w2 dim {w2.dim()} != 3, skipping")
            continue

        # Shape of the deserialized steering vector must match hidden_dim
        # the caller promised. Without this guard, a mis-sized v would
        # silently broadcast-explode inside einsum.
        if v.dim() != 1 or v.shape[0] != hidden_dim:
            errors.append(
                f"layer {idx}: v shape {tuple(v.shape)} != (hidden={hidden_dim},)"
            )
            continue

        d0, d1 = w2.shape[1], w2.shape[2]

        # Axis detection: find which axis matches hidden_dim.
        # gpt-oss has hidden == intermediate (both 2880) so it's ambiguous.
        # Convention: default to axis=1 (standard MoE layout). Callers of
        # known-transposed models (gpt-oss) pass "transposed" flag.
        transposed = bool(entry.get("transposed", False))
        if transposed:
            if d1 != hidden_dim:
                errors.append(
                    f"layer {idx}: transposed=True but d1={d1} != hidden={hidden_dim}"
                )
                continue
            axis_is_in = True  # hidden is the INPUT axis (last dim)
        else:
            if d0 == hidden_dim and d1 != hidden_dim:
                axis_is_in = False  # hidden is the OUTPUT axis (dim 1)
            elif d1 == hidden_dim and d0 != hidden_dim:
                axis_is_in = True
            elif d0 == hidden_dim:
                # Ambiguous case — prefer standard (axis_is_in=False).
                axis_is_in = False
            else:
                errors.append(
                    f"layer {idx}: hidden={hidden_dim} matches neither axis "
                    f"({d0}, {d1})"
                )
                continue

        device = w2.device
        vf = v.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            W_all = w2.data.to(torch.float32)  # (E, d0, d1)
            if axis_is_in:
                proj = torch.matmul(W_all, vf)  # (E, d0)
                W_new = W_all - strength * (proj.unsqueeze(-1) * vf.view(1, 1, -1))
            else:
                proj = torch.einsum("o,eoi->ei", vf, W_all)  # (E, d1)
                W_new = W_all - strength * (vf.view(1, -1, 1) * proj.unsqueeze(1))

            if norm_preserve:
                orig_norms = torch.linalg.vector_norm(W_all, dim=-1, keepdim=True)
                new_norms = torch.linalg.vector_norm(W_new, dim=-1, keepdim=True).clamp(
                    min=1e-8
                )
                W_new = W_new * (orig_norms / new_norms)

            w2.data.copy_(W_new.to(w2.dtype))
            del W_all, W_new, proj

        applied += 1
        per_layer.append((idx, 2 if axis_is_in else 1, int(w2.shape[0])))

    return {"applied": applied, "errors": errors, "per_layer": per_layer}


class VLLMExpertEditor:
    """Driver-side facade for Expert-Granular Abliteration on vLLM.

    Usage::

        editor = VLLMExpertEditor(vllm_gen.llm, hidden_dim=2880, transposed=True)
        editor.probe()                      # diagnostic
        editor.backup(layer_indices)        # one-shot before trial 1
        # --- per trial:
        editor.apply_ega(layer_plan, norm_preserve=True)
        # ... generate via vllm ...
        editor.restore()                    # reset to pristine

    The editor keeps pristine BF16 copies on CPU pinned RAM per worker.
    Memory cost: ``num_layers × num_experts × hidden × intermediate/TP ×
    2 bytes`` per worker. For gpt-oss-120b on TP=3 this is ~25 GB CPU per
    worker — well within a normal pod's host RAM.

    Only runs cleanly on TRITON / BATCHED_TRITON unquantized backends
    (BF16). FLASHINFER_TRTLLM repacks w2_weight in
    ``process_weights_after_loading`` so in-place edits miss the kernel;
    force TRITON via ``VLLM_FUSED_MOE_UNQUANTIZED_BACKEND=triton`` before
    engine load.
    """

    def __init__(
        self,
        llm: "LLM",
        hidden_dim: int,
        transposed: bool = False,
    ):
        self.llm = llm
        self.hidden_dim = int(hidden_dim)
        self.transposed = bool(transposed)
        self._probed = False
        self._backed_up: set[int] = set()
        self._applied = False
        self._moe_layers: set[int] = set()

    def _rpc(self, fn, args: tuple = (), kwargs: dict | None = None):
        engine = self.llm.llm_engine
        if kwargs is None:
            return engine.collective_rpc(fn, args=args)
        return engine.collective_rpc(fn, args=args, kwargs=kwargs)

    def probe(self) -> None:
        """Query workers for MoE expert module locations (once)."""
        if self._probed:
            return
        results = self._rpc(_worker_probe_experts)
        if not results:
            return
        info = results[0]
        n_layers = info["n_layers"]
        found = [(i, p, sh) for (i, p, sh, _dt) in info["per_layer"] if p]
        self._moe_layers = {i for (i, _, _) in found}
        if not found:
            print(
                "  [yellow]VLLMExpertEditor.probe: no MoE experts found on any layer. "
                "EGA will be a no-op.[/]"
            )
        else:
            paths = sorted({p for (_, p, _) in found})
            # pickle roundtrip via collective_rpc may have demoted tuple → list.
            shapes = sorted({tuple(sh) for (_, _, sh) in found if sh})
            print(
                f"  [dim]VLLMExpertEditor.probe: {len(found)}/{n_layers} layers "
                f"expose FusedMoE at paths={paths}, w2 shapes={shapes}[/]"
            )
        self._probed = True

    def backup(self, layer_indices: list[int] | None = None) -> int:
        """Snapshot pristine w2_weight to CPU pinned RAM. Idempotent.

        ``layer_indices=None`` means "every MoE layer" (requires probe first).
        """
        if not self._probed:
            self.probe()
        if layer_indices is None:
            layer_indices = sorted(self._moe_layers)
        new_targets = [i for i in layer_indices if i not in self._backed_up]
        if not new_targets:
            return 0
        results = self._rpc(_worker_backup_experts, args=(new_targets,))
        self._backed_up.update(new_targets)
        return int(results[0]) if results else 0

    def apply_ega(
        self,
        plan: list[dict[str, Any]],
        norm_preserve: bool = True,
    ) -> dict[str, Any]:
        """Apply EGA projection for a trial.

        ``plan`` must contain one dict per layer-to-edit. Each dict keys:
          * ``layer_idx``: int
          * ``v``: bytes (torch.save of 1-D hidden-dim float tensor)
          * ``strength``: float

        Driver responsibilities (NOT done here): computing ``v`` per layer,
        computing per-layer strength from the decay kernel, pruning layers
        outside the ``[max_position ± min_distance]`` band.

        Returns the aggregated RPC result.
        """
        if not self._probed:
            self.probe()
        if not self._moe_layers:
            return {"applied": 0, "errors": [], "per_layer": []}

        # Auto-backup on first apply.
        needed_idx = [int(p["layer_idx"]) for p in plan]
        self.backup(needed_idx)

        # Inject hidden_dim + transposed flag into every plan entry.
        for entry in plan:
            entry.setdefault("hidden_dim", self.hidden_dim)
            entry.setdefault("transposed", self.transposed)

        results = self._rpc(
            _worker_apply_ega_batch,
            args=(plan, bool(norm_preserve)),
        )
        self._applied = True
        if not results:
            return {"applied": 0, "errors": ["no workers"], "per_layer": []}
        # All workers run the same plan (TP replicated), return the first.
        return results[0]

    def restore(self) -> int:
        """Reset every edited layer's w2_weight to the pristine backup."""
        if not self._applied:
            return 0
        results = self._rpc(_worker_restore_experts)
        self._applied = False
        return int(results[0]) if results else 0


# ---------------------------------------------------------------------------
# Attention weight editor (q/k/v/o_proj orthogonal projection under vLLM TP)
#
# vLLM FUSES q, k, v into a single ``QKVParallelLinear`` module holding one
# weight tensor shaped ``(q_size + 2 * kv_size, hidden)`` per rank, with
# layout ``[Q; K; V]`` along dim 0. abliteration wants SEPARATE steering
# strengths per component (sp.q_proj / sp.k_proj / sp.v_proj). We slice
# the fused weight, project each slice independently, and write back —
# which keeps kernel contracts intact since layout is preserved.
#
# ``o_proj`` is a ``RowParallelLinear`` with weight ``(hidden, total_head
# _dim / TP)`` — input sharded, output intact. The steering vector lives
# in hidden (the OUTPUT axis of o_proj) so projection stays local per rank.
#
# TP sharding summary — NO cross-rank sync required:
#   * qkv_proj: input (hidden) intact across ranks → projection on input
#     axis works locally. Row-norm preservation uses local row norms.
#   * o_proj: output (hidden) intact → projection on output axis works
#     locally. Row-norm preservation also local.
# ---------------------------------------------------------------------------


_ATTN_PATHS: tuple[str, ...] = (
    "self_attn",  # gpt-oss, Qwen3, Llama, Mixtral
    "attention",  # some variants
    "attn",  # some others
)


def _worker_locate_attention(layer_module: Any):
    """Return ``(attn_module, path)`` if the layer has a fused qkv_proj
    + o_proj pair recognisable to vLLM, else ``(None, None)``.

    The returned module must expose ``qkv_proj.weight``, ``o_proj.weight``
    AND integer attributes ``q_size`` + ``kv_size`` so we know how to
    slice the fused QKV tensor.
    """
    for path in _ATTN_PATHS:
        attn = getattr(layer_module, path, None)
        if attn is None:
            continue
        if not hasattr(attn, "qkv_proj") or not hasattr(attn, "o_proj"):
            continue
        if not hasattr(attn, "q_size") or not hasattr(attn, "kv_size"):
            continue
        return attn, path
    return None, None


def _worker_probe_attention(worker: Any) -> dict[str, Any]:
    """Diagnostic: report attention module locations + shapes."""
    decoder = _worker_resolve_model(worker)
    layers = decoder.layers
    per_layer: list[
        tuple[int, str | None, tuple[int, ...] | None, tuple[int, ...] | None, int, int]
    ] = []
    for idx, layer in enumerate(layers):
        attn, path = _worker_locate_attention(layer)
        if attn is None:
            per_layer.append((idx, None, None, None, 0, 0))
            continue
        qkv_shape = tuple(attn.qkv_proj.weight.shape)
        o_shape = tuple(attn.o_proj.weight.shape)
        per_layer.append(
            (idx, path, qkv_shape, o_shape, int(attn.q_size), int(attn.kv_size))
        )
    return {"n_layers": len(layers), "per_layer": per_layer}


def _worker_backup_attention(worker: Any, layer_indices: list[int]) -> int:
    """Snapshot qkv_proj + o_proj weights into CPU pinned RAM. Idempotent."""
    import torch

    decoder = _worker_resolve_model(worker)
    backup = getattr(worker, "_abliterix_attn_backup", None)
    if backup is None:
        backup = {}
        worker._abliterix_attn_backup = backup

    n_new = 0
    for idx in layer_indices:
        if idx in backup:
            continue
        layer = decoder.layers[idx]
        attn, _ = _worker_locate_attention(layer)
        if attn is None:
            continue

        qkv_w = attn.qkv_proj.weight
        o_w = attn.o_proj.weight
        qkv_cpu = qkv_w.data.detach().to(device="cpu", dtype=qkv_w.dtype, copy=True)
        o_cpu = o_w.data.detach().to(device="cpu", dtype=o_w.dtype, copy=True)
        if torch.cuda.is_available():
            try:
                qkv_cpu = qkv_cpu.pin_memory()
                o_cpu = o_cpu.pin_memory()
            except RuntimeError:
                pass
        backup[idx] = {"qkv": qkv_cpu, "o": o_cpu}
        n_new += 1
    return n_new


def _worker_restore_attention(worker: Any) -> int:
    decoder = _worker_resolve_model(worker)
    backup = getattr(worker, "_abliterix_attn_backup", None)
    if not backup:
        return 0
    n = 0
    for idx, pair in backup.items():
        layer = decoder.layers[idx]
        attn, _ = _worker_locate_attention(layer)
        if attn is None:
            continue
        attn.qkv_proj.weight.data.copy_(
            pair["qkv"].to(attn.qkv_proj.weight.device, non_blocking=True)
        )
        attn.o_proj.weight.data.copy_(
            pair["o"].to(attn.o_proj.weight.device, non_blocking=True)
        )
        n += 1
    return n


def _project_2d(
    W: "torch.Tensor", vf: "torch.Tensor", strength: float, norm_preserve: bool
):
    """Apply ``_apply_direct_steering`` math on a 2-D weight slice.

    W shape (out_f, in_f) with possibly ``in_f`` or ``out_f`` matching v.
    Returns the edited tensor in the same dtype as the original weight slice.
    Raises ``ValueError`` if v matches neither axis.
    """
    import torch

    W32 = W.to(torch.float32)
    out_f, in_f = W32.shape
    v = vf
    if v.shape[0] == out_f:
        proj = v @ W32  # (in_f,)
        W_new = W32 - strength * v.unsqueeze(1) * proj.unsqueeze(0)
    elif v.shape[0] == in_f:
        proj = W32 @ v  # (out_f,)
        W_new = W32 - strength * proj.unsqueeze(1) * v.unsqueeze(0)
    else:
        raise ValueError(
            f"v shape {v.shape[0]} matches neither axis of W ({out_f}, {in_f})"
        )

    if norm_preserve:
        orig_norms = torch.linalg.vector_norm(W32, dim=1, keepdim=True)
        new_norms = torch.linalg.vector_norm(W_new, dim=1, keepdim=True).clamp(min=1e-8)
        W_new = W_new * (orig_norms / new_norms)
    W_new = torch.nan_to_num(W_new, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        finfo = torch.finfo(W.dtype)
    except TypeError:
        finfo = None
    if finfo is not None and "float8" in str(W.dtype):
        W_new = W_new.clamp(min=finfo.min, max=finfo.max)
    return W_new.to(W.dtype)


def _worker_apply_attn_batch(
    worker: Any,
    plan: list[dict[str, Any]],
    norm_preserve: bool,
) -> dict[str, Any]:
    """Apply attention projection to a batch of (layer, component, v, strength).

    ``plan`` entries:
      * ``layer_idx`` (int)
      * ``component`` (one of ``"q_proj"``, ``"k_proj"``, ``"v_proj"``, ``"o_proj"``)
      * ``v`` (bytes — torch.save of 1-D hidden-dim float tensor)
      * ``strength`` (float)

    Returns ``{"applied": n, "errors": [...], "per_layer": [...]}``.
    """
    import io
    import torch

    decoder = _worker_resolve_model(worker)
    applied = 0
    errors: list[str] = []
    per_layer: list[tuple[int, str, tuple[int, int]]] = []

    for entry in plan:
        idx = int(entry["layer_idx"])
        component = str(entry["component"])
        strength = float(entry["strength"])
        try:
            v = torch.load(
                io.BytesIO(entry["v"]), map_location="cpu", weights_only=True
            )
        except Exception as e:
            errors.append(f"layer {idx} {component}: deserialize failed: {e}")
            continue
        if v.dim() != 1:
            errors.append(f"layer {idx} {component}: v must be 1-D, got {v.shape}")
            continue

        layer = decoder.layers[idx]
        attn, _ = _worker_locate_attention(layer)
        if attn is None:
            continue

        if component == "o_proj":
            W = attn.o_proj.weight.data
            device = W.device
            vf = v.to(device=device, dtype=torch.float32)
            try:
                with torch.no_grad():
                    W_new = _project_2d(W, vf, strength, norm_preserve)
                    attn.o_proj.weight.data.copy_(W_new)
                applied += 1
                per_layer.append((idx, component, (W.shape[0], W.shape[1])))
            except ValueError as e:
                errors.append(f"layer {idx} o_proj: {e}")
            continue

        # q_proj / k_proj / v_proj — slice the fused QKV weight.
        if component not in ("q_proj", "k_proj", "v_proj"):
            errors.append(f"layer {idx}: unknown component {component!r}")
            continue

        qkv_w = attn.qkv_proj.weight
        q_size = int(attn.q_size)
        kv_size = int(attn.kv_size)
        if component == "q_proj":
            lo, hi = 0, q_size
        elif component == "k_proj":
            lo, hi = q_size, q_size + kv_size
        else:  # v_proj
            lo, hi = q_size + kv_size, q_size + 2 * kv_size

        if hi > qkv_w.shape[0]:
            errors.append(
                f"layer {idx} {component}: slice [{lo}:{hi}] exceeds qkv rows "
                f"{qkv_w.shape[0]} — TP split mismatch?"
            )
            continue

        device = qkv_w.device
        vf = v.to(device=device, dtype=torch.float32)
        try:
            with torch.no_grad():
                W_slice = qkv_w.data[lo:hi]
                W_new = _project_2d(W_slice, vf, strength, norm_preserve)
                qkv_w.data[lo:hi].copy_(W_new)
            applied += 1
            per_layer.append((idx, component, (hi - lo, int(qkv_w.shape[1]))))
        except ValueError as e:
            errors.append(f"layer {idx} {component}: {e}")

    return {"applied": applied, "errors": errors, "per_layer": per_layer}


class VLLMAttentionEditor:
    """Driver-side facade for attention weight projection under vLLM TP.

    Usage::

        editor = VLLMAttentionEditor(vllm_gen.llm)
        editor.probe()                                      # diagnostic
        editor.backup(layer_indices)                        # one-shot
        # --- per trial:
        editor.apply(plan, norm_preserve=True)              # list of dicts
        # ... generate ...
        editor.restore()
    """

    def __init__(self, llm: "LLM"):
        self.llm = llm
        self._probed = False
        self._attn_layers: set[int] = set()
        self._backed_up: set[int] = set()
        self._applied = False
        # Cached sizes from probe — used only for diagnostic printing.
        self._last_q_size = 0
        self._last_kv_size = 0

    def _rpc(self, fn, args: tuple = ()):
        return self.llm.llm_engine.collective_rpc(fn, args=args)

    def probe(self) -> None:
        if self._probed:
            return
        results = self._rpc(_worker_probe_attention)
        if not results:
            return
        info = results[0]
        n_layers = info["n_layers"]
        found = [
            (i, p, tuple(qs), tuple(os_), q, kv)
            for (i, p, qs, os_, q, kv) in info["per_layer"]
            if p is not None
        ]
        self._attn_layers = {i for (i, *_rest) in found}
        if not found:
            print(
                "  [yellow]VLLMAttentionEditor.probe: no attention modules found "
                "(expected self_attn.qkv_proj + self_attn.o_proj).[/]"
            )
            self._probed = True
            return
        first = found[0]
        self._last_q_size = first[4]
        self._last_kv_size = first[5]
        qkv_shapes = sorted({qs for (_, _, qs, _os, _q, _k) in found})
        o_shapes = sorted({os_ for (_, _, _qs, os_, _q, _k) in found})
        print(
            f"  [dim]VLLMAttentionEditor.probe: {len(found)}/{n_layers} layers "
            f"expose self_attn. qkv shapes={qkv_shapes}, o shapes={o_shapes}, "
            f"q_size={self._last_q_size}, kv_size={self._last_kv_size}[/]"
        )
        self._probed = True

    def backup(self, layer_indices: list[int] | None = None) -> int:
        if not self._probed:
            self.probe()
        if layer_indices is None:
            layer_indices = sorted(self._attn_layers)
        new_targets = [i for i in layer_indices if i not in self._backed_up]
        if not new_targets:
            return 0
        results = self._rpc(_worker_backup_attention, args=(new_targets,))
        self._backed_up.update(new_targets)
        return int(results[0]) if results else 0

    def apply(
        self,
        plan: list[dict[str, Any]],
        norm_preserve: bool = True,
    ) -> dict[str, Any]:
        """Apply attention projection for this trial.

        ``plan`` — one dict per (layer, component): ``layer_idx``, ``component``
        (``"q_proj"``/``"k_proj"``/``"v_proj"``/``"o_proj"``), ``v`` (bytes),
        ``strength`` (float).
        """
        if not self._probed:
            self.probe()
        if not self._attn_layers:
            return {"applied": 0, "errors": [], "per_layer": []}

        # Auto-backup each touched layer on first apply.
        needed = sorted({int(p["layer_idx"]) for p in plan})
        self.backup(needed)

        results = self._rpc(_worker_apply_attn_batch, args=(plan, bool(norm_preserve)))
        self._applied = True
        if not results:
            return {"applied": 0, "errors": ["no workers"], "per_layer": []}
        return results[0]

    def restore(self) -> int:
        if not self._applied:
            return 0
        results = self._rpc(_worker_restore_attention)
        self._applied = False
        return int(results[0]) if results else 0
