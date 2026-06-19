# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Trial scoring: KL divergence, coherence measurement, and multi-objective evaluation.

The :class:`TrialScorer` orchestrates baseline capture during init and then
provides :meth:`score_trial` to evaluate each Optuna trial.
"""

import statistics

import torch
import torch.nn.functional as F
from torch import Tensor

from ..data import load_prompt_dataset
from ..settings import AbliterixConfig
from ..types import ChatMessage
from ..util import print
from .detector import RefusalDetector


def _finite_logprobs(logprobs: Tensor) -> Tensor:
    """Return normalized finite log-probabilities for KL scoring."""
    if torch.isfinite(logprobs).all():
        return logprobs
    cleaned = torch.nan_to_num(logprobs, nan=-30.0, posinf=0.0, neginf=-30.0)
    return F.log_softmax(cleaned, dim=-1)


def _safe_kl_divergence(current_logprobs: Tensor, baseline_logprobs: Tensor) -> float:
    """Compute finite KL even when a damaged trial emits NaN/Inf logits."""
    # Align devices: with inference.offload_outputs_to_cpu the two operands may
    # originate on different devices (e.g. baseline on GPU, current offloaded to
    # CPU). Move the baseline onto the current tensor's device before kl_div.
    if baseline_logprobs.device != current_logprobs.device:
        baseline_logprobs = baseline_logprobs.to(current_logprobs.device)
    kl = F.kl_div(
        _finite_logprobs(current_logprobs),
        _finite_logprobs(baseline_logprobs),
        reduction="batchmean",
        log_target=True,
    ).item()
    if torch.isfinite(torch.tensor(kl)):
        return float(kl)
    return float("inf")


class TrialScorer:
    """Measures model damage (KL divergence, coherence) and compliance.

    On construction the scorer records baseline logprobs, response lengths,
    and refusal counts against the un-steered model.  Each call to
    :meth:`score_trial` then returns a multi-objective tuple that Optuna
    minimises.
    """

    config: AbliterixConfig
    detector: RefusalDetector
    benign_msgs: list[ChatMessage]
    target_msgs: list[ChatMessage]
    baseline_logprobs: Tensor
    baseline_refusal_count: int
    baseline_mean_length: float
    baseline_stdev_length: float

    def __init__(
        self,
        config: AbliterixConfig,
        engine,
        detector: RefusalDetector,
        defer_baseline: bool = False,
    ):
        self.config = config
        self.detector = detector

        print()
        print(
            f"Loading benign evaluation prompts from [bold]{config.benign_eval_prompts.dataset}[/]..."
        )
        self.benign_msgs = load_prompt_dataset(config, config.benign_eval_prompts)
        print(f"* [bold]{len(self.benign_msgs)}[/] prompts loaded")

        print()
        print(
            f"Loading target evaluation prompts from [bold]{config.target_eval_prompts.dataset}[/]..."
        )
        self.target_msgs = load_prompt_dataset(config, config.target_eval_prompts)
        print(f"* [bold]{len(self.target_msgs)}[/] prompts loaded")

        if defer_baseline:
            # Baseline capture deferred until capture_baseline() is called
            # after the TP backend is loaded.  This avoids running expensive
            # generation on HF pipeline parallelism (~4 tok/s) before the
            # fast TP backend is available.
            self.baseline_logprobs = None
            self.baseline_mean_length = 1.0
            self.baseline_stdev_length = 1.0
            self.baseline_refusal_count = 0
            print("* [dim]Baseline capture deferred to TP backend phase[/]")
        else:
            self._capture_baseline(engine)

    def _capture_baseline(self, engine):
        """Capture baseline logprobs, response lengths, and refusal count.

        Automatically routes to the TP backend (vLLM/SGLang) if available,
        avoiding the slow HF pipeline-parallel generation path.
        """
        # Capture baseline logprobs and response lengths in a single pass.
        # Route to TP backend if available.
        print("* Obtaining probability distributions and baseline response lengths...")
        vllm_gen = getattr(engine, "_vllm_gen", None)
        if vllm_gen is not None:
            base_responses, self.baseline_logprobs = (
                vllm_gen.generate_and_score_batched(
                    self.benign_msgs,
                    max_new_tokens=self.config.inference.max_gen_tokens,
                    kl_token_count=self.config.kl.token_count,
                    skip_special_tokens=True,
                    min_new_tokens=self.config.inference.min_gen_tokens,
                    adapter_path=None,
                )
            )
            if self._supports_vllm_continuation_kl(vllm_gen):
                self.baseline_continuations = base_responses
                print("* Scoring baseline continuations for vLLM in-place KL...")
                self.baseline_continuation_nll = vllm_gen.score_continuations_nll(
                    self.benign_msgs,
                    self.baseline_continuations,
                    adapter_path=None,
                )
        else:
            base_responses, self.baseline_logprobs = engine.generate_and_score_batched(
                self.benign_msgs,
                max_new_tokens=self.config.inference.max_gen_tokens,
                kl_token_count=self.config.kl.token_count,
                skip_special_tokens=True,
                min_new_tokens=self.config.inference.min_gen_tokens,
            )
        if not hasattr(self, "baseline_continuations"):
            self.baseline_continuations = None
            self.baseline_continuation_nll = None
        base_lengths = [len(r.split()) for r in base_responses]
        self.baseline_mean_length = (
            statistics.mean(base_lengths) if base_lengths else 1.0
        )
        self.baseline_stdev_length = (
            statistics.stdev(base_lengths) if len(base_lengths) > 1 else 1.0
        )
        print(
            f"* Baseline response length: [bold]{self.baseline_mean_length:.1f}[/] "
            f"+/- {self.baseline_stdev_length:.1f} words"
        )

        print("* Counting model refusals...")
        self.baseline_refusal_count = self.detector.evaluate_compliance(
            engine,
            self.target_msgs,
        )
        print(
            f"* Initial refusals: [bold]{self.baseline_refusal_count}[/]"
            f"/{len(self.target_msgs)}"
        )

    # ------------------------------------------------------------------
    # Individual metric helpers
    # ------------------------------------------------------------------

    def measure_kl_divergence(self, engine) -> float:
        """Compute KL divergence between the steered and baseline logprobs."""
        print("  * Obtaining probability distributions...")
        vllm_gen = getattr(engine, "_vllm_gen", None)
        adapter_path = getattr(engine, "_current_adapter_path", None)
        if self._use_vllm_continuation_kl(vllm_gen):
            kl = self._measure_vllm_continuation_kl(vllm_gen, adapter_path)
            print(f"  * KL divergence: [bold]{kl:.4f}[/] (continuation NLL)")
            return kl
        if vllm_gen is not None:
            logprobs = vllm_gen.compute_logprobs_batched(
                self.benign_msgs,
                adapter_path=adapter_path,
            )
        else:
            logprobs = engine.compute_logprobs_batched(self.benign_msgs)
        kl = _safe_kl_divergence(logprobs, self.baseline_logprobs)
        print(f"  * KL divergence: [bold]{kl:.4f}[/]")
        return kl

    def measure_coherence(self, engine) -> float:
        """Compute how much steered response lengths deviate from baseline.

        Returns the mean absolute z-score of word counts relative to the
        un-steered model.  Values near 0 indicate unchanged fluency; values
        above 2 suggest degenerate repetition or truncation.
        """
        vllm_gen = getattr(engine, "_vllm_gen", None)
        adapter_path = getattr(engine, "_current_adapter_path", None)
        if vllm_gen is not None:
            responses = vllm_gen.generate_text_batched(
                self.benign_msgs,
                skip_special_tokens=True,
                max_new_tokens=self.config.inference.max_gen_tokens,
                min_new_tokens=self.config.inference.min_gen_tokens,
                adapter_path=adapter_path,
            )
        else:
            responses = engine.generate_text_batched(
                self.benign_msgs,
                skip_special_tokens=True,
                max_new_tokens=self.config.inference.max_gen_tokens,
                min_new_tokens=self.config.inference.min_gen_tokens,
            )
        lengths = [len(r.split()) for r in responses]
        if not lengths or self.baseline_stdev_length == 0:
            return 0.0
        current_mean = statistics.mean(lengths)
        return abs(current_mean - self.baseline_mean_length) / max(
            self.baseline_stdev_length,
            1.0,
        )

    def measure_kl_and_coherence(self, engine) -> tuple[float, float]:
        """Compute KL divergence and coherence in one inference pass.

        Combines :meth:`measure_kl_divergence` and :meth:`measure_coherence`
        so that benign_msgs only go through the model once.
        """
        print("  * Obtaining probability distributions and response lengths...")

        vllm_gen = getattr(engine, "_vllm_gen", None)
        adapter_path = getattr(engine, "_current_adapter_path", None)

        if vllm_gen is not None:
            responses, logprobs = vllm_gen.generate_and_score_batched(
                self.benign_msgs,
                max_new_tokens=self.config.inference.max_gen_tokens,
                kl_token_count=self.config.kl.token_count,
                skip_special_tokens=True,
                min_new_tokens=self.config.inference.min_gen_tokens,
                adapter_path=adapter_path,
            )
        else:
            responses, logprobs = engine.generate_and_score_batched(
                self.benign_msgs,
                max_new_tokens=self.config.inference.max_gen_tokens,
                kl_token_count=self.config.kl.token_count,
                skip_special_tokens=True,
                min_new_tokens=self.config.inference.min_gen_tokens,
            )

        if self._use_vllm_continuation_kl(vllm_gen):
            kl = self._measure_vllm_continuation_kl(vllm_gen, adapter_path)
            print(f"  * KL divergence: [bold]{kl:.4f}[/] (continuation NLL)")
        else:
            kl = _safe_kl_divergence(logprobs, self.baseline_logprobs)
            print(f"  * KL divergence: [bold]{kl:.4f}[/]")

        lengths = [len(r.split()) for r in responses]
        if not lengths or self.baseline_stdev_length == 0:
            deviation = 0.0
        else:
            current_mean = statistics.mean(lengths)
            deviation = abs(current_mean - self.baseline_mean_length) / max(
                self.baseline_stdev_length,
                1.0,
            )
        print(f"  * Response length deviation: [bold]{deviation:.2f}[/] std devs")

        return kl, deviation

    def _use_vllm_continuation_kl(self, vllm_gen) -> bool:
        """Use fixed-continuation NLL drift for vLLM in-place edits.

        Sparse top-k sampler KL is known to read as exactly zero on Gemma 4
        vLLM in-place runs even when refusal counts move.  This path keeps the
        ordinary KL estimator for HF/LoRA/SGLang and only swaps the metric for
        the edit mode affected by stale sampler logprobs.
        """
        return bool(
            self._supports_vllm_continuation_kl(vllm_gen)
            and getattr(self, "baseline_continuations", None) is not None
            and getattr(self, "baseline_continuation_nll", None) is not None
        )

    def _supports_vllm_continuation_kl(self, vllm_gen) -> bool:
        return bool(
            vllm_gen is not None
            and getattr(self.config.model, "use_in_place_editing", False)
            and hasattr(vllm_gen, "score_continuations_nll")
        )

    def _measure_vllm_continuation_kl(
        self, vllm_gen, adapter_path: str | None
    ) -> float:
        current_nll = vllm_gen.score_continuations_nll(
            self.benign_msgs,
            self.baseline_continuations,
            adapter_path=adapter_path,
        )
        baseline_nll = self.baseline_continuation_nll.to(current_nll.device)
        drift = torch.mean(torch.abs(current_nll - baseline_nll)).item()
        if torch.isfinite(torch.tensor(drift)):
            return float(drift)
        return float("inf")

    # ------------------------------------------------------------------
    # Multi-objective scoring
    # ------------------------------------------------------------------

    def _compute_objectives(
        self,
        kl_divergence: float,
        detected_refusals: int,
        length_deviation: float = 0.0,
    ) -> tuple[float, float]:
        """Turn raw metrics into a ``(divergence_objective, compliance_objective)`` pair."""
        scale = self.config.kl.scale

        compliance_objective = detected_refusals / self.baseline_refusal_count

        # Default ("independent"): treat KL as an independent objective. Tying
        # the divergence objective to compliance below ``target`` collapses the
        # 2-D Pareto frontier into a single axis whenever steering is
        # conservative (KL < target), causing the TPE sampler to explore blindly
        # once the first low-refusal trial is found. Using KL directly keeps the
        # two objectives independent so the optimizer can learn a real
        # KL-vs-refusals tradeoff curve. (The 2-D front + kl.prune_threshold
        # early-stop already dominate true no-op trials on the compliance axis.)
        #
        # Opt-in ("do_nothing_guard"): replicate Heretic's anti-"do-nothing"
        # trick — while KL < target, drive the divergence objective from the
        # compliance score so the sampler is pushed out of conservative regions.
        if (
            self.config.kl.objective_mode == "do_nothing_guard"
            and kl_divergence < self.config.kl.target
        ):
            divergence_objective = compliance_objective * self.config.kl.target / scale
        else:
            divergence_objective = kl_divergence / scale

        # Penalise degenerate output lengths beyond 2 standard deviations.
        # KL on early generated tokens cannot see long-form drift (model
        # collapses into "帮好帮好…" loops 50 tokens in but the prefix
        # logprobs look fine).  Length deviation catches both shrinkage and
        # bloating.  Threshold at 2σ keeps natural variation uncounted;
        # beyond that, multiply divergence by (1 + 0.1·(dev - 2)) so the
        # penalty scales with the KL's own magnitude rather than swamping
        # it (which an additive penalty did during v3 — it pushed KL
        # 2000× below target by accident).
        if length_deviation > 2.0:
            divergence_objective *= 1.0 + 0.1 * (length_deviation - 2.0)

        return (divergence_objective, compliance_objective)

    def score_trial(self, engine) -> tuple[tuple[float, float], float, int, float]:
        """Evaluate the current steered model and return the multi-objective score.

        Returns
        -------
        objectives : tuple[float, float]
            ``(divergence_objective, compliance_objective)`` to minimise.
        kl_divergence : float
            Raw KL divergence value.
        detected_refusals : int
            Number of target prompts classified as refusals.
        length_deviation : float
            Response-length z-score relative to the baseline.
        """
        kl_divergence, length_deviation = self.measure_kl_and_coherence(engine)

        print("  * Counting model refusals...")
        detected_refusals = self.detector.evaluate_compliance(
            engine,
            self.target_msgs,
        )
        print(f"  * Refusals: [bold]{detected_refusals}[/]/{len(self.target_msgs)}")

        objectives = self._compute_objectives(
            kl_divergence,
            detected_refusals,
            length_deviation,
        )

        return objectives, kl_divergence, detected_refusals, length_deviation
