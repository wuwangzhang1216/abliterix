# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

"""Trial scoring: KL divergence, coherence measurement, and multi-objective evaluation.

The :class:`TrialScorer` orchestrates baseline capture during init and then
provides :meth:`score_trial` to evaluate each Optuna trial.
"""

import statistics

import torch.nn.functional as F
from torch import Tensor

from ..data import load_prompt_dataset
from ..settings import PrometheusConfig
from ..types import ChatMessage
from ..util import print
from .detector import RefusalDetector


class TrialScorer:
    """Measures model damage (KL divergence, coherence) and compliance.

    On construction the scorer records baseline logprobs, response lengths,
    and refusal counts against the un-steered model.  Each call to
    :meth:`score_trial` then returns a multi-objective tuple that Optuna
    minimises.
    """

    config: PrometheusConfig
    detector: RefusalDetector
    benign_msgs: list[ChatMessage]
    target_msgs: list[ChatMessage]
    baseline_logprobs: Tensor
    baseline_refusal_count: int
    baseline_mean_length: float
    baseline_stdev_length: float

    def __init__(self, config: PrometheusConfig, engine, detector: RefusalDetector):
        self.config = config
        self.detector = detector

        print()
        print(
            f"Loading benign evaluation prompts from [bold]{config.benign_eval_prompts.dataset}[/]..."
        )
        self.benign_msgs = load_prompt_dataset(config, config.benign_eval_prompts)
        print(f"* [bold]{len(self.benign_msgs)}[/] prompts loaded")

        # Capture baseline logprobs and response lengths in a single pass
        # to avoid duplicating the prefill cost of benign_msgs.
        print("* Obtaining probability distributions and baseline response lengths...")
        base_responses, self.baseline_logprobs = engine.generate_and_score_batched(
            self.benign_msgs,
            max_new_tokens=config.inference.max_gen_tokens,
            kl_token_count=config.kl.token_count,
            skip_special_tokens=True,
        )
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

        print()
        print(
            f"Loading target evaluation prompts from [bold]{config.target_eval_prompts.dataset}[/]..."
        )
        self.target_msgs = load_prompt_dataset(config, config.target_eval_prompts)
        print(f"* [bold]{len(self.target_msgs)}[/] prompts loaded")

        print("* Counting model refusals...")
        self.baseline_refusal_count = detector.evaluate_compliance(
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
        logprobs = engine.compute_logprobs_batched(self.benign_msgs)
        kl = F.kl_div(
            logprobs,
            self.baseline_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
        print(f"  * KL divergence: [bold]{kl:.4f}[/]")
        return kl

    def measure_coherence(self, engine) -> float:
        """Compute how much steered response lengths deviate from baseline.

        Returns the mean absolute z-score of word counts relative to the
        un-steered model.  Values near 0 indicate unchanged fluency; values
        above 2 suggest degenerate repetition or truncation.
        """
        responses = engine.generate_text_batched(
            self.benign_msgs,
            skip_special_tokens=True,
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
        responses, logprobs = engine.generate_and_score_batched(
            self.benign_msgs,
            max_new_tokens=self.config.inference.max_gen_tokens,
            kl_token_count=self.config.kl.token_count,
            skip_special_tokens=True,
        )

        kl = F.kl_div(
            logprobs,
            self.baseline_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
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
        target = self.config.kl.target

        compliance_objective = detected_refusals / self.baseline_refusal_count

        if kl_divergence >= target:
            divergence_objective = kl_divergence / scale
        else:
            divergence_objective = compliance_objective * target / scale

        # Penalise degenerate output lengths: if the mean response length drifts
        # beyond 2 standard deviations, ramp up the divergence objective.
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
