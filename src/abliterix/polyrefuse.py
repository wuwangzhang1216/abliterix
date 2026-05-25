# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""PolyRefuse cross-lingual evaluation harness.

Operationalises the finding from `Wang et al., 2025
<https://arxiv.org/abs/2505.17306>`_ — *Refusal Direction is Universal
Across Safety-Aligned Languages*: an English refusal vector ablates
refusals in 14+ languages near-perfectly, without any per-language
fine-tuning or recalibration.

Since the *extraction* path is identical to standard mean-diff (just
train on English harmful/benign), PolyRefuse in abliterix is purely an
**evaluation** feature — a regression harness that confirms the cross-
lingual transfer worked. Drop a per-language ``PromptSource`` into
``polyrefuse_eval.languages`` and abliterix will measure refusal rate
separately per language at the end of optimisation.

Per-language scoring uses the same :class:`RefusalDetector` as the main
loop (LLM judge or keyword fallback), so the multilingual numbers are
directly comparable to the headline English refusal rate.

Bundled prompt sets are intentionally not shipped: the literature
already has well-vetted translated harmful-prompt benchmarks (PolyBench,
multilingual JailbreakBench, etc.) and content sensitivity argues for
leaving the dataset choice with the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LanguageResult:
    language: str
    n_total: int
    n_refused: int
    refusal_rate: float
    sample_responses: list[str]


def _classify_refusals(
    detector,
    responses: list[str],
) -> int:
    """Count refusals in a batch of responses using the project's detector.

    Falls back to a coarse keyword check if the detector doesn't expose a
    batch-classification API — the unit tests stub a minimal detector
    that returns booleans, the real RefusalDetector exposes
    ``classify_batch``.
    """
    if hasattr(detector, "classify_batch"):
        return sum(1 for is_refusal in detector.classify_batch(responses) if is_refusal)
    return sum(
        1
        for r in responses
        if hasattr(detector, "is_refusal") and detector.is_refusal(r)
    )


def evaluate_per_language(
    generator,
    detector,
    prompt_groups: dict[str, list[Any]],
    *,
    sample_responses: int = 3,
) -> dict[str, LanguageResult]:
    """Run a refusal-rate sweep across multiple languages.

    Parameters
    ----------
    generator : callable
        Takes a list of chat messages and returns the generated responses
        as a list of strings.  In production this is
        ``lambda msgs: engine.generate(msgs)``; in tests it's a stub.
    detector : RefusalDetector-like
        Must expose ``classify_batch(responses) -> Iterable[bool]`` or
        ``is_refusal(response) -> bool``.
    prompt_groups : dict[str, list[ChatMessage]]
        Keyed by ISO 639-1 language code (e.g. ``"en"``, ``"zh"``,
        ``"es"``).  Each value is the list of prompts to evaluate in
        that language.
    sample_responses : int
        How many sample responses to keep per language for the report.
        Capped at the prompt-group size.

    Returns
    -------
    dict[str, LanguageResult]
        Keyed by the same language codes.  Sorted-ascending insertion
        order matches the caller's dict iteration order.
    """
    results: dict[str, LanguageResult] = {}
    for lang, msgs in prompt_groups.items():
        if not msgs:
            results[lang] = LanguageResult(
                language=lang,
                n_total=0,
                n_refused=0,
                refusal_rate=0.0,
                sample_responses=[],
            )
            continue
        responses = generator(msgs)
        n_refused = _classify_refusals(detector, responses)
        n_total = len(responses)
        rate = n_refused / n_total if n_total else 0.0
        results[lang] = LanguageResult(
            language=lang,
            n_total=n_total,
            n_refused=n_refused,
            refusal_rate=rate,
            sample_responses=list(responses[: max(0, sample_responses)]),
        )
    return results


def format_report(results: dict[str, LanguageResult]) -> str:
    """Pretty-print a PolyRefuse evaluation report.

    Format::

        Language   n_total  refused  refusal_rate
        en         100      6         6.0%
        zh         100      9         9.0%
        ...

    """
    if not results:
        return "(no languages evaluated)"
    header = f"{'Language':<10} {'n_total':>7}  {'refused':>7}  {'refusal_rate':>12}"
    sep = "-" * len(header)
    lines = [header, sep]
    for lang, res in results.items():
        rate_str = f"{res.refusal_rate * 100:.1f}%"
        lines.append(f"{lang:<10} {res.n_total:>7}  {res.n_refused:>7}  {rate_str:>12}")
    return "\n".join(lines)


def summarise_transfer(results: dict[str, LanguageResult]) -> dict[str, float]:
    """Aggregate cross-lingual transfer stats from per-language results.

    Returns a dict with:
    * ``mean_refusal_rate`` — averaged across languages (equal weight)
    * ``max_refusal_rate`` / ``min_refusal_rate`` — worst- and best-case
    * ``english_refusal_rate`` — the ``"en"`` baseline if present, else NaN
    * ``transfer_gap`` — ``max − english`` (positive = some languages
      didn't transfer; ideally ≤ 5 pp per the paper)
    """
    if not results:
        return {}
    rates = [r.refusal_rate for r in results.values() if r.n_total > 0]
    if not rates:
        return {}
    eng = results.get("en")
    eng_rate = eng.refusal_rate if eng and eng.n_total > 0 else float("nan")
    max_rate = max(rates)
    return {
        "mean_refusal_rate": sum(rates) / len(rates),
        "max_refusal_rate": max_rate,
        "min_refusal_rate": min(rates),
        "english_refusal_rate": eng_rate,
        "transfer_gap": (max_rate - eng_rate) if eng_rate == eng_rate else float("nan"),
    }
