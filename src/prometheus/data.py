# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

import os
from importlib.metadata import version
from pathlib import Path

from datasets import DatasetDict, ReadInstruction, load_dataset, load_from_disk
from datasets.config import DATASET_STATE_JSON_FILENAME
from datasets.download.download_manager import DownloadMode
from datasets.utils.info_utils import VerificationMode
from optuna import Trial

from .settings import PrometheusConfig
from .types import ChatMessage, PromptSource


def load_prompt_dataset(
    config: PrometheusConfig,
    source: PromptSource,
) -> list[ChatMessage]:
    """Load a prompt dataset and wrap every entry in a ChatMessage.

    Handles three dataset flavours:
    1. Local directory saved via ``datasets.save_to_disk``
    2. Arbitrary local directory readable by ``load_dataset``
    3. Hugging Face Hub repository
    """
    path = source.dataset
    split_expr = source.split

    if os.path.isdir(path):
        if Path(path, DATASET_STATE_JSON_FILENAME).exists():
            dataset = load_from_disk(path)
            assert not isinstance(dataset, DatasetDict), (
                "Loading dataset dicts is not supported"
            )
            instruction = ReadInstruction.from_spec(split_expr)
            name2len = {str(dataset.split): len(dataset)}
            absolute = instruction.to_absolute(name2len)[0]
            dataset = dataset[absolute.from_ : absolute.to]
        else:
            dataset = load_dataset(
                path,
                split=split_expr,
                verification_mode=VerificationMode.NO_CHECKS,
                download_mode=DownloadMode.FORCE_REDOWNLOAD,
            )
    else:
        dataset = load_dataset(path, split=split_expr)

    texts = list(dataset[source.column])

    if source.prefix:
        texts = [f"{source.prefix} {t}" for t in texts]
    if source.suffix:
        texts = [f"{t} {source.suffix}" for t in texts]

    sys_prompt = (
        config.system_prompt if source.system_prompt is None else source.system_prompt
    )

    return [ChatMessage(system=sys_prompt, user=t) for t in texts]


# ---------------------------------------------------------------------------
# Trial formatting helpers
# ---------------------------------------------------------------------------


def format_trial_params(trial: Trial) -> dict[str, str]:
    """Extract human-readable parameter strings from a completed trial."""
    params = {}

    vec_idx = trial.user_attrs["vector_index"]
    params["vector_index"] = "per layer" if vec_idx is None else f"{vec_idx:.2f}"

    for component, values in trial.user_attrs["parameters"].items():
        for name, value in values.items():
            params[f"{component}.{name}"] = f"{value:.2f}"

    return params


def generate_model_card(
    config: PrometheusConfig,
    trial: Trial,
    baseline_refusal_count: int,
    target_prompts: list[ChatMessage],
) -> str:
    """Produce a Markdown model-card header for an exported model."""
    model_id = config.model.model_id
    if Path(model_id).exists():
        model_link = "a model"
    else:
        model_link = f"[{model_id}](https://huggingface.co/{model_id})"

    return f"""# This is a decensored version of {
        model_link
    }, made using [Prometheus](https://github.com/wuwangzhang1216/prometheus) v{
        version("prometheus-llm")
    }

## Steering parameters

| Parameter | Value |
| :-------- | :---: |
{
        chr(10).join(
            [
                f"| **{name}** | {value} |"
                for name, value in format_trial_params(trial).items()
            ]
        )
    }

## Performance

| Metric | This model | Original model ({model_link}) |
| :----- | :--------: | :---------------------------: |
| **KL divergence** | {trial.user_attrs["kl_divergence"]:.4f} | 0 *(by definition)* |
| **Refusals** | {trial.user_attrs["refusals"]}/{len(target_prompts)} | {
        baseline_refusal_count
    }/{len(target_prompts)} |

-----

"""
