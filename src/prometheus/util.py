# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

import gc
import getpass
import os
from typing import Any, TypeVar

import questionary
import torch
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_sdaa_available,
    is_xpu_available,
)
from psutil import Process
from questionary import Choice, Style
from rich.console import Console

print = Console(highlight=False).print


def report_memory():
    """Print a summary of resident RAM and accelerator memory usage."""

    def _line(label: str, size_bytes: int):
        print(f"[grey50]{label}: [bold]{size_bytes / (1024**3):.2f} GB[/][/]")

    _line("Resident system RAM", Process().memory_info().rss)

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        allocated = sum(torch.cuda.memory_allocated(d) for d in range(n))
        reserved = sum(torch.cuda.memory_reserved(d) for d in range(n))
        _line("Allocated GPU VRAM", allocated)
        _line("Reserved GPU VRAM", reserved)
    elif is_xpu_available():
        n = torch.xpu.device_count()
        allocated = sum(torch.xpu.memory_allocated(d) for d in range(n))
        reserved = sum(torch.xpu.memory_reserved(d) for d in range(n))
        _line("Allocated XPU memory", allocated)
        _line("Reserved XPU memory", reserved)
    elif torch.backends.mps.is_available():
        _line("Allocated MPS memory", torch.mps.current_allocated_memory())
        _line("Driver (reserved) MPS memory", torch.mps.driver_allocated_memory())


def running_in_notebook() -> bool:
    """Return True when executing inside a Jupyter-like environment."""
    if os.getenv("COLAB_GPU") or os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return True

    try:
        from IPython import get_ipython  # ty:ignore[unresolved-import]

        shell = get_ipython()
        if shell is None:
            return False

        name = shell.__class__.__name__
        if name in ["ZMQInteractiveShell", "Shell"]:
            return True
        if "google.colab" in str(shell.__class__):
            return True

        return False
    except (ImportError, NameError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# Interactive prompts (notebook-safe wrappers around questionary)
# ---------------------------------------------------------------------------


def ask_choice(message: str, choices: list[Any]) -> Any:
    if running_in_notebook():
        print()
        print(message)
        real = []
        for i, c in enumerate(choices, 1):
            if isinstance(c, Choice):
                print(f"[{i}] {c.title}")
                real.append(c.value)
            else:
                print(f"[{i}] {c}")
                real.append(c)
        while True:
            try:
                idx = int(input("Enter number: ")) - 1
                if 0 <= idx < len(real):
                    return real[idx]
                print(f"[red]Please enter a number between 1 and {len(real)}[/]")
            except ValueError:
                print("[red]Invalid input. Please enter a number.[/]")
    else:
        return questionary.select(
            message,
            choices=choices,
            style=Style([("highlighted", "reverse")]),
        ).ask()


def ask_text(
    message: str,
    default: str = "",
    qmark: str = "?",
    unsafe: bool = False,
) -> str:
    if running_in_notebook():
        print()
        result = input(f"{message} [{default}]: " if default else f"{message}: ")
        return result if result else default
    else:
        q = questionary.text(message, default=default, qmark=qmark)
        return q.unsafe_ask() if unsafe else q.ask()


def ask_path(message: str) -> str:
    if running_in_notebook():
        return ask_text(message)
    else:
        return questionary.path(message, only_directories=True).ask()


def ask_secret(message: str) -> str:
    if running_in_notebook():
        print()
        return getpass.getpass(message)
    else:
        return questionary.password(message).ask()


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def humanize_duration(seconds: float) -> str:
    """Format a duration as a compact human-readable string."""
    seconds = round(seconds)
    h, seconds = divmod(seconds, 3600)
    m, s = divmod(seconds, 60)
    if h > 0:
        return f"{h}h {m}m"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


T = TypeVar("T")


def chunk_batches(items: list[T], batch_size: int) -> list[list[T]]:
    """Split *items* into consecutive sub-lists of at most *batch_size*."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def flush_memory():
    """Release cached GPU / accelerator memory and run garbage collection.

    gc.collect() is called both before and after clearing the backend cache
    because Python's cycle collector must break reference loops before the
    backend allocator can actually reclaim the underlying buffers.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif is_xpu_available():
        torch.xpu.empty_cache()
    elif is_mlu_available():
        torch.mlu.empty_cache()  # ty:ignore[unresolved-attribute]
    elif is_sdaa_available():
        torch.sdaa.empty_cache()  # ty:ignore[unresolved-attribute]
    elif is_musa_available():
        torch.musa.empty_cache()  # ty:ignore[unresolved-attribute]
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    gc.collect()


def slugify_model_name(model_name: str) -> str:
    """Turn a model identifier into a filesystem-safe slug."""
    return "".join(c if (c.isalnum() or c in ["_", "-"]) else "--" for c in model_name)
