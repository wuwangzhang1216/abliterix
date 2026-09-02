# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Abliterix — Automated model steering and alignment adjustment via LoRA-based optimization."""

import torch
import torch.utils._pytree as _pytree

if not hasattr(_pytree, "register_constant"):

    def _compat_register_constant(cls):
        _reg = getattr(
            _pytree,
            "register_pytree_node",
            getattr(_pytree, "_register_pytree_node", None),
        )
        if _reg is not None:
            return _reg(cls, lambda x: ((), x), lambda children, context: context)
        return cls

    _pytree.register_constant = _compat_register_constant  # type: ignore[attr-defined]

from .core.engine import SteeringEngine
from .eval.detector import RefusalDetector
from .eval.scorer import TrialScorer
from .settings import AbliterixConfig
from .types import (
    ChatMessage,
    DecayKernel,
    ExpertRoutingConfig,
    QuantMode,
    SteeringProfile,
    VectorMethod,
    WeightNorm,
)

__all__ = [
    "ChatMessage",
    "DecayKernel",
    "ExpertRoutingConfig",
    "AbliterixConfig",
    "QuantMode",
    "RefusalDetector",
    "SteeringEngine",
    "SteeringProfile",
    "TrialScorer",
    "VectorMethod",
    "WeightNorm",
]
