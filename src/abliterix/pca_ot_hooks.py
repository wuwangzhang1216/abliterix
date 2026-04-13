# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""PCA-OT affine transformation hooks for inference-time intervention.

This module provides forward hooks that apply full affine transformations
h' = A_full @ h + b_full to model activations, implementing the PCA-OT
method from arxiv:2603.04355.
"""

from typing import Any

import torch
from torch import Tensor, nn

from .types import PCAOTTransform


class PCAOTHookManager:
    """Manages forward hooks for PCA-OT affine transformations.

    This class registers forward hooks on specified transformer layers that
    apply the full affine transformation T(x) = A_full @ x + b_full.

    Example
    -------
    >>> from abliterix.vectors import compute_pca_ot_transforms
    >>> transforms = compute_pca_ot_transforms(benign_states, harmful_states)
    >>> hook_manager = PCAOTHookManager(model, transforms, layer_indices=[10, 11])
    >>> hook_manager.register_hooks()
    >>> # ... run inference ...
    >>> hook_manager.remove_hooks()
    """

    def __init__(
        self,
        model: nn.Module,
        transforms: list[PCAOTTransform],
        layer_indices: list[int] | None = None,
        norm_preserving: bool = False,
    ):
        """Initialize the hook manager.

        Parameters
        ----------
        model : nn.Module
            The transformer model to apply hooks to.
        transforms : list[PCAOTTransform]
            PCA-OT transforms for each layer (from compute_pca_ot_transforms).
        layer_indices : list[int], optional
            Specific layer indices to apply transforms to. If None, applies to
            all layers. Paper recommends 1-2 layers at 40-60% depth.
        norm_preserving : bool, default False
            If True, scale the transformed activation to preserve its norm.
            This reduces distribution shift.
        """
        self.model = model
        self.transforms = {t.layer_idx: t for t in transforms}
        self.layer_indices = layer_indices
        self.norm_preserving = norm_preserving
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._layer_modules: dict[int, nn.Module] = {}

    def _find_layer_modules(self) -> dict[int, nn.Module]:
        """Find transformer layer modules in the model.

        Returns
        -------
        dict[int, nn.Module]
            Mapping from layer index to layer module.
        """
        # Try common transformer layer paths
        layer_paths = [
            "model.layers",  # Llama, Mistral, Qwen
            "transformer.h",  # GPT-2
            "transformer.layers",  # Generic
            "encoder.layer",  # BERT-style
        ]

        for path in layer_paths:
            try:
                parts = path.split(".")
                module = self.model
                for part in parts:
                    module = getattr(module, part)
                # Found it - return indexed layers
                return {i: layer for i, layer in enumerate(module)}
            except AttributeError:
                continue

        raise ValueError(
            "Could not find transformer layers in model. "
            "Supported architectures: Llama, Mistral, Qwen, GPT-2, BERT. "
            "Please file an issue if your model is not supported."
        )

    def _create_hook(self, layer_idx: int, transform: PCAOTTransform) -> callable:
        """Create a forward hook function for a specific layer.

        Parameters
        ----------
        layer_idx : int
            The transformer layer index.
        transform : PCAOTTransform
            The affine transformation to apply.

        Returns
        -------
        callable
            Hook function with signature (module, input, output) -> output.
        """
        # Get device from model parameters
        device = next(self.model.parameters()).device
        A_full = transform.A_full.to(device)
        b_full = transform.b_full.to(device)

        def hook_fn(module: nn.Module, input: tuple, output: Any) -> Any:
            """Apply affine transformation to layer output.

            The output can be:
            - A tensor (hidden_states)
            - A tuple (hidden_states, *other)
            - A dict-like object with 'hidden_states' or similar

            We transform the hidden states and return in the same format.
            """
            # Extract hidden states from output
            if isinstance(output, Tensor):
                hidden_states = output
                return_tuple = False
                return_dict = False
            elif isinstance(output, tuple):
                hidden_states = output[0]
                other_outputs = output[1:]
                return_tuple = True
                return_dict = False
            elif hasattr(output, "last_hidden_state"):
                # HuggingFace BaseModelOutput
                hidden_states = output.last_hidden_state
                return_tuple = False
                return_dict = True
                output_obj = output
            else:
                # Unknown format - skip transformation
                return output

            # Apply affine transformation: h' = A_full @ h + b_full
            # hidden_states shape: (batch, seq_len, hidden_dim)
            original_shape = hidden_states.shape
            original_dtype = hidden_states.dtype

            # Flatten batch and sequence dimensions
            h = hidden_states.view(-1, hidden_states.size(-1)).float()

            # Apply transformation
            h_transformed = h @ A_full.T + b_full  # (batch*seq, dim)

            # Norm-preserving scaling (optional)
            if self.norm_preserving:
                original_norm = h.norm(dim=1, keepdim=True)
                transformed_norm = h_transformed.norm(dim=1, keepdim=True)
                scale = original_norm / (transformed_norm + 1e-8)
                h_transformed = h_transformed * scale

            # Reshape back
            h_transformed = h_transformed.view(original_shape).to(original_dtype)

            # Return in original format
            if return_tuple:
                return (h_transformed,) + other_outputs
            elif return_dict:
                output_obj.last_hidden_state = h_transformed
                return output_obj
            else:
                return h_transformed

        return hook_fn

    def register_hooks(self) -> None:
        """Register forward hooks on the specified layers.

        This applies the PCA-OT affine transformations during inference.
        Call remove_hooks() when done to clean up.
        """
        if self.hooks:
            raise RuntimeError("Hooks already registered. Call remove_hooks() first.")

        # Find layer modules if not already cached
        if not self._layer_modules:
            self._layer_modules = self._find_layer_modules()

        # Determine which layers to hook
        if self.layer_indices is None:
            # Apply to all layers that have transforms
            layers_to_hook = sorted(self.transforms.keys())
        else:
            # Apply only to specified layers
            layers_to_hook = self.layer_indices

        # Register hooks
        for layer_idx in layers_to_hook:
            if layer_idx not in self.transforms:
                raise ValueError(
                    f"No transform available for layer {layer_idx}. "
                    f"Available layers: {sorted(self.transforms.keys())}"
                )

            if layer_idx not in self._layer_modules:
                raise ValueError(
                    f"Layer {layer_idx} not found in model. "
                    f"Available layers: {sorted(self._layer_modules.keys())}"
                )

            transform = self.transforms[layer_idx]
            layer_module = self._layer_modules[layer_idx]

            hook_fn = self._create_hook(layer_idx, transform)
            handle = layer_module.register_forward_hook(hook_fn)
            self.hooks.append(handle)

        print(
            f"✓ Registered PCA-OT hooks on {len(self.hooks)} layers: {layers_to_hook}"
        )

    def remove_hooks(self) -> None:
        """Remove all registered hooks.

        Call this to restore the model to its original behavior.
        """
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        print("✓ Removed all PCA-OT hooks")

    def __enter__(self):
        """Context manager entry - register hooks if not already registered."""
        if not self.hooks:
            self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - remove hooks."""
        self.remove_hooks()
        return False


def apply_pca_ot_hooks(
    model: nn.Module,
    transforms: list[PCAOTTransform],
    layer_indices: list[int] | None = None,
    norm_preserving: bool = False,
) -> PCAOTHookManager:
    """Convenience function to create and register PCA-OT hooks.

    Parameters
    ----------
    model : nn.Module
        The transformer model.
    transforms : list[PCAOTTransform]
        PCA-OT transforms from compute_pca_ot_transforms().
    layer_indices : list[int], optional
        Specific layers to apply transforms to. If None, applies to all.
        Paper recommends 1-2 layers at 40-60% depth.
    norm_preserving : bool, default False
        If True, preserve activation norms to reduce distribution shift.

    Returns
    -------
    PCAOTHookManager
        Hook manager with hooks already registered. Call .remove_hooks()
        when done, or use as a context manager.

    Example
    -------
    >>> hook_mgr = apply_pca_ot_hooks(model, transforms, layer_indices=[15])
    >>> outputs = model.generate(...)
    >>> hook_mgr.remove_hooks()

    Or as a context manager:
    >>> with apply_pca_ot_hooks(model, transforms, layer_indices=[15]):
    ...     outputs = model.generate(...)
    """
    manager = PCAOTHookManager(model, transforms, layer_indices, norm_preserving)
    manager.register_hooks()
    return manager
