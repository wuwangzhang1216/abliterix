#!/usr/bin/env python3
"""DeepSeek-V4 FP4 expert dequant helper (used by deploy_dsv4_flash.sh).

abliterix-dequant-fp8 only knows about block-wise FP8 (the non-expert weights
in DSV4-Flash). The expert tensors are FP4 (NVFP4 e2m1 packed in uint8 + ue8m0
power-of-2 scales). This script picks up the FP4 tensors that the FP8 stage
copied through unchanged and rewrites them as BF16 in place.

Strategy (tried in order — first one that works wins):

  1. **Trust the modeling code.** If ``modeling_deepseek_v4.py`` ships a
     ``dequantize_to_bf16`` / ``unpack_fp4`` method on the experts module,
     load the model with ``trust_remote_code=True`` + ``Mxfp4Config(
     dequantize=True)`` (the same path abliterix uses for gpt-oss MXFP4)
     and ``save_pretrained`` to the destination.

  2. **Manual NVFP4 unpack.** Walk every safetensors shard in the source
     dir, find tensors whose dtype is ``uint8`` whose name lives under
     ``*.experts.*``, locate the matching scale tensor, unpack two FP4
     values per byte, multiply by ``2 ** scale_byte``, write BF16 over
     the corresponding shard in the destination dir.

  3. **Bail loudly.** If neither path matches the on-disk layout, print
     a structured dump of every FP4-shaped tensor (shape, dtype, sibling
     scale tensors) and exit non-zero. The dump is enough for a human to
     write a one-line patch the next time DeepSeek changes the FP4 layout.

The first path is far cheaper (no manual bit-twiddling, follows the model
authors' own dequant), so we always attempt it first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _try_modeling_code_dequant(src: Path, dst: Path) -> bool:
    """Strategy 1: ask the modeling code to dequant for us."""
    try:
        from transformers import AutoModelForCausalLM
    except Exception as e:
        print(f"  [modeling-code path] transformers import failed: {e}")
        return False

    extra = {}
    try:
        from transformers import Mxfp4Config

        extra["quantization_config"] = Mxfp4Config(dequantize=True)
    except Exception:
        pass

    try:
        print("  [modeling-code path] loading via trust_remote_code=True...")
        model = AutoModelForCausalLM.from_pretrained(
            str(src),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cpu",  # keep load OOM-safe; we just save again
            low_cpu_mem_usage=True,
            **extra,
        )
    except Exception as e:
        print(f"  [modeling-code path] load failed: {e}")
        return False

    # Quick check: any FP4 / packed tensors still in the state_dict?
    fp4_remaining = 0
    for name, p in model.state_dict().items():
        if p.dtype in (torch.uint8,) and "expert" in name.lower():
            fp4_remaining += 1
    if fp4_remaining:
        print(
            f"  [modeling-code path] WARN: {fp4_remaining} FP4-packed expert "
            "tensors still present after load — modeling code did not unpack."
        )
        return False

    print("  [modeling-code path] saving BF16 model to dst...")
    model.save_pretrained(str(dst), safe_serialization=True)
    return True


# Lookup table for FP4 e2m1 (sign-exp-mantissa). Values per OCP MX spec.
# bits → real number. Matches what NVIDIA NVFP4 / DeepSeek FP4 store.
_FP4_E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _unpack_fp4_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """uint8 packed (low nibble first) → fp32 (2× as many elements)."""
    assert packed.dtype == torch.uint8
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=torch.float32)
    out[..., 0::2] = _FP4_E2M1_LUT[lo.long()]
    out[..., 1::2] = _FP4_E2M1_LUT[hi.long()]
    return out


def _try_manual_nvfp4_unpack(src: Path, dst: Path) -> bool:
    """Strategy 2: walk safetensors and unpack uint8-packed FP4 manually."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = Path(src)
    dst = Path(dst)

    idx_path = src / "model.safetensors.index.json"
    if not idx_path.exists():
        print("  [manual path] no index.json — cannot map shards.")
        return False
    idx = json.loads(idx_path.read_text())
    weight_map: dict[str, str] = idx["weight_map"]

    by_shard: dict[str, list[str]] = {}
    for k, fname in weight_map.items():
        by_shard.setdefault(fname, []).append(k)

    rewrote_any = False
    for fname, keys in by_shard.items():
        with safe_open(src / fname, framework="pt") as f:
            shard_keys = list(f.keys())
            packed_keys = [
                k
                for k in shard_keys
                if "expert" in k.lower() and f.get_slice(k).get_dtype() == "U8"
            ]
            if not packed_keys:
                continue
            print(
                f"  [manual path] shard {fname}: {len(packed_keys)} packed FP4 tensors"
            )

            tensors: dict[str, torch.Tensor] = {}
            for k in shard_keys:
                tensors[k] = f.get_tensor(k)

        # Find scale-paired keys. Convention guess: matching key with one of
        # these suffixes lives next to the packed tensor.
        scale_suffixes = ("_scale", "_scales", "_scale_inv", "_block_scale")

        def _find_scale(k: str) -> str | None:
            for suffix in scale_suffixes:
                cand = k + suffix
                if cand in tensors:
                    return cand
            # Sibling under same module: replace the leaf attr.
            prefix, _, _leaf = k.rpartition(".")
            for suffix in scale_suffixes:
                cand = f"{prefix}.weight{suffix}" if prefix else f"weight{suffix}"
                if cand in tensors:
                    return cand
            return None

        out_tensors: dict[str, torch.Tensor] = dict(tensors)
        unpacked_count = 0
        for k in packed_keys:
            scale_key = _find_scale(k)
            if scale_key is None:
                print(
                    f"    {k}: no scale tensor found — leaving packed (will fail at load)"
                )
                continue
            packed = tensors[k]
            scale = tensors[scale_key]
            # NVFP4: scale is uint8 ue8m0, real_scale = 2.0 ** (scale - 127).
            if scale.dtype == torch.uint8:
                exp = scale.to(torch.int32) - 127
                real_scale = torch.pow(2.0, exp.to(torch.float32))
            else:
                real_scale = scale.to(torch.float32)

            unpacked = _unpack_fp4_e2m1(packed)  # (..., 2*last_dim)
            # Broadcast scale across the unpacked tile. We don't know the
            # tile geometry without the modeling code, so try the most
            # common: scale shape (E, R/B, C/B) for fused (E, R, C/2) packed.
            try:
                if unpacked.dim() == 3 and real_scale.dim() == 3:
                    block_r = max(1, unpacked.shape[1] // real_scale.shape[1])
                    block_c = max(1, unpacked.shape[2] // real_scale.shape[2])
                    s_exp = real_scale.repeat_interleave(
                        block_r, dim=1
                    ).repeat_interleave(block_c, dim=2)
                    s_exp = s_exp[
                        : unpacked.shape[0], : unpacked.shape[1], : unpacked.shape[2]
                    ]
                    unpacked = unpacked * s_exp
                elif unpacked.dim() == 2 and real_scale.dim() == 2:
                    block_r = max(1, unpacked.shape[0] // real_scale.shape[0])
                    block_c = max(1, unpacked.shape[1] // real_scale.shape[1])
                    s_exp = real_scale.repeat_interleave(
                        block_r, dim=0
                    ).repeat_interleave(block_c, dim=1)
                    s_exp = s_exp[: unpacked.shape[0], : unpacked.shape[1]]
                    unpacked = unpacked * s_exp
                else:
                    print(
                        f"    {k}: shape mismatch (weight {tuple(unpacked.shape)} "
                        f"vs scale {tuple(real_scale.shape)}) — skipping"
                    )
                    continue
            except Exception as e:
                print(f"    {k}: scale broadcast failed: {e} — skipping")
                continue

            out_tensors[k] = unpacked.to(torch.bfloat16)
            out_tensors.pop(scale_key, None)
            unpacked_count += 1

        if unpacked_count:
            save_file(out_tensors, str(dst / fname), metadata={"format": "pt"})
            rewrote_any = True
            print(f"    rewrote {fname} ({unpacked_count} tensors unpacked)")

    return rewrote_any


def _diagnostic_dump(src: Path) -> None:
    """Strategy 3: print every candidate FP4 tensor for debugging."""
    from safetensors import safe_open

    print("\n=== FP4 expert tensor diagnostic dump ===")
    for shard in sorted(src.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                slc = f.get_slice(k)
                dt = slc.get_dtype()
                if "expert" in k.lower() or dt in ("U8", "F4_E2M1"):
                    print(
                        f"  {shard.name}::{k}  dtype={dt}  shape={list(slc.get_shape())}"
                    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument(
        "--diagnostic",
        action="store_true",
        help="Skip dequant; dump FP4 tensor inventory and exit.",
    )
    args = p.parse_args()

    if args.diagnostic:
        _diagnostic_dump(args.src)
        return 0

    print(f"== src: {args.src}")
    print(f"== dst: {args.dst}")
    args.dst.mkdir(parents=True, exist_ok=True)

    print("== Strategy 1: trust the modeling code")
    if _try_modeling_code_dequant(args.src, args.dst):
        print("== done (strategy 1)")
        return 0

    print("== Strategy 2: manual NVFP4 unpack on the safetensors shards")
    if _try_manual_nvfp4_unpack(args.src, args.dst):
        print("== done (strategy 2)")
        return 0

    print("== Both strategies failed; dumping diagnostics")
    _diagnostic_dump(args.src)
    print(
        "\nFP4 layout did not match NVFP4 e2m1 packed-uint8. "
        "Check the on-disk dtype & key names above, then patch "
        "_try_manual_nvfp4_unpack to match the actual layout."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
