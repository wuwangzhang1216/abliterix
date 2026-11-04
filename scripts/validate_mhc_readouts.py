"""Check native attention reads using small ranges from two pinned checkpoints.

Run with Transformers 5.20.2 and PyTorch installed, for example:
    python scripts/validate_mhc_readouts.py --output docs/research/mhc-readout-probe.json

The probe evaluates published mHC and RMSNorm modules on CPU. It never builds
a full model. Layer zero uses checkpoint embedding rows. The midpoint uses
seeded, synthetic streams so that its input needs no earlier decoder layers.
"""

import argparse
import hashlib
import inspect
import json
import math
import platform
import re
import struct
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import Request, urlopen

import torch
import transformers


# Locations from each pinned checkpoint's model.safetensors.index.json.
CHECKPOINTS = {
    "deepseek_v4": {
        "repo": "deepseek-ai/DeepSeek-V4-Flash",
        "revision": "60d8d70770c6776ff598c94bb586a859a38244f1",
        "embedding": "embed.weight",
        "layer_prefix": "layers",
        "norm": "attn_norm.weight",
        "shards": {"embedding": 1, "first": 2, "midpoint": 23},
        "shard_count": 46,
    },
    "glm5_next": {
        "repo": "zai-org/GLM-5.3-Flash",
        "revision": "46a050bec13002ad29ee795c16e61cf4fa3c4a9c",
        "embedding": "model.language_model.embed_tokens.weight",
        "layer_prefix": "model.language_model.layers",
        "norm": "input_layernorm.weight",
        "shards": {"embedding": 1, "first": 2, "midpoint": 19},
        "shard_count": 62,
    },
}
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
SEED = 20261102
TOKEN_IDS = [17, 18, 19]
ATOL, RTOL = 2e-6, 2e-5


class Downloads:
    """Record verified response bytes and enforce one budget across both repos."""

    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.bytes_read = 0
        self.records = []

    def read(self, spec, filename, purpose, bounds=None):
        url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{filename}"
        headers = {"Accept-Encoding": "identity", "User-Agent": "abliterix-mhc-probe/1"}
        limit = 150_000  # config.json only
        suffixes = ["?download=true", ""]
        if bounds is not None:
            start, end = bounds
            limit = end - start + 1
            if start < 0 or limit <= 0 or limit > 4_000_000:
                raise ValueError(f"Invalid or oversized range for {purpose}: {bounds}")
            headers["Range"] = f"bytes={start}-{end}"
            # Distinct URLs avoid proxy caches serving a previous range.
            suffixes = [
                f"?download=true&probe_range={start}-{end}",
                f"?range={start}-{end}",
            ]
        if limit > self.max_bytes - self.bytes_read:
            raise ValueError("Requested response would exceed the download budget")
        for attempt, suffix in enumerate(suffixes):
            try:
                with urlopen(
                    Request(url + suffix, headers=headers), timeout=25
                ) as response:
                    length = response.headers.get("Content-Length")
                    revision = response.headers.get("x-repo-commit")
                    if revision and revision != spec["revision"]:
                        raise ValueError(
                            "Hub returned an unexpected checkpoint revision"
                        )
                    if bounds is not None:
                        matched = re.fullmatch(
                            r"bytes (\d+)-(\d+)/(\d+)",
                            response.headers.get("Content-Range", ""),
                        )
                        if response.status != 206 or matched is None:
                            raise ValueError(
                                f"Server did not confirm requested byte range for {filename}"
                            )
                        received_start, received_end, total = map(int, matched.groups())
                        if (received_start, received_end) != bounds or end >= total:
                            raise ValueError(f"Wrong byte range for {filename}")
                        if length is not None and int(length) != limit:
                            raise ValueError("Range response length mismatch")
                    elif response.status != 200 or (
                        length is not None and int(length) > limit
                    ):
                        raise ValueError("Unexpected config.json response")
                    data = response.read(limit)
                    self.bytes_read += len(data)
                    if bounds is not None and len(data) != limit:
                        raise ValueError("Truncated byte range")
                    if (
                        bounds is None
                        and length is not None
                        and len(data) != int(length)
                    ):
                        raise ValueError("Truncated config.json")
                break
            except URLError:
                if attempt == len(suffixes) - 1:
                    raise
        record = {
            "repo": spec["repo"],
            "revision": spec["revision"],
            "file": filename,
            "purpose": purpose,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        if bounds is not None:
            record.update(byte_range_inclusive=list(bounds), shard_bytes=total)
        self.records.append(record)
        return data


class Shard:
    def __init__(self, downloads, spec, role):
        self.downloads, self.spec = downloads, spec
        self.name = (
            f"model-{spec['shards'][role]:05d}-of-{spec['shard_count']:05d}.safetensors"
        )

        def read(purpose, start, end):
            return downloads.read(spec, self.name, purpose, (start, end))

        header_bytes = struct.unpack("<Q", read("header_length", 0, 7))[0]
        self.header = json.loads(read("safetensors_header", 8, 7 + header_bytes))
        self.data_start = 8 + header_bytes

    def tensor(self, name, rows=None):
        info = self.header[name]
        shape = list(info["shape"])
        dtype = DTYPES[info["dtype"]]
        start, stop = info["data_offsets"]
        item_bytes = torch.empty((), dtype=dtype).element_size()
        if stop - start != math.prod(shape) * item_bytes:
            raise ValueError(f"Header shape/length mismatch for {name}")
        if rows is not None:
            first, last = rows
            if len(shape) != 2 or not 0 <= first <= last < shape[0]:
                raise ValueError(f"Invalid embedding row selection for {name}")
            row_bytes = shape[1] * item_bytes
            stop = start + (last + 1) * row_bytes
            start += first * row_bytes
            shape[0] = last - first + 1
        data = self.downloads.read(
            self.spec,
            self.name,
            name,
            (self.data_start + start, self.data_start + stop - 1),
        )
        self.downloads.records[-1].update(
            storage_dtype=info["dtype"], selected_shape=shape
        )
        return torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape).float()


def differences(actual, reference):
    a, b = actual.double(), reference.double()
    return {
        "max_abs": (a - b).abs().max().item(),
        "relative_l2": ((a - b).norm() / b.norm().clamp_min(1e-30)).item(),
        "mean_token_cosine": torch.nn.functional.cosine_similarity(a, b, dim=-1)
        .mean()
        .item(),
    }


def compare(gate, norm, weights, norm_weight, x, config):
    # Upstream PyTorch modules perform the reference checkpoint's native
    # attention read and subsequent normalization. No model edit is applied.
    gate.load_state_dict(weights)
    norm.load_state_dict({"weight": norm_weight})
    native = gate.eval()(x)[2]
    normalized = norm.eval()(native)

    # Independent FP64 computation uses only the pre-read rows of the gate
    # and contracts over streams explicitly. It does not call either module.
    streams = config.hc_mult
    flat = x.double().flatten(-2)
    unit = (
        flat
        / (
            (flat.square().sum(-1, keepdim=True) / flat.shape[-1]) + config.rms_norm_eps
        ).sqrt()
    )
    logits = torch.einsum("btd,sd->bts", unit, weights["fn"][:streams].double())
    pre = (
        torch.sigmoid(
            logits * weights["scale"][0].double() + weights["base"][:streams].double()
        )
        + config.hc_eps
    )
    reference = torch.einsum("bts,btsd->btd", pre, x.double())
    reference_norm = (
        norm_weight.double()
        * reference
        / (reference.square().mean(-1, keepdim=True) + config.rms_norm_eps).sqrt()
    )
    checks = {
        "pre_norm": torch.allclose(native.double(), reference, atol=ATOL, rtol=RTOL),
        "post_norm": torch.allclose(
            normalized.double(), reference_norm, atol=ATOL, rtol=RTOL
        ),
    }
    average = x.mean(dim=2)
    return {
        "pointwise_matches": checks,
        "native_vs_fp64_reference": differences(native, reference),
        "normalized_native_vs_fp64_reference": differences(normalized, reference_norm),
        "raw_mean_vs_native": differences(average, native),
        "normalized_raw_mean_vs_normalized_native": differences(
            norm(average), normalized
        ),
    }


@torch.inference_mode()
def probe(family, spec, downloads):
    if family == "deepseek_v4":
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4HyperConnection as Gate,
            DeepseekV4RMSNorm as Norm,
        )
    else:
        from transformers.models.glm5_next.modeling_glm5_next import (
            Glm5NextTextHyperConnection as Gate,
            Glm5NextTextRMSNorm as Norm,
        )
    raw = json.loads(downloads.read(spec, "config.json", "checkpoint_config"))
    cfg = SimpleNamespace(**raw.get("text_config", raw))
    embeds = Shard(downloads, spec, "embedding").tensor(
        spec["embedding"], (TOKEN_IDS[0], TOKEN_IDS[-1])
    )
    results = []
    for role, layer in (("first", 0), ("midpoint", cfg.num_hidden_layers // 2)):
        shard = Shard(downloads, spec, role)
        prefix = f"{spec['layer_prefix']}.{layer}."
        weights = {
            k: shard.tensor(f"{prefix}hc_attn_{k}") for k in ("fn", "base", "scale")
        }
        norm_weight = shard.tensor(prefix + spec["norm"])
        if role == "first":
            x = embeds[None, :, None, :].expand(-1, -1, cfg.hc_mult, -1).contiguous()
            source = (
                "published embedding rows broadcast to all streams as at decoder entry"
            )
        else:
            generator = torch.Generator().manual_seed(SEED)
            x = torch.randn(
                (1, len(TOKEN_IDS), cfg.hc_mult, cfg.hidden_size), generator=generator
            )
            source = "synthetic independent standard-normal streams, not a recorded model activation"
        metrics = compare(
            Gate(cfg),
            Norm(cfg.hidden_size, eps=cfg.rms_norm_eps),
            weights,
            norm_weight,
            x,
            cfg,
        )
        results.append(
            {
                "layer": layer,
                "site": "attention",
                "input_source": source,
                "shape": list(x.shape),
                **metrics,
            }
        )
    return {
        "repo": spec["repo"],
        "revision": spec["revision"],
        "index_url": f"https://huggingface.co/{spec['repo']}/blob/{spec['revision']}/model.safetensors.index.json",
        "upstream_module": Gate.__module__,
        "upstream_source_sha256": hashlib.sha256(
            Path(inspect.getfile(Gate)).read_bytes()
        ).hexdigest(),
        "layers": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if transformers.__version__ != "5.20.2":
        parser.error("This probe uses the pinned Transformers 5.20.2 native modules.")
    torch.set_num_threads(1)
    downloads = Downloads(max_bytes=12_000_000)
    results = [probe(family, spec, downloads) for family, spec in CHECKPOINTS.items()]
    report = {
        "scope": "Read-only CPU FP32 native attention gates and RMSNorm, compared with an independent FP64 formula. No full checkpoint forward or model-quality measurement.",
        "python": platform.python_version(),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "token_ids": TOKEN_IDS,
        "synthetic_seed": SEED,
        "atol": ATOL,
        "rtol": RTOL,
        "relative_l2_denominator": "second operand",
        "max_download_bytes": downloads.max_bytes,
        "response_body_bytes_read": downloads.bytes_read,
        "models": results,
        "retrievals": downloads.records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    ok = all(
        all(layer["pointwise_matches"].values())
        for model in results
        for layer in model["layers"]
    )
    print(
        f"{downloads.bytes_read} response bytes, {len(results)} checkpoints, pointwise checks passed={ok}"
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
