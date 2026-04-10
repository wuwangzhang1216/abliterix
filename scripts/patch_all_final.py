"""Final comprehensive patch for Step-3.5-Flash + transformers 4.55/5.x."""
import json, glob, shutil

import os
HF = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))

# 1. config.json: eos_token_id int, pad_token_id set, restore rope_scaling
for f in glob.glob(f"{HF}/hub/models--stepfun-ai--Step-3.5-Flash/snapshots/*/config.json"):
    d = json.load(open(f))
    changed = False
    if isinstance(d.get("eos_token_id"), list):
        d["eos_token_id"] = 128007; changed = True
    if d.get("pad_token_id") is None:
        d["pad_token_id"] = 1; changed = True
    if d.get("rope_scaling") is None:
        d["rope_scaling"] = {"rope_type":"llama3","factor":2.0,"original_max_position_embeddings":131072,"low_freq_factor":1.0,"high_freq_factor":32.0}
        changed = True
    if d.get("yarn_only_types") is None:
        d["yarn_only_types"] = ["full_attention"]; changed = True
    if "use_cache" not in d:
        d["use_cache"] = True; changed = True
    if changed:
        json.dump(d, open(f, "w"), indent=2)
        print(f"[1] config.json patched: {f}")

# 2. configuration_step3p5.py: truncate layer_types
for pat in [f"{HF}/modules/transformers_modules/stepfun_hyphen_ai/Step_hyphen_3_dot_5_hyphen_Flash/*/configuration_step3p5.py",
            f"{HF}/hub/models--stepfun-ai--Step-3.5-Flash/snapshots/*/configuration_step3p5.py"]:
    for f in glob.glob(pat):
        text = open(f).read()
        old = "        self.layer_types = layer_types"
        new = "        # Truncate MTP head layers\n        if layer_types is not None and len(layer_types) > num_hidden_layers:\n            layer_types = layer_types[:num_hidden_layers]\n        self.layer_types = layer_types"
        if old in text and "Truncate MTP" not in text:
            text = text.replace(old, new)
            open(f, "w").write(text)
            print(f"[2] config class patched: {f}")
        else:
            print(f"[2] skip (already patched or no match): {f}")

# 3. modeling_step3p5.py: default rope, rope_parameters theta, compute_default_rope_parameters, cache_position
for pat in [f"{HF}/modules/transformers_modules/stepfun_hyphen_ai/Step_hyphen_3_dot_5_hyphen_Flash/*/modeling_step3p5.py",
            f"{HF}/hub/models--stepfun-ai--Step-3.5-Flash/snapshots/*/modeling_step3p5.py"]:
    for f in glob.glob(pat):
        text = open(f).read()
        patches_applied = []

        # 3a. Register "default" rope type
        old_imp = 'from transformers.modeling_rope_utils import (ROPE_INIT_FUNCTIONS,\n                                              dynamic_rope_update)'
        new_imp = '''from transformers.modeling_rope_utils import (ROPE_INIT_FUNCTIONS,
                                              dynamic_rope_update)

if "default" not in ROPE_INIT_FUNCTIONS:
    def _default_rope_init(config, device=None, **kwargs):
        import torch
        dim = int(config.head_dim * getattr(config, 'partial_rotary_factor', 1.0))
        base = config.rope_theta if not isinstance(config.rope_theta, list) else 10000.0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, 1.0
    ROPE_INIT_FUNCTIONS["default"] = _default_rope_init'''
        if old_imp in text and "_default_rope_init" not in text:
            text = text.replace(old_imp, new_imp)
            patches_applied.append("default_rope")

        # 3b. Fix list rope_theta in rope_parameters
        old_rope = "        self.rope_theta = config.rope_theta\n        if isinstance(config.rope_theta, list):\n            self.rope_theta = config.rope_theta.copy()\n            config.rope_theta = self.rope_theta[self.layer_idx]"
        new_rope = """        self.rope_theta = config.rope_theta
        if isinstance(config.rope_theta, list):
            self.rope_theta = config.rope_theta.copy()
            config.rope_theta = self.rope_theta[self.layer_idx]
        if hasattr(config, 'rope_parameters') and config.rope_parameters is not None:
            rp = config.rope_parameters
            rp_theta = rp.get("rope_theta") if isinstance(rp, dict) else getattr(rp, "rope_theta", None)
            if isinstance(rp_theta, list) and self.layer_idx is not None:
                if isinstance(rp, dict):
                    rp["rope_theta"] = rp_theta[self.layer_idx]
                else:
                    rp.rope_theta = rp_theta[self.layer_idx]"""
        if old_rope in text and "rp_theta" not in text:
            text = text.replace(old_rope, new_rope)
            patches_applied.append("rope_parameters_theta")

        # 3c. Add compute_default_rope_parameters
        old_fwd = "    @torch.no_grad()\n    @dynamic_rope_update"
        new_fwd = "    def compute_default_rope_parameters(self, device=None, seq_len=None):\n        return self.inv_freq, self.attention_scaling\n\n    @torch.no_grad()\n    @dynamic_rope_update"
        if "compute_default_rope_parameters" not in text and old_fwd in text:
            text = text.replace(old_fwd, new_fwd, 1)
            patches_applied.append("compute_default_rope")

        # 3d. Fix apply_rotary_pos_emb missing unsqueeze (CRITICAL BUG in model code)
        # cos/sin are 3D (batch, seq, dim) but q/k are 4D (batch, heads, seq, dim)
        # Without unsqueeze, broadcasting fails when seq_len != num_heads
        old_rotary = "    rotary_dim = cos.shape[-1]\n    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]\n    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]"
        new_rotary = "    cos = cos.unsqueeze(unsqueeze_dim)\n    sin = sin.unsqueeze(unsqueeze_dim)\n    rotary_dim = cos.shape[-1]\n    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]\n    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]"
        if old_rotary in text and "cos.unsqueeze(unsqueeze_dim)" not in text:
            text = text.replace(old_rotary, new_rotary)
            patches_applied.append("rotary_unsqueeze")

        # 3e. Fix Step3p5ForCausalLM.forward() dropping hidden_states from output
        old_return = "        return Step3p5CausalLMOutputWithPast(logits=logits, )"
        new_return = "        return Step3p5CausalLMOutputWithPast(\n            logits=logits,\n            past_key_values=outputs.past_key_values,\n            hidden_states=outputs.hidden_states,\n            attentions=outputs.attentions,\n        )"
        if old_return in text and "hidden_states=outputs.hidden_states" not in text:
            text = text.replace(old_return, new_return)
            patches_applied.append("forward_return_hidden_states")

        # 3f. Fix cache_position None check
        old_cache = "        if cache_position[0] == 0:"
        new_cache = "        if cache_position is not None and cache_position[0] == 0:"
        if old_cache in text and "cache_position is not None and cache_position[0]" not in text:
            text = text.replace(old_cache, new_cache)
            patches_applied.append("cache_position")

        if patches_applied:
            open(f, "w").write(text)
            print(f"[3] modeling patched ({', '.join(patches_applied)}): {f}")
        else:
            print(f"[3] skip (all already patched): {f}")

# Clear pycache
for pat in [f"{HF}/modules/transformers_modules/stepfun_hyphen_ai/*/__pycache__",
            f"{HF}/hub/models--stepfun-ai--Step-3.5-Flash/snapshots/*/__pycache__"]:
    for d in glob.glob(pat):
        shutil.rmtree(d, ignore_errors=True)

print("\nAll patches applied.")
