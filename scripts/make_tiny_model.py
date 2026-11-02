"""Create a tiny random Qwen2 with the real Qwen tokenizer files (Build Guide 0.12).

  python scripts/make_tiny_model.py --out tests/assets/tiny_g2 --variant g2
  python scripts/make_tiny_model.py --out tests/assets/tiny_g7 --variant g7

g2: 4 heads / 2 KV heads (G = 2). g7: 14 heads / 2 KV heads, head_dim 16 (G = 7).
initializer_range=0.2 spreads the logits so greedy ties are rare.
"""

from __future__ import annotations

import argparse
import os
import shutil

import torch

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "generation_config.json"]


def make(out: str, variant: str = "g2") -> str:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    kw = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
              num_key_value_heads=2, vocab_size=151936, tie_word_embeddings=True, rope_theta=1e6,
              max_position_embeddings=4096, initializer_range=0.2, rms_norm_eps=1e-6)
    if variant == "g7":
        kw.update(hidden_size=224, intermediate_size=448, num_attention_heads=14)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(**kw))
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    src = os.environ.get("TS_MODEL_05")
    for fn in TOKENIZER_FILES:
        if src and os.path.exists(os.path.join(src, fn)):
            shutil.copy(os.path.join(src, fn), os.path.join(out, fn))
        else:
            from huggingface_hub import hf_hub_download

            hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct", fn, local_dir=out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=["g2", "g7"], default="g2")
    a = ap.parse_args()
    print(make(a.out, a.variant))
