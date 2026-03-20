"""Push a model card to HuggingFace Hub.

Reads the card body from a markdown file and sets metadata via CLI args.

Usage:
    python scripts/push_model_card.py \
        --repo wangzhang/Devstral-Small-2-24B-Instruct-abliterated \
        --base-model mistralai/Devstral-Small-2-24B-Instruct-2512 \
        --card-file cards/devstral.md \
        --tags prometheus uncensored abliterated mistral code \
        --license apache-2.0 \
        --language en zh

The HF token is read from --token or the HF_TOKEN environment variable.
"""

import argparse
import os
import sys

from huggingface_hub import ModelCard, ModelCardData


def main():
    parser = argparse.ArgumentParser(description="Push a model card to HuggingFace Hub")
    parser.add_argument("--repo", required=True, help="Target HuggingFace repo (e.g. wangzhang/Model-abliterated)")
    parser.add_argument("--base-model", required=True, help="Base model ID")
    parser.add_argument("--card-file", required=True, help="Path to markdown file with card body")
    parser.add_argument("--tags", nargs="+", default=["prometheus", "uncensored", "abliterated"],
                        help="Model card tags")
    parser.add_argument("--license", default="apache-2.0", help="License identifier")
    parser.add_argument("--language", nargs="+", default=["en", "zh"], help="Language codes")
    parser.add_argument("--token", default=None, help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: provide --token or set HF_TOKEN environment variable", file=sys.stderr)
        sys.exit(1)

    with open(args.card_file, encoding="utf-8") as f:
        card_text = f.read()

    card = ModelCard("")
    card.data = ModelCardData()
    card.data.tags = args.tags
    card.data.base_model = args.base_model
    card.data.license = args.license
    card.data.language = args.language
    card.text = card_text

    card.push_to_hub(args.repo, token=token)
    print(f"Model card uploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
