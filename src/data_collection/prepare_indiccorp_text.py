import argparse
import hashlib
import os
import re

import yaml
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare IndicCorp prompt files for TTS generation")
    parser.add_argument("--config", type=str, default="../../config/tts_config.yaml")
    return parser.parse_args()


def normalize_text(text):
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text


def is_valid_text(text, config):
    if len(text) < config["min_chars"] or len(text) > config["max_chars"]:
        return False
    if len(text.split()) > config["max_words"]:
        return False
    if any(char in text for char in ["http://", "https://", "www."]):
        return False
    if text.count("|"):
        return False
    return True


def assign_split(text, eval_fraction):
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < eval_fraction else "train"


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    corpus_config = config["corpus"]
    if corpus_config["source"] != "huggingface":
        raise ValueError(f"Unsupported corpus source: {corpus_config['source']}")

    dataset = load_dataset(
        corpus_config["dataset_name"],
        corpus_config.get("subset"),
        data_dir=corpus_config.get("data_dir"),
        split=corpus_config.get("split", "train"),
        streaming=True,
    )

    train_target = corpus_config["max_train_samples"]
    eval_target = corpus_config["max_eval_samples"]
    eval_fraction = eval_target / max(train_target + eval_target, 1)

    train_texts = []
    eval_texts = []
    seen = set()

    for row in dataset:
        text = normalize_text(row.get(corpus_config.get("text_column", "text"), ""))
        if not text or text in seen or not is_valid_text(text, corpus_config):
            continue

        target_split = assign_split(text, eval_fraction)
        if target_split == "eval" and len(eval_texts) < eval_target:
            eval_texts.append(text)
            seen.add(text)
        elif len(train_texts) < train_target:
            train_texts.append(text)
            seen.add(text)

        if len(train_texts) >= train_target and len(eval_texts) >= eval_target:
            break

    os.makedirs(os.path.dirname(corpus_config["train_output_path"]), exist_ok=True)
    with open(corpus_config["train_output_path"], "w", encoding="utf-8") as handle:
        handle.write("\n".join(train_texts) + ("\n" if train_texts else ""))
    with open(corpus_config["eval_output_path"], "w", encoding="utf-8") as handle:
        handle.write("\n".join(eval_texts) + ("\n" if eval_texts else ""))

    print(
        f"Prepared {len(train_texts)} train prompts and {len(eval_texts)} eval prompts "
        f"from {corpus_config['dataset_name']} ({corpus_config['data_dir']})."
    )


if __name__ == "__main__":
    main()
