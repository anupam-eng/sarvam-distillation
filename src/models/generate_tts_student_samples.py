import argparse
import json
import os
import subprocess

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Generate evaluation samples from a trained VITS student")
    parser.add_argument("--config", type=str, default="../../config/tts_config.yaml")
    parser.add_argument("--input_text_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = config["training"]["output_dir"]
    model_path = os.path.join(model_dir, config["training"].get("best_checkpoint_name", "best_model.pth"))
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing trained VITS checkpoint: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing Coqui config: {config_path}")

    with open(args.input_text_file, "r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]

    for index, prompt in enumerate(prompts):
        base_name = f"tts_student_{index:06d}"
        wav_path = os.path.join(args.output_dir, f"{base_name}.wav")
        json_path = os.path.join(args.output_dir, f"{base_name}.json")
        if os.path.exists(wav_path) and os.path.exists(json_path):
            continue

        subprocess.run(
            [
                "tts",
                "--text",
                prompt,
                "--model_path",
                model_path,
                "--config_path",
                config_path,
                "--out_path",
                wav_path,
            ],
            check=True,
        )
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "text": prompt,
                    "language": config.get("corpus", {}).get("language"),
                    "sample_rate": config["data"].get("sample_rate", 22050),
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )


if __name__ == "__main__":
    main()
