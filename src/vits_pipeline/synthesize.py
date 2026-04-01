from __future__ import annotations

import argparse
import json

from src.vits_pipeline.common import PROJECT_ROOT, ensure_dir, find_latest_matching, load_config, read_nonempty_lines, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize sample audio from a trained VITS checkpoint")
    parser.add_argument("--config", type=str, default="config/tts_config.yaml")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--model-config-path", type=str, default="")
    parser.add_argument("--input-text-file", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    import soundfile as sf

    args = parse_args()
    config, config_path = load_config(args.config)
    base_dir = PROJECT_ROOT

    training_cfg = config["training"]
    inference_cfg = config.get("inference", {})

    training_output_dir = resolve_path(training_cfg["output_dir"], base_dir)
    checkpoint_path = (
        resolve_path(args.checkpoint_path, base_dir)
        if args.checkpoint_path
        else find_latest_matching(training_output_dir, "best_model.pth")
    )
    model_config_path = (
        resolve_path(args.model_config_path, base_dir)
        if args.model_config_path
        else find_latest_matching(training_output_dir, "config.json")
    )
    input_text_file = resolve_path(args.input_text_file, base_dir) if args.input_text_file else resolve_path(inference_cfg["prompts_file"], base_dir)
    output_dir = ensure_dir(resolve_path(args.output_dir, base_dir) if args.output_dir else resolve_path(inference_cfg["output_dir"], base_dir))
    max_samples = int(inference_cfg.get("max_samples", 10))

    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if model_config_path is None or not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    if not input_text_file.exists():
        raise FileNotFoundError(f"Input text file not found: {input_text_file}")

    from TTS.utils.synthesizer import Synthesizer
    import torch

    use_cuda = torch.cuda.is_available() and not args.cpu

    synthesizer = Synthesizer(
        tts_checkpoint=str(checkpoint_path),
        tts_config_path=str(model_config_path),
        use_cuda=use_cuda,
    )

    prompt_lines = read_nonempty_lines(input_text_file)[:max_samples]
    written = []
    for index, text in enumerate(prompt_lines, start=1):
        wav = synthesizer.tts(text)
        output_path = output_dir / f"sample_{index:03d}.wav"
        sf.write(str(output_path), wav, synthesizer.output_sample_rate)
        written.append({"index": index, "text": text, "audio_path": str(output_path)})

    with (output_dir / "samples.json").open("w", encoding="utf-8") as handle:
        json.dump(written, handle, ensure_ascii=False, indent=2)

    print(json.dumps({"output_dir": str(output_dir), "samples_written": len(written)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
