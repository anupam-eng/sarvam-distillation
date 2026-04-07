from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.vits_pipeline.common import PROJECT_ROOT, ensure_dir, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize a single text string with a trained VITS model")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--model-config-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="reports/local_test_output.wav")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def sanitize_preview(text: str) -> str:
    compact = re.sub(r"\s+", "_", text.strip())
    compact = re.sub(r"[^\w\u0900-\u097F]+", "", compact)
    return compact[:40] or "tts_sample"


def main() -> None:
    import soundfile as sf
    import torch
    from TTS.utils.synthesizer import Synthesizer

    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint_path, PROJECT_ROOT)
    model_config_path = resolve_path(args.model_config_path, PROJECT_ROOT)
    output_path = resolve_path(args.output_path, PROJECT_ROOT)
    ensure_dir(output_path.parent)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    use_cuda = torch.cuda.is_available() and not args.cpu
    synthesizer = Synthesizer(
        tts_checkpoint=str(checkpoint_path),
        tts_config_path=str(model_config_path),
        use_cuda=use_cuda,
    )

    wav = synthesizer.tts(args.text)
    sf.write(str(output_path), wav, synthesizer.output_sample_rate)

    print(
        json.dumps(
            {
                "text": args.text,
                "output_path": str(output_path),
                "sample_rate": synthesizer.output_sample_rate,
                "device": "cuda" if use_cuda else "cpu",
                "preview_name": sanitize_preview(args.text),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
