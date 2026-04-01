import argparse
import math
import os
import glob
import json
import logging
import yaml
import shutil
from tqdm import tqdm

from api_client import SarvamAPIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="ASR pseudo-labeler worker")
    parser.add_argument("--config", type=str, default="../../config/asr_config.yaml")
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--language_code", type=str, default=None)
    return parser.parse_args()


def collect_audio_files(audio_dir):
    patterns = ["*.wav", "*.mp3", "*.flac", "*.m4a"]
    audio_files = []
    for pattern in patterns:
        audio_files.extend(glob.glob(os.path.join(audio_dir, f"**/{pattern}"), recursive=True))
    audio_files.sort()
    return audio_files


def build_sarvam_labeler(config, default_language):
    client = SarvamAPIClient(max_retries=config["api"]["max_retries"])
    endpoint = config["api"]["endpoint"]

    def label_audio(audio_path):
        with open(audio_path, "rb") as handle:
            files = {"file": (os.path.basename(audio_path), handle, "audio/wav")}
            data = {"model": "saaras:v1"}
            if default_language:
                data["language_code"] = default_language
            response = client.post(endpoint, files=files, data=data)

        if response.status_code != 200:
            raise RuntimeError(f"Sarvam request failed for {audio_path}: {response.text}")

        result = response.json()
        return {
            "transcript": result.get("transcript", result.get("text", "")).strip(),
            "confidence": result.get("confidence", 1.0),
            "language": result.get("language_code", default_language),
        }

    return label_audio


def build_faster_whisper_labeler(teacher_config, default_language):
    import torch
    from faster_whisper import WhisperModel

    device = teacher_config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    compute_type = teacher_config.get("compute_type", "auto")
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    model = WhisperModel(
        teacher_config.get("model_name", "openai/whisper-large-v3-turbo"),
        device=device,
        compute_type=compute_type,
    )

    def label_audio(audio_path):
        segments, info = model.transcribe(
            audio_path,
            beam_size=teacher_config.get("beam_size", 1),
            vad_filter=teacher_config.get("vad_filter", True),
            language=default_language,
            task="transcribe",
        )
        segments = list(segments)
        transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()

        confidence_scores = []
        for segment in segments:
            avg_logprob = getattr(segment, "avg_logprob", None)
            if avg_logprob is None:
                continue
            confidence_scores.append(math.exp(max(min(avg_logprob, 0.0), -10.0)))

        confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 1.0
        return {
            "transcript": transcript,
            "confidence": confidence,
            "language": getattr(info, "language", None) or default_language,
        }

    return label_audio

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    teacher_config = config.get("teacher", {})
    provider = teacher_config.get("provider", "sarvam")

    if provider == "sarvam":
        label_audio = build_sarvam_labeler(config, args.language_code)
    elif provider == "faster_whisper":
        label_audio = build_faster_whisper_labeler(teacher_config, args.language_code)
    else:
        raise ValueError(f"Unsupported ASR teacher provider: {provider}")

    audio_files = collect_audio_files(args.audio_dir)
    logger.info(f"Found {len(audio_files)} audio files to process.")

    for audio_path in tqdm(audio_files):
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        out_json_path = os.path.join(args.output_dir, f"{base_name}.json")
        out_audio_path = os.path.join(args.output_dir, f"{base_name}.wav")

        if os.path.exists(out_json_path):
            continue

        try:
            result = label_audio(audio_path)
            meta = {
                "transcript": result.get("transcript", ""),
                "confidence": result.get("confidence", 1.0),
                "language": result.get("language"),
                "source_audio": os.path.abspath(audio_path),
                "teacher_provider": provider,
                "teacher_model": teacher_config.get("model_name"),
            }

            with open(out_json_path, "w", encoding="utf-8") as jf:
                json.dump(meta, jf, ensure_ascii=False, indent=2)

            if not os.path.exists(out_audio_path):
                shutil.copy2(audio_path, out_audio_path)
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")

if __name__ == "__main__":
    main()
