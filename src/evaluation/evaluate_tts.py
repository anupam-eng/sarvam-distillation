import argparse
import glob
import json
import os

import jiwer
import librosa
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated TTS audio with frozen ASR backtranscription")
    parser.add_argument("--config", type=str, default="../../config/tts_config.yaml")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def normalize_text(text):
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    return transform(text or "")


def build_asr_predictor(model_name_or_path):
    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name_or_path, device=device, compute_type=compute_type)

    def predict(audio_path, language=None):
        segments, _ = model.transcribe(
            audio_path,
            beam_size=1,
            vad_filter=True,
            language=language,
            task="transcribe",
        )
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()

    return predict


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    json_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if args.limit:
        json_files = json_files[:args.limit]
    if not json_files:
        raise ValueError(f"No metadata files found in {args.input_dir}")

    predict = build_asr_predictor(config.get("eval", {}).get("asr_model_name", "openai/whisper-large-v3-turbo"))

    references = []
    predictions = []
    durations = []
    success_count = 0
    records = []

    for json_path in json_files:
        wav_path = json_path.replace(".json", ".wav")
        if not os.path.exists(wav_path):
            continue

        with open(json_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        reference = metadata.get("text", "")
        prediction = predict(wav_path, metadata.get("language"))
        duration = librosa.get_duration(path=wav_path)

        normalized_reference = normalize_text(reference)
        normalized_prediction = normalize_text(prediction)

        references.append(normalized_reference)
        predictions.append(normalized_prediction)
        durations.append(duration)
        success_count += 1
        records.append(
            {
                "audio_path": os.path.abspath(wav_path),
                "reference": reference,
                "prediction": prediction,
                "duration_seconds": duration,
            }
        )

    if not references:
        raise ValueError(f"No valid wav/json pairs found in {args.input_dir}")

    output = {
        "num_samples": len(references),
        "wer": jiwer.wer(references, predictions),
        "cer": jiwer.cer(references, predictions),
        "generation_success_rate": success_count / len(json_files),
        "empty_backtranscription_rate": sum(1 for prediction in predictions if not prediction) / len(predictions),
        "average_duration_seconds": sum(durations) / len(durations),
        "input_dir": os.path.abspath(args.input_dir),
        "records": records,
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))

    output_json = args.output_json or config.get("eval", {}).get("output_path")
    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
