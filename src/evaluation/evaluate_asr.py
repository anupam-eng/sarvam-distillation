import argparse
import json
import os

import jiwer
import torch
import torchaudio
import yaml

from transformers import WhisperForConditionalGeneration, WhisperProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an ASR model on a labeled manifest")
    parser.add_argument("--config", type=str, default="../../config/asr_config.yaml")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--backend", type=str, default="whisper_seq2seq", choices=["whisper_seq2seq", "faster_whisper"])
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


def load_manifest(manifest_path, limit=0):
    samples = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            audio_path = sample["audio_path"]
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(os.path.dirname(manifest_path), audio_path)
            sample["audio_path"] = os.path.abspath(audio_path)
            samples.append(sample)
            if limit and len(samples) >= limit:
                break
    return samples


def resample_waveform(waveform, sample_rate, target_sample_rate):
    if sample_rate == target_sample_rate:
        return waveform
    resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
    return resampler(waveform)


def build_whisper_predictor(model_name_or_path, default_language=None, task="transcribe"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
    model.eval()
    model.generation_config.forced_decoder_ids = None

    def predict(audio_path, language=None):
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = resample_waveform(waveform, sample_rate, 16000)
        inputs = processor.feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            predicted_ids = model.generate(
                **inputs,
                language=language or default_language,
                task=task,
                max_new_tokens=225,
            )
        transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcript.strip()

    return predict


def build_faster_whisper_predictor(model_name_or_path):
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

    samples = load_manifest(args.manifest, limit=args.limit)
    if not samples:
        raise ValueError(f"No samples found in manifest: {args.manifest}")

    if args.backend == "whisper_seq2seq":
        predict = build_whisper_predictor(
            args.model_name_or_path,
            default_language=config.get("training", {}).get("language"),
            task=config.get("training", {}).get("task", "transcribe"),
        )
    else:
        predict = build_faster_whisper_predictor(args.model_name_or_path)

    references = []
    predictions = []
    records = []

    for sample in samples:
        prediction = predict(sample["audio_path"], sample.get("language"))
        reference = sample["text"]
        normalized_prediction = normalize_text(prediction)
        normalized_reference = normalize_text(reference)
        references.append(normalized_reference)
        predictions.append(normalized_prediction)
        records.append(
            {
                "audio_path": sample["audio_path"],
                "reference": reference,
                "prediction": prediction,
            }
        )

    output = {
        "backend": args.backend,
        "model_name_or_path": args.model_name_or_path,
        "num_samples": len(samples),
        "wer": jiwer.wer(references, predictions),
        "cer": jiwer.cer(references, predictions),
        "empty_prediction_rate": sum(1 for prediction in predictions if not prediction) / len(predictions),
        "manifest": os.path.abspath(args.manifest),
        "report_path": os.path.abspath(args.output_json) if args.output_json else None,
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
