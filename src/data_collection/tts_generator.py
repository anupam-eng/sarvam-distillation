import argparse
import os
import json
import logging
import yaml
import base64
from tqdm import tqdm

from api_client import SarvamAPIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="TTS teacher generator")
    parser.add_argument("--config", type=str, default="../../config/tts_config.yaml")
    parser.add_argument("--input_text_file", type=str, required=True, help="Path to distinct sentences separated by newline")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--speaker", type=str, default="meera")
    parser.add_argument("--target_language_code", type=str, default=None)
    return parser.parse_args()


def build_sarvam_generator(config, speaker, target_language_code):
    client = SarvamAPIClient(max_retries=config["api"]["max_retries"])
    endpoint = config["api"]["endpoint"]

    def generate_audio(text):
        payload = {
            "inputs": [text],
            "speaker": speaker,
            "pitch": 0,
            "pace": 1.0,
            "loudness": 1.5,
            "speech_sample_rate": 22050,
            "enable_preprocessing": True,
            "model": "bulbul:v1",
        }
        if target_language_code:
            payload["target_language_code"] = target_language_code

        response = client.post(endpoint, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Sarvam request failed for text '{text}': {response.text}")

        result = response.json()
        audio_base64 = result.get("audios", [None])[0]
        if not audio_base64:
            raise RuntimeError("No audio returned by Sarvam")

        return base64.b64decode(audio_base64), 22050

    return generate_audio


def build_transformers_vits_generator(teacher_config):
    import torch
    from transformers import AutoTokenizer, VitsModel

    device = teacher_config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = teacher_config.get("model_name", "facebook/mms-tts-eng")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VitsModel.from_pretrained(model_name).to(device)
    model.eval()

    sample_rate = getattr(model.config, "sampling_rate", teacher_config.get("sample_rate", 16000))

    def generate_audio(text):
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            waveform = model(**inputs).waveform[0].detach().cpu().numpy()
        return waveform, sample_rate

    return generate_audio

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_text_file, "r", encoding="utf-8") as tf:
        lines = [l.strip() for l in tf.readlines() if l.strip()]

    teacher_config = config.get("teacher", {})
    provider = teacher_config.get("provider", "sarvam")

    if provider == "sarvam":
        generate_audio = build_sarvam_generator(config, args.speaker, args.target_language_code)
    elif provider == "transformers_vits":
        generate_audio = build_transformers_vits_generator(teacher_config)
    else:
        raise ValueError(f"Unsupported TTS teacher provider: {provider}")

    logger.info(f"Loaded {len(lines)} sentences for TTS generation.")

    for idx, sentence in enumerate(tqdm(lines)):
        base_name = f"tts_sample_{idx:06d}"
        out_audio_path = os.path.join(args.output_dir, f"{base_name}.wav")
        out_json_path = os.path.join(args.output_dir, f"{base_name}.json")

        if os.path.exists(out_audio_path) and os.path.exists(out_json_path):
            continue

        try:
            audio_data, sample_rate = generate_audio(sentence)
            if hasattr(audio_data, "tobytes"):
                import soundfile as sf

                sf.write(out_audio_path, audio_data, sample_rate)
            else:
                with open(out_audio_path, "wb") as bf:
                    bf.write(audio_data)

            meta = {
                "text": sentence,
                "speaker": args.speaker,
                "language": args.target_language_code,
                "sample_rate": sample_rate,
                "teacher_provider": provider,
                "teacher_model": teacher_config.get("model_name"),
            }
            with open(out_json_path, "w", encoding="utf-8") as jf:
                json.dump(meta, jf, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Exception during generation: {e}")

if __name__ == "__main__":
    main()
