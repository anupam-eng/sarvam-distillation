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
    parser = argparse.ArgumentParser(description="Sarvam TTS Teacher Generator")
    parser.add_argument("--config", type=str, default="../../config/tts_config.yaml")
    parser.add_argument("--input_text_file", type=str, required=True, help="Path to distinct sentences separated by newline")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--speaker", type=str, default="meera")
    parser.add_argument("--target_language_code", type=str, default="hi-IN")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.input_text_file, "r", encoding="utf-8") as tf:
        lines = [l.strip() for l in tf.readlines() if l.strip()]
        
    client = SarvamAPIClient(max_retries=config['api']['max_retries'])
    endpoint = config['api']['endpoint']
    
    logger.info(f"Loaded {len(lines)} sentences for TTS generation.")
    
    for idx, sentence in enumerate(tqdm(lines)):
        base_name = f"tts_sample_{idx:06d}"
        out_audio_path = os.path.join(args.output_dir, f"{base_name}.wav")
        out_json_path = os.path.join(args.output_dir, f"{base_name}.json")
        
        if os.path.exists(out_audio_path) and os.path.exists(out_json_path):
            continue
            
        try:
            payload = {
                "inputs": [sentence],
                "target_language_code": args.target_language_code,
                "speaker": args.speaker,
                "pitch": 0,
                "pace": 1.0,
                "loudness": 1.5,
                "speech_sample_rate": 22050,
                "enable_preprocessing": True,
                "model": "bulbul:v1"
            }
            response = client.post(endpoint, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                audio_base64 = result.get('audios', [])[0] if result.get('audios') else None
                
                if audio_base64:
                    audio_bytes = base64.b64decode(audio_base64)
                    
                    with open(out_audio_path, "wb") as bf:
                        bf.write(audio_bytes)
                        
                    meta = {
                        "text": sentence,
                        "speaker": args.speaker,
                        "language": args.target_language_code,
                        "sample_rate": 22050
                    }
                    with open(out_json_path, 'w', encoding='utf-8') as jf:
                        json.dump(meta, jf, ensure_ascii=False, indent=2)
                else:
                    logger.error("No audio returned by API")
            else:
                logger.error(f"API Error {response.status_code} for text '{sentence}': {response.text}")
                
        except Exception as e:
             logger.error(f"Exception during generation: {e}")

if __name__ == "__main__":
    main()
