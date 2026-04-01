import argparse
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
    parser = argparse.ArgumentParser(description="Sarvam ASR Pseudo-labeler Worker")
    parser.add_argument("--config", type=str, default="../../config/asr_config.yaml")
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--language_code", type=str, default="hi-IN")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    client = SarvamAPIClient(max_retries=config['api']['max_retries'])
    endpoint = config['api']['endpoint']
    
    audio_files = glob.glob(os.path.join(args.audio_dir, "**/*.wav"), recursive=True)
    logger.info(f"Found {len(audio_files)} audio files to process.")
    
    for audio_path in tqdm(audio_files):
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        out_json_path = os.path.join(args.output_dir, f"{base_name}.json")
        out_audio_path = os.path.join(args.output_dir, f"{base_name}.wav")
        
        if os.path.exists(out_json_path):
            continue # Skip processed
            
        try:
            with open(audio_path, 'rb') as f:
                # Based on Sarvam documentation, usually it is a multipart/form-data request
                files = {'file': (os.path.basename(audio_path), f, 'audio/wav')}
                data = {'model': 'saaras:v1', 'language_code': args.language_code}
                response = client.post(endpoint, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                meta = {
                    "transcript": result.get("transcript", result.get("text", "")),
                    "confidence": result.get("confidence", 1.0),
                    "language": result.get("language_code", args.language_code),
                    "source_audio": os.path.abspath(audio_path)
                }
                
                with open(out_json_path, 'w', encoding='utf-8') as jf:
                    json.dump(meta, jf, ensure_ascii=False, indent=2)
                
                if not os.path.exists(out_audio_path):
                     shutil.copy2(audio_path, out_audio_path)
            else:
                logger.error(f"Failed to process {audio_path}: {response.text}")
                
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")

if __name__ == "__main__":
    main()
