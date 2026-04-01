import argparse
import os
import glob
import json
import logging
import yaml
import shutil
from tqdm import tqdm
import librosa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Filter low quality pseudo-labels")
    parser.add_argument("--config", type=str, default="../../config/asr_config.yaml")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(args.input_dir, "*.json"))
    logger.info(f"Found {len(json_files)} metadata files to filter.")
    
    min_conf = config['filtering'].get('min_confidence', 0.8)
    min_len = config['data'].get('min_audio_length_seconds', 0.2)
    max_len = config['data'].get('max_audio_length_seconds', 30.0)
    drop_missing_lang = config['filtering'].get('drop_missing_language', True)
    
    kept = 0
    dropped = 0
    
    for jf in tqdm(json_files):
        with open(jf, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        audio_file = jf.replace(".json", ".wav")
        if not os.path.exists(audio_file):
            dropped += 1
            continue
            
        reason_to_drop = None
        
        # Confidence logic
        if 'confidence' in meta and meta['confidence'] < min_conf:
            reason_to_drop = f"Low confidence ({meta['confidence']} < {min_conf})"
            
        # Language logic
        if drop_missing_lang and not meta.get('language'):
             reason_to_drop = "Missing language tag"
             
        # Length logic
        if not reason_to_drop:
            try:
                duration = librosa.get_duration(path=audio_file)
                if duration < min_len or duration > max_len:
                    reason_to_drop = f"Invalid length ({duration}s)"
                meta['duration'] = duration
            except Exception as e:
                reason_to_drop = f"Audio load error: {e}"
        
        if reason_to_drop:
            dropped += 1
            continue
            
        # If passed, copy to processed
        base = os.path.basename(jf)
        out_jf = os.path.join(args.output_dir, base)
        out_af = os.path.join(args.output_dir, os.path.basename(audio_file))
        
        # Update json with duration
        with open(out_jf, 'w', encoding='utf-8') as fw:
            json.dump(meta, fw, indent=2, ensure_ascii=False)
            
        if not os.path.exists(out_af):
             shutil.copy2(audio_file, out_af)
             
        kept += 1
        
    logger.info(f"Filtering complete. Kept: {kept}, Dropped: {dropped}")

if __name__ == "__main__":
    main()
