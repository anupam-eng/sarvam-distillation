import argparse
import os
import glob
import json
import logging
import webdataset as wds
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Create WebDataset shards")
    parser.add_argument("--input_dir", type=str, required=True, help="Processed dir with wav and json")
    parser.add_argument("--output_dir", type=str, required=True, help="Dir to store .tar shards")
    parser.add_argument("--shard_prefix", type=str, default="data", help="Prefix for tar files")
    parser.add_argument("--max_size", type=float, default=1e9, help="Max size per shard in bytes (default 1GB)")
    parser.add_argument("--max_count", type=int, default=10000, help="Max samples per shard")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    pattern = os.path.join(args.output_dir, f"{args.shard_prefix}-%06d.tar")
    
    # We use ShardWriter for streaming shards
    sink = wds.ShardWriter(pattern, maxsize=args.max_size, maxcount=args.max_count)
    
    json_files = glob.glob(os.path.join(args.input_dir, "*.json"))
    logger.info(f"Packing {len(json_files)} samples into shards...")
    
    # Sort for deterministic sharding
    json_files.sort()
    
    for i, jf in enumerate(tqdm(json_files)):
        audio_file = jf.replace(".json", ".wav")
        if not os.path.exists(audio_file):
            continue
            
        with open(jf, "r", encoding='utf-8') as f:
            meta = json.load(f)
            
        with open(audio_file, "rb") as f:
            audio_data = f.read()
            
        key = os.path.splitext(os.path.basename(jf))[0]
        
        sample = {
            "__key__": key,
            "wav": audio_data,
            "json": meta
        }
        
        sink.write(sample)
        
    sink.close()
    logger.info("Sharding complete.")

if __name__ == "__main__":
    main()
