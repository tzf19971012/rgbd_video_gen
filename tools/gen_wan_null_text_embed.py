"""
Generate and save the null (empty-string) text embedding for Wan2.1.
This only needs to be run once. Output: empty_text_embed_wan.pt
Usage:
    python tools/gen_wan_null_text_embed.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from diffsynth.models.wan_video_text_encoder import WanTextEncoder

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
T5_FILE = "models_t5_umt5-xxl-enc-bf16.pth"
LOCAL_DIR = "./models/Wan-AI/Wan2.1-T2V-1.3B"
TOKENIZER_DIR = "./models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
OUTPUT_PATH = "./empty_text_embed_wan.pt"
MAX_SEQ_LEN = 512

# --- Download T5 weights if needed ---
t5_path = os.path.join(LOCAL_DIR, T5_FILE)
if not os.path.exists(t5_path):
    print(f"Downloading {T5_FILE} from {MODEL_ID} (~10 GB, please wait) ...", flush=True)
    os.makedirs(LOCAL_DIR, exist_ok=True)
    hf_hub_download(repo_id=MODEL_ID, filename=T5_FILE, local_dir=LOCAL_DIR)
    print("Download complete.", flush=True)
else:
    print(f"Found {t5_path}", flush=True)

# --- Load tokenizer ---
print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

# --- Load text encoder ---
print("Loading WanTextEncoder ...")
state_dict = torch.load(t5_path, map_location="cpu", weights_only=True)
text_encoder = WanTextEncoder()
missing, unexpected = text_encoder.load_state_dict(state_dict, strict=False)
print(f"  missing={len(missing)}, unexpected={len(unexpected)}")
text_encoder.eval()
text_encoder = text_encoder.to(torch.bfloat16)

# --- Tokenize empty string ---
print("Encoding empty string ...")
encoding = tokenizer(
    [""],
    padding="max_length",
    max_length=MAX_SEQ_LEN,
    truncation=True,
    return_tensors="pt",
)
input_ids = encoding["input_ids"]          # [1, 512]
attention_mask = encoding["attention_mask"]  # [1, 512]

with torch.no_grad():
    null_embed = text_encoder(input_ids, mask=attention_mask)  # [1, 512, 4096]

print(f"null_embed shape: {null_embed.shape}, dtype: {null_embed.dtype}")
print(f"min/max/mean: {null_embed.min().item():.4f} / {null_embed.max().item():.4f} / {null_embed.mean().item():.6f}")
print(f"all-zero: {(null_embed == 0).all().item()}")

torch.save(null_embed, OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
