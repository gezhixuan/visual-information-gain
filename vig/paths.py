"""Where data, models, runs and evaluation outputs live. Every location is an environment variable with a default
relative to the working directory.

  VIG_DATA    raw downloads (VIG_DATA/mm, VIG_DATA/text) and training shards (VIG_DATA/shards)     default ./data
  VIG_MODELS  Hugging Face snapshots, one folder per model, named after the repository             default ./models
  VIG_RUNS    one folder per training run (checkpoints, logs)                                       default ./runs
  VIG_OUT     evaluation outputs (validation batches, jsonl files)                                  default ./outputs
"""
import os

DATA = os.environ.get("VIG_DATA", "data")
MODELS = os.environ.get("VIG_MODELS", "models")
RUNS = os.environ.get("VIG_RUNS", "runs")
OUT = os.environ.get("VIG_OUT", "outputs")

SHARDS = os.path.join(DATA, "shards")
VAL_SETS = os.path.join(OUT, "val_sets_32.pt")           # the fixed validation batches every evaluation shares
SIGLIP = os.path.join(MODELS, "siglip2-so400m-patch16-naflex")
TOKENIZER = os.path.join(MODELS, "Qwen2.5-0.5B")
