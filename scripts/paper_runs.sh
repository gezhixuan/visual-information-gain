#!/usr/bin/env bash
# Every run and evaluation of the paper, in order. This file is a list of commands, not a job script: the shard
# builders are independent tasks and the trainings take days, so run the lines through your own scheduler.
# Locations come from VIG_DATA, VIG_MODELS, VIG_RUNS and VIG_OUT (see vig/paths.py).
set -euo pipefail
TRAIN="torchrun --standalone --nproc_per_node=8 -m vig.train"   # any number of GPUs, the batch per optimizer step does not depend on it

# ---------------------------------------------------------------------------------------------------- 1. data
python -m vig.dataprep.download --what data
python -m vig.dataprep.build_mm_shards all-index-plan
N=$(python -c "import json; from vig import paths; print(json.load(open(paths.SHARDS + '/mm/plan.json'))['n_tasks'])")
for k in $(seq 0 $((N - 1))); do python -m vig.dataprep.build_mm_shards run --task "$k"; done            # independent tasks
for s in cc12m-wds cc3m-wds the_cauldron LLaVA-OneVision-Data Docmatix aguvis-stage1 ChartGemma ChartQA ArxivQA MathV360K; do
  python -m vig.dataprep.build_mm_shards finalize --source "$s"
done
python -m vig.dataprep.build_text_bins all-index-plan
N=$(python -c "import json; from vig import paths; print(json.load(open(paths.SHARDS + '/text/plan.json'))['n_tasks'])")
for k in $(seq 0 $((N - 1))); do python -m vig.dataprep.build_text_bins tokenize --task "$k"; done       # independent tasks
for s in fineweb-edu open-web-math starcoder-python starcoder-markdown wikipedia-en; do
  python -m vig.dataprep.build_text_bins merge --source "$s"
done

# ---------------------------------------------------------------------------------------------------- 2. training
# encoder-free decoders, 158M to 1.04B, 40B tokens
for c in d1024_L12 d1280_L16 d1792_L16 d2048_L20; do $TRAIN --config "$c"; done
# frozen SigLIP 2 encoder in front of the same decoder: 323M on the 40B schedule, 158M on the 20B schedule
$TRAIN --config d1280_L16_siglip
$TRAIN --config d1024_L12_siglip --set total_tokens=20000000000
# baseline at 158M on the 20B schedule, two seeds
$TRAIN --config d1024_L12 --set run=d1024_L12_s20 --set total_tokens=20000000000
$TRAIN --config d1024_L12 --set run=d1024_L12_seed1 --set total_tokens=20000000000 --set seed=1 --set data.seed=1
# interventions at 158M: caption-only mixture, document-heavy mixture, 16-pixel patches
$TRAIN --config d1024_L12_mix_cap
$TRAIN --config d1024_L12_mix_doc
$TRAIN --config d1024_L12_p16
# batch size: 512 sequences per step on the 40B schedule, 32 sequences on the 20B schedule
$TRAIN --config d1024_L12 --set run=d1024_L12_bs512 --set global_batch_tokens=2097152
$TRAIN --config d1024_L12 --set run=d1024_L12_bs32 --set global_batch_tokens=131072 --set total_tokens=20000000000
# SigLIP 2 architecture with random weights, trained jointly. The run of the paper followed a 20B schedule to 3.8B tokens,
# continued on a 10B schedule and was stopped after 8.1B tokens.
$TRAIN --config d1024_L12_vitscratch --set stop_at_tokens=3800000000
$TRAIN --config d1024_L12_vitscratch --set total_tokens=10000000000 --set stop_at_tokens=8100000000

# ---------------------------------------------------------------------------------------------------- 3. VIG of every checkpoint
# The first call builds the validation batches (32 sequences of 4,096 tokens per image type) and stores them in VIG_OUT.
python -m vig.eval_vig --runs d1024_L12 d1280_L16 d1792_L16 d2048_L20 d1280_L16_siglip d1024_L12_siglip d1024_L12_s20 \
  d1024_L12_seed1 d1024_L12_mix_cap d1024_L12_mix_doc d1024_L12_bs512 d1024_L12_bs32 d1024_L12_vitscratch
# the 16-pixel run is scored on the same samples, cut into 16-pixel patches
python -m vig.val_cache
python -m vig.eval_vig --runs d1024_L12_p16 --val_cache "$(python -c 'from vig import paths; print(paths.VAL_SETS.replace("_32.pt", "_32_p16.pt"))')"
python -m vig.eval_vig --summarize

# ---------------------------------------------------------------------------------------------------- 4. answer accuracy
python -m vig.eval_answers --runs d1024_L12 d1280_L16 d1792_L16 d2048_L20 d1280_L16_siglip d1024_L12_bs512 --tokens 40
python -m vig.eval_answers --runs d1280_L16 d1280_L16_siglip --tokens 1,2,5,10,20,30
python -m vig.eval_answers --runs d1024_L12_s20 d1024_L12_seed1 d1024_L12_mix_doc d1024_L12_mix_cap d1024_L12_siglip --tokens 20
python -m vig.eval_answers --runs d1024_L12_p16 --tokens 20 --val_cache "$(python -c 'from vig import paths; print(paths.VAL_SETS.replace("_32.pt", "_32_p16.pt"))')"
python -m vig.eval_answers --summarize

# ---------------------------------------------------------------------------------------------------- 5. public models
python -m vig.dataprep.download --what public
python -m vig.public_vig --models gemma-4-12B:raw gemma-4-12B-it:chat gemma-4-26B-A4B-it:chat Qwen3-VL-8B-Instruct:chat \
  Qwen2.5-VL-7B-Instruct:chat Qwen2.5-VL-7B-Instruct:raw InternVL3-8B-hf:chat llava-1.5-7b-hf:chat
python -m vig.public_vig --models gemma-4-12B:raw gemma-4-12B-it:chat --budget 1120
python -m vig.public_vig_stages --models NEO1_0-2B-PT:chat NEO1_0-2B-MT:chat NEO1_0-2B-SFT:chat NEO1_5-2B-SFT:chat \
  NEO1_0-9B-PT:chat NEO1_0-9B-MT:chat NEO1_0-9B-SFT:chat NEO1_5-9B-SFT:chat \
  Mono-InternVL-2B-S1-1:chat Mono-InternVL-2B-S1-2:chat Mono-InternVL-2B-S1-3:chat Mono-InternVL-2B:chat
python -m vig.public_vig --summarize
