#!/usr/bin/env bash
# CPU smoke test: synthetic shards -> unit tests -> 30-step tiny run (loss must decrease,
# eval diagnostics present) -> stop/resume reproduces the same next-step losses.
# Usage: bash tests/smoke_cpu.sh [workdir]   (default: /tmp/vig_smoke_$USER)
set -euo pipefail
PY=${PYTHON:-python}
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"; export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
WORK=${1:-/tmp/vig_smoke_$USER}
export PYTHONWARNINGS="ignore::UserWarning"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
rm -rf "$WORK/runs"; mkdir -p "$WORK"
echo "== synthetic shards"
[ -d "$WORK/synth_shards/mm" ] || "$PY" tests/make_synthetic_shards.py --out "$WORK/synth_shards"
echo "== unit tests"
"$PY" tests/test_units.py
COMMON=(--config tiny --set data.shard_root="$WORK/synth_shards" --set out_root="$WORK/runs")
echo "== run A: 30 steps"
"$PY" -m vig.train "${COMMON[@]}" --set run=A
echo "== run B: stop after 10 steps, then resume to 14"
"$PY" -m vig.train "${COMMON[@]}" --set run=B --set stop_at_tokens=$((10*4*256)) | tail -2
"$PY" -m vig.train "${COMMON[@]}" --set run=B --set stop_at_tokens=$((14*4*256)) | grep -E "resumed|step=1[1-4] "
echo "== run C/D: DDP 2 ranks (gloo) x 2 loader workers, straight vs stop/resume"
DDP=(--config tiny --set data.shard_root="$WORK/synth_shards" --set out_root="$WORK/runs" --set data.num_workers=2
     --set global_batch_tokens=2048 --set micro_batch_seqs=2 --set warmup_tokens=10240 --set eval_every_tokens=0
     --set save_every_tokens=0 --set state_every_tokens=0 --set eval_at_start=false --set total_tokens=28672)
TR=("$PY" -m torch.distributed.run --standalone --nproc_per_node=2 -m vig.train)
"${TR[@]}" "${DDP[@]}" --set run=C | grep -E "world=|DONE"
"${TR[@]}" "${DDP[@]}" --set run=Dd --set stop_at_tokens=14336 | grep STOPPED
"${TR[@]}" "${DDP[@]}" --set run=Dd | grep -E "resumed|DONE"
echo "== run E: tiny_siglip (random 2-layer ViT, trainable + grad ckpt, mm mixture reweighted), 12 steps"
SG=(--config tiny_siglip --set data.shard_root="$WORK/synth_shards" --set out_root="$WORK/runs" --set eval_every_tokens=0
    --set save_every_tokens=0 --set state_every_tokens=0 --set eval_at_start=false)
"$PY" -m vig.train "${SG[@]}" --set run=E --set total_tokens=$((12*4*256)) \
    --set 'data.mm_source_weights={"synth_doc": 3.0}' --set 'data.mm_source_caps={"synth_cap": 1}' | grep -E "front end|step=12 |DONE"
echo "== run F/G: tiny_siglip with a frozen encoder, straight vs stop/resume (state.pt holds no ViT weights)"
FR=("${SG[@]}" --set model.vision.freeze_encoder=true --set model.vision.encoder_grad_checkpoint=false --set total_tokens=$((10*4*256)))
"$PY" -m vig.train "${FR[@]}" --set run=F | grep -E "DONE"
"$PY" -m vig.train "${FR[@]}" --set run=G --set stop_at_tokens=$((6*4*256)) | grep STOPPED
"$PY" -m vig.train "${FR[@]}" --set run=G | grep -E "resumed|DONE"
echo "== checks"
"$PY" - "$WORK/runs" <<'PYEOF'
import json, sys, math
root = sys.argv[1]
A = [json.loads(l) for l in open(f"{root}/A/train.jsonl")]
B = [json.loads(l) for l in open(f"{root}/B/train.jsonl")]
ev = [json.loads(l) for l in open(f"{root}/A/eval.jsonl")]
first, last = A[0]["loss"], sum(r["loss"] for r in A[-5:]) / 5
print(f"loss trajectory (A): " + " ".join(f"{r['loss']:.3f}" for r in A))
assert last < first - 0.5, f"loss did not decrease: {first:.3f} -> {last:.3f}"
assert all(math.isfinite(r["loss_mm_seq"]) and math.isfinite(r["loss_text_seq"]) for r in A)
a = {r["step"]: r for r in A}; b = {r["step"]: r for r in B}
for s in (11, 12, 13, 14):
    da = abs(a[s]["loss"] - b[s]["loss"])
    assert da < 1e-5, f"resume mismatch at step {s}: {a[s]['loss']} vs {b[s]['loss']}"
print("resume OK: steps 11-14 losses match after stop/resume:",
      " ".join(f"{a[s]['loss']:.5f}={b[s]['loss']:.5f}" for s in (11, 12, 13, 14)))
C = {json.loads(l)["step"]: json.loads(l)["loss"] for l in open(f"{root}/C/train.jsonl")}
Dd = {json.loads(l)["step"]: json.loads(l)["loss"] for l in open(f"{root}/Dd/train.jsonl")}
for s in range(8, 15):
    assert abs(C[s] - Dd[s]) < 1e-5, (s, C[s], Dd[s])
print("DDP x workers resume OK: steps 8-14 match", [round(C[s], 4) for s in range(8, 15)])
L = 2
assert len(ev) >= 3, ev
for e in ev:
    assert len(e["diag/attn_mass"]) == L and len(e["diag/cos_visual"]) == L + 1 and len(e["diag/cos_text"]) == L + 1
    assert all(k in e for k in ("val/mm", "val/text", "val/mm/caption", "val/mm/stem"))
print("eval OK:", [(round(e["tokens"]/1e3), round(e["val/mm"], 3), round(e["val/text"], 3),
                    [round(x, 3) for x in e["diag/attn_mass"]], [round(x, 3) for x in e["diag/cos_visual"]]) for e in ev])
import os
ck = sorted(f for f in os.listdir(f"{root}/A") if f.startswith("ckpt_"))
assert ck and os.path.exists(f"{root}/A/state.pt") and os.path.exists(f"{root}/A/DONE"), ck
print("checkpoints:", ck)
import torch
E = [json.loads(l) for l in open(f"{root}/E/train.jsonl")]
assert len(E) == 12 and all(math.isfinite(r["loss"]) for r in E)
assert all(r["flops_per_token"] > r["flops_per_token_decoder"] > 0 for r in E)
print(f"tiny_siglip (trainable) OK: loss {E[0]['loss']:.3f} -> {E[-1]['loss']:.3f}, flops/token full {E[-1]['flops_per_token']:.3e} "
      f"vs decoder {E[-1]['flops_per_token_decoder']:.3e}")
F = {json.loads(l)["step"]: json.loads(l)["loss"] for l in open(f"{root}/F/train.jsonl")}
G = {json.loads(l)["step"]: json.loads(l)["loss"] for l in open(f"{root}/G/train.jsonl")}
for s in range(7, 11):
    assert abs(F[s] - G[s]) < 1e-5, (s, F[s], G[s])
sd = torch.load(f"{root}/G/state.pt", map_location="cpu", weights_only=False)["model"]
assert not any(k.startswith("front_end.vit.") for k in sd) and any(k.startswith("front_end.conv_pool") for k in sd)
cfgE = json.load(open(f"{root}/E/config.json"))
assert cfgE["config"]["data"]["mm_source_weights"] == {"synth_doc": 3.0} and "encoder" in cfgE["params"]
print("tiny_siglip (frozen) resume OK: steps 7-10 match", [round(F[s], 4) for s in range(7, 11)], "; state.pt has no ViT keys")
print("SMOKE TEST PASSED")
PYEOF
