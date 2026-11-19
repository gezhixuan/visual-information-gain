#!/usr/bin/env python
"""Image-ablation evaluation over saved checkpoints: how much of the multimodal validation loss
depends on the image, as a function of training tokens.

For every ckpt_*.pt of a run and every mm topic, the per-token loss on text positions is computed
under four pixel conditions on the same packed validation batches:
  real      the stored image
  permuted  the image's 32x32 patches shuffled within the image (colour statistics kept, layout destroyed)
  blank     every pixel 128 (uniform grey; the image span and its token count are kept)
  noise     uniform random pixels (fixed seed)
loss(blank) - loss(real) is the "visual information gain": ~0 while the decoder ignores images.

Usage: python -m vig.eval_vig --runs d1024_L12 d1280_L16 [--val_seqs 32] [--out ...jsonl]
       python -m vig.eval_vig --summarize [--out ...jsonl]
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys, time
import torch
from . import data as D
from . import eval as E
from . import paths
from .model import EncFreeDecoder, ModelConfig

RUNS = paths.RUNS
OUT = os.path.join(paths.OUT, "vig.jsonl")
CONDS = ("real", "permuted", "blank", "noise")


def ablate(batch: dict, cond: str, seed: int = 0) -> dict:
    b = dict(batch)
    pix = batch["pixels"]
    if cond == "real" or pix.numel() == 0:
        return b
    g = torch.Generator().manual_seed(seed)
    if cond == "blank":
        b["pixels"] = torch.full_like(pix, 128)
    elif cond == "noise":
        b["pixels"] = torch.randint(0, 256, pix.shape, generator=g, dtype=torch.uint8)
    elif cond == "permuted":
        idx = batch["img_index"]                       # [N, 2] = (row, position)
        key = idx[:, 0] * (batch["input_ids"].shape[1] + 1) + batch["image_id"][idx[:, 0], idx[:, 1]]
        perm = torch.arange(pix.shape[0])
        for k in key.unique():
            sel = (key == k).nonzero().flatten()
            perm[sel] = sel[torch.randperm(sel.numel(), generator=g)]
        b["pixels"] = pix[perm]
    else:
        raise ValueError(cond)
    return b


def ckpt_tokens(path: str) -> float:
    m = re.search(r"ckpt_([0-9.]+)B\.pt$", path)
    return float(m.group(1)) if m else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--runs_root", default=RUNS)
    ap.add_argument("--val_seqs", type=int, default=32, help="packed 4096-token sequences per topic")
    ap.add_argument("--topics", default="", help="comma list to restrict (tests)")
    ap.add_argument("--max_ckpts", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--val_cache", default=paths.VAL_SETS,
                    help="fixed validation batches shared by every evaluation (built once, then reused on every host)")
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.out)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adt = torch.bfloat16 if dev.type == "cuda" else None
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l); done.add((r["run"], r["tokens_B"], r["cond"], r["topic"]))
    val_sets = None
    for run in a.runs:
        rd = os.path.join(a.runs_root, run)
        cfgj = json.load(open(os.path.join(rd, "config.json")))["config"]
        mcfg = ModelConfig(**cfgj["model"]); mcfg.attn_impl = "flex" if dev.type == "cuda" else "eager"
        if val_sets is None:                      # same data config for every run
            if a.val_cache and os.path.exists(a.val_cache):
                val_sets = torch.load(a.val_cache, weights_only=False)
                print(f"[ablation] val sets loaded from {a.val_cache}", flush=True)
            else:
                base = {k: v for k, v in cfgj["data"].items() if k not in ("mm_source_weights", "mm_source_caps", "mm_sources")}
                dcfg = D.DataConfig(**{**base, "val_seqs_per_topic": a.val_seqs, "num_workers": 0})
                t0 = time.time(); val_sets = D.build_val_sets(dcfg)
                print(f"[ablation] val sets: {a.val_seqs} seqs/topic, built in {time.time()-t0:.0f}s", flush=True)
                if a.val_cache:
                    os.makedirs(os.path.dirname(a.val_cache), exist_ok=True); torch.save(val_sets, a.val_cache)
                    print(f"[ablation] val sets saved to {a.val_cache}", flush=True)
        topics = [t for t in val_sets["mm"] if not a.topics or t in a.topics.split(",")]
        model = EncFreeDecoder(mcfg).to(dev).eval()
        ckpts = sorted(glob.glob(os.path.join(rd, "ckpt_*.pt")), key=ckpt_tokens)
        if a.max_ckpts: ckpts = ckpts[:a.max_ckpts]
        for cp in ckpts:
            tok = ckpt_tokens(cp)
            if all((run, tok, c, t) in done for c in CONDS for t in topics):
                continue
            sd = torch.load(cp, map_location="cpu", weights_only=False)["model"]
            model.load_checkpoint_state_dict({k: v.to(torch.float32) for k, v in sd.items()})   # frozen ViT weights are not stored
            t0 = time.time()
            for t in topics:
                for c in CONDS:
                    if (run, tok, c, t) in done: continue
                    loss, n = E.batch_loss(model, ablate(val_sets["mm"][t], c), dev, micro=8, autocast_dtype=adt)
                    E.append_jsonl(a.out, {"run": run, "tokens_B": tok, "topic": t, "cond": c, "loss": loss, "n_tokens": n})
            print(f"[ablation] {run} {tok:.2f}B done in {time.time()-t0:.0f}s", flush=True)
    return 0


def summarize(path: str):
    import collections
    rows = [json.loads(l) for l in open(path)]
    tab = collections.defaultdict(dict)
    for r in rows:
        tab[(r["run"], r["tokens_B"], r["topic"])][r["cond"]] = r["loss"]
    runs = sorted({k[0] for k in tab}); topics = sorted({k[2] for k in tab})
    for run in runs:
        toks = sorted({k[1] for k in tab if k[0] == run})
        print(f"\n== {run}: visual information gain = loss(blank) - loss(real)  [loss(permuted) - loss(real)]")
        print("tokens_B " + " ".join(f"{t:>22s}" for t in topics) + "   mean")
        for tk in toks:
            cells, vig = [], []
            for t in topics:
                d = tab.get((run, tk, t), {})
                if {"real", "blank", "permuted"} <= set(d):
                    g = d["blank"] - d["real"]; p = d["permuted"] - d["real"]; vig.append(g)
                    cells.append(f"{g:+.3f} [{p:+.3f}] r={d['real']:.2f}")
                else:
                    cells.append(" " * 22)
            print(f"{tk:8.2f} " + " ".join(f"{c:>22s}" for c in cells) + (f"   {sum(vig)/len(vig):+.3f}" if vig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
