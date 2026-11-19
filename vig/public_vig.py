#!/usr/bin/env python
"""Visual information gain of public vision-language models on the shared validation samples.

For every sample of the 32-pixel validation cache (same images and same text as for our own runs) the text is scored
by a public model under three image conditions: real, permuted (32x32 pixel blocks shuffled within the image, fixed
seed per sample, identical for every model) and uniform (every pixel 128). Only the tokens of our text are scored.
Formats: "raw" puts the text directly after the image tokens and scores all of it (the protocol of our own runs).
"qa" uses the model's own chat template: the image and each question go into user turns, each answer is a reply, and
only the answers are scored (a caption is scored as the reply to the image alone). "chat" scores the whole text as one
reply to the image, which is unnatural for question-answer text and kept only for comparison. Tokenizers differ between models, so every row stores the summed
loss together with the number of scored tokens and of characters (nats per character is tokenizer independent).

Rows are per sample and appended to <out_dir>/<tag>.jsonl, finished rows are skipped on restart.
Usage: python -m vig.public_vig --models gemma-4-12B-it:chat gemma-4-12B:raw Qwen2.5-VL-7B-Instruct:chat [--budget 1120] [--max_samples 2]
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, traceback
import numpy as np, torch
from PIL import Image

from . import paths

MODELS = paths.MODELS + "/"
OUT = os.path.join(paths.OUT, "public_vig") + "/"
CONDS = ("real", "permuted", "uniform")
EOS_QWEN = 151643


def val_mm() -> dict:
    """The validation batches of our own evaluation, one per image type."""
    return torch.load(paths.VAL_SETS, map_location="cpu", weights_only=False)["mm"]


def common_tokens() -> dict:
    """{topic: tokens our own evaluation scores per sample (text, end-of-image and end-of-sample tokens), in cache order}."""
    src = val_mm()
    out = {}
    for topic, batch in src.items():
        sid, tid, lm = batch["sample_id"].numpy(), batch["topic_id"].numpy(), batch["loss_mask"].numpy().astype(bool)
        out[topic] = np.array([int(lm[b][(sid[b] == s) & (tid[b] >= 0)].sum()) for b in range(sid.shape[0]) for s in np.unique(sid[b][tid[b] >= 0])])
        assert out[topic].sum() == lm.sum()
    return out


def load_samples(max_samples: int = 0, topics: str = "") -> dict:
    """{topic: [(text, uint8 image array)]} recovered from the shared cache, in cache order."""
    from transformers import AutoTokenizer
    from .val_cache import recover_samples
    tok = AutoTokenizer.from_pretrained(paths.TOKENIZER)
    src = val_mm()
    out = {}
    for topic, batch in src.items():
        if topics and topic not in topics.split(","): continue
        rec = recover_samples(batch)
        if max_samples: rec = rec[:max_samples]
        items = []
        for ids, _, arr in rec:
            ids = ids[ids != EOS_QWEN]
            items.append((tok.decode(ids.tolist(), skip_special_tokens=False), arr))
        out[topic] = items
    return out


def make_image(arr: np.ndarray, cond: str, seed: int) -> Image.Image:
    if cond == "real": return Image.fromarray(arr)
    if cond == "uniform": return Image.fromarray(np.full_like(arr, 128))
    h, w = arr.shape[0] // 32, arr.shape[1] // 32
    blocks = arr.reshape(h, 32, w, 32, 3).transpose(0, 2, 1, 3, 4).reshape(h * w, 32, 32, 3)
    blocks = blocks[np.random.default_rng(seed).permutation(h * w)]
    return Image.fromarray(blocks.reshape(h, w, 32, 32, 3).transpose(0, 2, 1, 3, 4).reshape(h * 32, w * 32, 3))


def prefix_for(proc, fmt: str) -> str:
    if fmt == "chat":
        p = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}]}], add_generation_prompt=True)
        return p + (" " if p.endswith(":") else "")
    it = proc.image_token
    return "<|vision_start|>" + it + "<|vision_end|>" if it == "<|image_pad|>" else it


def parse_qa(text: str):
    """'Question: q\nAnswer: a\nQuestion: ...' -> [(q, a), ...]; None for text without this structure (captions)."""
    if not text.startswith("Question:"): return None
    turns = []
    for seg in re.split(r"(?:^|\n)Question:", text)[1:]:
        if "\nAnswer:" not in seg: return None
        q, a = seg.split("\nAnswer:", 1)
        turns.append((q.strip(), a.strip()))
    return turns


def qa_inputs(proc, text: str, img: Image.Image):
    """Chat-template conversation with the answers (or the caption) as replies. -> (processor output, labels, scored characters)."""
    tok = proc.tokenizer; turns = parse_qa(text)
    if turns is None: turns = [("", text.strip())]
    msgs = []
    for j, (q, a) in enumerate(turns):
        content = ([{"type": "image"}] if j == 0 else []) + ([{"type": "text", "text": q}] if q or j else [])
        msgs += [{"role": "user", "content": content}, {"role": "assistant", "content": [{"type": "text", "text": a}]}]
    R = proc.apply_chat_template(msgs, add_generation_prompt=False)
    ast = not (tok.bos_token and R.startswith(tok.bos_token))
    full = proc(text=[R], images=[img], return_tensors="pt", add_special_tokens=ast)
    ids = full["input_ids"][0]
    rest = R[R.index(proc.image_token) + len(proc.image_token):]
    enc = tok(rest, add_special_tokens=False, return_offsets_mapping=True)
    rid, off = enc["input_ids"], enc["offset_mapping"]
    m = 0                                                   # the text after the image tokenizes the same inside the full sequence
    while m < min(len(rid), len(ids)) and rid[-1 - m] == int(ids[-1 - m]): m += 1
    labels = torch.full_like(ids, -100); pos = 0; n_char = 0
    for q, a in turns:
        if not a: continue
        i = rest.index(a, pos); pos = i + len(a); n_char += len(a)
        for j, (s, e) in enumerate(off):
            if e > s and s >= i and e <= pos:
                assert len(rid) - j <= m, "answer token outside the aligned suffix"
                labels[len(ids) - len(rid) + j] = ids[len(ids) - len(rid) + j]
    return full, labels, n_char


@torch.no_grad()
def score(model, proc, prefix: str, text: str, img: Image.Image, fmt: str = "raw") -> tuple[float, int, int, int]:
    tok = proc.tokenizer
    if fmt == "qa":
        full, labels, n_char = qa_inputs(proc, text, img); ids = full["input_ids"][0]
    else:
        ast = not (tok.bos_token and prefix.startswith(tok.bos_token))
        full = proc(text=[prefix + text], images=[img], return_tensors="pt", add_special_tokens=ast)
        pre = proc(text=[prefix], images=[img], return_tensors="pt", add_special_tokens=ast)["input_ids"][0]
        ids = full["input_ids"][0]
        k = 0
        while k < min(len(pre), len(ids)) and pre[k] == ids[k]: k += 1          # scored tokens start after the common prefix
        assert k >= len(pre) - 1 and k < len(ids), (len(pre), len(ids), k)
        labels = ids.clone(); labels[:k] = -100; n_char = len(text)
    full = full.to(model.device)
    for key, v in list(full.items()):
        if torch.is_tensor(v) and v.is_floating_point(): full[key] = v.to(model.dtype)
    logits = model(**full).logits[0, :-1].float()
    tgt = labels[1:].to(logits.device)
    loss = torch.nn.functional.cross_entropy(logits, tgt, ignore_index=-100, reduction="sum")
    n_img = int((ids == tok.convert_tokens_to_ids(proc.image_token)).sum()) if getattr(proc, "image_token", None) else -1
    return float(loss), int((tgt != -100).sum()), n_img, n_char


def run_model(spec: str, samples: dict, out_dir: str, budget: int):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    name, fmt = spec.split(":") if ":" in spec else (spec, "chat")
    tag = f"{name}_{fmt}" + (f"_b{budget}" if budget else "")
    path = os.path.join(out_dir, tag + ".jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); done.add((r["topic"], r["i"], r["cond"]))
    todo = sum(1 for t, items in samples.items() for i in range(len(items)) for c in CONDS if (t, i, c) not in done)
    if not todo: print(f"[public_vig] {tag}: complete", flush=True); return
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODELS + name)
    if budget:
        assert hasattr(proc.image_processor, "max_soft_tokens"), "--budget is for Gemma 4 processors"
        proc.image_processor.max_soft_tokens = budget
    model = AutoModelForImageTextToText.from_pretrained(MODELS + name, dtype=torch.bfloat16, device_map="auto").eval()
    prefix = prefix_for(proc, fmt) if fmt != "qa" else ""
    print(f"[public_vig] {tag}: loaded in {time.time() - t0:.0f}s, prefix {prefix!r}, {todo} forward passes", flush=True)
    with open(path, "a") as f:
        for ti, (topic, items) in enumerate(samples.items()):
            t1 = time.time()
            for i, (text, arr) in enumerate(items):
                for cond in CONDS:
                    if (topic, i, cond) in done: continue
                    loss, n_tok, n_img, n_char = score(model, proc, prefix, text, make_image(arr, cond, seed=1000 * ti + i), fmt)
                    f.write(json.dumps({"model": name, "format": fmt, "budget": budget, "topic": topic, "i": i, "cond": cond, "loss_sum": loss,
                                        "n_tok": n_tok, "n_char": n_char, "n_img_tok": n_img}) + "\n")
                f.flush()
            print(f"[public_vig] {tag} {topic}: {len(items)} samples in {time.time() - t1:.0f}s", flush=True)
    del model; torch.cuda.empty_cache()


def summarize(out_dir: str):
    import collections, glob
    ref = common_tokens()
    for path in sorted(glob.glob(out_dir + "*.jsonl")):
        agg = collections.defaultdict(lambda: [0.0, 0, 0, 0])      # summed loss, tokens of the model, common tokens, samples
        for l in open(path):
            r = json.loads(l); a = agg[(r["topic"], r["cond"])]; a[0] += r["loss_sum"]; a[1] += r["n_tok"]; a[2] += int(ref[r["topic"]][r["i"]]); a[3] += 1
        print(f"\n== {os.path.basename(path)[:-6]}   (nats per token; share = VIG_total / L_unif; layout fraction = VIG_layout / VIG_total)")
        print("topic      n   L_real  L_perm  L_unif   total  layout  global   share  layout_frac")
        for t in sorted({k[0] for k in agg}):
            if not all((t, c) in agg for c in CONDS): continue
            L = {c: agg[(t, c)][0] / agg[(t, c)][2] for c in CONDS}; tot = L["uniform"] - L["real"]; lay = L["permuted"] - L["real"]
            print(f"{t:8s} {agg[(t, 'real')][3]:4d}  {L['real']:6.3f}  {L['permuted']:6.3f}  {L['uniform']:6.3f}  {tot:6.3f}  {lay:6.3f}  {tot - lay:6.3f}  {100 * tot / L['uniform']:5.1f}%  {lay / tot if tot > 0 else float('nan'):6.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[]); ap.add_argument("--out_dir", default=OUT); ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=0); ap.add_argument("--topics", default=""); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.models:
        S = load_samples(a.max_samples, a.topics)
        print("[public_vig] samples per type:", {t: len(v) for t, v in S.items()}, flush=True)
        for spec in a.models:
            try: run_model(spec, S, a.out_dir, a.budget)
            except Exception:  # noqa: BLE001
                print(f"[public_vig] {spec} FAILED\n{traceback.format_exc()}", flush=True)
    if a.summarize or not a.models: summarize(a.out_dir)
