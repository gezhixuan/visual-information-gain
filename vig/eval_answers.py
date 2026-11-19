#!/usr/bin/env python
"""Behavioural check of reading: are the answers predicted correctly, with and without the image?

For chosen checkpoints every validation sample is scored under the real, the permuted and the uniform image exactly as in
eval_vig.py (same cache, same permutation), and the result is kept per token: the loss and whether the target
is the top-1 prediction (teacher forcing). The text of a sample is "Question: q\nAnswer: a\nQuestion: ..." and a caption
is treated as one answer. For every answer the row stores
  n, ok        answer tokens and how many of them are top-1
  exact        1 if every answer token is top-1, which is exactly the case in which greedy decoding from the question
               reproduces the answer
  nd, okd      answer tokens that contain a digit (numbers have to be read off the image)
  nx, okx      first decimal digit of every x= and y= coordinate (GUI: the target lies in the right cell of a 10 x 10 grid)
  loss         summed loss of the answer tokens
and per sample the count and summed loss of the other text tokens (the questions) and the loss of the two special targets
that every sample adds to the scored tokens, the end-of-image token and the end-of-sample token.

Rows are per sample: {run, tokens_B, topic, cond, i, rest_n, rest_loss, ve, eos, a: [[n, ok, exact, nd, okd, nx, okx, loss], ...]}.
Usage: python -m vig.eval_answers --runs d1280_L16 d1280_L16_siglip --tokens 40 [--out ...jsonl]
       python -m vig.eval_answers --summarize [--out ...jsonl]
"""
from __future__ import annotations
import argparse, collections, json, os, re, sys, time
import numpy as np, torch
import torch.nn.functional as F
from . import data as D
from . import eval as E
from .eval_vig import ablate, RUNS
from . import paths
from .model import EncFreeDecoder, ModelConfig

OUT = os.path.join(paths.OUT, "answers.jsonl")
CONDS = ("real", "permuted", "blank")
TOKENIZER = paths.TOKENIZER


def answer_char_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges of the answers, each including the newline that ends it. A caption is one range."""
    if not text.startswith("Question:"): return [(0, len(text))]
    spans = []
    for m in re.finditer(r"\nAnswer:", text):
        nq = text.find("\nQuestion:", m.end())
        spans.append((m.end(), nq + 1 if nq >= 0 else len(text)))
    return spans


def build_spans(batch: dict, tok) -> tuple[list, dict]:
    """-> ([{pos: flat predicting positions of all scored tokens of the sample, ans: [(positions, digit flags, xy flags)]}], stats).
    Samples come in the order of val_cache.recover_samples, which is the sample index of public_vig.py."""
    ids, sid, tid = batch["input_ids"].numpy(), batch["sample_id"].numpy(), batch["topic_id"].numpy()
    tg, lm, idx = batch["targets"].numpy(), batch["loss_mask"].numpy(), batch["img_index"].numpy()
    T = ids.shape[1]; out = []; st = collections.Counter()
    for b in range(ids.shape[0]):
        pos_img = idx[idx[:, 0] == b, 1]
        for s in np.unique(sid[b][tid[b] >= 0]):
            span = np.where((sid[b] == s) & (tid[b] >= 0))[0]; a, e = span[0], span[-1] + 1
            n = int(((pos_img >= a) & (pos_img < e)).sum())
            assert ids[b, a] == D.VISION_START and ids[b, a + 1 + n] == D.VISION_END and ids[b, e - 1] == D.EOS
            p = np.arange(a + 2 + n, e)                           # text tokens, the last one is the end of sample
            q = np.arange(a + 1 + n, e)                           # scored targets: end of image, text, end of sample
            assert (tg[b, q - 1] == ids[b, q]).all() and lm[b, q - 1].all(), "a scored token is not predicted at the position before it"
            text_ids = ids[b, a + 2 + n:e - 1].tolist(); text = tok.decode(text_ids)
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            rec = {"pos": b * T + (q - 1), "ans": []}
            if enc["input_ids"] != text_ids:
                st["retokenization_differs"] += 1                 # no character offsets: the sample keeps only its total
            else:
                off = np.array(enc["offset_mapping"]).reshape(-1, 2)
                for (c0, c1) in answer_char_spans(text):
                    j = np.where((off[:, 0] >= c0) & (off[:, 0] < c1))[0]
                    if not len(j): continue
                    dig = np.array([any(ch.isdigit() for ch in text[off[k, 0]:off[k, 1]]) for k in j])
                    xy = np.zeros(len(j), bool)
                    for m in re.finditer(r"[xy]=0\.(\d)", text[c0:c1]):
                        c = c0 + m.start(1); xy |= (off[j, 0] <= c) & (off[j, 1] > c)
                    rec["ans"].append((b * T + (p[j] - 1), dig, xy)); st["answers"] += 1; st["answer_tokens"] += len(j)
            st["samples"] += 1; st["scored_tokens"] += len(q)
            out.append(rec)
    assert st["scored_tokens"] == int(lm.sum()), (st["scored_tokens"], int(lm.sum()))
    return out, dict(st)


@torch.no_grad()
def token_maps(model, batch: dict, dev, micro: int = 8, adt=None) -> tuple[np.ndarray, np.ndarray]:
    """Flat [B*T] arrays: loss (nan where not scored) and top-1 correctness of every scored position."""
    B, T = batch["input_ids"].shape
    loss = np.full(B * T, np.nan, np.float64); ok = np.zeros(B * T, bool)
    for i in range(0, B, micro):
        sub = E._to(E.slice_batch(batch, i, min(i + micro, B)), dev)
        with torch.autocast(device_type=dev.type, dtype=adt, enabled=adt is not None):
            h = E._fwd(model, sub)
            sel = sub["loss_mask"].reshape(-1).nonzero(as_tuple=True)[0]
            hs = h.reshape(-1, h.shape[-1])[sel]; ts = sub["targets"].reshape(-1)[sel]
            ls, os_ = [], []
            for c in range(0, hs.shape[0], 8192):
                lg = model.logits(hs[c:c + 8192]).float()
                ls.append(F.cross_entropy(lg, ts[c:c + 8192], reduction="none")); os_.append(lg.argmax(-1) == ts[c:c + 8192])
        g = i * T + sel.cpu().numpy()
        loss[g] = torch.cat(ls).double().cpu().numpy(); ok[g] = torch.cat(os_).cpu().numpy()
    return loss, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[]); ap.add_argument("--runs_root", default=RUNS)
    ap.add_argument("--tokens", default="40", help="comma list of checkpoints in billions of tokens")
    ap.add_argument("--topics", default=""); ap.add_argument("--max_rows", type=int, default=0, help="first rows of every batch only (tests)")
    ap.add_argument("--out", default=OUT); ap.add_argument("--val_cache", default=paths.VAL_SETS); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize or not a.runs: return summarize(a.out)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); adt = torch.bfloat16 if dev.type == "cuda" else None
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l); done.add((r["run"], r["tokens_B"], r["topic"], r["cond"]))
    val = torch.load(a.val_cache, weights_only=False)["mm"]
    if a.max_rows: val = {t: E.slice_batch(b, 0, a.max_rows) | {"n_samples": -1} for t, b in val.items()}
    topics = [t for t in val if not a.topics or t in a.topics.split(",")]
    spans = {}
    for t in topics:
        spans[t], st = build_spans(val[t], tok); print(f"[answers] {t}: {st}", flush=True)
    for run in a.runs:
        rd = os.path.join(a.runs_root, run)
        mcfg = ModelConfig(**json.load(open(os.path.join(rd, "config.json")))["config"]["model"]); mcfg.attn_impl = "flex" if dev.type == "cuda" else "eager"
        model = EncFreeDecoder(mcfg).to(dev).eval()
        for tk in [float(x) for x in a.tokens.split(",")]:
            cp = os.path.join(rd, f"ckpt_{tk:.2f}B.pt")
            if not os.path.exists(cp): print(f"[answers] {run}: no checkpoint at {tk}B", flush=True); continue
            if all((run, tk, t, c) in done for t in topics for c in CONDS): continue
            sd = torch.load(cp, map_location="cpu", weights_only=False)["model"]
            model.load_checkpoint_state_dict({k: v.to(torch.float32) for k, v in sd.items()})
            t0 = time.time()
            for t in topics:
                for c in CONDS:
                    if (run, tk, t, c) in done: continue
                    loss, ok = token_maps(model, ablate(val[t], c), dev, adt=adt)
                    with open(a.out, "a") as f:
                        for i, s in enumerate(spans[t]):
                            ans = [[len(p), int(ok[p].sum()), int(ok[p].all()), int(dg.sum()), int(ok[p][dg].sum()), int(xy.sum()), int(ok[p][xy].sum()),
                                    float(loss[p].sum())] for p, dg, xy in s["ans"]]
                            ve, eos = float(loss[s["pos"][0]]), float(loss[s["pos"][-1]])       # end-of-image and end-of-sample targets
                            tot_n, tot_l = len(s["pos"]) - 2, float(loss[s["pos"]].sum()) - ve - eos
                            f.write(json.dumps({"run": run, "tokens_B": tk, "topic": t, "cond": c, "i": i, "rest_n": tot_n - sum(x[0] for x in ans),
                                                "rest_loss": tot_l - sum(x[7] for x in ans), "ve": ve, "eos": eos, "a": ans}) + "\n")
                    print(f"[answers] {run} {tk:.2f}B {t} {c}: mean loss {np.nanmean(loss):.4f} over {int((~np.isnan(loss)).sum())} tokens", flush=True)
            print(f"[answers] {run} {tk:.2f}B done in {time.time() - t0:.0f}s", flush=True)
    return 0


def summarize(path: str):
    agg = collections.defaultdict(lambda: np.zeros(11))
    for l in open(path):
        r = json.loads(l); k = (r["run"], r["tokens_B"], r["topic"], r["cond"]); x = agg[k]
        for n, ok, ex, nd, okd, nx, okx, ls in r["a"]:
            x += [n, ok, 1, ex, nd, okd, nx, okx, ls, 0, 0]
        x[9] += r["rest_n"]; x[10] += r["rest_loss"]
    print("run                 tokens topic    cond       answers  exact%  tok%   digit%  (n_digit)  xy%  (n_xy)   loss/answer_tok  loss/other_tok")
    for k in sorted(agg):
        n, ok, na, ex, nd, okd, nx, okx, ls, rn, rl = agg[k]
        print(f"{k[0]:19s} {k[1]:5.1f}  {k[2]:8s} {k[3]:9s} {int(na):6d}  {100 * ex / max(na, 1):5.1f}  {100 * ok / max(n, 1):5.1f}  "
              f"{100 * okd / nd if nd else float('nan'):5.1f}  ({int(nd):5d})  {100 * okx / nx if nx else float('nan'):5.1f} ({int(nx):4d})   {ls / max(n, 1):6.3f}   {rl / max(rn, 1):6.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
