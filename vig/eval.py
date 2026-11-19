"""Validation losses per topic and two diagnostics of Chen et al. (their Fig. 7C and Fig. 9).

evaluate(model, val_sets, device) ->
  {"val/mm": mean over topics, "val/mm/<topic>": ..., "val/text": mean over sources,
   "val/text/<source>": ..., "diag/attn_mass": [L], "diag/cos_visual": [L+1],
   "diag/cos_text": [L+1], "diag/n_text_queries": int, "diag/n_visual": int}

(a) attention mass on visual tokens per layer: for every text query position in the diag
    batch that can see at least one image key, the attention probability it puts on
    image keys, averaged over heads and such queries (eager attention path).
(b) cosine similarity between each layer's residual-stream state and the layer-0 input
    (token/patch embeddings), averaged over visual tokens and, separately, text tokens.
"""
from __future__ import annotations

import json
import time
from typing import Optional

import torch
import torch.nn.functional as F


def _to(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _fwd(model, b: dict, attn_impl: Optional[str] = None, on_layer=None):
    return model(b["input_ids"], b["sample_id"], b["image_id"], b["pixels"], b["grid_xy"], b["img_index"],
                 attn_impl=attn_impl, on_layer=on_layer)


def slice_batch(batch: dict, i: int, j: int) -> dict:
    """Rows [i, j) of a collated batch, including the row-indexed pixel tensors."""
    B = batch["input_ids"].shape[0]
    sub = {k: v[i:j] for k, v in batch.items()
           if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == B and k not in ("pixels", "grid_xy", "img_index")}
    sel = (batch["img_index"][:, 0] >= i) & (batch["img_index"][:, 0] < j)
    sub["pixels"] = batch["pixels"][sel]
    sub["grid_xy"] = batch["grid_xy"][sel]
    sub["img_index"] = batch["img_index"][sel].clone()
    sub["img_index"][:, 0] -= i
    return sub


@torch.no_grad()
def batch_loss(model, batch: dict, device, micro: int = 8, autocast_dtype=None) -> tuple[float, int]:
    """Mean per-token loss over loss positions of a (possibly large) val batch."""
    tot, n = 0.0, 0
    B = batch["input_ids"].shape[0]
    for i in range(0, B, micro):
        sub = _to(slice_batch(batch, i, min(i + micro, B)), device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            h = _fwd(model, sub)
            losses, _ = model.token_losses(h, sub["targets"], sub["loss_mask"])
        tot += float(losses.sum())
        n += int(losses.numel())
    return (tot / max(n, 1)), n


@torch.no_grad()
def diagnostics(model, batch: dict, device, autocast_dtype=None) -> dict:
    b = _to({k: v for k, v in batch.items() if k != "state"}, device)
    is_img = b["image_id"] >= 0                               # [B, T]
    is_text = (~is_img) & (b["topic_id"] >= 0)
    n_layers = len(model.blocks)
    attn_mass = [0.0] * n_layers
    cos_vis, cos_txt = [1.0] * (n_layers + 1), [1.0] * (n_layers + 1)
    state = {"h0": None, "q_sel": None}

    def on_layer(l, x, probs, mask):
        if l == 0:
            state["h0"] = x.float()
            vis_img = (mask[:, 0] & is_img[:, None, :]).any(-1)   # text queries that see an image
            state["q_sel"] = is_text & vis_img
            return
        cs = F.cosine_similarity(x.float(), state["h0"], dim=-1)   # [B, T]
        cos_vis[l] = float(cs[is_img].mean()) if is_img.any() else float("nan")
        cos_txt[l] = float(cs[is_text].mean())
        mass = (probs * is_img[:, None, None, :].to(probs.dtype)).sum(-1)   # [B, H, T]
        q = state["q_sel"]
        attn_mass[l - 1] = float(mass.permute(0, 2, 1)[q].mean()) if q.any() else float("nan")

    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        _fwd(model, b, attn_impl="eager", on_layer=on_layer)
    return {"diag/attn_mass": attn_mass, "diag/cos_visual": cos_vis, "diag/cos_text": cos_txt,
            "diag/n_text_queries": int(state["q_sel"].sum()), "diag/n_visual": int(is_img.sum())}


@torch.no_grad()
def evaluate(model, val_sets: dict, device, autocast_dtype=None, micro: int = 8) -> dict:
    model.eval()
    t0 = time.time()
    out: dict = {}
    mm_vals = []
    for topic, batch in val_sets["mm"].items():
        v, n = batch_loss(model, batch, device, micro, autocast_dtype)
        out[f"val/mm/{topic}"] = v
        out[f"val/mm/{topic}/n_tokens"] = n
        mm_vals.append(v)
    if mm_vals:
        out["val/mm"] = sum(mm_vals) / len(mm_vals)      # topics weighted uniformly (Chen et al., App. A.4)
    tx_vals = []
    for name, batch in val_sets["text"].items():
        v, n = batch_loss(model, batch, device, micro, autocast_dtype)
        out[f"val/text/{name}"] = v
        out[f"val/text/{name}/n_tokens"] = n
        tx_vals.append(v)
    if tx_vals:
        out["val/text"] = sum(tx_vals) / len(tx_vals)
    if val_sets.get("diag") is not None:
        out.update(diagnostics(model, val_sets["diag"], device, autocast_dtype))
    out["eval_seconds"] = time.time() - t0
    model.train()
    return out


def append_jsonl(path: str, rec: dict):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
