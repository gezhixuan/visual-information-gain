#!/usr/bin/env python
"""Build a 16-pixel validation cache that holds exactly the samples of the shared 32-pixel cache.

The 32-pixel cache (val_sets_32.pt) stores the raw pixels of every image as 32x32 patches, so each image
can be reassembled and cut into 16x16 patches without touching the shards. Every sample keeps its text tokens and gets
four times the image-pad tokens. Samples are repacked in their original order into 8,192-token sequences (as many as
needed, whole samples only). Document masking makes the loss of a sample independent of how it is packed, so the
16-pixel run is then scored on the same images and the same text tokens as every 32-pixel run.

Checks before writing: (1) repacking the recovered samples at 32 pixels reproduces the source cache tensor for tensor,
(2) the image rebuilt from the 16-pixel patches equals the image rebuilt from the 32-pixel patches, (3) the number of
scored text tokens per image type is unchanged.
Usage: python -m vig.val_cache [--src ...val_sets_32.pt] [--dst ...val_sets_32_p16.pt]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch
from . import data as D
from . import paths


def image_from_patches(pix: np.ndarray, grid: np.ndarray, patch: int) -> np.ndarray:
    gw, gh = int(grid[:, 0].max()) + 1, int(grid[:, 1].max()) + 1
    assert len(pix) == gw * gh and (grid == D.grid_coords(gw * patch, gh * patch, patch)).all(), "patches are not a full row-major grid"
    return pix.reshape(gh, gw, patch, patch, 3).transpose(0, 2, 1, 3, 4).reshape(gh * patch, gw * patch, 3)


def patches_from_image(arr: np.ndarray, patch: int) -> np.ndarray:
    gh, gw = arr.shape[0] // patch, arr.shape[1] // patch
    return arr.reshape(gh, patch, gw, patch, 3).transpose(0, 2, 1, 3, 4).reshape(gh * gw, patch * patch * 3)


def recover_samples(batch: dict) -> list[tuple[np.ndarray, int, np.ndarray]]:
    """-> [(text ids incl. EOS, topic id, image array)] in packing order."""
    ids, sid, tid = batch["input_ids"].numpy(), batch["sample_id"].numpy(), batch["topic_id"].numpy()
    pix, grid, idx = batch["pixels"].numpy(), batch["grid_xy"].numpy(), batch["img_index"].numpy()
    out = []
    for b in range(ids.shape[0]):
        rows = np.where(idx[:, 0] == b)[0]; pos = idx[rows, 1]
        for s in np.unique(sid[b][tid[b] >= 0]):
            span = np.where((sid[b] == s) & (tid[b] >= 0))[0]; a, e = span[0], span[-1] + 1
            assert (np.diff(span) == 1).all() and ids[b, a] == D.VISION_START
            r = rows[(pos >= a) & (pos < e)]; n = len(r)
            assert n > 0 and (pos[(pos >= a) & (pos < e)] == np.arange(a + 1, a + 1 + n)).all()
            assert (ids[b, a + 1:a + 1 + n] == D.IMAGE_PAD).all() and ids[b, a + 1 + n] == D.VISION_END and ids[b, e - 1] == D.EOS
            out.append((ids[b, a + 2 + n:e].copy(), int(tid[b, a]), image_from_patches(pix[r], grid[r], 32)))
    return out


def make_sample(text: np.ndarray, topic: int, arr: np.ndarray, patch: int) -> D.Sample:
    h, w = arr.shape[:2]; n = (h // patch) * (w // patch)
    ids = np.concatenate([[D.VISION_START], np.full(n, D.IMAGE_PAD, dtype=np.int64), [D.VISION_END], text])
    return D.Sample(ids=ids, topic=topic, kind=1, pixels=patches_from_image(arr, patch), grid=D.grid_coords(w, h, patch), img_offset=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", default=paths.VAL_SETS); ap.add_argument("--dst", default=paths.VAL_SETS.replace("_32.pt", "_32_p16.pt"))
    ap.add_argument("--seq_len", type=int, default=8192); a = ap.parse_args()
    src = torch.load(a.src, map_location="cpu", weights_only=False)
    out = {"mm": {}, "text": {}, "diag": None}
    for topic, batch in src["mm"].items():
        rec = recover_samples(batch)
        # check 1: the recovered samples repack to the source cache at 32 pixels
        back = D.collate(D._pack_list([make_sample(t, k, arr, 32) for t, k, arr in rec], batch["input_ids"].shape[1], kind=1))
        for key, v in batch.items():
            same = torch.equal(v, back[key]) if torch.is_tensor(v) else (v == back[key] or key == "n_dropped")
            assert same, f"{topic}: 32-pixel round trip differs in {key}"
        s16 = [make_sample(t, k, arr, 16) for t, k, arr in rec]
        # check 2: same pixels
        for (t, k, arr), s in zip(rec, s16):
            assert (image_from_patches(s.pixels, s.grid, 16) == arr).all()
        too_long = [len(s) for s in s16 if len(s) > a.seq_len]
        assert not too_long, f"{topic}: {len(too_long)} samples exceed {a.seq_len} tokens at 16 pixels: {too_long[:5]}"
        seqs = D._pack_list(s16, a.seq_len, kind=1)
        b16 = D.collate(seqs)
        # check 3: same samples and same scored text tokens
        assert b16["n_samples"] == batch["n_samples"] == len(rec), (topic, b16["n_samples"], batch["n_samples"], len(rec))
        n32, n16 = int(batch["loss_mask"].sum()), int(b16["loss_mask"].sum())
        assert n32 == n16, (topic, n32, n16)
        assert torch.equal(batch["targets"][batch["loss_mask"]], b16["targets"][b16["loss_mask"]]), f"{topic}: scored targets differ"
        out["mm"][topic] = b16
        print(f"{topic:8s} samples {len(rec):4d}  scored text tokens {n16:6d}  32-px seqs {batch['input_ids'].shape[0]} x {batch['input_ids'].shape[1]}"
              f"  ->  16-px seqs {b16['input_ids'].shape[0]} x {a.seq_len}  image tokens {batch['pixels'].shape[0]} -> {b16['pixels'].shape[0]}", flush=True)
    torch.save(out, a.dst); print("saved", a.dst)


if __name__ == "__main__":
    main()
