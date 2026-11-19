"""Synthetic shards in the schema of the README for CPU tests.

mm/<source>/{train,val}-000000.tar with <key>.jpg + <key>.json; text/<source>/{train,val}.bin
(uint32, EOS-separated) + .idx (int64 doc starts) + meta.json.
"""
import argparse
import io
import json
import os
import tarfile

import numpy as np
from PIL import Image

EOS = 151643
TOPICS = ["caption", "general", "ocr", "chart", "gui", "stem"]


def write_mm(root, source, split, n, rng, max_side, max_text, topics):
    os.makedirs(os.path.join(root, "mm", source), exist_ok=True)
    path = os.path.join(root, "mm", source, f"{split}-000000.tar")
    with tarfile.open(path, "w") as tf:
        for i in range(n):
            w = 32 * int(rng.integers(1, max_side // 32 + 1))
            h = 32 * int(rng.integers(1, max_side // 32 + 1))
            # structured image: smooth gradient + colour blocks so the loss can use it
            yy, xx = np.mgrid[0:h, 0:w]
            topic = topics[i % len(topics)]
            base = np.stack([(xx * 255 // max(w, 1)), (yy * 255 // max(h, 1)), np.full_like(xx, 40 * (i % 6))], -1)
            img = Image.fromarray(base.astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            n_text = int(rng.integers(4, max_text + 1))
            # text correlated with topic & image size so a tiny model can learn something
            text_ids = (rng.integers(0, 200, size=n_text) + 1000 * TOPICS.index(topic)).tolist()
            # n_img is the number of images in the sample (schema: always 1; the image-token count is derived from
            # w, h). It used to hold the token count, which made data.mm_sample drop every image larger than 32x32
            # and let a loader worker whose stride held no 32x32 image spin forever.
            meta = {"text_ids": text_ids, "source": source, "topic": topic, "w": w, "h": h,
                    "n_img": 1, "split": split}
            key = f"{source}_{split}_{i:06d}"
            for name, data in ((f"{key}.jpg", buf.getvalue()), (f"{key}.json", json.dumps(meta).encode())):
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
    return path


def write_text(root, source, topic, n_docs, rng, max_len, split):
    d = os.path.join(root, "text", source)
    os.makedirs(d, exist_ok=True)
    docs, starts, pos = [], [], 0
    for _ in range(n_docs):
        n = int(rng.integers(3, max_len))
        ids = (rng.integers(0, 300, size=n) + 5000 * (["web", "math", "code", "wiki"].index(topic) + 1)).astype(np.uint32)
        ids = np.concatenate([ids, np.array([EOS], dtype=np.uint32)])
        starts.append(pos)
        pos += len(ids)
        docs.append(ids)
    arr = np.concatenate(docs)
    arr.tofile(os.path.join(d, f"{split}.bin"))
    np.asarray(starts, dtype=np.int64).tofile(os.path.join(d, f"{split}.idx"))
    if split == "train":
        json.dump({"tokenizer": "Qwen3-0.6B-Base", "n_tokens": int(len(arr)), "n_docs": n_docs, "topic": topic},
                  open(os.path.join(d, "meta.json"), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_mm", type=int, default=120)
    ap.add_argument("--n_text", type=int, default=200)
    ap.add_argument("--max_side", type=int, default=96)
    ap.add_argument("--max_text", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for src, topics in (("synth_cap", ["caption", "general", "ocr"]), ("synth_doc", ["chart", "gui", "stem"])):
        for split in ("train", "val"):
            write_mm(a.out, src, split, a.n_mm, rng, a.max_side, a.max_text, topics)
    for src, topic in (("synth_web", "web"), ("synth_code", "code")):
        for split in ("train", "val"):
            write_text(a.out, src, topic, a.n_text, rng, a.max_text * 3, split)
    print("synthetic shards written to", a.out)


if __name__ == "__main__":
    main()
