#!/usr/bin/env python
"""Tokenize the text sources into {train,val}.bin (uint32) + .idx (int64 doc starts) + meta.json (see the README).

Sub-commands (--pilot works on 2 files x 20,000 rows per source under shards/text_pilot/):
  index    --source S [--n-files K]   list the first K parquet files, count rows, draw val (20,000 docs, seed 0)
  plan     [--files-per-task F]       write <out>/plan.json: one task per (source, file range) for a job array
  tokenize --task K | --source S --files a:b   tokenize files into parts/ (resumable per file)
  merge    --source S [--budget 60B]  concatenate parts in file order up to the token budget -> train.bin/idx, val.bin/idx, meta.json
  all-index-plan                      index every source, then plan

Layout under <out>/<source>/:
  parts/train-FFFFF.bin, parts/train-FFFFF.idx (int64 doc starts, no trailing total), parts/val-FFFFF.{bin,idx}, parts/FFFFF.json
  train.bin, train.idx (n_docs+1 int64 offsets: doc i = bin[idx[i]:idx[i+1]], last token of every doc is EOS 151643),
  val.bin, val.idx, meta.json = {tokenizer, n_tokens, n_docs, topic, val: {...}, files_merged, budget, ...}

Budgets (first pass): fineweb-edu 60B (index default --n-files 96 ~ 68B available), open-web-math all,
  starcoder-python all, starcoder-markdown 10B (--n-files 40), wikipedia-en all.
Extending later: (1) `index --source fineweb-edu --n-files 140 --keep-val` keeps the existing val positions and
  only appends files; (2) `plan` + `tokenize` the new file range (existing parts are skipped);
  (3) `merge --budget 100B`: merge is incremental, it appends the parts not yet listed in meta.json["files_merged"].
  The budget is applied at file granularity: files are appended while the merged total is below the budget, so the
  final size overshoots by at most one file (~0.7B tokens for fineweb-edu).
"""
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from . import common as C

DEFAULT_N_FILES = {"fineweb-edu": 96, "starcoder-markdown": 40}
DEFAULT_BUDGET = {"fineweb-edu": 60e9, "starcoder-markdown": 10e9}
DEFAULT_FILES_PER_TASK = {"fineweb-edu": 8, "open-web-math": 20, "starcoder-python": 10, "starcoder-markdown": 8, "wikipedia-en": 8}
PILOT_FILES, PILOT_ROWS = 2, 20_000
DOCS_PER_JOB = 64        # docs per pool job


def out_root(pilot):
    return C.TEXT_OUT_PILOT if pilot else C.TEXT_OUT


def parse_budget(s):
    if s is None or s in ("", "all", "none", "None"):
        return None
    s = str(s).strip().upper()
    mult = {"B": 1e9, "M": 1e6, "K": 1e3}.get(s[-1], None)
    return int(float(s[:-1]) * mult) if mult else int(float(s))


# ------------------------------------------------------------------------------------- index
def build_index(source, n_files, pilot, keep_val=False):
    pat, col, topic = C.TEXT_SOURCES[source]
    files = sorted(glob.glob(pat))
    if pilot:
        files = files[:PILOT_FILES]
    elif n_files:
        files = files[:n_files]
    rows = [min(C.parquet_num_rows(f), PILOT_ROWS) if pilot else C.parquet_num_rows(f) for f in files]
    offs = np.cumsum([0] + rows)
    N = int(offs[-1])
    d = f"{out_root(pilot)}/{source}"
    os.makedirs(d, exist_ok=True)
    ipath = f"{d}/index.json"
    old = C.load_json(ipath) if keep_val and os.path.exists(ipath) else None
    if old is not None:
        val_by_file = {int(k): v for k, v in old["val_by_file"].items()}
        n_val = old["n_val"]
    else:
        rng = np.random.default_rng(C.SPLIT_SEED)
        n_val = min(C.N_VAL_TEXT if not pilot else 500, N)
        val = np.sort(rng.permutation(N)[:n_val])
        fi = np.searchsorted(offs, val, side="right") - 1
        val_by_file = {}
        for f_idx, g in zip(fi, val):
            val_by_file.setdefault(int(f_idx), []).append(int(g - offs[f_idx]))
    idx = {"source": source, "pilot": pilot, "column": col, "topic": topic, "files": files, "rows": rows, "N": N,
           "n_val": n_val, "seed": C.SPLIT_SEED, "row_cap": PILOT_ROWS if pilot else None,
           "val_by_file": {str(k): v for k, v in val_by_file.items()}, "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    C.write_json(ipath, idx)
    C.log(f"[index] {source}: {len(files)} files, {N:,} docs, {C.human_bytes(sum(os.path.getsize(f) for f in files))}, val {n_val} docs")
    return idx


def load_index(source, pilot):
    return C.load_json(f"{out_root(pilot)}/{source}/index.json")


def build_plan(sources, pilot, files_per_task=None):
    tasks = []
    for s in sources:
        idx = load_index(s, pilot)
        step = files_per_task or DEFAULT_FILES_PER_TASK.get(s, 8)
        if pilot:
            step = PILOT_FILES
        for a in range(0, len(idx["files"]), step):
            tasks.append({"source": s, "files": [a, min(len(idx["files"]), a + step)]})
    for i, t in enumerate(tasks):
        t["plan_id"] = i
    C.write_json(f"{out_root(pilot)}/plan.json", {"pilot": pilot, "n_tasks": len(tasks), "tasks": tasks})
    C.log(f"[plan] {len(tasks)} tokenize tasks -> {out_root(pilot)}/plan.json")
    return tasks


# ------------------------------------------------------------------------------------- tokenize
def _init():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    C.get_tokenizer()


def _tok_job(texts):
    ids = C.encode_batch(texts)
    return [np.array(x + [C.EOS], dtype=np.uint32) for x in ids]


def tokenize_file(idx, f_idx, pool, pilot):
    import pyarrow.parquet as pq
    src = idx["source"]
    d = f"{out_root(pilot)}/{src}/parts"
    os.makedirs(d, exist_ok=True)
    meta_path = f"{d}/{f_idx:05d}.json"
    if os.path.exists(meta_path):
        C.log(f"[tokenize] {src} file {f_idx}: parts exist, skipping")
        return
    path, col = idx["files"][f_idx], idx["column"]
    val_rows = set(idx["val_by_file"].get(str(f_idx), []))
    cap = idx.get("row_cap")
    t0 = time.time()
    outs = {}
    for split in ("train", "val"):
        outs[split] = {"bin": open(f"{d}/{split}-{f_idx:05d}.bin.tmp", "wb"), "starts": [], "tok": 0, "docs": 0}
    pf = pq.ParquetFile(path)
    row = 0
    pending = []      # (split, texts) jobs

    def submit(batch_texts, batch_splits):
        # one pool job per DOCS_PER_JOB docs; splits tracked alongside
        jobs = []
        for i in range(0, len(batch_texts), DOCS_PER_JOB):
            jobs.append((batch_texts[i:i + DOCS_PER_JOB], batch_splits[i:i + DOCS_PER_JOB]))
        for arrs, (_, splits) in zip(pool.imap(_tok_job, [j[0] for j in jobs], chunksize=1), jobs):
            for arr, sp in zip(arrs, splits):
                o = outs[sp]
                o["starts"].append(o["tok"])
                o["bin"].write(arr.tobytes())
                o["tok"] += len(arr)
                o["docs"] += 1

    texts, splits = [], []
    for batch in pf.iter_batches(batch_size=1024, columns=[col]):
        vals = batch.column(0).to_pylist()
        for v in vals:
            if cap is not None and row >= cap:
                break
            sp = "val" if row in val_rows else "train"
            row += 1
            if not v:
                continue
            texts.append(v)
            splits.append(sp)
        if len(texts) >= 16 * DOCS_PER_JOB * max(1, pool._processes // 4):
            submit(texts, splits)
            texts, splits = [], []
        if cap is not None and row >= cap:
            break
    if texts:
        submit(texts, splits)
    for sp, o in outs.items():
        o["bin"].close()
        os.replace(f"{d}/{sp}-{f_idx:05d}.bin.tmp", f"{d}/{sp}-{f_idx:05d}.bin")
        np.array(o["starts"], dtype=np.int64).tofile(f"{d}/{sp}-{f_idx:05d}.idx")
    meta = {"file": path, "file_idx": f_idx, "rows_read": row, "train": {"n_tokens": outs["train"]["tok"], "n_docs": outs["train"]["docs"]},
            "val": {"n_tokens": outs["val"]["tok"], "n_docs": outs["val"]["docs"]}, "seconds": round(time.time() - t0, 1)}
    C.write_json(meta_path, meta)
    dt = time.time() - t0
    C.log(f"[tokenize] {src} file {f_idx} ({os.path.basename(path)}): {row:,} rows, train {C.human_count(outs['train']['tok'])} tok / {outs['train']['docs']:,} docs, "
          f"val {outs['val']['docs']} docs, {dt:.0f}s ({outs['train']['tok'] / max(dt, 1e-9) / 1e6:.2f} Mtok/s)")


def tokenize_range(source, a, b, pilot, workers):
    idx = load_index(source, pilot)
    b = min(b, len(idx["files"]))
    C.log(f"[tokenize] {source} files [{a},{b}) workers={workers} node={os.uname().nodename}")
    C.get_tokenizer()   # forked workers inherit the loaded tokenizer
    pool = mp.Pool(workers, initializer=_init)
    try:
        for f in range(a, b):
            tokenize_file(idx, f, pool, pilot)
    finally:
        pool.close()
        pool.join()


# ------------------------------------------------------------------------------------- merge
def merge(source, budget, pilot, keep_parts=False):
    idx = load_index(source, pilot)
    d = f"{out_root(pilot)}/{source}"
    parts = f"{d}/parts"
    meta_path = f"{d}/meta.json"
    meta = C.load_json(meta_path) if os.path.exists(meta_path) else None
    merged = list(meta["files_merged"]) if meta else []
    total = meta["n_tokens"] if meta else 0
    n_docs = meta["n_docs"] if meta else 0
    if meta and os.path.exists(f"{d}/train.idx"):
        starts = [np.fromfile(f"{d}/train.idx", dtype=np.int64)[:-1]]
    else:
        starts, merged, total, n_docs = [], [], 0, 0
        for p in (f"{d}/train.bin", f"{d}/train.idx"):
            if os.path.exists(p):
                os.remove(p)
    to_add = []
    for f in range(len(idx["files"])):
        if f in merged:
            continue
        pm = f"{parts}/{f:05d}.json"
        if not os.path.exists(pm):
            C.log(f"[merge] {source}: file {f} not tokenized yet, stopping here (files are merged in order)")
            break
        if budget is not None and total + sum(C.load_json(f"{parts}/{x:05d}.json")["train"]["n_tokens"] for x in to_add) >= budget:
            C.log(f"[merge] {source}: budget {C.human_count(budget)} reached")
            break
        to_add.append(f)
    if not to_add and meta:
        C.log(f"[merge] {source}: nothing new to merge ({C.human_count(total)} tokens)")
    with open(f"{d}/train.bin", "ab") as out:
        for f in to_add:
            pm = C.load_json(f"{parts}/{f:05d}.json")
            st = np.fromfile(f"{parts}/train-{f:05d}.idx", dtype=np.int64) + total
            starts.append(st)
            with open(f"{parts}/train-{f:05d}.bin", "rb") as src:
                while True:
                    buf = src.read(64 << 20)
                    if not buf:
                        break
                    out.write(buf)
            total += pm["train"]["n_tokens"]
            n_docs += pm["train"]["n_docs"]
            merged.append(f)
            C.log(f"[merge] {source}: + file {f} -> {C.human_count(total)} tokens, {n_docs:,} docs")
    all_starts = np.concatenate(starts + [np.array([total], dtype=np.int64)]) if starts else np.array([0], dtype=np.int64)
    assert all_starts.shape[0] == n_docs + 1, (all_starts.shape, n_docs)
    all_starts.astype(np.int64).tofile(f"{d}/train.idx.tmp")
    os.replace(f"{d}/train.idx.tmp", f"{d}/train.idx")
    # val: rebuilt from the val parts of all merged files
    vtot, vdocs, vstarts = 0, 0, []
    with open(f"{d}/val.bin.tmp", "wb") as out:
        for f in merged:
            pm = C.load_json(f"{parts}/{f:05d}.json")
            if pm["val"]["n_docs"] == 0:
                continue
            vstarts.append(np.fromfile(f"{parts}/val-{f:05d}.idx", dtype=np.int64) + vtot)
            with open(f"{parts}/val-{f:05d}.bin", "rb") as src:
                out.write(src.read())
            vtot += pm["val"]["n_tokens"]
            vdocs += pm["val"]["n_docs"]
    os.replace(f"{d}/val.bin.tmp", f"{d}/val.bin")
    np.concatenate(vstarts + [np.array([vtot], dtype=np.int64)]).astype(np.int64).tofile(f"{d}/val.idx")
    meta = {"tokenizer": C.TOKENIZER_NAME, "n_tokens": int(total), "n_docs": int(n_docs), "topic": idx["topic"],
            "val": {"n_tokens": int(vtot), "n_docs": int(vdocs)}, "eos": C.EOS, "dtype": "uint32",
            "idx": "int64 document start offsets, n_docs+1 entries (last = n_tokens); every document ends with EOS",
            "files_merged": merged, "files_merged_paths": [idx["files"][f] for f in merged],
            "budget": budget, "merged": time.strftime("%Y-%m-%d %H:%M:%S")}
    C.write_json(meta_path, meta)
    if not keep_parts:
        for f in to_add:
            os.remove(f"{parts}/train-{f:05d}.bin")   # the idx/json/val parts are kept (small) for incremental re-merges
    C.log(f"[merge] {source}: train {C.human_count(total)} tokens / {n_docs:,} docs ({C.human_bytes(total * 4)}), val {C.human_count(vtot)} tokens / {vdocs:,} docs, files {merged}")


# ------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["index", "plan", "tokenize", "merge", "all-index-plan"])
    ap.add_argument("--source")
    ap.add_argument("--sources", default=",".join(C.TEXT_SOURCES))
    ap.add_argument("--n-files", type=int, default=None, help="index: only the first K files (default per source, see DEFAULT_N_FILES)")
    ap.add_argument("--keep-val", action="store_true", help="index: keep the existing val positions when extending")
    ap.add_argument("--files", default=None, help="tokenize: file range a:b")
    ap.add_argument("--task", type=int, default=None, help="tokenize: plan task id")
    ap.add_argument("--files-per-task", type=int, default=None)
    ap.add_argument("--budget", default=None, help="merge: token budget like 60B (default per source: fineweb 60B, markdown 10B, others all)")
    ap.add_argument("--keep-parts", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    a = ap.parse_args()
    os.makedirs(out_root(a.pilot), exist_ok=True)
    if a.cmd == "index":
        build_index(a.source, a.n_files or DEFAULT_N_FILES.get(a.source), a.pilot, a.keep_val)
    elif a.cmd == "plan":
        build_plan(a.sources.split(","), a.pilot, a.files_per_task)
    elif a.cmd == "all-index-plan":
        srcs = a.sources.split(",")
        for s in srcs:
            build_index(s, a.n_files or DEFAULT_N_FILES.get(s), a.pilot, a.keep_val)
        build_plan(srcs, a.pilot, a.files_per_task)
    elif a.cmd == "tokenize":
        if a.task is not None:
            t = C.load_json(f"{out_root(a.pilot)}/plan.json")["tasks"][a.task]
            tokenize_range(t["source"], t["files"][0], t["files"][1], a.pilot, a.workers)
        else:
            x, y = a.files.split(":")
            tokenize_range(a.source, int(x), int(y), a.pilot, a.workers)
    elif a.cmd == "merge":
        budget = parse_budget(a.budget) if a.budget is not None else (None if a.pilot else DEFAULT_BUDGET.get(a.source))
        merge(a.source, budget, a.pilot, a.keep_parts)


if __name__ == "__main__":
    main()
