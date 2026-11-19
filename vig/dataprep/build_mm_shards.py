#!/usr/bin/env python
"""Build the multimodal tar shards (see the README) from the raw downloads.

Sub-commands (all take --pilot to work on a small slice under shards/mm_pilot/):
  index    --source S        discover input units, count samples, draw the val split (seed 0, 2,000/source)
  plan     [--sources a,b]   group units into worker tasks -> <out>/plan.json (one job array task each)
  run      --task K          process plan task K (resumable: complete shards are skipped, .tmp then rename)
  finalize --source S        merge the per-task val tars into val-000000.tar and write summary.json
  all-index-plan             index every source, then plan (used by the launcher)

Units and tasks
  A unit is one input file (wds tar, parquet file) or one annotation set (aguvis json + zip, MathV360K json +
  zip, ArxivQA jsonl + tgz, LLaVA-OV *_processed.json + image tar.gz). Samples get global positions
  0..N-1 per source in unit order; val = first 2,000 of rng(0).permutation(N).
  Random-access units (tar on disk, parquet, zip) are cut into tasks of <= TASK_N samples / TASK_BYTES input;
  each task writes train shards for fixed position slices of 2,000, so an existing complete shard can be
  skipped and only its val positions are re-read. Stream units (gzip tars) are one task per archive;
  shards are filled in arrival order and the task restarts from scratch if interrupted.
  Shard names: train-TTTKKK.tar (task TTT, slice KKK), val-TTT001.tar per task (merged by `finalize` into val-000000.tar).

Sample rules
  one image per sample: rows with 0 or >1 images are skipped and counted, except aguvis multi-image
  records whose k human turns each carry one <image> tag, which are split into k samples.
  Conversations -> "Question: ...\\nAnswer: ...\\n" per pair; captions raw; text_ids <= 1,024 tokens.
  Images: PIL decode, EXIF transpose, alpha flattened on white, RGB, LANCZOS resize to multiples of 32 with
  long side <= 512 (caption/general) or <= 1024 (ocr/chart/gui/stem), JPEG q90. Decode failures are skipped and counted.
"""
import argparse
import glob
import io
import json
import math
import multiprocessing as mp
import os
import sys
import tarfile
import time
import traceback
import zipfile
from collections import Counter, OrderedDict, defaultdict

import numpy as np

from . import common as C
from . import topics as T

TASK_N = 150_000          # max samples per random-access task
TASK_BYTES = 12 << 30     # max input bytes per random-access task
PILOT_UNITS_PER_KIND = 2  # --pilot: first 2 units of every kind per source
PILOT_CAP = 3000          # --pilot: max positions per unit (-> ~2 shards)
STREAM_KINDS = {"json_targz"}


def out_root(pilot):
    return C.MM_OUT_PILOT if pilot else C.MM_OUT


# ============================================================================ unit discovery
def _unit(kind, subset, topic, n, nbytes, **kw):
    d = {"kind": kind, "subset": subset, "topic": topic, "n": int(n), "nbytes": int(nbytes)}
    d.update(kw)
    return d


def discover_units(source):
    root = f"{C.MM_ROOT}/{source}"
    units = []
    if source in ("cc12m-wds", "cc3m-wds"):
        info = C.load_json(f"{root}/_info.json")
        for split in info["splits"].values():
            for fn, n in zip(split["filenames"], split["shard_lengths"]):
                p = f"{root}/{fn}"
                if os.path.exists(p):
                    units.append(_unit("wds", split["name"], "caption", n, os.path.getsize(p), path=p))
    elif source == "the_cauldron":
        for sub in sorted(d for d in os.listdir(root) if os.path.isdir(f"{root}/{d}") and not d.startswith(".")):
            for p in sorted(glob.glob(f"{root}/{sub}/*.parquet")):
                units.append(_unit("parquet_cauldron", sub, T.topic_for(source, sub), C.parquet_num_rows(p), os.path.getsize(p), path=p))
    elif source == "LLaVA-OneVision-Data":
        for sub in sorted(d for d in os.listdir(root) if os.path.isdir(f"{root}/{d}") and not d.startswith(".")):
            if sub in T.TEXT_ONLY:
                continue
            topic = T.topic_for(source, sub)
            pqs = sorted(glob.glob(f"{root}/{sub}/*.parquet"))
            if pqs:
                for p in pqs:
                    units.append(_unit("parquet_llava", sub, topic, C.parquet_num_rows(p), os.path.getsize(p), path=p))
            else:
                ann = sorted(glob.glob(f"{root}/{sub}/*_processed.json"))
                parts = [[p] for p in sorted(glob.glob(f"{root}/{sub}/*.tar.gz"))]
                groups = load_groups({"kind": "json_targz", "ann_kind": "llava_json", "ann": ann})
                units.append(_unit("json_targz", sub, topic, len(groups), sum(os.path.getsize(p[0]) for p in parts),
                                   ann_kind="llava_json", ann=ann, parts=parts))
    elif source == "Docmatix":
        for p in sorted(glob.glob(f"{root}/data/*.parquet")):
            units.append(_unit("parquet_cauldron", "Docmatix", "ocr", C.parquet_num_rows(p), os.path.getsize(p), path=p))
    elif source == "ChartGemma":
        for p in sorted(glob.glob(f"{root}/data/*.parquet")):
            units.append(_unit("parquet_chartgemma", "ChartGemma", "chart", C.parquet_num_rows(p), os.path.getsize(p), path=p))
    elif source == "ChartQA":
        for p in sorted(glob.glob(f"{root}/data/train-*.parquet")) + sorted(glob.glob(f"{root}/data/val-*.parquet")):
            units.append(_unit("parquet_chartqa", "ChartQA", "chart", C.parquet_num_rows(p), os.path.getsize(p), path=p))
    elif source == "aguvis-stage1":
        zips = {"guienv": "guienvs.zip", "omniact_fix": "omniact.zip"}
        for js in sorted(glob.glob(f"{root}/*.json")):
            stem = os.path.basename(js)[:-5]
            if stem in ("seeclick", "seeclick_mi"):
                continue
            z = f"{root}/{zips.get(stem, stem + '.zip')}"
            u = {"kind": "json_zip", "ann_kind": "llava_json", "ann": [js], "archive": z}
            groups = load_groups(u)
            units.append(_unit("json_zip", stem, "gui", len(groups), os.path.getsize(z), ann_kind="llava_json", ann=[js], archive=z))
        ann = [f"{root}/seeclick.json", f"{root}/seeclick_mi.json"]
        parts = [sorted(glob.glob(f"{root}/seeclick.tar.gz.part_*"))]   # one gzip stream split into parts
        groups = load_groups({"kind": "json_targz", "ann_kind": "llava_json", "ann": ann})
        units.append(_unit("json_targz", "seeclick", "gui", len(groups), sum(os.path.getsize(p) for p in parts[0]),
                           ann_kind="llava_json", ann=ann, parts=parts))
    elif source == "ArxivQA":
        ann = [f"{root}/arxivqa.jsonl"]
        groups = load_groups({"kind": "json_targz", "ann_kind": "arxivqa", "ann": ann})
        units.append(_unit("json_targz", "ArxivQA", "stem", len(groups), os.path.getsize(f"{root}/images.tgz"),
                           ann_kind="arxivqa", ann=ann, parts=[[f"{root}/images.tgz"]]))
    elif source == "MathV360K":
        ann = [f"{root}/train_samples_all_tuning.json"]
        z = f"{root}/data_images.zip"
        groups = load_groups({"kind": "json_zip", "ann_kind": "mathv", "ann": ann, "archive": z})
        units.append(_unit("json_zip", "MathV360K", "stem", len(groups), os.path.getsize(z), ann_kind="mathv", ann=ann, archive=z))
    else:
        raise ValueError(source)
    for i, u in enumerate(units):
        u["id"] = i
    return units


# ============================================================================ annotation groups
_GROUP_CACHE = {}


def load_groups(unit):
    """Annotation set -> list of (image_key, [(q, a), ...]) grouped by image in first-appearance order.
    Also returns via unit['group_skips'] the number of unsplittable multi-image records."""
    ck = tuple(unit["ann"])
    if ck in _GROUP_CACHE:
        return _GROUP_CACHE[ck]
    groups = OrderedDict()
    skips = Counter()
    kind = unit["ann_kind"]
    for path in unit["ann"]:
        if kind == "arxivqa":
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    opts = "\n".join(r.get("options") or [])
                    q = (r.get("question") or "").strip() + ("\n" + opts if opts else "")
                    a = (r.get("label") or "").strip()
                    rat = (r.get("rationale") or "").strip()
                    if rat:
                        a = f"{a}. {rat}" if a else rat
                    groups.setdefault(r["image"], []).append((q, a))
            continue
        data = C.load_json(path)
        for r in data:
            img = r.get("image")
            conv = r.get("conversations") or []
            if not img:
                skips["no_image"] += 1
                continue
            if isinstance(img, list):
                if len(img) == 1:
                    img = img[0]
                else:
                    sp = C.split_multi_image_llava(img, conv)
                    if sp is None:
                        skips["multi_image"] += 1
                        continue
                    for im, pairs in sp:
                        groups.setdefault(im, []).extend(pairs)
                    continue
            groups.setdefault(img, []).extend(C.pairs_from_llava(conv))
    out = list(groups.items())
    _GROUP_CACHE[ck] = out
    if skips:
        C.log(f"[groups] {os.path.basename(unit['ann'][0])}: skipped records {dict(skips)}")
    return out


# ============================================================================ index / plan
def build_index(source, pilot):
    t0 = time.time()
    units = discover_units(source)
    if pilot:
        keep, seen = [], Counter()
        for u in units:
            if seen[u["kind"]] < PILOT_UNITS_PER_KIND:
                seen[u["kind"]] += 1
                u["n"] = min(u["n"], PILOT_CAP)
                keep.append(u)
        units = keep
        for i, u in enumerate(units):
            u["id"] = i
    off = 0
    for u in units:
        u["offset"] = off
        off += u["n"]
    N = off
    rng = np.random.default_rng(C.SPLIT_SEED)
    n_val = min(C.N_VAL_MM, N)
    val = np.sort(rng.permutation(N)[:n_val]).tolist()
    idx = {"source": source, "pilot": pilot, "n_units": len(units), "N": N, "n_val": n_val, "seed": C.SPLIT_SEED,
           "val_positions": val, "units": units, "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    d = f"{out_root(pilot)}/{source}"
    os.makedirs(d, exist_ok=True)
    C.write_json(f"{d}/index.json", idx)
    kinds = Counter(u["kind"] for u in units)
    C.log(f"[index] {source}: {len(units)} units {dict(kinds)}, N={N:,}, {C.human_bytes(sum(u['nbytes'] for u in units))} input, {time.time() - t0:.0f}s")
    return idx


def load_index(source, pilot):
    return C.load_json(f"{out_root(pilot)}/{source}/index.json")


def build_plan(sources, pilot):
    tasks = []
    for source in sources:
        idx = load_index(source, pilot)
        tid = 0
        cur = None   # [start, end, bytes]

        def flush():
            nonlocal cur, tid
            if cur is not None and cur[1] > cur[0]:
                tasks.append({"source": source, "task": tid, "kind": "random", "start": cur[0], "end": cur[1], "nbytes": cur[2]})
                tid += 1
            cur = None

        for u in idx["units"]:
            if u["kind"] in STREAM_KINDS:
                flush()
                for pi in range(len(u["parts"])):
                    tasks.append({"source": source, "task": tid, "kind": "stream", "unit": u["id"], "part": pi,
                                  "nbytes": sum(os.path.getsize(p) for p in u["parts"][pi])})
                    tid += 1
                continue
            if u["n"] > TASK_N or u["nbytes"] > TASK_BYTES:   # big random-access unit: split by positions
                flush()
                n_chunks = max(math.ceil(u["n"] / TASK_N), math.ceil(u["nbytes"] / TASK_BYTES))
                step = math.ceil(u["n"] / n_chunks)
                for a in range(0, u["n"], step):
                    b = min(u["n"], a + step)
                    tasks.append({"source": source, "task": tid, "kind": "random", "start": u["offset"] + a, "end": u["offset"] + b,
                                  "nbytes": int(u["nbytes"] * (b - a) / u["n"])})
                    tid += 1
                continue
            if cur is None:
                cur = [u["offset"], u["offset"], 0]
            if cur[1] - cur[0] + u["n"] > TASK_N or cur[2] + u["nbytes"] > TASK_BYTES:
                flush()
                cur = [u["offset"], u["offset"], 0]
            cur[1] = u["offset"] + u["n"]
            cur[2] += u["nbytes"]
        flush()
    for i, t in enumerate(tasks):
        t["plan_id"] = i
    plan = {"pilot": pilot, "n_tasks": len(tasks), "tasks": tasks, "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    C.write_json(f"{out_root(pilot)}/plan.json", plan)
    per = Counter(t["source"] for t in tasks)
    C.log(f"[plan] {len(tasks)} tasks: {dict(per)} -> {out_root(pilot)}/plan.json")
    return plan


# ============================================================================ raw sample iteration
def raw(unit, pos, key, image, pairs=None, caption=None):
    return {"unit": unit["id"], "subset": unit["subset"], "topic": unit["topic"], "pos": pos, "key": key,
            "image": image, "pairs": pairs, "caption": caption}


def iter_random(units, a, b, wanted, skips):
    """Yield raw samples for global positions in [a, b) (only `wanted` positions if given)."""
    for u in units:
        ua, ub = u["offset"], u["offset"] + u["n"]
        if ub <= a or ua >= b:
            continue
        la, lb = max(a, ua) - ua, min(b, ub) - ua
        lw = None if wanted is None else sorted(p - ua for p in wanted if ua <= p < ub)
        if lw is not None and not lw:
            continue
        kind = u["kind"]
        if kind == "wds":
            w = set(lw) if lw is not None else (set(range(la, lb)) if (la, lb) != (0, u["n"]) else None)
            for pos, key, d in C.iter_wds_tar(u["path"], wanted=w):
                if pos >= u["n"] or pos < la or pos >= lb:
                    continue
                if "jpg" not in d:
                    skips["no_image"] += 1
                    continue
                cap = d.get("txt")
                if cap is None:
                    try:
                        cap = json.loads(d.get("json", b"{}")).get("caption", "").encode()
                    except Exception:
                        cap = b""
                yield raw(u, ua + pos, f"{u['id']:04d}_{ua + pos:08d}", d["jpg"], caption=cap.decode("utf-8", "replace"))
        elif kind in ("parquet_cauldron", "parquet_llava", "parquet_chartgemma", "parquet_chartqa"):
            cols = {"parquet_cauldron": ["images", "texts"], "parquet_llava": ["image", "conversations"],
                    "parquet_chartgemma": ["image", "input", "output"], "parquet_chartqa": ["image", "query", "label"]}[kind]
            for r, row in C.iter_parquet_rows(u["path"], cols, la, lb, lw):
                gpos = ua + r
                key = f"{u['id']:04d}_{gpos:08d}"
                if kind == "parquet_cauldron":
                    ims = row["images"] or []
                    if len(ims) != 1:
                        skips["multi_image" if len(ims) > 1 else "no_image"] += 1
                        continue
                    pairs = [(t.get("user") or "", t.get("assistant") or "") for t in (row["texts"] or [])]
                    yield raw(u, gpos, key, ims[0]["bytes"], pairs=pairs)
                elif kind == "parquet_llava":
                    im = row["image"]
                    if not im or not im.get("bytes"):
                        skips["no_image"] += 1
                        continue
                    yield raw(u, gpos, key, im["bytes"], pairs=C.pairs_from_llava(row["conversations"]))
                elif kind == "parquet_chartgemma":
                    yield raw(u, gpos, key, row["image"], pairs=[(row["input"] or "", row["output"] or "")])
                else:
                    im = row["image"]
                    lab = row["label"] or []
                    yield raw(u, gpos, key, im["bytes"] if isinstance(im, dict) else im,
                              pairs=[(row["query"] or "", ", ".join(str(x) for x in lab))])
        elif kind == "json_zip":
            groups = load_groups(u)
            with zipfile.ZipFile(u["archive"]) as z:
                members = {os.path.basename(n): n for n in z.namelist() if not n.endswith("/")}
                full = set(members.values())
                rng_ = lw if lw is not None else range(la, lb)
                for p in rng_:
                    img, pairs = groups[p]
                    name = img if img in full else members.get(os.path.basename(img))
                    if name is None:
                        skips["missing_in_archive"] += 1
                        continue
                    try:
                        data = z.read(name)
                    except Exception:
                        skips["archive_read_error"] += 1
                        continue
                    yield raw(u, ua + p, f"{u['id']:04d}_{ua + p:08d}", data, pairs=pairs)
        else:
            raise ValueError(kind)


def iter_stream(u, part, cap, skips):
    """Yield raw samples from one gzip tar stream of a json_targz unit (positions from annotation order)."""
    groups = load_groups(u)
    by_full = {img: i for i, (img, _) in enumerate(groups)}
    bn = Counter(os.path.basename(img) for img, _ in groups)
    by_base = {os.path.basename(img): i for i, (img, _) in enumerate(groups) if bn[os.path.basename(img)] == 1}
    ua = u["offset"]
    n_out = 0
    tf = C.open_tar_stream(u["parts"][part], gz=True)
    for m in tf:
        if not m.isfile():
            continue
        i = by_full.get(m.name)
        if i is None:
            i = by_base.get(os.path.basename(m.name))
        if i is None:
            skips["unannotated_member"] += 1
            continue
        data = tf.extractfile(m).read()
        gpos = ua + (i if cap is None else n_out)   # pilot: sequential pseudo-positions inside the capped unit
        yield raw(u, gpos, f"{u['id']:04d}_{gpos:08d}", data, pairs=groups[i][1])
        n_out += 1
        if cap is not None and n_out >= cap:
            break


# ============================================================================ worker
def _init_worker():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    C.get_tokenizer()


def process_sample(s):
    """Decode/resize/encode the image and tokenize the text. Returns dict or {'error': reason}."""
    try:
        jpg, w, h = C.process_image(s["image"], C.MAX_SIDE[s["topic"]])
    except Exception as e:
        return {"error": "decode_fail", "key": s["key"], "msg": repr(e)[:200]}
    text = s["caption"] if s["caption"] is not None else C.render_pairs(s["pairs"])
    text = text.strip() if s["caption"] is not None else text
    if not text.strip():
        return {"error": "empty_text", "key": s["key"]}
    ids, trunc = C.encode(text, C.MAX_TEXT_TOKENS)
    meta = {"text_ids": ids, "source": s["source"], "topic": s["topic"], "w": w, "h": h, "n_img": 1,
            "split": s["split"], "subset": s["subset"], "key": s["key"]}
    return {"key": s["key"], "jpg": jpg, "json": json.dumps(meta, ensure_ascii=False).encode("utf-8"),
            "img_tok": C.image_tokens(w, h), "txt_tok": len(ids), "trunc": trunc, "topic": s["topic"],
            "subset": s["subset"], "split": s["split"]}


class Stats:
    def __init__(self):
        self.n = 0
        self.img_tok = 0
        self.txt_tok = 0
        self.trunc = 0
        self.per_topic = defaultdict(lambda: {"samples": 0, "image_tokens": 0, "text_tokens": 0})
        self.per_subset = Counter()
        self.skips = Counter()

    def add(self, r):
        self.n += 1
        self.img_tok += r["img_tok"]
        self.txt_tok += r["txt_tok"]
        self.trunc += int(r["trunc"])
        t = self.per_topic[r["topic"]]
        t["samples"] += 1
        t["image_tokens"] += r["img_tok"]
        t["text_tokens"] += r["txt_tok"]
        self.per_subset[r["subset"]] += 1

    def merge(self, o):
        self.n += o.n; self.img_tok += o.img_tok; self.txt_tok += o.txt_tok; self.trunc += o.trunc
        for k, v in o.per_topic.items():
            for kk in v:
                self.per_topic[k][kk] += v[kk]
        self.per_subset.update(o.per_subset)
        self.skips.update(o.skips)

    def to_dict(self):
        return {"n_samples": self.n, "image_tokens": self.img_tok, "text_tokens": self.txt_tok, "n_truncated": self.trunc,
                "per_topic": dict(self.per_topic), "per_subset": dict(self.per_subset), "skipped": dict(self.skips)}


class ShardWriter:
    """Writes <key>.jpg + <key>.json members to path.tmp, renames on close, writes the sidecar json."""

    def __init__(self, path, source, split, pos_range=None):
        self.path, self.source, self.split, self.pos_range = path, source, split, pos_range
        self.tmp = path + ".tmp"
        self.tf = tarfile.open(self.tmp, "w")
        self.stats = Stats()

    def _add(self, name, data):
        ti = tarfile.TarInfo(name)
        ti.size = len(data)
        ti.mtime = 0
        ti.mode = 0o644
        self.tf.addfile(ti, io.BytesIO(data))

    def add(self, r):
        self._add(r["key"] + ".jpg", r["jpg"])
        self._add(r["key"] + ".json", r["json"])
        self.stats.add(r)

    def close(self, extra_skips=None):
        self.tf.close()
        if extra_skips:
            self.stats.skips.update(extra_skips)
        if self.stats.n > 0:
            os.replace(self.tmp, self.path)
        else:
            os.remove(self.tmp)
            if os.path.exists(self.path):
                os.remove(self.path)
        side = self.stats.to_dict()
        side.update({"shard": os.path.basename(self.path), "source": self.source, "split": self.split,
                     "pos_range": self.pos_range, "written": time.strftime("%Y-%m-%d %H:%M:%S")})
        C.write_json(self.path[:-4] + ".json", side)
        return self.stats


def shard_complete(path):
    side = path[:-4] + ".json"
    if not os.path.exists(side):
        return False
    try:
        n = C.load_json(side)["n_samples"]
    except Exception:
        return False
    return n == 0 or os.path.exists(path)


def pump(pool, gen, source, val_set, on_result, skips, chunk=256):
    """Feed raw samples through the pool in bounded chunks (keeps memory flat), preserving order."""
    buf = []

    def flush():
        for r in pool.imap(process_sample, buf, chunksize=4):
            if "error" in r:
                skips[r["error"]] += 1
            else:
                on_result(r)
        buf.clear()

    for s in gen:
        s["source"] = source
        s["split"] = "val" if s["pos"] in val_set else "train"
        del s["pos"]
        buf.append(s)
        if len(buf) >= chunk:
            flush()
    if buf:
        flush()


def run_task(plan_id, pilot, workers):
    plan = C.load_json(f"{out_root(pilot)}/plan.json")
    task = plan["tasks"][plan_id]
    source, tid = task["source"], task["task"]
    idx = load_index(source, pilot)
    units = idx["units"]
    val_set = set(idx["val_positions"])
    out_dir = f"{out_root(pilot)}/{source}"
    os.makedirs(f"{out_dir}/tasks", exist_ok=True)
    done_path = f"{out_dir}/tasks/task-{tid:03d}.json"
    if os.path.exists(done_path):
        C.log(f"[run] {source} task {tid}: already done, skipping")
        return
    t0 = time.time()
    C.log(f"[run] {source} task {tid} ({task['kind']}) plan_id={plan_id} workers={workers} node={os.uname().nodename}")
    for stale in glob.glob(f"{out_dir}/*-{tid:03d}???.tar.tmp") + glob.glob(f"{out_dir}/*-{tid:03d}???.json.tmp"):
        os.remove(stale)
    total = Stats()
    val_buf = []
    C.get_tokenizer()   # loaded once here; forked workers inherit it (avoids 16 concurrent imports per task)
    pool = mp.Pool(workers, initializer=_init_worker)
    try:
        if task["kind"] == "random":
            a0, b0 = task["start"], task["end"]
            n_slices = math.ceil((b0 - a0) / C.SAMPLES_PER_SHARD)
            for k in range(n_slices):
                a = a0 + k * C.SAMPLES_PER_SHARD
                b = min(b0, a + C.SAMPLES_PER_SHARD)
                path = f"{out_dir}/train-{tid:03d}{k:03d}.tar"
                skips = Counter()
                if shard_complete(path):
                    wanted = [p for p in range(a, b) if p in val_set]
                    side = C.load_json(path[:-4] + ".json")
                    s = Stats(); s.n = side["n_samples"]; s.img_tok = side["image_tokens"]; s.txt_tok = side["text_tokens"]
                    s.trunc = side.get("n_truncated", 0)
                    for kk, v in side.get("per_topic", {}).items():
                        s.per_topic[kk].update(v)
                    s.per_subset.update(side.get("per_subset", {})); s.skips.update(side.get("skipped", {}))
                    total.merge(s)
                    if wanted:
                        pump(pool, iter_random(units, a, b, wanted, skips), source, val_set, val_buf.append, skips)
                    C.log(f"[run]   slice {k}: shard exists ({side['n_samples']} samples), re-read {len(wanted)} val positions")
                    continue
                w = ShardWriter(path, source, "train", [a, b])

                def on_result(r, w=w):
                    (val_buf.append if r["split"] == "val" else w.add)(r)

                pump(pool, iter_random(units, a, b, None, skips), source, val_set, on_result, skips)
                st = w.close(skips)
                total.merge(st)
                C.log(f"[run]   slice {k} [{a},{b}): {st.n} train samples, img_tok {st.img_tok}, txt_tok {st.txt_tok}, skips {dict(skips)}, val so far {len(val_buf)}, {time.time() - t0:.0f}s")
        else:
            u = units[task["unit"]]
            for old in glob.glob(f"{out_dir}/train-{tid:03d}???.tar") + glob.glob(f"{out_dir}/train-{tid:03d}???.json"):
                os.remove(old)
            cap = u["n"] if idx["pilot"] else None
            skips = Counter()
            state = {"k": 0, "w": None}

            def new_writer():
                state["w"] = ShardWriter(f"{out_dir}/train-{tid:03d}{state['k']:03d}.tar", source, "train", None)
                state["k"] += 1

            new_writer()

            def on_result(r):
                if r["split"] == "val":
                    val_buf.append(r)
                    return
                state["w"].add(r)
                if state["w"].stats.n >= C.SAMPLES_PER_SHARD:
                    st = state["w"].close(skips)
                    skips.clear()
                    total.merge(st)
                    C.log(f"[run]   shard {state['k'] - 1}: {st.n} samples, img_tok {st.img_tok}, txt_tok {st.txt_tok}, skips {dict(st.skips)}, {time.time() - t0:.0f}s")
                    new_writer()

            pump(pool, iter_stream(u, task["part"], cap, skips), source, val_set, on_result, skips)
            st = state["w"].close(skips)
            total.merge(st)
        # val tar for this task
        vpath = f"{out_dir}/val-{tid:03d}001.tar"
        vw = ShardWriter(vpath, source, "val", None)
        for r in val_buf:
            vw.add(r)
        vst = vw.close()
    finally:
        pool.close()
        pool.join()
    done = {"source": source, "task": tid, "plan": task, "train": total.to_dict(), "val": vst.to_dict(),
            "seconds": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    C.write_json(done_path, done)
    C.log(f"[run] {source} task {tid} done: train {total.n} samples ({C.human_count(total.img_tok)} img tok, {C.human_count(total.txt_tok)} txt tok), "
          f"val {vst.n}, skips {dict(total.skips)}, {time.time() - t0:.0f}s")


# ============================================================================ finalize
def finalize(source, pilot):
    out_dir = f"{out_root(pilot)}/{source}"
    plan = C.load_json(f"{out_root(pilot)}/plan.json")
    my = [t for t in plan["tasks"] if t["source"] == source]
    done = [t for t in my if os.path.exists(f"{out_dir}/tasks/task-{t['task']:03d}.json")]
    if len(done) < len(my):
        C.log(f"[finalize] {source}: WARNING {len(my) - len(done)} of {len(my)} tasks not done; merging what exists")
    parts = sorted(p for p in glob.glob(f"{out_dir}/val-*.tar") if os.path.basename(p) != "val-000000.tar")
    if parts:
        merged = f"{out_dir}/val-000000.tar"
        w = ShardWriter(merged, source, "val", None)
        seen = set()
        for p in ([merged] if os.path.exists(merged) else []) + parts:   # previously merged content is kept; keys deduped
            with tarfile.open(p, "r:") as tf:
                for key, jpg, meta in iter_tar_samples(tf):
                    if key in seen:
                        continue
                    seen.add(key)
                    w.add(_result_from_members(key, jpg, meta))
        st = w.close()
        for p in parts:
            os.remove(p)
            side = p[:-4] + ".json"
            if os.path.exists(side):
                os.remove(side)
        C.log(f"[finalize] {source}: merged {len(parts)} val tars -> val-000000.tar ({st.n} samples)")
    total, val = Stats(), Stats()
    for side in glob.glob(f"{out_dir}/train-*.json"):
        d = C.load_json(side)
        s = Stats(); s.n = d["n_samples"]; s.img_tok = d["image_tokens"]; s.txt_tok = d["text_tokens"]; s.trunc = d.get("n_truncated", 0)
        for k, v in d.get("per_topic", {}).items():
            s.per_topic[k].update(v)
        s.per_subset.update(d.get("per_subset", {})); s.skips.update(d.get("skipped", {}))
        total.merge(s)
    vside = f"{out_dir}/val-000000.json"
    if os.path.exists(vside):
        d = C.load_json(vside)
        val.n, val.img_tok, val.txt_tok = d["n_samples"], d["image_tokens"], d["text_tokens"]
    summary = {"source": source, "tasks_total": len(my), "tasks_done": len(done), "train": total.to_dict(), "val": val.to_dict(),
               "n_train_shards": len(glob.glob(f"{out_dir}/train-*.tar")), "finalized": time.strftime("%Y-%m-%d %H:%M:%S")}
    C.write_json(f"{out_dir}/summary.json", summary)
    C.log(f"[finalize] {source}: train {total.n:,} samples, {C.human_count(total.img_tok)} image tokens, {C.human_count(total.txt_tok)} text tokens, "
          f"val {val.n}, skips {dict(total.skips)}")


def iter_tar_samples(tf):
    """(key, jpg bytes, json bytes) triples from an open shard tar."""
    pend = {}
    for m in tf:
        if not m.isfile():
            continue
        key, _, ext = m.name.rpartition(".")
        pend.setdefault(key, {})[ext] = tf.extractfile(m).read()
        if "jpg" in pend[key] and "json" in pend[key]:
            d = pend.pop(key)
            yield key, d["jpg"], d["json"]


def _result_from_members(key, jpg, meta_bytes):
    meta = json.loads(meta_bytes)
    return {"key": key, "jpg": jpg, "json": meta_bytes, "img_tok": C.image_tokens(meta["w"], meta["h"]), "txt_tok": len(meta["text_ids"]),
            "trunc": len(meta["text_ids"]) >= C.MAX_TEXT_TOKENS, "topic": meta["topic"], "subset": meta.get("subset", ""), "split": meta["split"]}


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["index", "plan", "run", "finalize", "all-index-plan"])
    ap.add_argument("--source")
    ap.add_argument("--sources", default=",".join(C.MM_SOURCES))
    ap.add_argument("--task", type=int)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    a = ap.parse_args()
    os.makedirs(out_root(a.pilot), exist_ok=True)
    if a.cmd == "index":
        build_index(a.source, a.pilot)
    elif a.cmd == "plan":
        build_plan(a.sources.split(","), a.pilot)
    elif a.cmd == "all-index-plan":
        srcs = a.sources.split(",")
        for s in srcs:
            build_index(s, a.pilot)
        build_plan(srcs, a.pilot)
    elif a.cmd == "run":
        run_task(a.task, a.pilot, a.workers)
    elif a.cmd == "finalize":
        finalize(a.source, a.pilot)


if __name__ == "__main__":
    main()
