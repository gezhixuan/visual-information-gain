"""Shared constants and helpers for the data pipeline (tokenizer and shard schema: see the README).

Everything here is stdlib + numpy + PIL (+ pyarrow for parquet, transformers for the tokenizer).
No webdataset: tar shards are written and read with `tarfile`.
"""
import glob
import gzip
import io
import json
import os
import re
import sys
import tarfile
import zipfile
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageOps

from .. import paths

DATA_ROOT = paths.DATA
MM_ROOT = f"{DATA_ROOT}/mm"
TEXT_ROOT = f"{DATA_ROOT}/text"
SHARDS_ROOT = f"{DATA_ROOT}/shards"
MM_OUT = f"{SHARDS_ROOT}/mm"
MM_OUT_PILOT = f"{SHARDS_ROOT}/mm_pilot"
TEXT_OUT = f"{SHARDS_ROOT}/text"
TEXT_OUT_PILOT = f"{SHARDS_ROOT}/text_pilot"
LOG_DIR = f"{DATA_ROOT}/logs"

TOKENIZER_PATH = paths.TOKENIZER
TOKENIZER_NAME = "Qwen3-0.6B-Base"
VISION_START, VISION_END, IMAGE_PAD, EOS = 151652, 151653, 151655, 151643
VOCAB_SIZE = 151936

PATCH = 32                      # one <|image_pad|> per 32x32 region
TOPICS = ("caption", "general", "ocr", "chart", "gui", "stem")
MAX_SIDE = {"caption": 512, "general": 512, "ocr": 1024, "chart": 1024, "gui": 1024, "stem": 1024}
MAX_TEXT_TOKENS = 1024
SAMPLES_PER_SHARD = 2000
N_VAL_MM = 2000
N_VAL_TEXT = 20000
JPEG_QUALITY = 90
SPLIT_SEED = 0

MM_SOURCES = ["cc12m-wds", "cc3m-wds", "the_cauldron", "LLaVA-OneVision-Data", "Docmatix",
              "aguvis-stage1", "ChartGemma", "ChartQA", "ArxivQA", "MathV360K"]

# text sources: name -> (glob, text column, topic)
TEXT_SOURCES = OrderedDict([
    ("fineweb-edu", (f"{TEXT_ROOT}/fineweb-edu/sample/100BT/*.parquet", "text", "web")),
    ("open-web-math", (f"{TEXT_ROOT}/open-web-math/data/*.parquet", "text", "math")),
    ("starcoder-python", (f"{TEXT_ROOT}/starcoderdata/python/*.parquet", "content", "code")),
    ("starcoder-markdown", (f"{TEXT_ROOT}/starcoderdata/markdown/*.parquet", "content", "code")),
    ("wikipedia-en", (f"{TEXT_ROOT}/wikipedia-en/20231101.en/*.parquet", "text", "wiki")),
])

Image.MAX_IMAGE_PIXELS = 120_000_000   # above this PIL raises DecompressionBombError -> counted as a decode failure


def log(*a, **k):
    print(*a, **k, flush=True)


# ----------------------------------------------------------------------------- tokenizer
_TOK = None


def get_tokenizer():
    """Qwen3 tokenizer (fast). Verifies the three vision tokens and EOS on first load."""
    global _TOK
    if _TOK is None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        ids = tok.convert_tokens_to_ids(["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|endoftext|>"])
        assert ids == [VISION_START, VISION_END, IMAGE_PAD, EOS], f"unexpected special ids {ids}"
        assert len(tok) <= VOCAB_SIZE, len(tok)
        _TOK = tok
    return _TOK


def encode(text, max_tokens=None):
    ids = get_tokenizer()(text, add_special_tokens=False)["input_ids"]
    truncated = False
    if max_tokens is not None and len(ids) > max_tokens:
        ids = ids[:max_tokens]
        truncated = True
    return ids, truncated


def encode_batch(texts):
    """Batched tokenization, returns list of lists (no special tokens)."""
    return get_tokenizer()(texts, add_special_tokens=False)["input_ids"]


# ----------------------------------------------------------------------------- text rendering
_IMG_TAG = re.compile(r"<image>\s*")
_PREFIX_Q = re.compile(r"^\s*Question:\s*", re.I)
_PREFIX_A = re.compile(r"^\s*Answer:\s*", re.I)
_HINT_LINE = re.compile(r"^\s*Hint:[^\n]*\n", re.I)   # MathV360K-style boilerplate before the question


def strip_image_tags(s):
    return _IMG_TAG.sub("", s or "").strip()


def render_pairs(pairs):
    """pairs: list of (question, answer) -> "Question: ...\\nAnswer: ...\\n" per pair.
    Leading "Question:"/"Answer:" already present in the source text (the_cauldron) are removed to
    avoid "Question: Question: ..."; a leading "Hint: ...\n" line (MathV360K) is dropped."""
    out = []
    for q, a in pairs:
        q = _PREFIX_Q.sub("", _HINT_LINE.sub("", strip_image_tags(q), count=1), count=1).strip()
        a = _PREFIX_A.sub("", strip_image_tags(a), count=1).strip()
        if not q and not a:
            continue
        out.append(f"Question: {q}\nAnswer: {a}\n")
    return "".join(out)


def pairs_from_llava(conv):
    """LLaVA conversation list [{'from': 'human'|'gpt', 'value': str}] -> [(q, a), ...]."""
    pairs, q = [], None
    for t in conv or []:
        who = t.get("from")
        if who in ("human", "user"):
            q = t.get("value") or ""
        elif who in ("gpt", "assistant"):
            pairs.append((q or "", t.get("value") or ""))
            q = None
    return pairs


def count_image_turns(conv):
    """Number of human turns that contain an <image> tag (for per-image splitting of multi-image records)."""
    return sum(1 for t in conv or [] if t.get("from") in ("human", "user") and "<image>" in (t.get("value") or ""))


def split_multi_image_llava(images, conv):
    """A record with k images whose k human turns each carry one <image> tag is split into k
    single-image (image, [(q, a)]) samples (aguvis seeclick_mi). Returns None if not splittable."""
    if not isinstance(images, (list, tuple)) or len(images) < 2:
        return None
    pairs = pairs_from_llava(conv)
    img_turns = [i for i, (q, _) in enumerate(pairs) if "<image>" in q]
    if len(img_turns) != len(images) or any(q.count("<image>") != 1 for i, (q, _) in enumerate(pairs) if i in img_turns):
        return None
    out = []
    bounds = img_turns + [len(pairs)]
    for j, img in enumerate(images):
        out.append((img, pairs[bounds[j]:bounds[j + 1]]))
    return out


# ----------------------------------------------------------------------------- images
def target_size(w, h, max_side):
    """Both sides multiples of 32 (min 32), long side <= max_side, no upscaling beyond the 32-rounding."""
    scale = min(1.0, max_side / float(max(w, h)))
    tw = max(PATCH, int(round(w * scale / PATCH)) * PATCH)
    th = max(PATCH, int(round(h * scale / PATCH)) * PATCH)
    return min(tw, max_side), min(th, max_side)


def to_rgb(im):
    """EXIF-transpose, flatten alpha onto white (screenshots/charts), convert to RGB."""
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def process_image(data, max_side):
    """bytes -> (jpeg bytes, w, h). Raises on undecodable input (caller counts and skips)."""
    im = Image.open(io.BytesIO(data))
    im.load()
    im = to_rgb(im)
    w, h = im.size
    tw, th = target_size(w, h, max_side)
    if (tw, th) != (w, h):
        im = im.resize((tw, th), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue(), tw, th


def image_tokens(w, h):
    return (w // PATCH) * (h // PATCH)


def image_size_only(data):
    """(w, h, format, mode) without decoding pixels."""
    im = Image.open(io.BytesIO(data))
    return im.size[0], im.size[1], im.format, im.mode


# ----------------------------------------------------------------------------- parquet
def parquet_num_rows(path):
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata.num_rows


def parquet_row_group_offsets(pf):
    md = pf.metadata
    offs = [0]
    for i in range(md.num_row_groups):
        offs.append(offs[-1] + md.row_group(i).num_rows)
    return offs


def iter_parquet_rows(path, columns, start=0, end=None, wanted=None):
    """Yield (row_idx, dict) for rows in [start, end) (file-local indices), optionally only `wanted` rows."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    offs = parquet_row_group_offsets(pf)
    n = offs[-1]
    end = n if end is None else min(end, n)
    wanted = None if wanted is None else set(wanted)
    for rg in range(pf.metadata.num_row_groups):
        a, b = offs[rg], offs[rg + 1]
        if b <= start or a >= end:
            continue
        if wanted is not None and not any(a <= p < b for p in wanted):
            continue
        tbl = pf.read_row_group(rg, columns=columns)
        cols = {c: tbl.column(c).to_pylist() for c in columns}
        for j in range(b - a):
            r = a + j
            if r < start or r >= end:
                continue
            if wanted is not None and r not in wanted:
                continue
            yield r, {c: cols[c][j] for c in columns}


# ----------------------------------------------------------------------------- streams / archives
class ConcatReader(io.RawIOBase):
    """Read several files as one sequential stream (seeclick.tar.gz.part_00..05)."""

    def __init__(self, paths):
        self.paths = list(paths)
        self.i = 0
        self.f = open(self.paths[0], "rb") if self.paths else None

    def readable(self):
        return True

    def readinto(self, b):
        while self.f is not None:
            n = self.f.readinto(b)
            if n:
                return n
            self.f.close()
            self.i += 1
            self.f = open(self.paths[self.i], "rb") if self.i < len(self.paths) else None
        return 0

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None
        super().close()


def open_tar_stream(paths, gz=True):
    """tarfile in streaming mode over one or several concatenated files (optionally gzip)."""
    raw = io.BufferedReader(ConcatReader(paths), buffer_size=16 << 20)
    if gz:
        return tarfile.open(fileobj=gzip.GzipFile(fileobj=raw), mode="r|")
    return tarfile.open(fileobj=raw, mode="r|")


def iter_wds_tar(path, wanted=None, start_pos=0):
    """Iterate a webdataset tar; yield (pos, key, {ext: bytes}) grouping members by key.
    pos counts samples in file order from start_pos. If `wanted` (set of pos) is given, non-wanted
    samples are skipped without reading their data."""
    wanted = None if wanted is None else set(wanted)
    with tarfile.open(path, "r:") as tf:
        cur_key, cur, pos = None, {}, start_pos
        for m in tf:
            if not m.isfile():
                continue
            key, _, ext = m.name.rpartition(".")
            if key != cur_key:
                if cur_key is not None:
                    if wanted is None or pos in wanted:
                        yield pos, cur_key, cur
                    pos += 1
                cur_key, cur = key, {}
            if wanted is None or pos in wanted:
                cur[ext] = tf.extractfile(m).read()
        if cur_key is not None and (wanted is None or pos in wanted):
            yield pos, cur_key, cur


def zip_member_index(zpath):
    """basename -> member name for all files in a zip."""
    with zipfile.ZipFile(zpath) as z:
        return {os.path.basename(n): n for n in z.namelist() if not n.endswith("/")}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=int)
    os.replace(tmp, path)


def human_bytes(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f}{u}" if u != "B" else f"{n}B"
        n /= 1024.0


def human_count(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return str(n)
