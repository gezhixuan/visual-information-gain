"""Data pipeline for the trainer: shard readers, packing, 1:1 mixture, val sets.

Shard schema (the README): mm tar shards with <key>.jpg + <key>.json
{text_ids, source, topic, w, h, n_img, split}; text bins uint32 + int64 idx + meta.json.

A multimodal sample becomes  <|vision_start|> + n_img x <|image_pad|> + <|vision_end|>
+ text_ids + <|endoftext|>;  a text document becomes  ids + <|endoftext|>.
Sequences of `seq_len` tokens are packed from whole samples (samples longer than
seq_len are dropped, the tail is padded). Text and multimodal sequences alternate, so the
mixture is 1:1 by token count. Per-token metadata: sample_id (packing boundary; the pad
tail is its own sample), image_id (-1 outside images, unique per image in a sequence),
topic_id (index into TOPICS, -1 for pads). Loss mask at position t is true iff the
target t+1 is in the same sample and is not <|image_pad|> and not padding (so the loss
covers <|vision_end|>, the text and the EOS; it never covers image tokens).

Pixels: each 32x32 region of the (already 32-aligned) JPEG becomes one 3072-vector of
raw uint8 (row-major over the region: py, px, c) in row-major token order (gy, gx),
which is the order of the <|image_pad|> placeholders. The model normalises to
(x/255 - 0.5)/0.5. The exact permutation inside the 3072 values is absorbed by the first
linear layer, so this is equivalent to the "16x16 patches merged 2x2" of Chen et al. The SigLIP 2
front end (model.SiglipFrontEnd) splits the same rows into their four 16x16 patches.
`DataConfig.patch_px = 16` (linear front end only) instead cuts 16x16 regions: 768-vectors,
4x the image tokens, grid coords over the (h/16, w/16) grid; the packing, loss mask, image
spans and <|image_pad|> count follow from the token count, nothing else changes.

Mixture: text sources are mixed in proportion to n_tokens; mm sources in proportion to their
shard count times `DataConfig.mm_source_weights[source]` (default 1.0), after restricting a
source to its first `DataConfig.mm_source_caps[source]` shards (default: all). Defaults
reproduce the original behaviour exactly. Validation sets ignore both (fixed, per topic).

Resumability: every batch carries the data state after producing it (per worker);
the trainer saves the latest state of each DataLoader worker and the index of the next
worker, and `make_loader` hands those back so the stream continues exactly.
"""
from __future__ import annotations

import glob
import io
import json
import os
import tarfile
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
import torch
from PIL import Image

from . import paths

VISION_START, VISION_END, IMAGE_PAD, EOS = 151652, 151653, 151655, 151643
PAD_ID = EOS
MM_TOPICS = ["caption", "general", "ocr", "chart", "gui", "stem"]
TEXT_TOPICS = ["web", "math", "code", "wiki"]
TOPICS = MM_TOPICS + TEXT_TOPICS
TOPIC_ID = {t: i for i, t in enumerate(TOPICS)}
PATCH = 32
PATCH_DIM = PATCH * PATCH * 3


# ------------------------------------------------------------------ shard reader
def iter_shard(path: str) -> Iterator[tuple[str, bytes, dict]]:
    """Yield (key, jpg_bytes, meta) per sample from one tar, in file order."""
    pending: dict[str, dict] = {}
    with tarfile.open(path, "r|*") as tf:      # streaming mode: sequential, no seeks
        for m in tf:
            if not m.isfile():
                continue
            key, ext = os.path.splitext(m.name)
            key = os.path.basename(key)
            if ext not in (".jpg", ".json"):
                continue
            data = tf.extractfile(m).read()
            e = pending.setdefault(key, {})
            e["jpg" if ext == ".jpg" else "json"] = data
            if "jpg" in e and "json" in e:
                del pending[key]
                yield key, e["jpg"], json.loads(e["json"])


# ------------------------------------------------------------------ images
def jpg_to_patches(jpg: bytes, w: int, h: int, patch: int = PATCH) -> np.ndarray:
    """-> uint8 [(h/patch)*(w/patch), patch*patch*3], token order row-major (gy, gx); inside a
    token the pixels are row-major (py, px, c)."""
    img = Image.open(io.BytesIO(jpg)).convert("RGB")
    arr = np.asarray(img)
    if arr.shape[0] != h or arr.shape[1] != w:
        raise ValueError(f"image {arr.shape} != meta (h={h}, w={w})")
    gh, gw = h // patch, w // patch
    return arr.reshape(gh, patch, gw, patch, 3).transpose(0, 2, 1, 3, 4).reshape(gh * gw, patch * patch * 3)


def grid_coords(w: int, h: int, patch: int = PATCH) -> np.ndarray:
    gh, gw = h // patch, w // patch
    gy, gx = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], 1).astype(np.int64)   # [N, 2] = (gx, gy)


# ------------------------------------------------------------------ samples
@dataclass
class Sample:
    ids: np.ndarray                      # int64 tokens incl. special tokens and EOS
    topic: int
    kind: int                            # 0 text, 1 mm
    pixels: Optional[np.ndarray] = None  # uint8 [n_img, 3072]
    grid: Optional[np.ndarray] = None    # int64 [n_img, 2]
    img_offset: int = 0                  # position of the first <|image_pad|> in ids

    def __len__(self):
        return len(self.ids)


def mm_sample(jpg: bytes, meta: dict, decode: bool = True, patch: int = PATCH) -> Optional[Sample]:
    w, h = int(meta["w"]), int(meta["h"])
    # Image-token count is derived from the (32-aligned) image size. The shard json's "n_img"
    # is the number of images in the sample (always 1 in these shards), not a token count.
    if w % PATCH or h % PATCH or int(meta.get("n_img", 1)) != 1:
        return None
    assert patch in (16, 32), patch
    n_img = (w // patch) * (h // patch)
    text = np.asarray(meta["text_ids"], dtype=np.int64)
    ids = np.concatenate([[VISION_START], np.full(n_img, IMAGE_PAD, dtype=np.int64), [VISION_END], text, [EOS]])
    topic = TOPIC_ID.get(meta.get("topic", ""), TOPIC_ID["general"])
    s = Sample(ids=ids, topic=topic, kind=1, img_offset=1)
    if decode:
        try:
            s.pixels = jpg_to_patches(jpg, w, h, patch)
        except Exception as e:  # noqa: BLE001
            print(f"[data] bad image skipped: {e}")
            return None
        s.grid = grid_coords(w, h, patch)
    return s


# ------------------------------------------------------------------ sources
class MMSource:
    """Iterates train samples of one mm source over its tar shards, sharded over
    `n_workers` global workers, resumable via (epoch, shard_pos, offset)."""

    def __init__(self, shards: list[str], topic_hint: str, worker: int, n_workers: int, seed: int,
                 patch: int = PATCH):
        assert shards, "no shards"
        self.all_shards = sorted(shards)
        self.topic_hint = topic_hint
        self.worker, self.n_workers, self.seed = worker, n_workers, seed
        self.patch = patch
        self.epoch, self.shard_pos, self.offset = 0, 0, 0
        self._it = None
        self.n_shards = len(self.all_shards)
        # guard against spinning forever when this worker's share of a source has no usable sample
        # (e.g. every sample rejected by mm_sample): counts samples yielded since the start of a full pass
        self._pass_yield, self._full_pass = 0, True

    def _order(self, epoch: int) -> list[str]:
        rng = np.random.default_rng([self.seed, epoch, 7])
        perm = rng.permutation(len(self.all_shards))
        shards = [self.all_shards[i] for i in perm]
        if len(shards) >= self.n_workers:
            return shards[self.worker::self.n_workers]
        return shards            # few shards: every worker reads all, strides samples

    def _sample_stride(self) -> tuple[int, int]:
        if len(self.all_shards) >= self.n_workers:
            return 1, 0
        return self.n_workers, self.worker

    def state(self) -> dict:
        return {"epoch": self.epoch, "shard_pos": self.shard_pos, "offset": self.offset}

    def load_state(self, st: dict):
        self.epoch, self.shard_pos, self.offset = st["epoch"], st["shard_pos"], st["offset"]
        self._it = None
        self._pass_yield, self._full_pass = 0, (self.shard_pos == 0 and self.offset == 0)

    def _open(self):
        order = self._order(self.epoch)
        while self.shard_pos >= len(order):
            if self._full_pass and self._pass_yield == 0:
                raise RuntimeError(f"[data] mm source {self.topic_hint}: worker {self.worker}/{self.n_workers} found no "
                                   f"usable train sample in a full pass over its {len(order)} shard(s) of epoch "
                                   f"{self.epoch} (every sample rejected by mm_sample: n_img != 1, size not a multiple "
                                   f"of 32, or undecodable); refusing to spin")
            self.epoch += 1
            self.shard_pos, self.offset = 0, 0
            self._pass_yield, self._full_pass = 0, True
            order = self._order(self.epoch)
        stride, phase = self._sample_stride()
        path = order[self.shard_pos]

        def gen():
            for j, (key, jpg, meta) in enumerate(iter_shard(path)):
                if j < self.offset or (j % stride) != phase:
                    continue
                if meta.get("split", "train") != "train":
                    continue
                yield j, jpg, meta
        self._it = gen()

    def next(self) -> Sample:
        while True:
            if self._it is None:
                self._open()
            try:
                j, jpg, meta = next(self._it)
            except StopIteration:
                self._it = None
                self.shard_pos += 1
                self.offset = 0
                continue
            self.offset = j + 1
            s = mm_sample(jpg, meta, patch=self.patch)
            if s is not None:
                self._pass_yield += 1
                return s


class TextSource:
    """One text source (train.bin/idx/meta.json); worker w reads docs w::n_workers."""

    def __init__(self, root: str, worker: int, n_workers: int, split: str = "train", truncate: bool = False):
        self.bin = np.memmap(os.path.join(root, f"{split}.bin"), dtype=np.uint32, mode="r")
        self.idx = np.fromfile(os.path.join(root, f"{split}.idx"), dtype=np.int64)
        self.meta = json.load(open(os.path.join(root, "meta.json")))
        self.topic = TOPIC_ID.get(self.meta.get("topic", "web"), TOPIC_ID["web"])
        self.n_tokens = int(self.meta.get("n_tokens", len(self.bin)))
        self.worker, self.n_workers, self.truncate = worker, n_workers, truncate
        self.n_docs = len(self.idx)
        self.pos = 0        # count of own docs consumed (across epochs)

    def state(self) -> dict:
        return {"pos": self.pos}

    def load_state(self, st: dict):
        self.pos = st["pos"]

    def doc(self, i: int) -> np.ndarray:
        a = int(self.idx[i])
        b = int(self.idx[i + 1]) if i + 1 < self.n_docs else len(self.bin)
        ids = np.asarray(self.bin[a:b], dtype=np.int64)
        if len(ids) == 0 or ids[-1] != EOS:
            ids = np.concatenate([ids, [EOS]])
        return ids

    def next(self, max_len: int) -> Sample:
        own = max(1, (self.n_docs - self.worker + self.n_workers - 1) // self.n_workers)
        while True:
            i = self.worker + (self.pos % own) * self.n_workers
            self.pos += 1
            if i >= self.n_docs:
                continue
            ids = self.doc(i)
            if len(ids) > max_len:
                if not self.truncate:
                    continue
                ids = np.concatenate([ids[:max_len - 1], [EOS]])
            return Sample(ids=ids, topic=self.topic, kind=0)


class Mixture:
    """Chooses among sources with fixed probabilities (seeded, resumable)."""

    def __init__(self, sources: list, weights: list[float], seed: int):
        self.sources = sources
        w = np.asarray(weights, dtype=np.float64)
        self.p = w / w.sum()
        self.rng = np.random.default_rng(seed)

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "sources": [s.state() for s in self.sources]}

    def load_state(self, st: dict):
        self.rng.bit_generator.state = st["rng"]
        for s, ss in zip(self.sources, st["sources"]):
            s.load_state(ss)

    def next(self, max_len: int) -> Sample:
        i = int(self.rng.choice(len(self.sources), p=self.p))
        src = self.sources[i]
        return src.next(max_len) if isinstance(src, TextSource) else src.next()


# ------------------------------------------------------------------ packing
@dataclass
class PackedSeq:
    ids: np.ndarray
    sample_id: np.ndarray
    image_id: np.ndarray
    topic_id: np.ndarray
    loss_mask: np.ndarray
    kind: int
    pixels: list = field(default_factory=list)   # list of uint8 [n, 3072]
    grid: list = field(default_factory=list)     # list of int64 [n, 2]
    img_pos: list = field(default_factory=list)  # list of int64 [n] positions in seq
    n_samples: int = 0
    n_dropped: int = 0


def pack(samples: list[Sample], seq_len: int, kind: int) -> PackedSeq:
    T = seq_len
    ids = np.full(T, PAD_ID, dtype=np.int64)
    sample_id = np.full(T, len(samples), dtype=np.int32)   # pad tail = its own sample
    image_id = np.full(T, -1, dtype=np.int32)
    topic_id = np.full(T, -1, dtype=np.int32)
    seq = PackedSeq(ids, sample_id, image_id, topic_id, np.zeros(T, bool), kind, n_samples=len(samples))
    t, n_img = 0, 0
    for si, s in enumerate(samples):
        n = len(s)
        ids[t:t + n] = s.ids
        sample_id[t:t + n] = si
        topic_id[t:t + n] = s.topic
        if s.pixels is not None:
            a = t + s.img_offset
            k = s.pixels.shape[0]
            image_id[a:a + k] = n_img
            seq.pixels.append(s.pixels)
            seq.grid.append(s.grid)
            seq.img_pos.append(np.arange(a, a + k, dtype=np.int64))
            n_img += 1
        t += n
    # loss on position t iff target t+1 is same sample, not image pad, not padding
    nxt_same = sample_id[1:] == sample_id[:-1]
    nxt_text = (ids[1:] != IMAGE_PAD) & (topic_id[1:] >= 0)
    seq.loss_mask[:-1] = nxt_same & nxt_text
    return seq


class Packer:
    """Greedy packer over a sample stream (whole samples; a sample that does not fit
    closes the current sequence and opens the next). State is exactly resumable: the
    pending (carried-over) sample is re-read from the source after a resume."""

    def __init__(self, mixture: Mixture, seq_len: int, kind: int):
        self.mix, self.T, self.kind = mixture, seq_len, kind
        self.pending: Optional[Sample] = None
        self.pending_state: Optional[dict] = None
        self.n_dropped = 0

    def state(self) -> dict:
        return self.pending_state if self.pending is not None else self.mix.state()

    def load_state(self, st: dict):
        self.mix.load_state(st)
        self.pending, self.pending_state = None, None

    def _pull(self) -> tuple[dict, Sample]:
        while True:
            st = self.mix.state()
            s = self.mix.next(self.T)
            if len(s) <= self.T:
                return st, s
            self.n_dropped += 1

    def next(self) -> PackedSeq:
        buf, used = [], 0
        if self.pending is not None:
            buf.append(self.pending)
            used = len(self.pending)
            self.pending = self.pending_state = None
        while True:
            st, s = self._pull()
            if used + len(s) > self.T:
                self.pending, self.pending_state = s, st
                break
            buf.append(s)
            used += len(s)
            if used == self.T:
                break
        seq = pack(buf, self.T, self.kind)
        seq.n_dropped, self.n_dropped = self.n_dropped, 0
        return seq


def collate(seqs: list[PackedSeq]) -> dict:
    B = len(seqs)
    out = {
        "input_ids": torch.from_numpy(np.stack([s.ids for s in seqs])),
        "sample_id": torch.from_numpy(np.stack([s.sample_id for s in seqs])),
        "image_id": torch.from_numpy(np.stack([s.image_id for s in seqs])),
        "topic_id": torch.from_numpy(np.stack([s.topic_id for s in seqs])),
        "loss_mask": torch.from_numpy(np.stack([s.loss_mask for s in seqs])),
        "kind": torch.tensor([s.kind for s in seqs], dtype=torch.int64),
        "n_samples": sum(s.n_samples for s in seqs),
        "n_dropped": sum(s.n_dropped for s in seqs),
    }
    ids = out["input_ids"]
    tgt = torch.full_like(ids, -100)
    tgt[:, :-1] = ids[:, 1:]
    tgt[~out["loss_mask"]] = -100
    out["targets"] = tgt
    pix, grid, idx = [], [], []
    for b, s in enumerate(seqs):
        for p, g, pos in zip(s.pixels, s.grid, s.img_pos):
            pix.append(p)
            grid.append(g)
            idx.append(np.stack([np.full_like(pos, b), pos], 1))
    if pix:
        out["pixels"] = torch.from_numpy(np.concatenate(pix))
        out["grid_xy"] = torch.from_numpy(np.concatenate(grid))
        out["img_index"] = torch.from_numpy(np.concatenate(idx))
    else:
        out["pixels"] = torch.zeros(0, PATCH_DIM, dtype=torch.uint8)
        out["grid_xy"] = torch.zeros(0, 2, dtype=torch.int64)
        out["img_index"] = torch.zeros(0, 2, dtype=torch.int64)
    return out


def attended_keys(sample_id: torch.Tensor, image_id: torch.Tensor, bidirectional: bool = True) -> torch.Tensor:
    """Number of keys each query attends to under the mask, [B, T] (for eq. 6's l)."""
    B, T = sample_id.shape
    ar = torch.arange(T, device=sample_id.device).expand(B, T)
    chg = torch.ones(B, T, dtype=torch.bool, device=sample_id.device)
    chg[:, 1:] = sample_id[:, 1:] != sample_id[:, :-1]
    start = torch.where(chg, ar, torch.zeros_like(ar)).cummax(1).values
    n = ar - start + 1
    if bidirectional:
        is_img = image_id >= 0
        chg_i = torch.ones(B, T, dtype=torch.bool, device=sample_id.device)
        chg_i[:, :-1] = image_id[:, :-1] != image_id[:, 1:]
        end = torch.where(chg_i, ar, torch.full_like(ar, T)).flip(1).cummin(1).values.flip(1)
        n = n + torch.where(is_img, end - ar, torch.zeros_like(ar))
    return n


# ------------------------------------------------------------------ dataset
@dataclass
class DataConfig:
    shard_root: str = paths.SHARDS
    mm_sources: str = "*"            # glob under shard_root/mm/
    text_sources: str = "*"          # glob under shard_root/text/
    seq_len: int = 4096
    truncate_text_docs: bool = True  # long text docs are truncated to seq_len rather than dropped (avoids a short-document bias; multimodal samples longer than seq_len are dropped)
    seed: int = 0
    num_workers: int = 4
    patch_px: int = 32               # pixels per visual token: 32 (default, both front ends) or 16 (linear front end only)
    # mm mixture: sampling weight = n_shards(source, after cap) * mm_source_weights.get(source, 1.0);
    # mm_source_caps[source] = use only the first k train shards (~2,000 samples each). {} = original behaviour.
    mm_source_weights: dict = field(default_factory=dict)
    mm_source_caps: dict = field(default_factory=dict)
    # validation
    val_seqs_per_topic: int = 8      # packed mm sequences per topic
    val_seqs_per_text_source: int = 4
    diag_seqs: int = 2               # mm sequences for the attention/cosine diagnostics


def list_mm_sources(cfg: DataConfig, split: str = "train") -> dict[str, list[str]]:
    out = {}
    for d in sorted(glob.glob(os.path.join(cfg.shard_root, "mm", cfg.mm_sources))):
        shards = sorted(glob.glob(os.path.join(d, f"{split}-*.tar")))
        if shards:
            out[os.path.basename(d)] = shards
    return out


def mm_source_plan(cfg: DataConfig, mm: dict[str, list[str]]) -> tuple[list[str], list[list[str]], list[float]]:
    """Apply mm_source_caps and mm_source_weights to {source: shards}. Returns (names, shard lists,
    sampling weights); sources with weight 0 are dropped; unknown names are an error."""
    unknown = [k for k in list(cfg.mm_source_weights) + list(cfg.mm_source_caps) if k not in mm]
    assert not unknown, f"unknown mm sources {unknown}; available: {sorted(mm)}"
    names, shards, weights = [], [], []
    for name, sh in mm.items():
        cap = int(cfg.mm_source_caps.get(name, 0))
        if cap > 0:
            sh = sh[:cap]
        w = len(sh) * float(cfg.mm_source_weights.get(name, 1.0))
        assert w >= 0, (name, w)
        if w == 0:
            continue
        names.append(name)
        shards.append(sh)
        weights.append(w)
    assert names, "every mm source has weight 0"
    return names, shards, weights


def list_text_sources(cfg: DataConfig) -> list[str]:
    return sorted(d for d in glob.glob(os.path.join(cfg.shard_root, "text", cfg.text_sources))
                  if os.path.exists(os.path.join(d, "train.bin")))


class PackedDataset(torch.utils.data.IterableDataset):
    """Yields collated batches of `batch_seqs` sequences alternating text / mm, each with
    the worker's data state after the batch."""

    def __init__(self, cfg: DataConfig, batch_seqs: int, rank: int, world: int,
                 worker_states: Optional[list] = None):
        super().__init__()
        self.cfg, self.B, self.rank, self.world = cfg, batch_seqs, rank, world
        self.worker_states = worker_states

    def _build(self, worker: int, n_workers: int):
        cfg = self.cfg
        g = self.rank * n_workers + worker
        G = self.world * n_workers
        mm = list_mm_sources(cfg, "train")
        assert mm, f"no mm train shards under {cfg.shard_root}/mm/{cfg.mm_sources}"
        names, shard_lists, weights = mm_source_plan(cfg, mm)
        mm_src = [MMSource(sh, name, g, G, cfg.seed, patch=cfg.patch_px) for name, sh in zip(names, shard_lists)]
        mm_mix = Mixture(mm_src, weights, seed=cfg.seed * 1_000_003 + 2 * g + 1)
        tx_roots = list_text_sources(cfg)
        assert tx_roots, f"no text sources under {cfg.shard_root}/text/{cfg.text_sources}"
        tx_src = [TextSource(r, g, G, truncate=cfg.truncate_text_docs) for r in tx_roots]
        tx_mix = Mixture(tx_src, [s.n_tokens for s in tx_src], seed=cfg.seed * 1_000_003 + 2 * g)
        return Packer(tx_mix, cfg.seq_len, 0), Packer(mm_mix, cfg.seq_len, 1)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        worker, n_workers = (wi.id, wi.num_workers) if wi else (0, 1)
        text_p, mm_p = self._build(worker, n_workers)
        turn = 0
        if self.worker_states is not None and self.worker_states[worker] is not None:
            st = self.worker_states[worker]
            text_p.load_state(st["text"])
            mm_p.load_state(st["mm"])
            turn = st.get("turn", 0)
        while True:
            seqs = []
            for _ in range(self.B):
                seqs.append(text_p.next() if turn % 2 == 0 else mm_p.next())
                turn += 1
            batch = collate(seqs)
            batch["state"] = {"text": text_p.state(), "mm": mm_p.state(), "turn": turn, "worker": worker}
            yield batch


def make_loader(cfg: DataConfig, batch_seqs: int, rank: int, world: int,
                data_state: Optional[dict] = None, num_workers: Optional[int] = None):
    """data_state (from a checkpoint) = {"workers": {wid: state}, "next_worker": int}.
    New worker w resumes old worker (w + next_worker) % W so the batch order continues."""
    W = cfg.num_workers if num_workers is None else num_workers
    worker_states = None
    if data_state is not None:
        Wn = max(W, 1)
        old = data_state["workers"]
        nw = data_state.get("next_worker", 0)
        worker_states = [old.get((w + nw) % Wn) for w in range(Wn)]
    ds = PackedDataset(cfg, batch_seqs, rank, world, worker_states)
    return torch.utils.data.DataLoader(ds, batch_size=None, num_workers=W, pin_memory=torch.cuda.is_available(),
                                       prefetch_factor=(4 if W > 0 else None), persistent_workers=False)


class DataStateTracker:
    """Keeps the latest state reported by each worker + the next worker index."""

    def __init__(self, num_workers: int):
        self.W = max(num_workers, 1)
        self.workers: dict[int, dict] = {}
        self.n_batches = 0

    def update(self, batch_state: dict):
        self.workers[batch_state["worker"]] = batch_state
        self.n_batches += 1

    def state(self) -> dict:
        return {"workers": dict(self.workers), "next_worker": self.n_batches % self.W}


# ------------------------------------------------------------------ validation sets
def build_val_sets(cfg: DataConfig) -> dict:
    """Fixed held-out batches: per mm topic `val_seqs_per_topic` packed sequences (from
    split 'val' shards of every mm source, round-robin over sources), per text source
    `val_seqs_per_text_source` sequences from val.bin. Also a small mm 'diag' batch.
    Returns {"mm": {topic: batch}, "text": {source_name: batch}, "diag": batch}."""
    T = cfg.seq_len
    mm = list_mm_sources(cfg, "val")
    buckets: dict[str, list[Sample]] = {t: [] for t in MM_TOPICS}
    need = {t: cfg.val_seqs_per_topic * T for t in MM_TOPICS}
    iters = {name: (x for sh in shards for x in iter_shard(sh)) for name, shards in mm.items()}
    tokens = {t: 0 for t in MM_TOPICS}
    while iters and any(tokens[t] < need[t] for t in MM_TOPICS):
        for name in list(iters):
            try:
                key, jpg, meta = next(iters[name])
            except StopIteration:
                del iters[name]
                continue
            topic = meta.get("topic", "general")
            if topic not in buckets or tokens[topic] >= need[topic]:
                continue
            s = mm_sample(jpg, meta, patch=cfg.patch_px)
            if s is None or len(s) > T:
                continue
            buckets[topic].append(s)
            tokens[topic] += len(s)
    out = {"mm": {}, "text": {}, "diag": None}
    diag_seqs = []
    for topic, samples in buckets.items():
        if not samples:
            continue
        seqs = _pack_list(samples, T, kind=1)[: cfg.val_seqs_per_topic]
        out["mm"][topic] = collate(seqs)
        diag_seqs.extend(seqs[:1])
    out["diag"] = collate(diag_seqs[: cfg.diag_seqs]) if diag_seqs else None
    for root in list_text_sources(cfg):
        if not os.path.exists(os.path.join(root, "val.bin")):
            continue
        src = TextSource(root, 0, 1, split="val", truncate=cfg.truncate_text_docs)
        samples, used = [], 0
        for i in range(src.n_docs):
            ids = src.doc(i)
            if len(ids) > T:
                if not cfg.truncate_text_docs:
                    continue
                ids = np.concatenate([ids[:T - 1], [EOS]])
            samples.append(Sample(ids=ids, topic=src.topic, kind=0))
            used += len(ids)
            if used >= cfg.val_seqs_per_text_source * T:
                break
        if samples:
            seqs = _pack_list(samples, T, kind=0)[: cfg.val_seqs_per_text_source]
            out["text"][os.path.basename(root)] = collate(seqs)
    return out


def _pack_list(samples: list[Sample], T: int, kind: int) -> list[PackedSeq]:
    seqs, buf, used = [], [], 0
    for s in samples:
        if used + len(s) > T:
            seqs.append(pack(buf, T, kind))
            buf, used = [], 0
        buf.append(s)
        used += len(s)
    if buf:
        seqs.append(pack(buf, T, kind))
    return seqs
