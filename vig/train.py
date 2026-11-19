"""torchrun-ready trainer for the ladder (DDP, bf16 autocast, AdamW, token-based
schedule and checkpointing).

    torchrun --nproc_per_node 8 -m vig.train --config d1024_L12 [--set lr=1e-3 --set run=foo]

Everything is scheduled in tokens *processed* (sequences x seq_len, padding and image
tokens included, as D in Chen et al.). Checkpoints go to <out_root>/<run>/:
  ckpt_<tokens>B.pt        bf16 weights every save_every_tokens
  state.pt                 rolling full state (fp32 weights, optimizer, tokens, rng) for resume
  data_state_rank<r>.pt    per-rank data position (per DataLoader worker)
  train.jsonl / eval.jsonl logs;  DONE marker when total_tokens is reached.
Exit code 3 = wall-time limit reached after saving state (a job script can resubmit on it).
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from . import data as D
from . import eval as E
from . import paths
from .model import EncFreeDecoder, ModelConfig, count_params


@dataclass
class TrainConfig:
    run: str = "debug"
    out_root: str = paths.RUNS
    model: ModelConfig = field(default_factory=ModelConfig)
    data: D.DataConfig = field(default_factory=D.DataConfig)
    # batch / budget (tokens)
    global_batch_tokens: int = 524_288      # 128 x 4096
    micro_batch_seqs: int = 8               # sequences per GPU per micro-step
    total_tokens: int = 40_000_000_000
    # optimizer
    lr: float = 1.5e-3
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0
    schedule: str = "cosine"                # "cosine" | "wsd"
    warmup_tokens: int = 1_000_000_000      # 2,000 warmup steps at batch 128
    wsd_decay_frac: float = 0.2             # WSD: linear decay over the last fraction
    # cadence (tokens)
    log_every_tokens: int = 50_000_000
    log_first_steps: int = 10
    eval_every_tokens: int = 500_000_000
    save_every_tokens: int = 1_000_000_000
    state_every_tokens: int = 500_000_000
    eval_at_start: bool = True
    # system
    seed: int = 0
    dtype: str = "bf16"                     # autocast dtype on CUDA; "fp32" disables autocast
    peak_flops: float = 989e12              # peak FLOP/s of one accelerator, for the utilisation log
    max_hours: float = 23.5                 # save + exit(3) before the wall-time limit of a job
    stop_at_tokens: int = 0                 # tests: stop (exit 0) after this many tokens
    eval_micro: int = 8


# ----------------------------------------------------------------------- config utils
def load_config(name: str) -> TrainConfig:
    """A name from vig/configs (d1024_L12) or the path of a configuration file that defines `config`."""
    if name.endswith(".py"):
        spec = importlib.util.spec_from_file_location("vig_user_config", name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(f"vig.configs.{name}")
    return mod.config


def apply_overrides(cfg, items: list[str]):
    for it in items:
        key, val = it.split("=", 1)
        obj, parts = cfg, key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        f = {x.name: x for x in fields(obj)}[parts[-1]]
        cur = getattr(obj, parts[-1])
        typ = type(cur) if cur is not None else f.type
        if typ is bool:
            v = val.lower() in ("1", "true", "yes")
        elif typ is dict:
            v = json.loads(val)        # e.g. --set data.mm_source_weights='{"Docmatix": 3}'
        elif typ is tuple:
            v = tuple(float(x) for x in val.strip("()").split(","))
        elif typ is int:
            v = int(float(val))
        elif typ is float:
            v = float(val)
        else:
            v = val
        setattr(obj, parts[-1], v)
    return cfg


def to_dict(obj):
    return dataclasses.asdict(obj) if is_dataclass(obj) else obj


# ----------------------------------------------------------------------- schedule
def lr_at(cfg: TrainConfig, tokens: int) -> float:
    if tokens < cfg.warmup_tokens:
        return cfg.lr * (tokens + 1) / cfg.warmup_tokens
    lo = cfg.lr * cfg.min_lr_ratio
    if cfg.schedule == "cosine":
        p = min(1.0, (tokens - cfg.warmup_tokens) / max(1, cfg.total_tokens - cfg.warmup_tokens))
        return lo + 0.5 * (cfg.lr - lo) * (1 + math.cos(math.pi * p))
    if cfg.schedule == "wsd":
        start = cfg.total_tokens * (1 - cfg.wsd_decay_frac)
        if tokens < start:
            return cfg.lr
        p = min(1.0, (tokens - start) / max(1, cfg.total_tokens - start))
        return cfg.lr - (cfg.lr - lo) * p
    raise ValueError(cfg.schedule)


# ----------------------------------------------------------------------- checkpoint io
def atomic_save(obj, path: str):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def fmt_tok(n: int) -> str:
    return f"{n/1e9:.4f}B" if n >= 1e8 else (f"{n/1e6:.2f}M" if n >= 1e5 else f"{n/1e3:.1f}K")


def ckpt_name(tokens: int) -> str:
    return f"ckpt_{tokens / 1e9:.2f}B.pt"


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="dotted override, e.g. --set lr=1e-3")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)

    # ---- distributed
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local_rank = 0, 1, 0
    cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    autocast_dtype = torch.bfloat16 if (cuda and cfg.dtype == "bf16") else None
    is_main = rank == 0
    log = (lambda *a: print(*a, flush=True)) if is_main else (lambda *a: None)

    run_dir = os.path.join(cfg.out_root, cfg.run)
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)

    # ---- model
    if not cuda and cfg.model.attn_impl == "flex":
        log("[train] CPU: forcing attn_impl=eager")
        cfg.model.attn_impl = "eager"
    cfg.model.seq_len = cfg.data.seq_len
    cfg.model.patch_px = cfg.data.patch_px       # one source of truth for the token size
    if cfg.model.max_grid * cfg.model.patch_px < 2048:
        log(f"[train] note: max_grid {cfg.model.max_grid} x patch_px {cfg.model.patch_px} covers "
            f"{cfg.model.max_grid * cfg.model.patch_px} px per side")
    model = EncFreeDecoder(cfg.model).to(device)
    pc = model.param_counts()
    assert pc == count_params(cfg.model), (pc, count_params(cfg.model))
    log(f"[train] params: {json.dumps(pc)}")
    if model.is_siglip:
        log(f"[train] front end: siglip2 {cfg.model.vision.siglip_path or '(random init)'} "
            f"frozen={cfg.model.vision.freeze_encoder} grad_ckpt={cfg.model.vision.encoder_grad_checkpoint} "
            f"encoder FLOPs multiplier={model.encoder_flops_mult()}")
    if is_main:
        json.dump({"config": to_dict(cfg), "params": pc}, open(os.path.join(run_dir, "config.json"), "w"), indent=1)

    # ---- optimizer: decoupled weight decay on matrices only (no decay on norms, biases,
    # embeddings and position tables)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:          # frozen encoder
            continue
        (decay if (p.dim() >= 2 and "emb" not in n and "pos_" not in n) else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps, fused=cuda)

    # ---- batch arithmetic
    T = cfg.data.seq_len
    micro_tokens = cfg.micro_batch_seqs * T
    grad_accum = cfg.global_batch_tokens // (micro_tokens * world)
    assert grad_accum >= 1 and grad_accum * micro_tokens * world == cfg.global_batch_tokens, \
        f"global_batch_tokens {cfg.global_batch_tokens} must equal micro_batch_seqs*seq_len*world*k"
    log(f"[train] world={world} micro={cfg.micro_batch_seqs}x{T} grad_accum={grad_accum} "
        f"global_batch_tokens={cfg.global_batch_tokens}")

    # ---- resume
    tokens, mm_tokens, step = 0, 0, 0
    data_state = None
    state_path = os.path.join(run_dir, "state.pt")
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_checkpoint_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        tokens, mm_tokens, step = st["tokens"], st["mm_tokens"], st["step"]
        torch.set_rng_state(st["rng_cpu"])
        ds_path = os.path.join(run_dir, f"data_state_rank{rank}.pt")
        dsd = torch.load(ds_path, map_location="cpu", weights_only=False)
        assert dsd["world"] == world and dsd["num_workers"] == cfg.data.num_workers, \
            "resume requires the same world size and num_workers"
        data_state = dsd["state"]
        log(f"[train] resumed from {state_path}: step={step} tokens={fmt_tok(tokens)}")

    ddp = DDP(model, device_ids=[local_rank] if cuda else None) if distributed else model
    raw = model

    # ---- data
    loader = D.make_loader(cfg.data, cfg.micro_batch_seqs, rank, world, data_state)
    tracker = D.DataStateTracker(cfg.data.num_workers)
    if data_state is not None:
        tracker.workers = dict(data_state["workers"])
        tracker.n_batches = data_state["next_worker"]
    data_iter = iter(loader)

    val_sets = None
    if is_main:
        vpath = os.path.join(run_dir, "val_sets.pt")
        if os.path.exists(vpath):
            val_sets = torch.load(vpath, weights_only=False)
        else:
            t0 = time.time()
            val_sets = D.build_val_sets(cfg.data)
            atomic_save(val_sets, vpath)
            log(f"[train] built val sets in {time.time()-t0:.1f}s: mm topics={list(val_sets['mm'])} "
                f"text={list(val_sets['text'])}")

    def save_data_state():
        atomic_save({"state": tracker.state(), "world": world, "num_workers": cfg.data.num_workers},
                    os.path.join(run_dir, f"data_state_rank{rank}.pt"))

    def save_state():
        save_data_state()
        if is_main:
            atomic_save({"model": raw.checkpoint_state_dict(), "opt": opt.state_dict(), "tokens": tokens,
                         "mm_tokens": mm_tokens, "step": step, "rng_cpu": torch.get_rng_state(),
                         "config": to_dict(cfg)}, state_path)
        if distributed:
            dist.barrier()

    def save_snapshot():
        if is_main:
            sd = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in raw.checkpoint_state_dict().items()}
            atomic_save({"model": sd, "tokens": tokens, "step": step, "model_config": to_dict(cfg.model)},
                        os.path.join(run_dir, ckpt_name(tokens)))

    def run_eval():
        if is_main:
            rec = {"step": step, "tokens": tokens, "mm_tokens": mm_tokens, "lr": lr_at(cfg, tokens)}
            rec.update(E.evaluate(raw, val_sets, device, autocast_dtype, cfg.eval_micro))
            E.append_jsonl(os.path.join(run_dir, "eval.jsonl"), rec)
            am = rec.get("diag/attn_mass", [])
            log(f"[eval] tok={fmt_tok(tokens)} val/mm={rec.get('val/mm', float('nan')):.4f} "
                f"val/text={rec.get('val/text', float('nan')):.4f} attn_mass(L)={[round(a, 3) for a in am]} "
                f"({rec['eval_seconds']:.1f}s)")
        if distributed:
            dist.barrier()

    def boundary(before: int, after: int, every: int) -> bool:
        return every > 0 and (after // every) > (before // every)

    # ---- loop
    if cfg.eval_at_start and tokens == 0:
        run_eval()
    model.train()
    t_start = time.time()
    win = {"loss": 0.0, "n": 0, "loss_t": 0.0, "n_t": 0, "loss_m": 0.0, "n_m": 0, "flops": 0.0, "flops_dec": 0.0,
           "tokens": 0, "t0": time.time(), "gn": 0.0, "steps": 0, "n_dropped": 0}
    last_eval_tokens = -1
    exit_code = 0
    while tokens < cfg.total_tokens:
        lr = lr_at(cfg, tokens)
        for g in opt.param_groups:
            g["lr"] = lr
        for k in range(grad_accum):
            batch = next(data_iter)
            tracker.update(batch["state"])
            b = {k2: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k2, v in batch.items() if k2 != "state"}
            sync = (k == grad_accum - 1) or not distributed
            ctx = ddp.no_sync() if (distributed and not sync) else torch.enable_grad()
            with ctx:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    h = ddp(b["input_ids"], b["sample_id"], b["image_id"], b["pixels"], b["grid_xy"], b["img_index"])
                    losses, sel = raw.token_losses(h, b["targets"], b["loss_mask"])
                loss = losses.mean() / grad_accum
                loss.backward()
            with torch.no_grad():
                kind_tok = b["kind"][sel // T]
                ls = losses.detach()
                win["loss"] += float(ls.sum()); win["n"] += ls.numel()
                mt = kind_tok == 1
                win["loss_m"] += float(ls[mt].sum()); win["n_m"] += int(mt.sum())
                win["loss_t"] += float(ls[~mt].sum()); win["n_t"] += int((~mt).sum())
                n_mm_seq = int((b["kind"] == 1).sum())
                mm_tokens += n_mm_seq * T * world          # same composition on every rank
                keys = D.attended_keys(b["sample_id"], b["image_id"], cfg.model.image_bidirectional)
                fl = raw.flops_per_token(0.0)
                fe = raw.front_end_flops(b["grid_xy"])     # linear: 6 * front-end params * visual tokens; siglip2: + encoder
                dec = fl["m_base"] * b["input_ids"].numel() + 12 * cfg.model.n_layers * cfg.model.d_model * float(keys.sum())
                win["flops"] += (dec + fe["adapter"] + fe["encoder"]) * world
                win["flops_dec"] += dec * world
                win["n_dropped"] += batch["n_dropped"]
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        prev = tokens
        tokens += cfg.global_batch_tokens
        step += 1
        win["gn"] += float(gn); win["steps"] += 1; win["tokens"] += cfg.global_batch_tokens

        if boundary(prev, tokens, cfg.log_every_tokens) or step <= cfg.log_first_steps or tokens >= cfg.total_tokens:
            agg = torch.tensor([win["loss"], win["n"], win["loss_t"], win["n_t"], win["loss_m"], win["n_m"]],
                               dtype=torch.float64, device=device)
            if distributed:
                dist.all_reduce(agg)
            dt = time.time() - win["t0"]
            rec = {"step": step, "tokens": tokens, "mm_tokens": mm_tokens, "lr": lr,
                   "loss": float(agg[0] / max(agg[1], 1)), "loss_text_seq": float(agg[2] / max(agg[3], 1)),
                   "loss_mm_seq": float(agg[4] / max(agg[5], 1)), "grad_norm": win["gn"] / win["steps"],
                   "tokens_per_s": win["tokens"] / dt, "flops_per_token": win["flops"] / max(win["tokens"], 1),
                   "flops_per_token_decoder": win["flops_dec"] / max(win["tokens"], 1),
                   "flops_per_token_6N_nonemb": 6 * pc["non_embedding"],
                   "mfu": win["flops"] / dt / (cfg.peak_flops * world), "n_dropped_samples": win["n_dropped"],
                   "elapsed_h": (time.time() - t_start) / 3600}
            if is_main:
                E.append_jsonl(os.path.join(run_dir, "train.jsonl"), rec)
            log(f"[train] step={step} tok={fmt_tok(tokens)} mm={fmt_tok(mm_tokens)} loss={rec['loss']:.4f} "
                f"(text {rec['loss_text_seq']:.4f} | mm {rec['loss_mm_seq']:.4f}) lr={lr:.2e} gn={rec['grad_norm']:.2f} "
                f"tok/s={rec['tokens_per_s']:.0f} mfu={rec['mfu']:.3f}")
            win = {"loss": 0.0, "n": 0, "loss_t": 0.0, "n_t": 0, "loss_m": 0.0, "n_m": 0, "flops": 0.0, "flops_dec": 0.0,
                   "tokens": 0, "t0": time.time(), "gn": 0.0, "steps": 0, "n_dropped": 0}

        if boundary(prev, tokens, cfg.eval_every_tokens):
            run_eval()
            last_eval_tokens = tokens
        if boundary(prev, tokens, cfg.save_every_tokens):
            save_snapshot()
        if boundary(prev, tokens, cfg.state_every_tokens):
            save_state()

        if cfg.stop_at_tokens and tokens >= cfg.stop_at_tokens:
            save_state()
            log(f"[train] STOPPED at {tokens} tokens (stop_at_tokens)")
            break
        hours = (time.time() - t_start) / 3600
        if distributed:   # all ranks must agree on the time-limit decision
            t = torch.tensor([hours], device=device); dist.all_reduce(t, op=dist.ReduceOp.MAX); hours = float(t)
        if hours >= cfg.max_hours:
            save_state()
            log(f"[train] TIME_LIMIT after {hours:.2f}h at {fmt_tok(tokens)} tokens; exiting 3 for re-chain")
            exit_code = 3
            break
    else:
        if last_eval_tokens != tokens:
            run_eval()
        save_snapshot()
        save_state()
        if is_main:
            open(os.path.join(run_dir, "DONE"), "w").write(f"{tokens}\n")
        log(f"[train] DONE {fmt_tok(tokens)} tokens in {(time.time()-t_start)/3600:.2f}h")

    if distributed:
        dist.destroy_process_group()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
