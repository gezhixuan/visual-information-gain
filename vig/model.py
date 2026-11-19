"""Dense decoder for the ladder, with two interchangeable visual front ends.

Decoder: pre-norm, RMSNorm, RoPE, SwiGLU, optional GQA, tied/untied embeddings.
Front end "linear" (encoder-free, Chen et al., App. A.1; 2x2 merge of 16x16 patches = one token per 32x32 region):
    raw 32*32*3 pixels -> LayerNorm -> Linear -> LayerNorm -> + pos_x[gx] + pos_y[gy]
    -> LayerNorm -> RMSNorm -> Linear(d_model).
Front end "siglip2" (frozen-encoder condition): the same uint8 [N, 3072] blocks are split into
    their four 16x16 patches, each image's 16-px patch sequence (row-major over the 2*h32 x 2*w32
    grid) goes through a pretrained SigLIP 2 NaFlex ViT (Siglip2VisionModel, variable resolution,
    padded per group of images, position embeddings interpolated to the image's spatial shape),
    then a 2x2 stride-2 ConvPool over the ViT output grid and a 2-layer MLP projector, so the
    decoder receives exactly one token per 32x32 region: the same token count, positions and
    attention mask as the linear front end. Only the front end differs between the two.
Patch embeddings replace the <|image_pad|> token embeddings; <|vision_start|> and
<|vision_end|> stay ordinary text tokens.

Attention mask (per token metadata sample_id, image_id):
    allowed(q, k) = same_sample(q, k) and (k <= q or same_image(q, k))
with image_id = -1 for non-image tokens, so same_image is only true inside one image
span (bidirectional), everything else is causal, nothing crosses a packed-sample
boundary. Two implementations with identical semantics: flex_attention with a
create_block_mask (GPU) and an eager path with an explicit boolean mask (CPU tests
and the diagnostics, which need the attention probabilities).

RoPE positions are the absolute index in the packed 4,096 sequence. RoPE is relative,
and the mask never lets a token see another sample, so this is identical to restarting
positions at each sample. Image tokens just take their 1D sequence position (no 2D RoPE);
the 2D information enters only through the learned factorized position embeddings.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from . import paths

_flex_compiled: Optional[callable] = None


def _get_flex():
    """flex_attention must be torch.compile'd on GPU; uncompiled it falls back to a
    slow materialised implementation. On CPU we use it uncompiled (tests only)."""
    global _flex_compiled
    if _flex_compiled is None:
        if torch.cuda.is_available():
            _flex_compiled = torch.compile(flex_attention, dynamic=False)
        else:
            _flex_compiled = flex_attention
    return _flex_compiled


@dataclass
class VisionConfig:
    """Visual front end. "linear" is the encoder-free patch projection (VisionFrontEnd);
    "siglip2" is the pretrained-encoder front end (SiglipFrontEnd). Everything else in the model is shared."""
    front_end: str = "linear"                 # "linear" | "siglip2"
    siglip_path: str = paths.SIGLIP   # "" -> random init from rand_* (tests)
    freeze_encoder: bool = True               # frozen: no grad, eval mode, forward under no_grad
    encoder_grad_checkpoint: bool = False     # trainable encoder: checkpoint every ViT layer (non-reentrant)
    adapter_width: int = 0                    # ConvPool output width; 0 -> d_model
    encoder_attn_impl: str = "sdpa"           # HF attn_implementation for the ViT ("sdpa" | "eager" | "flash_attention_2")
    encoder_weight_dtype: str = "fp32"        # "bf16" keeps the frozen ViT weights in bf16 (frozen only); matmul dtype comes from autocast
    encoder_chunk_tokens: int = 16384         # images are sorted by patch count and run in groups whose padded size <= this
    # random stand-in for the checkpoint when siglip_path == "" (CPU tests): small Siglip2VisionConfig
    rand_layers: int = 2
    rand_width: int = 64
    rand_heads: int = 2
    rand_ff: int = 128
    rand_num_patches: int = 256               # pretrained position grid is sqrt(num_patches) square (16x16 for so400m)


@dataclass
class ModelConfig:
    d_model: int = 1024
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 8          # == n_heads -> MHA; < n_heads -> GQA
    d_ff: int = 2816             # SwiGLU hidden width (3 matrices of d_model x d_ff)
    vocab_size: int = 151_936    # Qwen3: 151,669 used, padded to 151,936
    tie_embeddings: bool = True
    rope_theta: float = 10_000.0
    seq_len: int = 4096
    norm_eps: float = 1e-6
    init_std: float = 0.02
    # front end
    patch_px: int = 32           # pixels per side of one visual token (32, or 16 with the linear front end; set from data.patch_px)
    vision_width: int = 0        # intermediate width of the front end; 0 -> d_model
    max_grid: int = 64           # max patches per image side (64*32 = 2048 px)
    # attention
    image_bidirectional: bool = True   # False -> fully causal variant
    attn_impl: str = "flex"            # "flex" | "eager"
    # training-time options
    compile_blocks: bool = False
    # visual front end selection (linear | siglip2)
    vision: VisionConfig = field(default_factory=VisionConfig)

    def __post_init__(self):
        if isinstance(self.vision, dict):          # ModelConfig(**json) round trip
            self.vision = VisionConfig(**self.vision)

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @property
    def patch_dim(self) -> int:
        return self.patch_px * self.patch_px * 3


# ----------------------------------------------------------------------------- masks
def build_eager_mask(sample_id: torch.Tensor, image_id: torch.Tensor, bidirectional: bool = True) -> torch.Tensor:
    """Boolean [B, 1, T, T] mask, True = may attend."""
    same_sample = sample_id[:, :, None] == sample_id[:, None, :]
    T = sample_id.shape[1]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=sample_id.device))[None]
    allowed = same_sample & causal
    if bidirectional:
        same_image = (image_id[:, :, None] >= 0) & (image_id[:, :, None] == image_id[:, None, :])
        allowed = allowed | (same_sample & same_image)
    return allowed[:, None]


def build_flex_block_mask(sample_id: torch.Tensor, image_id: torch.Tensor, bidirectional: bool = True, block_size: int = 128):
    B, T = sample_id.shape
    sid, iid = sample_id, image_id

    if bidirectional:
        def mask_mod(b, h, q, k):
            same = sid[b, q] == sid[b, k]
            img = (iid[b, q] >= 0) & (iid[b, q] == iid[b, k])
            return same & ((k <= q) | img)
    else:
        def mask_mod(b, h, q, k):
            return (sid[b, q] == sid[b, k]) & (k <= q)

    return create_block_mask(mask_mod, B, None, T, T, device=sid.device, BLOCK_SIZE=block_size,
                             _compile=sid.is_cuda)


# ----------------------------------------------------------------------------- layers
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def rope_cos_sin(T: int, head_dim: int, theta: float, device, dtype=torch.float32):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(T, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv)                      # [T, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)            # [T, hd]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, hd]; cos/sin: [T, hd]
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    rot = torch.cat([-x2, x1], dim=-1)
    # Rotate in fp32 (cos/sin buffers are fp32) and cast back: under bf16 autocast the elementwise
    # product would otherwise promote q/k to fp32 while v stays bf16, which flex/SDPA reject.
    return (x.float() * cos[None, None] + rot.float() * sin[None, None]).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        hd = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * hd, bias=False)
        self.wo = nn.Linear(cfg.n_heads * hd, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, mask, attn_impl: str, need_probs: bool = False):
        B, T, _ = x.shape
        cfg = self.cfg
        hd, H, KV = cfg.head_dim, cfg.n_heads, cfg.n_kv_heads
        q = self.wq(x).view(B, T, H, hd).transpose(1, 2)
        k = self.wk(x).view(B, T, KV, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, KV, hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        probs = None
        if attn_impl == "flex":
            out = _get_flex()(q, k, v, block_mask=mask, enable_gqa=(KV != H))
        else:
            if KV != H:
                rep = H // KV
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            if need_probs:
                scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(hd)
                scores = scores.masked_fill(~mask, float("-inf"))
                probs = scores.softmax(-1)
                out = (probs.to(v.dtype) @ v)
            else:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, T, H * hd)
        return self.wo(out), probs


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, mask, attn_impl, need_probs=False):
        a, probs = self.attn(self.norm1(x), cos, sin, mask, attn_impl, need_probs)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, probs


class VisionFrontEnd(nn.Module):
    """Patch projection of Chen et al., App. A.1. Input: [N, 3072] normalised raw pixels of one 32x32
    region; grid coords [N, 2] = (gx, gy) of the token in its image."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        vw = cfg.vision_width or cfg.d_model
        self.ln_in = nn.LayerNorm(cfg.patch_dim)
        self.proj = nn.Linear(cfg.patch_dim, vw)
        self.ln_mid = nn.LayerNorm(vw)
        self.pos_x = nn.Embedding(cfg.max_grid, vw)
        self.pos_y = nn.Embedding(cfg.max_grid, vw)
        self.ln_pos = nn.LayerNorm(vw)
        self.rms = RMSNorm(vw, cfg.norm_eps)
        self.out = nn.Linear(vw, cfg.d_model)

    def forward(self, pixels: torch.Tensor, grid_xy: torch.Tensor) -> torch.Tensor:
        if pixels.dtype == torch.uint8:          # data.py ships raw bytes; normalise here
            pixels = pixels.float() / 127.5 - 1.0   # mean 0.5, std 0.5 on [0, 1]
        pixels = pixels.to(self.proj.weight.dtype)
        h = self.ln_mid(self.proj(self.ln_in(pixels)))
        h = h + self.pos_x(grid_xy[:, 0]) + self.pos_y(grid_xy[:, 1])
        h = self.rms(self.ln_pos(h))
        return self.out(h)


# ----------------------------------------------------------------------------- SigLIP 2 front end
SIGLIP_PATCH = 16


def siglip_vision_config(vc: VisionConfig):
    """Siglip2VisionConfig of the encoder (from the checkpoint's config.json, or the rand_* stand-in).
    The pooling head is disabled: it is not part of the front end and would be an unused parameter."""
    from transformers import Siglip2VisionConfig
    if vc.siglip_path:
        cfg = Siglip2VisionConfig.from_pretrained(vc.siglip_path)
    else:
        cfg = Siglip2VisionConfig(hidden_size=vc.rand_width, intermediate_size=vc.rand_ff,
                                  num_hidden_layers=vc.rand_layers, num_attention_heads=vc.rand_heads,
                                  num_patches=vc.rand_num_patches, patch_size=SIGLIP_PATCH)
    assert cfg.patch_size == SIGLIP_PATCH and cfg.num_channels == 3, cfg
    cfg.vision_use_head = False
    return cfg


def siglip_param_count(cfg) -> int:
    """Closed-form parameter count of Siglip2VisionModel without the pooling head."""
    w, ff, L = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
    pd = cfg.num_channels * cfg.patch_size * cfg.patch_size
    layer = 2 * w + 4 * (w * w + w) + 2 * w + (w * ff + ff) + (ff * w + w)
    return (pd * w + w) + cfg.num_patches * w + L * layer + 2 * w


def siglip_matmul_params(cfg) -> int:
    """Parameters that take part in a per-patch matmul (everything but the position table)."""
    return siglip_param_count(cfg) - cfg.num_patches * cfg.hidden_size


def load_siglip(vc: VisionConfig, cfg):
    from transformers import Siglip2VisionModel
    from transformers.utils import logging as hf_logging
    if not vc.siglip_path:
        return Siglip2VisionModel._from_config(cfg, attn_implementation=vc.encoder_attn_impl)
    lvl = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()        # the checkpoint also holds the text tower and the head: skip the UNEXPECTED report
    hf_logging.disable_progress_bar()
    try:
        return Siglip2VisionModel.from_pretrained(vc.siglip_path, config=cfg, dtype=torch.float32,
                                                 attn_implementation=vc.encoder_attn_impl)
    finally:
        hf_logging.set_verbosity(lvl)


def image_layout(grid_xy: torch.Tensor):
    """Per-image layout of the visual-token rows. Rows are grouped per image, contiguous and
    row-major (data.collate), so a row with (gx, gy) == (0, 0) starts a new image.
    Returns seg [N] (image index of each row), counts [n_img], gw [n_img], gh [n_img]."""
    gx, gy = grid_xy[:, 0], grid_xy[:, 1]
    starts = (gx == 0) & (gy == 0)
    seg = torch.cumsum(starts.to(torch.int64), 0) - 1
    n_img = int(seg[-1].item()) + 1 if grid_xy.shape[0] else 0
    counts = torch.bincount(seg, minlength=n_img)
    gw = torch.zeros(n_img, dtype=torch.int64, device=grid_xy.device).scatter_reduce(0, seg, gx, reduce="amax") + 1
    gh = counts // gw
    assert bool((gh * gw == counts).all()), "visual rows are not complete row-major image grids"
    return seg, counts, gw, gh


class SiglipFrontEnd(nn.Module):
    """Pretrained SigLIP 2 NaFlex ViT -> 2x2 ConvPool -> 2-layer MLP projector.

    Input is identical to VisionFrontEnd: uint8 (or normalised float) pixels [N, 3072], one row per
    32x32 block in (py, px, c) order, and grid_xy [N, 2] = (gx, gy); rows of one image are contiguous
    and row-major. Each block is split into its four 16x16 patches (sy, sx), flattened in
    (patch_h, patch_w, channel) order exactly like transformers' Siglip2ImageProcessor
    .convert_image_to_patches (verified in tests/test_units.py), and the image's P = 4*h32*w32 patches
    are laid out row-major over the (2*h32, 2*w32) grid. Images are sorted by P and run through the
    ViT in groups padded to the group's max P (pixel_attention_mask masks the padded keys, so the
    result of an image is independent of its group); spatial_shapes = (2*h32, 2*w32) drives the
    NaFlex position-embedding interpolation. The last_hidden_state rows of the four patches of each
    block are gathered back into [N, 4, width]: a 2x2 stride-2 conv over the (2*h32, 2*w32) feature
    grid is exactly a linear map of this [N, 4*width] tensor (kernel positions (ky, kx) = (sy, sx)),
    so ConvPool is an nn.Linear(4*width, adapter_width). Then LayerNorm, Linear(adapter_width,
    d_model), GELU, Linear(d_model, d_model). Output [N, d_model] in the input row order."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        vc = cfg.vision
        assert cfg.patch_px == 32, "the SigLIP 2 front end needs 32-px blocks (4 x 16-px patches)"
        self.vcfg = siglip_vision_config(vc)
        self.vit = load_siglip(vc, self.vcfg)
        w = self.vcfg.hidden_size
        aw = vc.adapter_width or cfg.d_model
        self.conv_pool = nn.Linear(4 * w, aw)          # == Conv2d(w, aw, kernel 2, stride 2) on the ViT grid
        self.ln = nn.LayerNorm(aw)
        self.proj_in = nn.Linear(aw, cfg.d_model)
        self.proj_out = nn.Linear(cfg.d_model, cfg.d_model)
        self.frozen = bool(vc.freeze_encoder)
        self.chunk_tokens = int(vc.encoder_chunk_tokens)
        if self.frozen:
            self.vit.requires_grad_(False)
            self.vit.eval()
            if vc.encoder_weight_dtype == "bf16":
                self.vit.to(torch.bfloat16)
            else:
                assert vc.encoder_weight_dtype == "fp32", vc.encoder_weight_dtype
        else:
            assert vc.encoder_weight_dtype == "fp32", "a trainable encoder keeps fp32 master weights"
            if vc.encoder_grad_checkpoint:
                self.vit.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.vit.eval()
        return self

    # ---- patch assembly
    @staticmethod
    def normalise(pixels: torch.Tensor) -> torch.Tensor:
        if pixels.dtype == torch.uint8:
            pixels = pixels.float() / 127.5 - 1.0        # (x/255 - 0.5) / 0.5, SigLIP 2 image_mean/std = 0.5
        return pixels.float()

    @staticmethod
    def split_blocks(pixels: torch.Tensor) -> torch.Tensor:
        """[N, 3072] blocks (py, px, c) -> [N, 4, 768] 16-px patches, sub-patch order (sy, sx) =
        (0,0), (0,1), (1,0), (1,1), each flattened as (ph, pw, c)."""
        N = pixels.shape[0]
        return pixels.view(N, 2, SIGLIP_PATCH, 2, SIGLIP_PATCH, 3).permute(0, 1, 3, 2, 4, 5).reshape(N, 4, 3 * SIGLIP_PATCH ** 2)

    def _groups(self, P: torch.Tensor) -> list[list[int]]:
        """Images sorted by patch count (descending), greedily grouped so that
        len(group) * max_P(group) <= chunk_tokens (a single image always forms a valid group)."""
        order = torch.argsort(P, descending=True).tolist()
        P_list = P.tolist()
        groups, cur, cur_max = [], [], 0
        for i in order:
            if cur and (len(cur) + 1) * max(cur_max, P_list[i]) > self.chunk_tokens:
                groups.append(cur)
                cur, cur_max = [], 0
            cur.append(i)
            cur_max = max(cur_max, P_list[i])
        if cur:
            groups.append(cur)
        return groups

    def vit_inputs(self, pixels: torch.Tensor, grid_xy: torch.Tensor):
        """Everything the ViT needs, per group. Yields (rows, dest, pixel_values, mask, shapes):
        rows [n] = indices of the 32-px rows in this group, dest [n, 4] = flat index of each row's
        four patches into pixel_values.view(-1, 768) (and into last_hidden_state.view(-1, w)),
        pixel_values [B, Pmax, 768] normalised, mask [B, Pmax] bool, shapes [B, 2] = (2*h32, 2*w32)."""
        dev = grid_xy.device
        seg, counts, gw, gh = image_layout(grid_xy)
        P, ph, pw = 4 * counts, 2 * gh, 2 * gw
        gx, gy = grid_xy[:, 0], grid_xy[:, 1]
        sy = torch.tensor([0, 0, 1, 1], device=dev)
        sx = torch.tensor([0, 1, 0, 1], device=dev)
        local = (2 * gy[:, None] + sy[None]) * pw[seg][:, None] + (2 * gx[:, None] + sx[None])   # [N, 4]
        sub = self.split_blocks(self.normalise(pixels))
        for group in self._groups(P):
            g = torch.tensor(group, device=dev)
            B, Pmax = len(group), int(P[g].max().item())
            slot = torch.full((counts.shape[0],), -1, dtype=torch.int64, device=dev)
            slot[g] = torch.arange(B, device=dev)
            rows = (slot[seg] >= 0).nonzero(as_tuple=True)[0]
            dest = slot[seg[rows]][:, None] * Pmax + local[rows]
            pv = sub.new_zeros(B * Pmax, sub.shape[-1])
            pv[dest.reshape(-1)] = sub[rows].reshape(-1, sub.shape[-1])
            mask = torch.arange(Pmax, device=dev)[None] < P[g][:, None]
            shapes = torch.stack([ph[g], pw[g]], 1)
            yield rows, dest, pv.view(B, Pmax, -1), mask, shapes

    def patch_sequences(self, pixels: torch.Tensor, grid_xy: torch.Tensor) -> list[torch.Tensor]:
        """Per image (in order of appearance) the [P, 768] patch sequence exactly as fed to the ViT
        (tests compare this with Siglip2ImageProcessor)."""
        seg, counts, gw, gh = image_layout(grid_xy)
        out = [None] * counts.shape[0]
        for rows, dest, pv, mask, shapes in self.vit_inputs(pixels, grid_xy):
            imgs = seg[rows].unique().tolist()
            Pmax = pv.shape[1]
            for b, i in enumerate(imgs):
                slot = (dest[seg[rows] == i][0, 0] // Pmax).item()
                out[i] = pv[slot, : int(4 * counts[i])]
        return out

    def _run_vit(self, pv, mask, shapes):
        ctx = torch.no_grad() if self.frozen else nullcontext()
        with ctx:
            return self.vit(pixel_values=pv, pixel_attention_mask=mask, spatial_shapes=shapes).last_hidden_state

    def forward(self, pixels: torch.Tensor, grid_xy: torch.Tensor) -> torch.Tensor:
        N, w = pixels.shape[0], self.vcfg.hidden_size
        feats = torch.zeros(N, 4, w, dtype=torch.float32, device=pixels.device)
        for rows, dest, pv, mask, shapes in self.vit_inputs(pixels, grid_xy):
            last = self._run_vit(pv, mask, shapes)                       # [B, Pmax, w]
            feats = feats.index_put((rows,), last.reshape(-1, w)[dest].to(feats.dtype))
        h = self.conv_pool(feats.reshape(N, 4 * w).to(self.conv_pool.weight.dtype))
        h = self.ln(h)
        return self.proj_out(F.gelu(self.proj_in(h)))

    def adapter_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("vit.")]


class EncFreeDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.is_siglip = cfg.vision.front_end == "siglip2"
        assert cfg.vision.front_end in ("linear", "siglip2"), cfg.vision.front_end
        self.front_end = SiglipFrontEnd(cfg) if self.is_siglip else VisionFrontEnd(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        cos, sin = rope_cos_sin(cfg.seq_len, cfg.head_dim, cfg.rope_theta, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        # the pretrained (or HF-initialised) ViT keeps its own weights; the adapter gets the decoder init
        self._skip_init = {id(m) for m in self.front_end.vit.modules()} if self.is_siglip else set()
        self.apply(self._init)
        # scale residual-branch output projections by 1/sqrt(2L)
        for blk in self.blocks:
            for p in (blk.attn.wo.weight, blk.mlp.w_down.weight):
                nn.init.normal_(p, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))

    def _init(self, m):
        if id(m) in self._skip_init:
            return
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    # ---- parameter accounting
    def param_counts(self) -> dict:
        """blocks, front_end (trainable front-end parameters: the linear projection, or the ConvPool +
        projector adapter), non_embedding = blocks + front_end, tok_emb, lm_head, total; for the
        siglip2 front end also encoder (ViT parameters, frozen or not), included in total."""
        n_blocks = sum(p.numel() for p in self.blocks.parameters()) + self.norm_f.weight.numel()
        n_emb = self.tok_emb.weight.numel()
        n_head = 0 if self.lm_head is None else self.lm_head.weight.numel()
        if self.is_siglip:
            n_fe = sum(p.numel() for p in self.front_end.adapter_parameters())
            n_enc = sum(p.numel() for p in self.front_end.vit.parameters())
            return {"blocks": n_blocks, "front_end": n_fe, "non_embedding": n_blocks + n_fe,
                    "tok_emb": n_emb, "lm_head": n_head, "encoder": n_enc,
                    "total": n_blocks + n_fe + n_emb + n_head + n_enc}
        n_fe = sum(p.numel() for p in self.front_end.parameters())
        return {"blocks": n_blocks, "front_end": n_fe, "non_embedding": n_blocks + n_fe,
                "tok_emb": n_emb, "lm_head": n_head, "total": n_blocks + n_fe + n_emb + n_head}

    def encoder_flops_mult(self) -> int:
        """Training FLOPs multiplier of the encoder forward: 1 when frozen (forward only, nothing
        before the ViT output needs a gradient), 3 when trainable (forward + backward)."""
        return 1 if (self.is_siglip and self.front_end.frozen) else 3

    def flops_per_token(self, mean_keys: float, frac_visual: float = 0.0, mean_img_patches: float = 0.0) -> dict:
        """Chen et al., Eq. 6: M = M_base + M_attn * l. Training FLOPs (fwd+bwd = 3x fwd, 2 FLOPs
        per MAC): M_base = 6 * (block params + d*V output matmul) [+ 6 * front-end params
        on visual tokens]; M_attn * l = 12 * L * d * l where l = mean number of keys a
        query attends to (exact under the document/image mask), QK^T and PV each
        2*d*l forward per layer.
        siglip2 front end, full accounting (Chen et al., App. C): m_encoder = frac_visual * mult * 4 * (2 * N_enc
        + 4 * L_enc * w_enc * P) per token, where each visual token is 4 ViT patches, P =
        mean_img_patches is the patch count of the token's image (attention over the image) and
        mult = 1 (frozen) or 3 (trainable). "decoder" = m_base + m_attn is the decoder-only number."""
        c = self.cfg
        pc = self.param_counts()
        m_base = 6 * (pc["blocks"] + c.d_model * c.vocab_size)
        m_fe = 6 * pc["front_end"] * frac_visual
        m_attn = 12 * c.n_layers * c.d_model * mean_keys
        m_enc = 0.0
        if self.is_siglip:
            vc = self.front_end.vcfg
            m_enc = frac_visual * self.encoder_flops_mult() * 4 * (
                2 * siglip_matmul_params(vc) + 4 * vc.num_hidden_layers * vc.hidden_size * mean_img_patches)
        return {"m_base": m_base, "m_front_end": m_fe, "m_attn": m_attn, "m_encoder": m_enc,
                "decoder": m_base + m_attn, "total": m_base + m_fe + m_attn + m_enc}

    def front_end_flops(self, grid_xy: torch.Tensor) -> dict:
        """Exact training FLOPs of the front end for one batch of visual rows (grid_xy [N, 2]):
        adapter = 6 * front_end params * N (the whole linear front end, or the ConvPool +
        projector), encoder = mult * sum_images (2 * N_enc * P_i + 4 * L_enc * w_enc * P_i^2) with
        P_i = 4 * (32-px blocks of image i); 0 for the linear front end."""
        N = grid_xy.shape[0]
        out = {"adapter": 6 * self.param_counts()["front_end"] * N, "encoder": 0.0}
        if self.is_siglip and N > 0:
            vc = self.front_end.vcfg
            P = (4 * image_layout(grid_xy)[1]).double()
            out["encoder"] = self.encoder_flops_mult() * (
                2.0 * siglip_matmul_params(vc) * float(P.sum())
                + 4.0 * vc.num_hidden_layers * vc.hidden_size * float((P * P).sum()))
        return out

    # ---- checkpoint helpers: a frozen encoder is reloaded from siglip_path, not stored
    def _frozen_prefix(self) -> Optional[str]:
        return "front_end.vit." if (self.is_siglip and self.front_end.frozen) else None

    def checkpoint_state_dict(self) -> dict:
        sd = self.state_dict()
        pre = self._frozen_prefix()
        if pre is None:
            return sd
        return {k: v for k, v in sd.items() if not k.startswith(pre)}

    def load_checkpoint_state_dict(self, sd: dict):
        pre = self._frozen_prefix()
        if pre is None or any(k.startswith(pre) for k in sd):
            return self.load_state_dict(sd)
        res = self.load_state_dict(sd, strict=False)
        bad = [k for k in res.missing_keys if not k.startswith(pre)]
        assert not bad and not res.unexpected_keys, (bad, res.unexpected_keys)
        return res

    # ---- forward
    def embed(self, input_ids, pixels, grid_xy, img_index):
        """img_index: [N, 2] (batch, position) of each visual token; pixels [N, 3072]."""
        x = self.tok_emb(input_ids)
        if pixels is not None and pixels.shape[0] > 0:
            v = self.front_end(pixels, grid_xy)
            x = x.index_put((img_index[:, 0], img_index[:, 1]), v.to(x.dtype))
        return x

    def make_mask(self, sample_id, image_id, attn_impl: str):
        bi = self.cfg.image_bidirectional
        if attn_impl == "flex":
            return build_flex_block_mask(sample_id, image_id, bi)
        return build_eager_mask(sample_id, image_id, bi)

    def forward(self, input_ids, sample_id, image_id, pixels=None, grid_xy=None, img_index=None,
                attn_impl: Optional[str] = None, on_layer=None):
        """Returns final hidden states [B, T, d] (after the final norm, before lm_head).
        on_layer(l, x, probs, mask): diagnostics callback, called with l=0 for the block
        input (probs None) and after every block l=1..L with that block's attention probs
        [B, H, T, T] (eager path only); nothing is retained between calls."""
        impl = attn_impl or self.cfg.attn_impl
        T = input_ids.shape[1]
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        mask = self.make_mask(sample_id, image_id, impl)
        x = self.embed(input_ids, pixels, grid_xy, img_index)
        collect = on_layer is not None
        if collect:
            on_layer(0, x, None, mask)
        for l, blk in enumerate(self.blocks):
            x, probs = blk(x, cos, sin, mask, impl, need_probs=collect)
            if collect:
                on_layer(l + 1, x, probs, mask)
                del probs
        return self.norm_f(x)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        w = self.tok_emb.weight if self.lm_head is None else self.lm_head.weight
        return F.linear(h, w)

    def token_losses(self, h: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor, chunk: int = 8192):
        """Per-token CE (float32) only at loss positions, chunked over the selected
        positions so the [n, V] logits never exist for the whole batch.
        Returns (losses [n], flat_index [n]) with flat_index into B*T."""
        sel = loss_mask.reshape(-1).nonzero(as_tuple=True)[0]
        hs = h.reshape(-1, h.shape[-1])[sel]
        ts = targets.reshape(-1)[sel]
        out = []
        for i in range(0, hs.shape[0], chunk):
            lg = self.logits(hs[i:i + chunk]).float()
            out.append(F.cross_entropy(lg, ts[i:i + chunk], reduction="none"))
        losses = torch.cat(out) if out else h.new_zeros(0, dtype=torch.float32)
        return losses, sel


def count_params(cfg: ModelConfig) -> dict:
    """Closed-form parameter counts (no instantiation)."""
    d, L, hd = cfg.d_model, cfg.n_layers, cfg.head_dim
    attn = d * cfg.n_heads * hd * 2 + d * cfg.n_kv_heads * hd * 2
    mlp = 3 * d * cfg.d_ff
    norms = 2 * d
    blocks = L * (attn + mlp + norms) + d
    emb = cfg.vocab_size * d
    head = 0 if cfg.tie_embeddings else cfg.vocab_size * d
    if cfg.vision.front_end == "siglip2":
        vcfg = siglip_vision_config(cfg.vision)
        w = vcfg.hidden_size
        aw = cfg.vision.adapter_width or d
        fe = (4 * w * aw + aw) + 2 * aw + (aw * d + d) + (d * d + d)
        enc = siglip_param_count(vcfg)
        return {"blocks": blocks, "front_end": fe, "non_embedding": blocks + fe, "tok_emb": emb,
                "lm_head": head, "encoder": enc, "total": blocks + fe + emb + head + enc}
    vw = cfg.vision_width or d
    pd = cfg.patch_dim
    fe = (2 * pd) + (pd * vw + vw) + (2 * vw) + 2 * cfg.max_grid * vw + (2 * vw) + vw + (vw * d + d)
    return {"blocks": blocks, "front_end": fe, "non_embedding": blocks + fe, "tok_emb": emb,
            "lm_head": head, "total": blocks + fe + emb + head}
