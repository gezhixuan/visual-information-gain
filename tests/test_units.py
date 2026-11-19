"""Unit tests (CPU or GPU): attention mask (flex block mask vs eager boolean mask vs an
independent reference), loss mask excludes <|image_pad|>, attended_keys, model forward;
encoder front end: 16-px patch order vs Siglip2ImageProcessor on real shard JPEGs, tiny random
SigLIP front end (batched == per-image, grads, frozen/trainable, NaFlex 512/1024 px, grouping,
checkpoint filtering), patch_px=16 packing, mm mixture weights/caps."""
import glob
import io
import itertools
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

from vig import data as D, paths
from vig.model import (EncFreeDecoder, ModelConfig, VisionConfig, SiglipFrontEnd, build_eager_mask,
                   build_flex_block_mask, count_params, image_layout, _get_flex)

CC3M = os.path.join(paths.SHARDS, "mm", "cc3m-wds")

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def packed_example(T=128):
    """Three samples, two images: text doc | mm (2x3 image) | mm (2x2 image) | pad tail."""
    s0 = D.Sample(ids=np.r_[np.arange(10, 30), D.EOS].astype(np.int64), topic=6, kind=0)
    def mm(gw, gh, n_text, topic):
        n = gw * gh
        ids = np.r_[D.VISION_START, np.full(n, D.IMAGE_PAD), D.VISION_END, np.arange(100, 100 + n_text), D.EOS].astype(np.int64)
        return D.Sample(ids=ids, topic=topic, kind=1, pixels=np.random.randint(0, 255, (n, D.PATCH_DIM), dtype=np.uint8),
                        grid=D.grid_coords(gw * 32, gh * 32), img_offset=1)
    s1, s2 = mm(3, 2, 25, 0), mm(2, 2, 12, 3)
    return D.pack([s0, s1, s2], T, kind=1), [s0, s1, s2]


def reference_mask(seq: D.PackedSeq, bidirectional=True):
    T = len(seq.ids)
    m = np.zeros((T, T), dtype=bool)
    for q in range(T):
        for k in range(T):
            same = seq.sample_id[q] == seq.sample_id[k]
            img = bidirectional and seq.image_id[q] >= 0 and seq.image_id[q] == seq.image_id[k]
            m[q, k] = same and (k <= q or img)
    return torch.from_numpy(m)


def test_masks():
    seq, _ = packed_example()
    b = D.collate([seq])
    sid, iid = b["sample_id"].to(dev), b["image_id"].to(dev)
    for bi in (True, False):
        ref = reference_mask(seq, bi).to(dev)
        eager = build_eager_mask(sid, iid, bi)[0, 0]
        assert torch.equal(eager, ref), f"eager mask != reference (bidirectional={bi})"
        # flex: compare attention outputs under the block mask with eager masked softmax
        torch.manual_seed(0)
        H, hd = 2, 16
        q, k, v = (torch.randn(1, H, 128, hd, device=dev) for _ in range(3))
        bm = build_flex_block_mask(sid, iid, bi, block_size=128 if dev.type == "cuda" else 16)  # Triton flex kernel needs >=128 on GPU (smoke 315662 failed on 16)
        out_flex = _get_flex()(q, k, v, block_mask=bm)
        scores = (q @ k.transpose(-1, -2)) / hd ** 0.5
        scores = scores.masked_fill(~ref[None, None], float("-inf"))
        out_eager = scores.softmax(-1) @ v
        err = (out_flex - out_eager).abs().max().item()
        assert err < 1e-4, f"flex vs eager output mismatch {err} (bidirectional={bi})"
        # sanity on structure: image tokens see each other both ways, text does not look ahead
        img_pos = (iid[0] == 0).nonzero().flatten()
        assert bool(ref[img_pos[0], img_pos[-1]]) == bi
        txt = (seq.sample_id == 2).nonzero()[0]
        assert not ref[txt[-5], txt[-1]]
        assert not ref[txt[0], 0]   # no cross-sample attention
    print("mask tests OK (eager == reference, flex == eager, bidirectional and causal)")


def test_loss_mask_and_keys():
    seq, samples = packed_example()
    b = D.collate([seq])
    lm, tgt, ids = b["loss_mask"][0], b["targets"][0], b["input_ids"][0]
    assert not (tgt[lm] == D.IMAGE_PAD).any(), "image pad used as a target"
    assert (tgt[~lm] == -100).all()
    # positions whose target is an image pad are off; the <|vision_end|> target (after the image) is on
    pad_pos = (ids == D.IMAGE_PAD).nonzero().flatten()
    assert not lm[pad_pos - 1].any()
    vend = (ids == D.VISION_END).nonzero().flatten()
    assert lm[vend - 1].all(), "predicting <|vision_end|> from the last image token should count"
    assert lm[vend].all(), "first text token after the image should count"
    # EOS is predicted, the token after EOS (next sample / pad) is not
    eos = (ids == D.EOS).nonzero().flatten()
    eos = eos[eos < sum(len(s) for s in samples)]
    assert lm[eos - 1].all() and not lm[eos].any()
    expected = sum(len(s) - 1 - (0 if s.pixels is None else s.pixels.shape[0]) for s in samples)
    assert int(lm.sum()) == expected, (int(lm.sum()), expected)
    nk = D.attended_keys(b["sample_id"], b["image_id"])
    em = build_eager_mask(b["sample_id"], b["image_id"])[:, 0]
    assert torch.equal(nk, em.sum(-1)), "attended_keys != mask row sums"
    assert torch.equal(D.attended_keys(b["sample_id"], b["image_id"], False),
                       build_eager_mask(b["sample_id"], b["image_id"], False)[:, 0].sum(-1))
    print(f"loss-mask test OK ({expected} loss positions, 10 image tokens excluded); attended_keys OK")


def test_model_forward():
    seq, _ = packed_example()
    b = D.collate([seq, seq])
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=2, n_kv_heads=1, d_ff=128, seq_len=128, max_grid=8, attn_impl="eager")
    m = EncFreeDecoder(cfg).to(dev)
    bb = {k: v.to(dev) for k, v in b.items() if torch.is_tensor(v)}
    h = m(bb["input_ids"], bb["sample_id"], bb["image_id"], bb["pixels"], bb["grid_xy"], bb["img_index"])
    assert h.shape == (2, 128, 64)
    losses, sel = m.token_losses(h, bb["targets"], bb["loss_mask"])
    assert losses.numel() == int(bb["loss_mask"].sum())
    losses.mean().backward()
    assert m.front_end.proj.weight.grad is not None and m.front_end.proj.weight.grad.abs().sum() > 0
    # flex path (GQA) gives the same hidden states as eager
    m.zero_grad()
    with torch.no_grad():
        h2 = m(bb["input_ids"], bb["sample_id"], bb["image_id"], bb["pixels"], bb["grid_xy"], bb["img_index"], attn_impl="flex")
    err = (h - h2).abs().max().item()
    assert err < 1e-3, f"flex vs eager hidden mismatch {err}"
    assert m.param_counts()["non_embedding"] > 0
    print(f"model forward OK (GQA, flex vs eager hidden max err {err:.2e}, grads reach the front end)")


# ----------------------------------------------------------------------------- encoder front end
def tiny_siglip_cfg(frozen=False, grad_ckpt=False, T=128, **kw):
    vision = VisionConfig(front_end="siglip2", siglip_path="", freeze_encoder=frozen, encoder_grad_checkpoint=grad_ckpt,
                          rand_layers=2, rand_width=64, rand_heads=2, rand_ff=128, rand_num_patches=256, **kw)
    return ModelConfig(d_model=64, n_layers=2, n_heads=2, n_kv_heads=2, d_ff=128, seq_len=T, max_grid=64,
                       attn_impl="eager", vision=vision)


def test_siglip_patch_order():
    """The 16-px patch sequences SiglipFrontEnd feeds to the ViT equal Siglip2ImageProcessor's
    pixel_values (do_resize=False) on the same JPEGs: same (patch_h, patch_w, channel) flattening,
    same row-major order, same normalisation; spatial_shapes = (h/16, w/16)."""
    tars = sorted(glob.glob(os.path.join(CC3M, "train-*.tar")))
    if not tars:
        print(f"patch-order test SKIPPED (no shards under {CC3M})")
        return
    from transformers import Siglip2ImageProcessor
    samples = list(itertools.islice(D.iter_shard(tars[0]), 3))
    sizes = {(m["w"], m["h"]) for _, _, m in samples}
    assert len(sizes) >= 2, f"want images of different sizes, got {sizes}"
    pix, grid = [], []
    for _, jpg, meta in samples:
        s = D.mm_sample(jpg, meta)
        pix.append(s.pixels)
        grid.append(s.grid)
    pixels, grid_xy = torch.from_numpy(np.concatenate(pix)), torch.from_numpy(np.concatenate(grid))
    torch.manual_seed(0)
    fe = SiglipFrontEnd(tiny_siglip_cfg())          # only the (parameter-free) patch assembly is used
    seqs = fe.patch_sequences(pixels, grid_xy)
    proc = Siglip2ImageProcessor()
    worst = 0.0
    for (_, jpg, meta), seq in zip(samples, seqs):
        img = Image.open(io.BytesIO(jpg)).convert("RGB")
        P = (meta["w"] // 16) * (meta["h"] // 16)
        out = proc(images=[img], do_resize=False, max_num_patches=P, return_tensors="pt")
        assert tuple(out["spatial_shapes"][0].tolist()) == (meta["h"] // 16, meta["w"] // 16)
        assert int(out["pixel_attention_mask"][0].sum()) == P and seq.shape == (P, 768), (seq.shape, P)
        worst = max(worst, (seq - out["pixel_values"][0]).abs().max().item())
    assert worst < 1e-5, worst
    # masks and spatial shapes handed to the ViT
    seg, counts, gw, gh = image_layout(grid_xy)
    for rows, dest, pv, mask, shapes in fe.vit_inputs(pixels, grid_xy):
        imgs = seg[rows].unique()
        assert torch.equal(mask.sum(1).sort().values, (4 * counts[imgs]).sort().values)
        assert torch.equal(shapes.sort(0).values, torch.stack([2 * gh[imgs], 2 * gw[imgs]], 1).sort(0).values)
        assert int(dest.max()) < pv.shape[0] * pv.shape[1] and dest.unique().numel() == dest.numel()
    print(f"patch-order test OK: {len(samples)} real JPEGs {sorted(sizes)} == Siglip2ImageProcessor (max abs err {worst:.1e})")


def test_siglip_forward():
    seq, _ = packed_example()
    b = D.collate([seq])
    bb = {k: v.to(dev) for k, v in b.items() if torch.is_tensor(v)}
    seg, counts, gw, gh = image_layout(bb["grid_xy"])
    assert counts.tolist() == [6, 4] and gw.tolist() == [3, 2] and gh.tolist() == [2, 2]
    for frozen, ckpt in ((False, False), (False, True), (True, False)):
        torch.manual_seed(0)
        cfg = tiny_siglip_cfg(frozen, ckpt)
        m = EncFreeDecoder(cfg).to(dev)
        assert m.param_counts() == count_params(cfg), (m.param_counts(), count_params(cfg))
        m.train()
        assert m.front_end.vit.training == (not frozen)
        h = m(bb["input_ids"], bb["sample_id"], bb["image_id"], bb["pixels"], bb["grid_xy"], bb["img_index"])
        assert h.shape == (1, 128, 64) and torch.isfinite(h).all()
        # batched front end == each image alone (row order kept, padding does not leak)
        with torch.no_grad():
            v_all = m.front_end(bb["pixels"], bb["grid_xy"])
            off = 0
            for i in range(len(counts)):
                n = int(counts[i])
                v_i = m.front_end(bb["pixels"][off:off + n], bb["grid_xy"][off:off + n])
                err = (v_all[off:off + n] - v_i).abs().max().item()
                assert err < 1e-4, (i, err)
                off += n
            # the visual rows landed on the <|image_pad|> positions
            x0 = m.embed(bb["input_ids"], bb["pixels"], bb["grid_xy"], bb["img_index"])
            assert torch.allclose(x0[bb["img_index"][:, 0], bb["img_index"][:, 1]], v_all, atol=1e-5)
        losses, sel = m.token_losses(h, bb["targets"], bb["loss_mask"])
        losses.mean().backward()
        for p in m.front_end.adapter_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
        assert m.front_end.conv_pool.weight.grad.abs().sum() > 0 and m.front_end.proj_out.weight.grad.abs().sum() > 0
        q = m.front_end.vit.encoder.layers[0].self_attn.q_proj.weight
        pe = m.front_end.vit.embeddings.position_embedding.weight
        if frozen:
            assert not q.requires_grad and q.grad is None and pe.grad is None
            assert all(not p.requires_grad for p in m.front_end.vit.parameters())
        else:
            assert q.grad is not None and q.grad.abs().sum() > 0 and pe.grad.abs().sum() > 0
            assert m.front_end.vit.encoder.layers[0].gradient_checkpointing == ckpt
        # checkpoint filtering: a frozen ViT is not stored and reloads from the config
        sd = m.checkpoint_state_dict()
        has_vit = any(k.startswith("front_end.vit.") for k in sd)
        assert has_vit == (not frozen)
        m.load_checkpoint_state_dict(sd)
        # FLOPs accounting: encoder term present, 1x when frozen and 3x when trainable
        fe = m.front_end_flops(bb["grid_xy"])
        assert fe["adapter"] == 6 * m.param_counts()["front_end"] * 10 and fe["encoder"] > 0
        if frozen:
            assert m.encoder_flops_mult() == 1
            enc_frozen = fe["encoder"]
        else:
            assert m.encoder_flops_mult() == 3
            enc_train = fe["encoder"]
    assert abs(enc_train / enc_frozen - 3.0) < 1e-9
    vc = m.front_end.vcfg
    nmm = count_params(cfg)["encoder"] - vc.num_patches * vc.hidden_size
    expect = sum(2 * nmm * P + 4 * vc.num_hidden_layers * vc.hidden_size * P * P for P in (24, 16))
    assert abs(enc_frozen - expect) < 1e-6 * expect, (enc_frozen, expect)
    # NaFlex interpolation of the 16x16 pretrained grid to 512 px (32x32 patches) and 1024 px (64x64) images,
    # and the size grouping: both images in one padded group vs one group each give the same rows
    torch.manual_seed(0)
    m = EncFreeDecoder(tiny_siglip_cfg(True, T=4096)).to(dev).eval()
    pix, grd, shapes_seen = [], [], []
    for px in (512, 1024):
        g = px // 32
        pix.append(torch.randint(0, 256, (g * g, 3072), dtype=torch.uint8))
        grd.append(torch.from_numpy(D.grid_coords(px, px)))
    pixels, grid_xy = torch.cat(pix).to(dev), torch.cat(grd).to(dev)
    with torch.no_grad():
        m.front_end.chunk_tokens = 16384
        groups = [(pv.shape[0], pv.shape[1], shapes.tolist()) for _, _, pv, _, shapes in m.front_end.vit_inputs(pixels, grid_xy)]
        assert groups == [(2, 4096, [[64, 64], [32, 32]])], groups
        v1 = m.front_end(pixels, grid_xy)
        m.front_end.chunk_tokens = 4096
        groups = [(pv.shape[0], pv.shape[1]) for _, _, pv, _, _ in m.front_end.vit_inputs(pixels, grid_xy)]
        assert groups == [(1, 4096), (1, 1024)], groups
        v2 = m.front_end(pixels, grid_xy)
    assert v1.shape == (256 + 1024, 64) and torch.isfinite(v1).all()
    err = (v1 - v2).abs().max().item()
    assert err < 1e-4, err
    print(f"siglip forward OK: tiny ViT, batched == per-image, grads reach adapter and ViT (trainable, "
          f"with and without grad ckpt), frozen ViT has no grads and is left out of the checkpoint, "
          f"encoder FLOPs 1x/3x, NaFlex 512 px (32x32) and 1024 px (64x64) ok, one group vs two groups err {err:.1e}")


def test_patch16_packing():
    w, h = 96, 64
    rng = np.random.default_rng(0)
    arr = (rng.integers(0, 256, (h, w, 3))).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    jpg = buf.getvalue()
    meta = {"text_ids": list(range(100, 110)), "topic": "ocr", "w": w, "h": h, "n_img": 1}
    s16, s32 = D.mm_sample(jpg, meta, patch=16), D.mm_sample(jpg, meta)
    assert s16.pixels.shape == (24, 768) and s32.pixels.shape == (6, 3072)
    assert int((s16.ids == D.IMAGE_PAD).sum()) == 24 and len(s16) == 24 + 2 + 10 + 1 and s16.img_offset == 1
    assert s16.grid.max(0).tolist() == [5, 3] and s16.grid.tolist()[:3] == [[0, 0], [1, 0], [2, 0]]
    dec = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB"))
    for n, (gx, gy) in enumerate(s16.grid):
        assert np.array_equal(s16.pixels[n].reshape(16, 16, 3), dec[16 * gy:16 * gy + 16, 16 * gx:16 * gx + 16])
    # the four 16-px tokens of each 32-px block are the block's sub-patches as SiglipFrontEnd splits them
    sub = SiglipFrontEnd.split_blocks(torch.from_numpy(s32.pixels).float())      # [6, 4, 768]
    for n32, (bx, by) in enumerate(s32.grid):
        for k, (sy, sx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            n16 = (2 * by + sy) * 6 + (2 * bx + sx)
            assert torch.equal(sub[n32, k], torch.from_numpy(s16.pixels[n16]).float())
    # packing: image span, loss mask, collate
    s0 = D.Sample(ids=np.r_[np.arange(10, 30), D.EOS].astype(np.int64), topic=6, kind=0)
    seq = D.pack([s0, s16], 128, kind=1)
    b = D.collate([seq])
    ids, lm = b["input_ids"][0], b["loss_mask"][0]
    img_pos = (b["image_id"][0] == 0).nonzero().flatten()
    assert len(img_pos) == 24 and img_pos[0] == len(s0) + 1 and torch.equal(img_pos, torch.arange(img_pos[0], img_pos[0] + 24))
    assert (ids[img_pos] == D.IMAGE_PAD).all() and (ids[(b["image_id"][0] < 0)] != D.IMAGE_PAD).all()
    assert not lm[img_pos - 1].any() and lm[img_pos[-1]] and not (b["targets"][0][lm] == D.IMAGE_PAD).any()
    assert int(lm.sum()) == (len(s0) - 1) + (len(s16) - 1 - 24)
    assert b["pixels"].shape == (24, 768) and b["grid_xy"].shape == (24, 2) and torch.equal(b["img_index"][:, 1], img_pos)
    assert torch.equal(D.attended_keys(b["sample_id"], b["image_id"]), build_eager_mask(b["sample_id"], b["image_id"])[:, 0].sum(-1))
    # the linear front end consumes 768-dim rows; closed-form counts and FLOPs follow patch_dim
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=2, n_kv_heads=2, d_ff=128, seq_len=128, max_grid=8, attn_impl="eager", patch_px=16)
    m = EncFreeDecoder(cfg).to(dev)
    assert cfg.patch_dim == 768 and m.param_counts() == count_params(cfg)
    bb = {k: v.to(dev) for k, v in b.items() if torch.is_tensor(v)}
    hdn = m(bb["input_ids"], bb["sample_id"], bb["image_id"], bb["pixels"], bb["grid_xy"], bb["img_index"])
    assert hdn.shape == (1, 128, 64) and m.front_end_flops(bb["grid_xy"]) == {"adapter": 6 * m.param_counts()["front_end"] * 24, "encoder": 0.0}
    print("patch_px=16 packing OK: 24 tokens of 768 for 96x64, pixels == 16-px crops, == SigLIP sub-patches, "
          "span/loss mask/collate consistent, linear model forward OK")


class _Stub:
    def __init__(self, name):
        self.name = name

    def next(self):
        return self.name

    def state(self):
        return {}

    def load_state(self, st):
        pass


def test_mixture_weights():
    mm = {"a": [f"a{i}" for i in range(4)], "b": ["b0", "b1"], "c": ["c0"]}
    assert D.mm_source_plan(D.DataConfig(), mm) == (["a", "b", "c"], [mm["a"], mm["b"], mm["c"]], [4.0, 2.0, 1.0])
    cfg = D.DataConfig(mm_source_weights={"b": 3.0, "c": 0.0}, mm_source_caps={"a": 2})
    names, shards, weights = D.mm_source_plan(cfg, mm)
    assert names == ["a", "b"] and shards == [["a0", "a1"], ["b0", "b1"]] and weights == [2.0, 6.0], (names, shards, weights)
    try:
        D.mm_source_plan(D.DataConfig(mm_source_weights={"zzz": 1.0}), mm)
        raise RuntimeError("unknown source accepted")
    except AssertionError:
        pass
    mix = D.Mixture([_Stub("a"), _Stub("b")], weights, seed=1)
    n = 4000
    draws = [mix.next(0) for _ in range(n)]
    fa = draws.count("a") / n
    tol = 3 * math.sqrt(0.25 * 0.75 / n)
    assert abs(fa - 0.25) < tol, (fa, tol)
    st = mix.state()
    s1 = [mix.next(0) for _ in range(50)]
    mix.load_state(st)
    s2 = [mix.next(0) for _ in range(50)]
    assert s1 == s2
    print(f"mixture test OK: caps/weights plan, p(a)={fa:.3f} (expect 0.25 +- {tol:.3f}), state resumable")


if __name__ == "__main__":
    test_masks()
    test_loss_mask_and_keys()
    test_model_forward()
    test_siglip_patch_order()
    test_siglip_forward()
    test_patch16_packing()
    test_mixture_weights()
    print("ALL UNIT TESTS PASSED")
