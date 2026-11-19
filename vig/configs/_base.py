"""Shared constructor for the ladder configs (import-only; not a runnable config)."""
from ..data import DataConfig
from ..model import ModelConfig, VisionConfig
from ..train import TrainConfig

SEQ = 4096
# head_dim is 64 for every size (heads = d/64): torch 2.6 + triton 3.2 cannot compile the flex_attention
# BACKWARD kernel for head_dim 128 in bf16 (NoValidChoicesError), while
# head_dim 64 bf16 compiles and runs fwd+bwd in 5 ms at B=8, T=4096, H=8. Parameter counts and
# attention FLOPs are unchanged by this choice.


def ladder(run: str, d: int, L: int, heads: int, d_ff: int, batch_seqs: int, micro_seqs: int,
           lr: float, warmup_steps: int = 2000, total_tokens: int = 40_000_000_000,
           front_end: str = "linear", patch_px: int = 32, freeze_encoder: bool = True,
           encoder_grad_checkpoint: bool = False, **kw) -> TrainConfig:
    """One decoder size. batch_seqs = global batch in 4,096-token sequences (8 GPUs assumed
    for micro_seqs; grad_accum = batch_seqs / (8 * micro_seqs) is derived at runtime from
    WORLD_SIZE, so the same config also runs on fewer GPUs with more accumulation).
    front_end: "linear" (encoder-free) or "siglip2" (frozen SigLIP 2 so400m + ConvPool + projector);
    patch_px: 32, or 16 for the 4x-token variant of the linear front end (max_grid keeps 2048 px)."""
    assert front_end == "linear" or patch_px == 32
    vision = VisionConfig(front_end=front_end, freeze_encoder=freeze_encoder,
                          encoder_grad_checkpoint=encoder_grad_checkpoint)
    model = ModelConfig(d_model=d, n_layers=L, n_heads=heads, n_kv_heads=heads, d_ff=d_ff, seq_len=SEQ,
                        tie_embeddings=True, image_bidirectional=True, attn_impl="flex",
                        patch_px=patch_px, max_grid=2048 // patch_px, vision=vision)
    data = DataConfig(seq_len=SEQ, num_workers=6, patch_px=patch_px)
    return TrainConfig(run=run, model=model, data=data,
                       global_batch_tokens=batch_seqs * SEQ, micro_batch_seqs=micro_seqs,
                       total_tokens=total_tokens, lr=lr, warmup_tokens=warmup_steps * batch_seqs * SEQ,
                       log_every_tokens=50_000_000, eval_every_tokens=500_000_000,
                       save_every_tokens=1_000_000_000, state_every_tokens=500_000_000, **kw)
