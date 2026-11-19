# CPU smoke-test config for the encoder front end: the tiny decoder (d=64, L=2, seq 256) with a randomly initialised
# small Siglip2VisionConfig (2 layers, width 64, 2 heads, ff 128, 16x16 position grid; siglip_path="" = no
# checkpoint needed), trainable with gradient checkpointing by default; tests/smoke_cpu.sh also runs it frozen
# (--set model.vision.freeze_encoder=true) to exercise the checkpoint filtering of a frozen encoder.
from ..data import DataConfig
from ..model import ModelConfig, VisionConfig
from ..train import TrainConfig

SEQ = 256
vision = VisionConfig(front_end="siglip2", siglip_path="", freeze_encoder=False, encoder_grad_checkpoint=True,
                      rand_layers=2, rand_width=64, rand_heads=2, rand_ff=128, rand_num_patches=256)
config = TrainConfig(
    run="tiny_siglip",
    model=ModelConfig(d_model=64, n_layers=2, n_heads=2, n_kv_heads=2, d_ff=192, seq_len=SEQ, max_grid=8,
                      attn_impl="eager", vision=vision),
    data=DataConfig(seq_len=SEQ, num_workers=0, val_seqs_per_topic=2, val_seqs_per_text_source=2, diag_seqs=2),
    global_batch_tokens=4 * SEQ, micro_batch_seqs=4, total_tokens=30 * 4 * SEQ,
    lr=3e-3, warmup_tokens=5 * 4 * SEQ, weight_decay=0.1,
    log_every_tokens=4 * SEQ, log_first_steps=0, eval_every_tokens=15 * 4 * SEQ,
    save_every_tokens=15 * 4 * SEQ, state_every_tokens=15 * 4 * SEQ, eval_at_start=True,
    dtype="fp32", eval_micro=4, max_hours=1.0,
)
