# CPU smoke-test config: d=64, L=2, seq 256, full padded vocab. Overridden by tests/smoke_cpu.sh
# for shard_root / out_root / run.
from ..data import DataConfig
from ..model import ModelConfig
from ..train import TrainConfig

SEQ = 256
config = TrainConfig(
    run="tiny",
    model=ModelConfig(d_model=64, n_layers=2, n_heads=2, n_kv_heads=2, d_ff=192, seq_len=SEQ, max_grid=8,
                      attn_impl="eager"),
    data=DataConfig(seq_len=SEQ, num_workers=0, val_seqs_per_topic=2, val_seqs_per_text_source=2, diag_seqs=2),
    global_batch_tokens=4 * SEQ, micro_batch_seqs=4, total_tokens=30 * 4 * SEQ,
    lr=3e-3, warmup_tokens=5 * 4 * SEQ, weight_decay=0.1,
    log_every_tokens=4 * SEQ, log_first_steps=0, eval_every_tokens=15 * 4 * SEQ,
    save_every_tokens=15 * 4 * SEQ, state_every_tokens=15 * 4 * SEQ, eval_at_start=True,
    dtype="fp32", eval_micro=4, max_hours=1.0,
)
