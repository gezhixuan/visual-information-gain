# Token-granularity experiment: the 158M decoder with one visual token per 16x16 pixels (4x the image tokens of the
# 32-px baseline; same pixels, same decoder, same mixture). seq_len is raised to 8192 so that 1024-px documents
# (64x64 = 4096 image tokens) still fit with their text; the global batch stays at 0.52M tokens per step (64 x 8192).
# Tests whether reading (ocr / gui VIG_layout) is limited by the 32-px resolution floor or by supervision.
from ._base import ladder

config = ladder("d1024_L12_p16", d=1024, L=12, heads=16, d_ff=2816, batch_seqs=64, micro_seqs=8, lr=1.5e-3,
                total_tokens=20_000_000_000, patch_px=16)
config.model.seq_len = 8192
config.data.seq_len = 8192
config.global_batch_tokens = 64 * 8192        # 0.52M tokens/step, same as the 32-px baseline (ladder() used SEQ=4096)
config.warmup_tokens = 2000 * 128 * 4096      # same warmup tokens as the 32-px baseline (1.05B)
