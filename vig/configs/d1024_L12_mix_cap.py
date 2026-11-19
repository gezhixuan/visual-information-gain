# Supervision-density experiment, sparse side: the 158M decoder trained on captions only (cc12m + cc3m), i.e. about
# 24 image-dependent text tokens per 191 image tokens. Same decoder, schedule, batch and lr as d1024_L12; the
# validation sets (all six topics) are unchanged so per-topic VIG is comparable across mixtures.
from ._base import ladder

config = ladder("d1024_L12_mix_cap", d=1024, L=12, heads=16, d_ff=2816, batch_seqs=128, micro_seqs=16, lr=1.5e-3,
                total_tokens=20_000_000_000)
config.data.mm_sources = "cc*-wds"     # glob under shards/mm: cc12m-wds, cc3m-wds
