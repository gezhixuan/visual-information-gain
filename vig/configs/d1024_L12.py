# 158M decoder: blocks 154.2M, + front end 4.3M = 158.5M non-embedding; tied embedding 155.6M.
from ._base import ladder
config = ladder("d1024_L12", d=1024, L=12, heads=16, d_ff=2816, batch_seqs=128, micro_seqs=16, lr=1.5e-3)
