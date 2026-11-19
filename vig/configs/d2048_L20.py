# 1.04B decoder: blocks 1027.7M, + front end 10.8M = 1038.5M non-embedding; tied embedding 311.2M.
from ._base import ladder
config = ladder("d2048_L20", d=2048, L=20, heads=32, d_ff=5632, batch_seqs=512, micro_seqs=8, lr=5e-4)
