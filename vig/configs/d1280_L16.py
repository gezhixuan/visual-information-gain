# 323M decoder: blocks 317.2M, + front end 5.8M = 323.0M non-embedding; tied embedding 194.5M.
from ._base import ladder
config = ladder("d1280_L16", d=1280, L=16, heads=20, d_ff=3456, batch_seqs=256, micro_seqs=16, lr=1.0e-3)
