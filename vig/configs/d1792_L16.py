# 633M decoder: blocks 624.0M, + front end 9.0M = 632.9M non-embedding; tied embedding 272.3M.
from ._base import ladder
config = ladder("d1792_L16", d=1792, L=16, heads=28, d_ff=4864, batch_seqs=256, micro_seqs=8, lr=7e-4)
