# Frozen-encoder condition of the 158M decoder: d1024_L12 with the frozen SigLIP 2 so400m front end (ConvPool + projector adapter
# 6.8M trainable, encoder 412.6M frozen).
from ._base import ladder
config = ladder("d1024_L12_siglip", d=1024, L=12, heads=16, d_ff=2816, batch_seqs=128, micro_seqs=16, lr=1.5e-3,
                front_end="siglip2")
