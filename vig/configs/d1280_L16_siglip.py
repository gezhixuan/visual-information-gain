# Frozen-encoder condition of the 323M decoder: identical to d1280_L16 (decoder 317.2M blocks, tied embedding 194.5M, data,
# 1:1 mixture, bidirectional attention inside each image, schedule, batch, lr) except the visual front end:
# frozen SigLIP 2 so400m NaFlex ViT (412.6M, 27 layers, width 1152, patch 16) -> 2x2 ConvPool -> 2-layer MLP
# projector (adapter 9.2M trainable), one token per 32x32 region like the linear front end. The difference
# between the two runs measures the value of the pretrained visual prior at matched decoder/data/mask.
from ._base import ladder
config = ladder("d1280_L16_siglip", d=1280, L=16, heads=20, d_ff=3456, batch_seqs=256, micro_seqs=16, lr=1.0e-3,
                front_end="siglip2")
