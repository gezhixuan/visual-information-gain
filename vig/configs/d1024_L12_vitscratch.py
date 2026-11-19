# Architecture-only control for the frozen-encoder condition: the SigLIP 2 so400m architecture (27 layers, width 1152, 16 heads,
# ff 4304, 16x16 position grid) randomly initialised and trained jointly from scratch, with the same ConvPool +
# projector as d1024_L12_siglip. The difference d1024_L12_siglip - d1024_L12_vitscratch isolates the pretrained
# prior from the extra encoder architecture/capacity; d1024_L12_vitscratch - d1024_L12 isolates the architecture.
# Cost: trainable encoder = 3x its forward FLOPs (about two thirds of the run's FLOPs at this size).
from ._base import ladder

config = ladder("d1024_L12_vitscratch", d=1024, L=12, heads=16, d_ff=2816, batch_seqs=128, micro_seqs=8, lr=1.5e-3,
                total_tokens=20_000_000_000, front_end="siglip2", freeze_encoder=False, encoder_grad_checkpoint=True)
v = config.model.vision
v.siglip_path = ""                    # random init from the rand_* dims below
v.rand_layers, v.rand_width, v.rand_heads, v.rand_ff, v.rand_num_patches = 27, 1152, 16, 4304, 256
