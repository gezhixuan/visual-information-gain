# Supervision-density experiment, dense side: the 158M decoder trained on an OCR/document-heavy multimodal mixture.
# Same decoder, schedule, batch and lr as d1024_L12; only the per-source sampling weights change. Docmatix carries
# ~361 text tokens per image (vs 24 for cc12m captions), aguvis ~320, the conversation sources 138-241, so this
# mixture raises the image-dependent text supervision per image several-fold while the validation sets stay fixed.
from ._base import ladder

config = ladder("d1024_L12_mix_doc", d=1024, L=12, heads=16, d_ff=2816, batch_seqs=128, micro_seqs=16, lr=1.5e-3,
                total_tokens=20_000_000_000)
# Per-source sampling weights. Token shares of the resulting image-text mixture: 43% Docmatix, 19% LLaVA-OneVision,
# 15% the Cauldron, 8% Aguvis, 4% chart and STEM sources, 11% captions.
config.data.mm_source_weights = {"Docmatix": 7.9, "LLaVA-OneVision-Data": 4.4, "the_cauldron": 4.6, "aguvis-stage1": 4.8,
                                 "ChartGemma": 5.3, "ChartQA": 7.6, "ArxivQA": 8.6, "MathV360K": 5.0,
                                 "cc12m-wds": 1.0, "cc3m-wds": 1.0}
