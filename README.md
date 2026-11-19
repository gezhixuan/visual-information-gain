# Visual Information Gain

Code for the paper **What Does the Visual Encoder Buy? Visual Information Gain in Encoder-Free Multimodal Pretraining**
(Zhixuan Ge, Anqi Li, Wei Qiu).

Visual information gain (VIG) measures how much of a model's next-token loss on the text of image-text data depends on
the image. The same text is scored under three versions of the image:

| condition | image | loss |
|---|---|---|
| real | the image as it is | L_real |
| shuffled | its 32x32 pixel patches permuted within the image | L_perm |
| uniform | a gray image of the same size | L_unif |

```
VIG_total  = L_unif - L_real
VIG_global = L_unif - L_perm      what the model takes from the bag of patches
VIG_layout = L_perm - L_real      what it takes from their arrangement, which includes reading
```

The measurement needs three forward passes and no labels, so it runs on every checkpoint of a training run and on
released models. This repository holds what is needed to reproduce the paper: data preparation, the encoder-free
decoders and their training, the frozen-encoder condition and the interventions, the VIG evaluation, the answer
accuracy check, and the evaluation of public checkpoints.

## Setup

```bash
pip install -r requirements.txt
bash tests/smoke_cpu.sh        # unit tests and a few training steps on synthetic data, about two minutes on a CPU
```

The compiled flex attention of the decoder needs a GPU and PyTorch 2.6 or newer. On a CPU the code falls back to an
explicit attention mask, which is what the smoke test uses.

Data, models, runs and outputs live where four environment variables point (defaults in brackets):
`VIG_DATA` (`./data`), `VIG_MODELS` (`./models`), `VIG_RUNS` (`./runs`), `VIG_OUT` (`./outputs`).

## Reproducing the paper

`scripts/paper_runs.sh` lists every command in order. The steps are:

1. **Data.** `python -m vig.dataprep.download --what data` fetches the public sources from the Hugging Face Hub.
   `vig.dataprep.build_mm_shards` turns the image-text sources into tar shards (one JPEG and one JSON per sample, images
   resized to multiples of 32 pixels) and `vig.dataprep.build_text_bins` tokenizes the text sources. Both split their
   work into independent tasks.
2. **Training.** `torchrun --standalone --nproc_per_node=8 -m vig.train --config d1024_L12` trains the 158M decoder.
   The configurations are in `vig/configs`, and any field can be overridden with `--set a.b=c`. A run resumes from its
   `state.pt` when started again.
3. **VIG of the checkpoints.** `python -m vig.eval_vig --runs d1024_L12 d1280_L16_siglip` scores every checkpoint of the
   runs under the image conditions and appends one row per run, checkpoint, image type and condition to
   `outputs/vig.jsonl`. `--summarize` prints VIG_total and VIG_layout per checkpoint.
4. **Answer accuracy.** `python -m vig.eval_answers --runs d1280_L16 d1280_L16_siglip --tokens 40` records for every
   answer token whether the model ranks it first, under the same three conditions.
5. **Public models.** `python -m vig.public_vig --models gemma-4-12B:raw Qwen3-VL-8B-Instruct:chat` scores the same
   validation samples with models that load through `AutoProcessor` and `AutoModelForImageTextToText`.
   `vig.public_vig_stages` does the same for the released training stages of NEO and Mono-InternVL, which ship their own
   model code.

| run in the paper | configuration |
|---|---|
| encoder-free decoders, 158M, 323M, 633M, 1.04B | `d1024_L12`, `d1280_L16`, `d1792_L16`, `d2048_L20` |
| frozen SigLIP 2 encoder, 323M and 158M | `d1280_L16_siglip`, `d1024_L12_siglip` |
| caption-only and document-heavy mixtures | `d1024_L12_mix_cap`, `d1024_L12_mix_doc` |
| 16-pixel patches | `d1024_L12_p16` |
| SigLIP 2 architecture with random weights | `d1024_L12_vitscratch` |
| baseline seeds and batch sizes | `d1024_L12` with the overrides in `scripts/paper_runs.sh` |

Repeated evaluations on one GPU give identical outputs. Between GPU models the evaluation, which runs in bfloat16,
moves the losses by about 0.001 nats and the answer accuracies by a few tenths of a point.

## Measuring VIG on another model

`vig/public_vig.py` is the place to start. `make_image` builds the three conditions for an image, `score` returns the
summed loss of a text given an image, and `run_model` loops over the validation samples. A model that needs its own
preprocessing gets an adapter as in `vig/public_vig_stages.py`. Tokenizers differ between models, so compare loss
differences on the same text and divide the summed loss by one common token count, as the paper does.

## Names in the code

| paper | code |
|---|---|
| uniform image, L_unif | `blank` in our evaluation, `uniform` for public models |
| shuffled image, L_perm | `permuted` |
| image types | `topic`: `caption`, `chart`, `general`, `ocr`, `stem`, `gui` |
| linear embedding, frozen-encoder condition | `front_end="linear"`, `front_end="siglip2"` |

`eval_vig` also scores a fourth condition, `noise` (uniformly random pixels), which the paper uses as a check.

## Layout

```
vig/model.py              decoder and the two front ends
vig/data.py               shard readers, packing into 4,096-token sequences, validation batches
vig/train.py              training loop (DDP, token-based schedule and checkpoints)
vig/eval.py               validation losses during training
vig/eval_vig.py           VIG of saved checkpoints
vig/eval_answers.py       answer accuracy under the three conditions
vig/val_cache.py          the validation samples as images, and their 16-pixel version
vig/public_vig.py         VIG of public models
vig/public_vig_stages.py  adapters for NEO and Mono-InternVL
vig/configs/              one file per run
vig/dataprep/             download, shards, text bins
scripts/paper_runs.sh     every command of the paper
tests/                    unit tests and the CPU smoke test
```

## To do

- [ ] Release the checkpoints of the runs in the paper.

## Citation

```bibtex
@article{ge2026vig,
  title={What Does the Visual Encoder Buy? Visual Information Gain in Encoder-Free Multimodal Pretraining},
  author={Zhixuan Ge and Anqi Li and Wei Qiu},
  year={2026}
}
```

## License

MIT, see `LICENSE`. The datasets and models keep their own licenses.
