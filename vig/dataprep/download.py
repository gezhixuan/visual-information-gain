"""Download the public datasets and models the paper uses from the Hugging Face Hub.

Datasets go to VIG_DATA/{mm,text}/<name>, models to VIG_MODELS/<name> (see vig/paths.py). Finished repositories are
skipped on a second call. Gated repositories need `huggingface-cli login` first.

Usage: python -m vig.dataprep.download --what data      image-text and text sources, the tokenizer and the SigLIP 2 encoder
       python -m vig.dataprep.download --what public    the public checkpoints of Sec. 5
       python -m vig.dataprep.download --only google/gemma-4-12B
"""
import argparse
import os

from huggingface_hub import snapshot_download

from .. import paths

# (group, repository, type, local folder, allow_patterns)
JOBS = [
    ("data", "pixparse/cc12m-wds", "dataset", "mm/cc12m-wds", None),
    ("data", "pixparse/cc3m-wds", "dataset", "mm/cc3m-wds", None),
    ("data", "HuggingFaceM4/the_cauldron", "dataset", "mm/the_cauldron", None),
    ("data", "lmms-lab/LLaVA-OneVision-Data", "dataset", "mm/LLaVA-OneVision-Data", None),
    ("data", "HuggingFaceM4/Docmatix", "dataset", "mm/Docmatix", ["data/*", "README.md"]),      # the image configuration, not the PDFs
    ("data", "xlangai/aguvis-stage1", "dataset", "mm/aguvis-stage1", None),
    ("data", "ahmed-masry/ChartGemma", "dataset", "mm/ChartGemma", None),
    ("data", "HuggingFaceM4/ChartQA", "dataset", "mm/ChartQA", None),
    ("data", "MMInstruction/ArxivQA", "dataset", "mm/ArxivQA", None),
    ("data", "Zhiqiang007/MathV360K", "dataset", "mm/MathV360K", None),
    ("data", "HuggingFaceFW/fineweb-edu", "dataset", "text/fineweb-edu", ["sample/100BT/*", "README.md"]),
    ("data", "open-web-math/open-web-math", "dataset", "text/open-web-math", None),
    ("data", "bigcode/starcoderdata", "dataset", "text/starcoderdata", ["python/*", "markdown/*", "README.md"]),
    ("data", "wikimedia/wikipedia", "dataset", "text/wikipedia-en", ["20231101.en/*", "README.md"]),
    ("data", "Qwen/Qwen2.5-0.5B", "model", "Qwen2.5-0.5B", ["*.json", "*.txt"]),                # tokenizer only
    ("data", "google/siglip2-so400m-patch16-naflex", "model", "siglip2-so400m-patch16-naflex", None),
    ("public", "google/gemma-4-12B", "model", "gemma-4-12B", None),
    ("public", "google/gemma-4-12B-it", "model", "gemma-4-12B-it", None),
    ("public", "google/gemma-4-26B-A4B-it", "model", "gemma-4-26B-A4B-it", None),
    ("public", "Qwen/Qwen3-VL-8B-Instruct", "model", "Qwen3-VL-8B-Instruct", None),
    ("public", "Qwen/Qwen2.5-VL-7B-Instruct", "model", "Qwen2.5-VL-7B-Instruct", None),
    ("public", "OpenGVLab/InternVL3-8B-hf", "model", "InternVL3-8B-hf", None),
    ("public", "llava-hf/llava-1.5-7b-hf", "model", "llava-1.5-7b-hf", None),
    ("public", "Paranioar/NEO1_0-2B-PT", "model", "NEO1_0-2B-PT", None),
    ("public", "Paranioar/NEO1_0-2B-MT", "model", "NEO1_0-2B-MT", None),
    ("public", "Paranioar/NEO1_0-2B-SFT", "model", "NEO1_0-2B-SFT", None),
    ("public", "Paranioar/NEO1_5-2B-SFT", "model", "NEO1_5-2B-SFT", None),
    ("public", "Paranioar/NEO1_0-9B-PT", "model", "NEO1_0-9B-PT", None),
    ("public", "Paranioar/NEO1_0-9B-MT", "model", "NEO1_0-9B-MT", None),
    ("public", "Paranioar/NEO1_0-9B-SFT", "model", "NEO1_0-9B-SFT", None),
    ("public", "Paranioar/NEO1_5-9B-SFT", "model", "NEO1_5-9B-SFT", None),
    ("public", "OpenGVLab/Mono-InternVL-2B-S1-1", "model", "Mono-InternVL-2B-S1-1", None),
    ("public", "OpenGVLab/Mono-InternVL-2B-S1-2", "model", "Mono-InternVL-2B-S1-2", None),
    ("public", "OpenGVLab/Mono-InternVL-2B-S1-3", "model", "Mono-InternVL-2B-S1-3", None),
    ("public", "OpenGVLab/Mono-InternVL-2B", "model", "Mono-InternVL-2B", None),
]
MODEL_IGNORE = ["*.gguf", "*.onnx", "*.msgpack", "*.h5", "*.ot", "*.ckpt", "original/*"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["data", "public"])
    ap.add_argument("--only", help="one repository id")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    jobs = [j for j in JOBS if j[1] == a.only] if a.only else [j for j in JOBS if j[0] == a.what]
    if not jobs:
        ap.error("give --what or a repository id from the list with --only")
    for _, repo, kind, sub, patterns in jobs:
        dst = os.path.join(paths.MODELS if kind == "model" else paths.DATA, sub)
        marker = os.path.join(dst, ".complete")
        if os.path.exists(marker):
            print(f"[skip] {sub}"); continue
        os.makedirs(dst, exist_ok=True)
        print(f"[get ] {sub} <- {repo}", flush=True)
        snapshot_download(repo, repo_type=kind, local_dir=dst, allow_patterns=patterns,
                          ignore_patterns=MODEL_IGNORE if kind == "model" else None, max_workers=a.workers)
        open(marker, "w").write(repo + "\n")
        print(f"[done] {sub}", flush=True)


if __name__ == "__main__":
    main()
