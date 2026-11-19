#!/usr/bin/env python
"""Visual information gain of public encoder-free vision-language models that ship their own modelling code.

Companion of public_vig.py for models that do not load through AutoProcessor + AutoModelForImageTextToText. Samples,
sample order, image conditions and permutation seeds come from public_vig.py unchanged (load_samples, make_image, CONDS),
and the rows have the same fields, so the two outputs can be pooled. Every family has an adapter that rebuilds the input
of the model's own chat() code path (its image preprocessing, its image-token layout, its conversation template), runs
one forward pass and returns the summed next-token loss over the tokens of our text only.

Families (detected from the files in VIG_MODELS/<name>):
  neo    NEO1_0-*, NEO1_5-*          native resolution, 32x32 pixels per image token, <img><IMG_CONTEXT>*n</img>
  mono   Mono-InternVL-2B[-S1-*]     448-pixel tiles + thumbnail, 256 tokens per tile, <img><IMG_CONTEXT>*256*tiles</img>
HoVLE and VoRA are recognised but have no adapter (see ADAPTERS), EVE and SAIL ship no code.

Formats: "raw" puts the text directly after the image tokens, no conversation template. "chat" makes the image alone
the user turn of the model's conversation template (as chat() does with an empty question) and scores the whole text as
the reply. The end-of-turn token after the text is not part of the sequence and not scored.

The scored positions are checked for every forward pass: the token ids after the prefix are decoded and compared with
the text. The counts are printed per model and stored with the preprocessing settings in <out_dir>/<tag>.meta.json.

The model directories hold code written for transformers 4. It is imported in place and the weights are loaded by
build_and_load, not by from_pretrained (see there). The few other adjustments for transformers 5 are marked "shim".

--budget N  maximal number of tiles of the tiled family (default 12, the value of its quick start load_image). The
            tag then ends in _bN as in public_vig.py.
--attn      eager (default): the plain attention code of each model, the path that was tested. native: on a GPU, flash
            attention for mono and the compiled flex attention of NEO1_5 (same masks, same result up to rounding, untested).
--dtype     overrides the default (bfloat16 on a GPU, float32 on the CPU).

Usage: python -m vig.public_vig_stages --models NEO1_0-2B-SFT:chat NEO1_0-2B-PT:raw Mono-InternVL-2B:chat [--max_samples 2] [--topics ocr]
"""
from __future__ import annotations
import argparse, contextlib, glob, importlib, io, json, math, os, re, sys, time, traceback, types

sys.dont_write_bytecode = True                              # the model directories are imported in place and must stay untouched
import torch
import torch.nn.functional as F
from PIL import Image

from . import paths
from .public_vig import CONDS, MODELS, load_samples, make_image, summarize

OUT = os.path.join(paths.OUT, "public_vig") + "/"
IMG_START, IMG_END, IMG_CONTEXT = "<img>", "</img>", "<IMG_CONTEXT>"
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
LOG = "[public_vig_stages]"


def import_model_dir(path: str):
    """The python files of a model directory as a private package, imported in place (nothing is copied or written)."""
    alias = "_pvc_" + re.sub(r"\W", "_", os.path.basename(path.rstrip("/")))
    if alias not in sys.modules:
        pkg = types.ModuleType(alias); pkg.__path__ = [path]; pkg.__package__ = alias
        sys.modules[alias] = pkg
    return lambda sub: importlib.import_module(f"{alias}.{sub}")


def rope_default_shim():
    """shim: transformers 5 dropped the "default" entry of ROPE_INIT_FUNCTIONS and moved rope_theta into config.rope_parameters.
    The NEO code (written for 4.5x) looks the entry up and sets rope_theta on a copy of the config for its height/width
    rotary embedding, so the entry is restored with the 4.x behaviour: base = config.rope_theta, dim = config.head_dim."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    def default_rope(config, device=None, seq_len=None, **kwargs):
        base = getattr(config, "rope_theta", None)
        if base is None: base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS.setdefault("default", default_rope)


@contextlib.contextmanager
def construction_context(dtype: torch.dtype):
    """Default dtype = model dtype while the constructors run (what from_pretrained of transformers 4 did) and the random
    initialisers switched off, since every weight is read from the checkpoint afterwards."""
    import torch.nn.init as tinit
    from transformers.modeling_utils import PreTrainedModel
    names = [n for n in ("uniform_", "normal_", "trunc_normal_", "constant_", "ones_", "zeros_", "xavier_uniform_", "xavier_normal_",
                         "kaiming_uniform_", "kaiming_normal_", "orthogonal_") if hasattr(tinit, n)]
    saved, saved_init, saved_dtype = {n: getattr(tinit, n) for n in names}, PreTrainedModel.init_weights, torch.get_default_dtype()
    try:
        for n in names: setattr(tinit, n, lambda tensor, *args, **kwargs: tensor)
        PreTrainedModel.init_weights = lambda self: None
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(saved_dtype)
        PreTrainedModel.init_weights = saved_init
        for n, f in saved.items(): setattr(tinit, n, f)


def build_and_load(cls, config, path: str, dtype: torch.dtype):
    """The model built by its own constructors on the CPU, then every parameter and persistent buffer copied from the
    safetensors files with an exact key and shape match.

    from_pretrained of transformers 5 is not used: it builds the model on the meta device and afterwards leaves every
    non-persistent buffer uninitialised unless the model's _init_weights restores it. Code written for transformers 4
    computes such buffers in the constructors only (rotary position tables in NEO and in InternLM2), so they would be
    garbage, silently. Building the model normally keeps them as the authors' code computes them."""
    from safetensors import safe_open
    with construction_context(dtype):
        model = cls(config)
    target = model.state_dict(keep_vars=True)
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    assert files, f"no safetensors files in {path}"
    seen = set()
    with torch.no_grad():
        for file in files:
            with safe_open(file, framework="pt") as f:
                for key in f.keys():
                    assert key in target, f"{key} of the checkpoint is not in the model"
                    assert key not in seen, f"{key} appears twice in the checkpoint"
                    value = f.get_tensor(key)
                    assert value.shape == target[key].shape, (key, tuple(value.shape), tuple(target[key].shape))
                    target[key].copy_(value); seen.add(key)
    missing = sorted(set(target) - seen)
    assert not missing, f"{len(missing)} model tensors are not in the checkpoint, e.g. {missing[:5]}"
    return model, len(seen)


ARCH_KEYS = ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
             "hidden_act", "max_position_embeddings", "rms_norm_eps", "rope_theta", "rope_theta_hw", "max_position_embeddings_hw", "attention_bias",
             "bias", "tie_word_embeddings", "use_sliding_window", "patch_size", "image_size", "num_channels", "downsample_ratio", "llm_hidden_size",
             "rope_theta_vision", "max_position_embeddings_vision", "force_image_size", "ps_version", "select_layer")


def load_config(cls, path: str):
    """shim: the composite configs here declare is_composition = True, the transformers 4 flag for "cannot be built without
    arguments". transformers 5 renamed it to has_no_defaults_at_init and otherwise builds a default instance while loading,
    which the stricter config class of the Mono-InternVL S1 checkpoints refuses."""
    if getattr(cls, "is_composition", False): cls.has_no_defaults_at_init = True
    return cls.from_pretrained(path)


def config_guard(obj, raw: dict, where: str):
    """The architecture values of config.json must be the ones the loaded config object reports. The config classes of the
    model directories were written for transformers 4, and transformers 5 renames and moves some keys underneath them."""
    bad = []
    for k in ARCH_KEYS:
        if k not in raw or raw[k] is None: continue
        got = getattr(obj, k, None)
        if got is None and k == "rope_theta": got = (getattr(obj, "rope_parameters", None) or {}).get("rope_theta")
        if isinstance(got, tuple) and len(got) == 1: got = got[0]       # NEOVisionConfig keeps two values as 1-tuples (trailing commas in its code)
        if got != raw[k]: bad.append((k, raw[k], got))
    assert not bad, f"{where}: config.json and the loaded config disagree (key, json, loaded): {bad}"


def family_of(path: str) -> str:
    for fam, marker in (("neo", "modeling_neo_chat.py"), ("mono", "modeling_internlm2_ve.py"),
                        ("hovle", "modeling_holistic_embedding.py"), ("vora", "modeling_vora.py")):
        if os.path.exists(os.path.join(path, marker)): return fam
    raise ValueError(f"no adapter for {path} (no known modelling file in the directory)")


def to_pixels(img: Image.Image) -> torch.Tensor:
    """ToTensor + ImageNet normalisation, the transform of the NEO and InternVL quick starts."""
    import torchvision.transforms as T
    return T.Compose([T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])(img.convert("RGB"))


class NoCache:
    """Stand-in for an empty key-value cache: the model code asks for one, a single forward pass has nothing to reuse."""

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return key_states, value_states


class Adapter:
    family = ""

    def __init__(self, name: str, device: torch.device, budget: int = 0, attn: str = "eager", dtype: str = "auto"):
        self.name, self.device, self.budget = name, device, budget
        self.native_attn = attn == "native" and device.type == "cuda"
        self.path = os.path.join(MODELS, name)
        self.dtype = (torch.bfloat16 if device.type == "cuda" else torch.float32) if dtype == "auto" else getattr(torch, dtype)
        self.prep = {}                                      # image preprocessing settings, stored in the meta file
        self.check = {"passes": 0, "exact": 0, "equal_ignoring_whitespace": 0, "mismatch": 0, "boundary_merged": 0, "examples": []}
        self.layout = {}                                    # one example prefix per format, image tokens collapsed
        self.load()
        self.model.to(self.device).eval()
        self.img_context_id = self.tok.convert_tokens_to_ids(IMG_CONTEXT)

    # ---- per family -------------------------------------------------------------------------------------------------
    def load(self): raise NotImplementedError

    def image_inputs(self, img: Image.Image) -> tuple[dict, str]:
        """-> (tensors for the forward pass, the image-token string that replaces <image> in the prompt)."""
        raise NotImplementedError

    def logits(self, ids: torch.Tensor, vis: dict) -> torch.Tensor:
        """ids [1, L] -> logits [L, vocabulary]."""
        raise NotImplementedError

    # ---- shared -----------------------------------------------------------------------------------------------------
    def from_pretrained(self, cls, config):
        model, n = build_and_load(cls, config, self.path, self.dtype)
        self.prep["weights"] = f"{n} tensors, all keys and shapes matched"
        return model

    def conv_prompt(self, system_message=None) -> str:
        """The prompt chat() builds for a first turn with an image and an empty question, up to the start of the reply."""
        template = self.get_conv_template(self.model.template)
        template.system_message = self.model.system_message if system_message is None else system_message
        template.append_message(template.roles[0], "<image>\n")          # chat(): question = '<image>\n' + question
        template.append_message(template.roles[1], None)
        return template.get_prompt()

    def prefix(self, image_str: str, fmt: str) -> str:
        if fmt == "raw": return image_str
        assert fmt == "chat", f"format {fmt!r} is not implemented for {self.family}"
        return self.conv_prompt().replace("<image>", image_str, 1)

    def encode(self, s: str) -> torch.Tensor:
        return self.tok(s, return_tensors="pt")["input_ids"][0]         # the call of chat(), special tokens as there

    def detok(self, ids: list) -> str:
        return self.tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def verify(self, text: str, ids: torch.Tensor, k: int, n_pre: int):
        dec = self.detok(ids[k:].tolist())
        c = self.check; c["passes"] += 1
        if k < n_pre: c["boundary_merged"] += 1
        if dec == text and k == n_pre: c["exact"] += 1; return
        if "".join(dec.split()) == "".join(text.split()) and k == n_pre: c["equal_ignoring_whitespace"] += 1
        else: c["mismatch"] += 1
        if len(c["examples"]) < 3:
            j = next((i for i, (a, b) in enumerate(zip(dec, text)) if a != b), min(len(dec), len(text)))
            c["examples"].append({"at_char": j, "decoded": dec[max(0, j - 20):j + 40], "text": text[max(0, j - 20):j + 40]})

    def tokens(self, text: str, image_str: str, fmt: str) -> tuple[torch.Tensor, int]:
        """-> (token ids of prefix + text, index of the first text token), with the scored-token check."""
        P = self.prefix(image_str, fmt)
        pre, ids = self.encode(P), self.encode(P + text)
        k = 0
        while k < min(len(pre), len(ids)) and pre[k] == ids[k]: k += 1  # scored tokens start after the common prefix
        assert k >= len(pre) - 1 and 0 < k < len(ids), (len(pre), len(ids), k)
        self.verify(text, ids, k, len(pre))
        if fmt not in self.layout:
            self.layout[fmt] = re.sub(f"(?:{re.escape(IMG_CONTEXT)})+", lambda m: f"{IMG_CONTEXT}*{len(m.group(0)) // len(IMG_CONTEXT)}", P) + "{text}"
            self.layout[fmt + "_first_ids"] = ids[:3].tolist()
            print(f"{LOG} {self.name}: {fmt} layout {self.layout[fmt]!r}, first ids {ids[:3].tolist()}, {len(ids)} tokens, {len(ids) - k} scored", flush=True)
        return ids, k

    @torch.no_grad()
    def score(self, text: str, img: Image.Image, fmt: str) -> tuple[float, int, int, int]:
        vis, image_str = self.image_inputs(img)
        ids, k = self.tokens(text, image_str, fmt)
        logits = self.logits(ids[None].to(self.device), vis)
        assert logits.shape[0] == len(ids), (logits.shape, len(ids))
        tgt = ids[k:].to(logits.device)
        loss = F.cross_entropy(logits[k - 1:-1].float(), tgt, reduction="sum")
        return float(loss), len(tgt), int((ids == self.img_context_id).sum()), len(text)


# ---------------------------------------------------------------------------------------------------------------------
# NEO (EvolvingLMMs-Lab/NEO). Preprocessing follows VLMEvalKit/vlmeval/vlm/neo/utils.py load_image_native of that
# repository (the model directories contain no preprocessing code), the prompt follows NEOChatModel.chat().
# ---------------------------------------------------------------------------------------------------------------------
def smart_resize(height: int, width: int, factor: int = 32, min_pixels: int = 65536, max_pixels: int = 4194304) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200: raise ValueError("aspect ratio above 200")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class NEOAdapter(Adapter):
    family = "neo"

    def load(self):
        from transformers import AutoTokenizer
        assert not self.budget, "--budget is for the tiled families"
        mod = import_model_dir(self.path)
        raw = json.load(open(os.path.join(self.path, "config.json")))
        # NEOChatConfig is read directly, so a config.json whose auto_map names classes that are not shipped
        # (NEO1_0-2B-PT: InternVLChatModel / InternVLChatConfig) loads like the others. The weight names are the same.
        config = load_config(mod("configuration_neo_chat").NEOChatConfig, self.path)
        config_guard(config, raw, "config"); config_guard(config.llm_config, raw["llm_config"], "llm_config")
        config_guard(config.vision_config, raw["vision_config"], "vision_config")
        if config.template != "neo1_0":                     # NEO1_0-2B-PT: internvl2_5. Same roles and separators, only the system message differs
            self.prep["template_in_config"] = config.template
            config.template = "neo1_0"
        rope_default_shim()
        with contextlib.redirect_stdout(io.StringIO()):     # transformers prints "[ERROR] ... not documented" lines about the docstrings of this file
            qwen3 = mod("modeling_qwen3")
        flex = hasattr(qwen3, "create_flex_mask_qwen3_style")            # NEO1_5 runs the prefix through torch.compile'd flex attention
        if flex and not self.native_attn:
            # shim: without a flex mask its layers fall back to the plain attention of NEO1_0 with the same block-causal mask and scale
            qwen3.create_flex_mask_qwen3_style = lambda index, **kw: (None, 0)
        self.model = self.from_pretrained(mod("modeling_neo_chat").NEOChatModel, config)
        self.model.system_message = ""                      # as the official evaluation wrapper (and the training data): no system turn
        self.get_conv_template = mod("conversation").get_conv_template
        # Qwen2 tokenizer files, no custom tokenizer code. fix_mistral_regex=False: transformers 5 otherwise warns about a Mistral
        # tokenizer for a config.json without transformers_version (NEO1_0-2B-PT). The tokenizer is not changed either way.
        # The merges.txt of the NEO directories is the Qwen3 file without its 96 merges that begin with "#", so "##" is two
        # tokens here (4 validation texts get 1 to 3 tokens more than with a Qwen tokenizer). Kept as released.
        self.tok = AutoTokenizer.from_pretrained(self.path, trust_remote_code=False, fix_mistral_regex=False)
        self.img_start_id = self.tok.convert_tokens_to_ids(IMG_START)
        vc = raw.get("vision_config", {})
        self.patch = int(raw.get("patch_size", vc.get("patch_size", 16)))
        self.ds = float(raw.get("downsample_ratio", 0.5))
        self.min_pixels = int(raw.get("min_pixels", vc.get("min_pixels", 65536)))
        self.max_pixels = int(raw.get("max_pixels", vc.get("max_pixels", 4194304)))
        self.prep.update(resize="smart_resize to multiples of %d, PIL default resampling" % int(self.patch / self.ds), patch_size=self.patch,
                         pixels_per_image_token=int(self.patch / self.ds) ** 2, min_pixels=self.min_pixels, max_pixels=self.max_pixels,
                         normalisation="ImageNet mean/std", system_message="", architectures_in_config=raw.get("architectures"),
                         attention="flex (compiled)" if flex and self.native_attn else "eager, block-causal mask")

    def image_inputs(self, img):
        img = img.convert("RGB")
        h, w = smart_resize(img.height, img.width, factor=int(self.patch / self.ds), min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        x = to_pixels(img.resize((w, h)))
        p, gh, gw = self.patch, h // self.patch, w // self.patch
        pv = x.reshape(3, gh, p, gw, p).permute(1, 3, 0, 2, 4).reshape(gh * gw, 3 * p * p)
        grid_hw = torch.tensor([[gh, gw]])
        n = int(gh * gw * self.ds ** 2)
        return {"pixel_values": pv.to(self.device, self.dtype), "grid_hw": grid_hw.to(self.device)}, IMG_START + IMG_CONTEXT * n + IMG_END

    def logits(self, ids, vis):
        m = self.model                                      # the steps of NEOChatModel.generate() up to language_model.generate()
        m.img_context_token_id, m.img_start_token_id = self.img_context_id, self.img_start_id
        indexes = m.get_thw_indexes(ids[0], vis["grid_hw"])
        vit = m.extract_feature(vis["pixel_values"], grid_hw=vis["grid_hw"])
        emb = m.language_model.get_input_embeddings()(ids)
        B, N, C = emb.shape
        emb = emb.reshape(B * N, C)
        sel = ids.reshape(B * N) == self.img_context_id
        assert int(sel.sum()) == vit.reshape(-1, C).shape[0], (int(sel.sum()), vit.shape)
        emb[sel] = vit.reshape(-1, C).to(emb.device, emb.dtype)
        pos = torch.arange(N, device=ids.device)
        out = m.language_model(inputs_embeds=emb.reshape(B, N, C), indexes=indexes, attention_mask=torch.ones_like(ids),
                               position_ids=pos[None], cache_position=pos, past_key_values=NoCache(), use_cache=False)
        return out.logits[0]


# ---------------------------------------------------------------------------------------------------------------------
# InternVL-style tiling, from the quick start of OpenGVLab/Mono-InternVL: load_image(input_size=448, max_num=12)
# ---------------------------------------------------------------------------------------------------------------------
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff, best_ratio, area = float("inf"), (1, 1), width * height
    for ratio in target_ratios:
        ratio_diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if ratio_diff < best_ratio_diff:
            best_ratio_diff, best_ratio = ratio_diff, ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]: best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    ratio = find_closest_aspect_ratio(orig_width / orig_height, target_ratios, orig_width, orig_height, image_size)
    target_width, target_height, blocks = image_size * ratio[0], image_size * ratio[1], ratio[0] * ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1: processed_images.append(image.resize((image_size, image_size)))
    return processed_images


class TiledAdapter(Adapter):
    """InternVL-derived models: 448-pixel tiles + thumbnail, num_image_token tokens per tile."""
    max_num_default = 12

    def setup_tiling(self, raw: dict):
        self.image_size = int(raw.get("force_image_size") or 448)
        self.max_num = self.budget or self.max_num_default
        self.prep.update(tile_size=self.image_size, min_tiles=1, max_tiles=self.max_num, thumbnail=True, tokens_per_tile=self.model.num_image_token,
                         resize="closest tile grid, PIL default resampling, then bicubic resize of every tile to the tile size (a no-op)",
                         normalisation="ImageNet mean/std", max_dynamic_patch_in_config=raw.get("max_dynamic_patch"),
                         system_message=self.model.system_message, template=self.model.template)

    def image_inputs(self, img):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        tiles = dynamic_preprocess(img.convert("RGB"), image_size=self.image_size, use_thumbnail=True, max_num=self.max_num)
        resize = T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC)
        pv = torch.stack([to_pixels(resize(t)) for t in tiles])
        return {"pixel_values": pv.to(self.device, self.dtype)}, IMG_START + IMG_CONTEXT * self.model.num_image_token * len(tiles) + IMG_END


class MonoAdapter(TiledAdapter):
    family = "mono"

    def load(self):
        mod = import_model_dir(self.path)
        raw = json.load(open(os.path.join(self.path, "config.json")))
        config = load_config(mod("configuration_internvl_chat").InternVLChatConfig, self.path)
        config_guard(config, raw, "config"); config_guard(config.llm_config, raw["llm_config"], "llm_config")
        config_guard(config.vision_config, raw["vision_config"], "vision_config")
        if raw["llm_config"].get("rope_scaling") is None:
            # shim: transformers 5 turns rope_scaling into an alias of its own rope_parameters dict. The InternLM2 code tests
            # `config.rope_scaling is None` to choose the plain rotary embedding, so the null of config.json is restored.
            config.llm_config.rope_parameters = None
            assert config.llm_config.rope_scaling is None
        if not self.native_attn: config.llm_config.attn_implementation = "eager"   # the config asks for flash attention
        self.model = self.from_pretrained(mod("modeling_internvl_chat").InternVLChatModel, config)
        self.get_conv_template = mod("conversation").get_conv_template
        self.tok = mod("tokenization_internlm2").InternLM2Tokenizer.from_pretrained(self.path)
        self.setup_tiling(raw)
        self.prep["attention"] = self.model.language_model.config.attn_implementation

    def detok(self, ids):
        # decode() of this tokenizer always removes the space before punctuation (its convert_tokens_to_string cleans up),
        # so the pieces of the text are joined by sentencepiece itself, which is exact
        sp = self.tok.sp_model
        return sp.decode(ids) if all(i < sp.get_piece_size() for i in ids) else self.tok.decode(ids, skip_special_tokens=False)

    def logits(self, ids, vis):
        m = self.model                                      # the steps of InternVLChatModel.generate() up to language_model.generate()
        m.img_context_token_id = self.img_context_id
        vit = m.extract_feature(vis["pixel_values"])
        emb = m.language_model.get_input_embeddings()(ids)
        B, N, C = emb.shape
        emb = emb.reshape(B * N, C)
        sel = ids.reshape(B * N) == self.img_context_id
        assert int(sel.sum()) == vit.reshape(-1, C).shape[0], (int(sel.sum()), vit.shape)
        emb[sel] = vit.reshape(-1, C).to(emb.device, emb.dtype)
        emb = emb.reshape(B, N, C)
        kw = {"visual_token_mask": sel.reshape(B, N, 1).to(emb.dtype)} if m.use_visual_token_mask else {}
        out = m.language_model(inputs_embeds=emb, attention_mask=torch.ones_like(ids), use_cache=False, output_hidden_states=False,
                               return_dict=True, **kw)
        return out.logits[0]


ADAPTERS = {"neo": NEOAdapter, "mono": MonoAdapter}


def run_model(spec: str, samples: dict, out_dir: str, budget: int, attn: str = "eager", dtype: str = "auto"):
    name, fmt = spec.split(":") if ":" in spec else (spec, "chat")
    tag = f"{name}_{fmt}" + (f"_b{budget}" if budget else "")
    path = os.path.join(out_dir, tag + ".jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); done.add((r["topic"], r["i"], r["cond"]))
    todo = sum(1 for t, items in samples.items() for i in range(len(items)) for c in CONDS if (t, i, c) not in done)
    if not todo: print(f"{LOG} {tag}: complete", flush=True); return
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fam = family_of(os.path.join(MODELS, name))
    assert fam in ADAPTERS, f"the {fam} adapter is not available"
    ad = ADAPTERS[fam](name, device, budget, attn, dtype)
    print(f"{LOG} {tag}: {fam} adapter loaded in {time.time() - t0:.0f}s on {device} ({ad.dtype}), {todo} forward passes\n{LOG} {tag}: preprocessing {ad.prep}", flush=True)
    try:
        with open(path, "a") as f:
            for ti, (topic, items) in enumerate(samples.items()):
                t1 = time.time()
                for i, (text, arr) in enumerate(items):
                    for cond in CONDS:
                        if (topic, i, cond) in done: continue
                        loss, n_tok, n_img, n_char = ad.score(text, make_image(arr, cond, seed=1000 * ti + i), fmt)
                        f.write(json.dumps({"model": name, "format": fmt, "budget": budget, "topic": topic, "i": i, "cond": cond, "loss_sum": loss,
                                            "n_tok": n_tok, "n_char": n_char, "n_img_tok": n_img}) + "\n")
                    f.flush()
                print(f"{LOG} {tag} {topic}: {len(items)} samples in {time.time() - t1:.0f}s", flush=True)
    finally:
        c = ad.check
        print(f"{LOG} {tag}: scored-token check over {c['passes']} forward passes: {c['exact']} exact, {c['equal_ignoring_whitespace']} equal ignoring "
              f"whitespace, {c['mismatch']} mismatch, {c['boundary_merged']} with a merged boundary token", flush=True)
        meta_path = os.path.join(out_dir, tag + ".meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else []
        import transformers
        meta.append({"model": name, "format": fmt, "budget": budget, "family": fam, "device": str(device), "dtype": str(ad.dtype),
                     "transformers": transformers.__version__, "torch": torch.__version__, "preprocessing": ad.prep,
                     "layout": ad.layout, "scored_token_check": c, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        json.dump(meta, open(meta_path, "w"), indent=1, ensure_ascii=False)
        del ad
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[]); ap.add_argument("--out_dir", default=OUT); ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=0); ap.add_argument("--topics", default=""); ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--attn", default="eager", choices=["eager", "native"]); ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"])
    a = ap.parse_args()
    if not a.out_dir.endswith("/"): a.out_dir += "/"
    os.makedirs(a.out_dir, exist_ok=True)
    if not torch.cuda.is_available(): torch.set_num_threads(min(8, torch.get_num_threads()))     # shared login nodes
    if a.models:
        S = load_samples(a.max_samples, "")              # all types, so that the topic index and with it the permutation seed match the full runs
        if a.topics: S = {t: (v if t in a.topics.split(",") else []) for t, v in S.items()}
        print(f"{LOG} samples per type:", {t: len(v) for t, v in S.items()}, flush=True)
        for spec in a.models:
            try: run_model(spec, S, a.out_dir, a.budget, a.attn, a.dtype)
            except Exception:  # noqa: BLE001
                print(f"{LOG} {spec} FAILED\n{traceback.format_exc()}", flush=True)
    if a.summarize or not a.models: summarize(a.out_dir)
