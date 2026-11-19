"""Subset -> topic mapping for the multimodal shards (see the README).

Topics: caption | general | ocr | chart | gui | stem.  The topic decides the long-side cap
(512 for caption/general, 1024 otherwise) and the per-topic validation loss in the trainer.

Decisions:
- caption: natural-image captioning / detailed-description subsets (cc12m, cc3m, localized_narratives,
  sharegpt4v*, sharegpt4o, image_textualization, cambrian's description-heavy mix stays "general").
- general: natural-image VQA / instruction following (vqav2, okvqa, aokvqa, visual7w, clevr scenes, ...).
- ocr: documents, scene text, rendered tables, handwriting, infographics, webpage reading (Docmatix, docvqa,
  textvqa, textcaps, rendered_text, ureader_*, robut_*, hitab, finqa, tat_qa, iam, hme100k, k12_printing, ...).
- chart: charts, plots, maps as charts (chartqa, chart2text, dvqa, plotqa, figureqa, mapqa, vistext, lrv_chart).
- gui: screenshots (aguvis-stage1, screen2words, websight).
- stem: diagrams, science/textbook QA, geometry, abstract reasoning, math word problems, medical figures
  (ai2d, scienceqa, tqa, iconqa, intergps, geomverse, geo*, mavis*, raven, tabmwp, clevr_math, PMC-VQA, vqarad, datikz).
Text-only LLaVA-OneVision subsets (no image column content) are listed in TEXT_ONLY and excluded.
"""
import sys

SOURCE_TOPIC = {
    "cc12m-wds": "caption",
    "cc3m-wds": "caption",
    "Docmatix": "ocr",
    "aguvis-stage1": "gui",
    "ChartGemma": "chart",
    "ChartQA": "chart",
    "ArxivQA": "stem",
    "MathV360K": "stem",
}

CAULDRON_TOPIC = {
    "ai2d": "stem",
    "aokvqa": "general",
    "chart2text": "chart",
    "chartqa": "chart",
    "clevr": "general",
    "clevr_math": "stem",
    "cocoqa": "general",
    "datikz": "stem",
    "diagram_image_to_text": "stem",
    "docvqa": "ocr",
    "dvqa": "chart",
    "figureqa": "chart",
    "finqa": "ocr",
    "geomverse": "stem",
    "hateful_memes": "general",
    "hitab": "ocr",
    "iam": "ocr",
    "iconqa": "stem",
    "infographic_vqa": "ocr",
    "intergps": "stem",
    "localized_narratives": "caption",
    "mapqa": "chart",
    "mimic_cgd": "general",       # 2 images per row -> skipped by the multi-image rule
    "multihiertt": "ocr",         # 3-7 table images per row -> skipped
    "nlvr2": "general",           # 2 images per row -> skipped
    "ocrvqa": "ocr",
    "okvqa": "general",
    "plotqa": "chart",
    "raven": "stem",              # ~half the rows have 2 images -> those are skipped
    "rendered_text": "ocr",
    "robut_sqa": "ocr",
    "robut_wikisql": "ocr",
    "robut_wtq": "ocr",
    "scienceqa": "stem",
    "screen2words": "gui",
    "spot_the_diff": "general",   # 2 images per row -> skipped
    "st_vqa": "ocr",
    "tabmwp": "stem",
    "tallyqa": "general",
    "tat_qa": "ocr",
    "textcaps": "ocr",
    "textvqa": "ocr",
    "tqa": "stem",
    "vistext": "chart",
    "visual7w": "general",
    "visualmrc": "ocr",
    "vqarad": "stem",
    "vqav2": "general",
    "vsr": "general",
    "websight": "gui",
}

LLAVA_OV_TOPIC = {
    "CLEVR-Math(MathV360K)": "stem",
    "Evol-Instruct-GPT4-Turbo": "general",   # text-only
    "FigureQA(MathV360K)": "chart",
    "GEOS(MathV360K)": "stem",
    "GeoQA+(MathV360K)": "stem",
    "Geometry3K(MathV360K)": "stem",
    "IconQA(MathV360K)": "stem",
    "MapQA(MathV360K)": "chart",
    "MathV360K_TQA": "stem",
    "MathV360K_VQA-AS": "stem",
    "MathV360K_VQA-RAD": "stem",
    "PMC-VQA(MathV360K)": "stem",
    "Super-CLEVR(MathV360K)": "stem",
    "TabMWP(MathV360K)": "stem",
    "UniGeo(MathV360K)": "stem",
    "VisualWebInstruct(filtered)": "stem",
    "VizWiz(MathV360K)": "general",
    "ai2d(cauldron,llava_format)": "stem",
    "ai2d(gpt4v)": "stem",
    "ai2d(internvl)": "stem",
    "allava_instruct_laion4v": "general",
    "allava_instruct_vflan4v": "general",
    "aokvqa(cauldron,llava_format)": "general",
    "cambrian(filtered)": "general",
    "chart2text(cauldron)": "chart",
    "chartqa(cauldron,llava_format)": "chart",
    "chrome_writting": "ocr",
    "clevr(cauldron,llava_format)": "general",
    "diagram_image_to_text(cauldron)": "stem",
    "dvqa(cauldron,llava_format)": "chart",
    "figureqa(cauldron,llava_format)": "chart",
    "geo170k(align)": "stem",
    "geo170k(qa)": "stem",
    "geo3k": "stem",
    "geomverse(cauldron)": "stem",
    "hateful_memes(cauldron,llava_format)": "general",
    "hitab(cauldron,llava_format)": "ocr",
    "hme100k": "ocr",
    "iam(cauldron)": "ocr",
    "iconqa(cauldron,llava_format)": "stem",
    "iiit5k": "ocr",
    "image_textualization(filtered)": "caption",
    "infographic(gpt4v)": "ocr",
    "infographic_vqa": "ocr",
    "infographic_vqa_llava_format": "ocr",
    "intergps(cauldron,llava_format)": "stem",
    "k12_printing": "ocr",
    "llava_wild_4v_12k_filtered": "general",
    "llava_wild_4v_39k_filtered": "general",
    "llavar_gpt4_20k": "ocr",
    "lrv_chart": "chart",
    "lrv_normal(filtered)": "general",
    "magpie_pro(l3_80b_mt)": "general",      # text-only
    "magpie_pro(l3_80b_st)": "general",      # text-only
    "magpie_pro(qwen2_72b_st)": "general",   # text-only
    "mapqa(cauldron,llava_format)": "chart",
    "mathqa": "stem",                        # text-only
    "mavis_math_metagen": "stem",
    "mavis_math_rule_geo": "stem",
    "multihiertt(cauldron)": "ocr",
    "orand_car_a": "ocr",
    "raven(cauldron)": "stem",
    "rendered_text(cauldron)": "ocr",
    "robut_sqa(cauldron)": "ocr",
    "robut_wikisql(cauldron)": "ocr",
    "robut_wtq(cauldron,llava_format)": "ocr",
    "scienceqa(cauldron,llava_format)": "stem",
    "scienceqa(nona_context)": "stem",
    "screen2words(cauldron)": "gui",
    "sharegpt4o": "caption",
    "sharegpt4v(coco)": "caption",
    "sharegpt4v(knowledge)": "caption",
    "sharegpt4v(llava)": "caption",
    "sharegpt4v(sam)": "caption",
    "sroie": "ocr",
    "st_vqa(cauldron,llava_format)": "ocr",
    "tabmwp(cauldron)": "stem",
    "tallyqa(cauldron,llava_format)": "general",
    "textcaps": "ocr",
    "textocr(gpt4v)": "ocr",
    "tqa(cauldron,llava_format)": "stem",
    "ureader_cap": "ocr",
    "ureader_ie": "ocr",
    "ureader_kg": "ocr",
    "ureader_qa": "ocr",
    "vision_flan(filtered)": "general",
    "vistext(cauldron)": "chart",
    "visual7w(cauldron,llava_format)": "general",
    "visualmrc(cauldron)": "ocr",
    "vqarad(cauldron,llava_format)": "stem",
    "vsr(cauldron,llava_format)": "general",
    "websight(cauldron)": "gui",
}

# LLaVA-OneVision subsets whose rows carry no image: excluded from the multimodal shards.
TEXT_ONLY = {
    "Evol-Instruct-GPT4-Turbo",
    "magpie_pro(l3_80b_mt)",
    "magpie_pro(l3_80b_st)",
    "magpie_pro(qwen2_72b_st)",
    "mathqa",
}

# aguvis-stage1 annotation files (all GUI).
AGUVIS_TOPIC = {k: "gui" for k in
                ["guienv", "omniact_fix", "ricoig16k", "ricosca", "seeclick", "seeclick_mi",
                 "ui_refexp", "webui350k", "widget_captioning"]}

_warned = set()


def topic_for(source, subset=None):
    """Return the topic of (source, subset); unknown subsets map to 'general' with one warning."""
    if source == "the_cauldron":
        table = CAULDRON_TOPIC
    elif source == "LLaVA-OneVision-Data":
        table = LLAVA_OV_TOPIC
    elif source == "aguvis-stage1":
        table = AGUVIS_TOPIC
    else:
        return SOURCE_TOPIC[source]
    if subset in table:
        return table[subset]
    key = (source, subset)
    if key not in _warned:
        _warned.add(key)
        print(f"[topics] WARNING unknown subset {source}/{subset!r} -> 'general'", file=sys.stderr)
    return "general"


def check_against_disk(mm_root=None):
    """Report subsets on disk that are missing from the tables (and table entries missing on disk)."""
    import os
    from .. import paths
    mm_root = mm_root or os.path.join(paths.DATA, "mm")
    report = {}
    for source, table in (("the_cauldron", CAULDRON_TOPIC), ("LLaVA-OneVision-Data", LLAVA_OV_TOPIC)):
        on_disk = sorted(d for d in os.listdir(f"{mm_root}/{source}")
                         if os.path.isdir(f"{mm_root}/{source}/{d}") and not d.startswith("."))
        report[source] = {"unmapped": sorted(set(on_disk) - set(table)),
                          "not_on_disk": sorted(set(table) - set(on_disk)),
                          "n_disk": len(on_disk)}
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(check_against_disk(), indent=1))
