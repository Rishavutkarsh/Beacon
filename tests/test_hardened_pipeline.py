import csv

from sankat_saathi_dataset.hardened import make_hardened_dpo_pairs, make_hardened_text_examples
from sankat_saathi_dataset.manifests import load_image_manifest, load_source_manifest


def test_source_manifest_has_ready_sources():
    sources = load_source_manifest()
    assert sources
    assert all(source["source_ready"] for source in sources)


def test_image_manifest_is_external_and_split():
    images = load_image_manifest()
    assert images
    assert {image.split_group for image in images}.issuperset({"train", "eval"})


def test_dpo_prompts_are_not_exact_sft_prompts():
    sft_prompts = {example.user_prompt for example in make_hardened_text_examples(24, "train")}
    dpo_prompts = {pair.prompt for pair in make_hardened_dpo_pairs(24)}
    assert not sft_prompts.intersection(dpo_prompts)


def test_water_answer_foregrounds_chemical_caveat():
    example = make_hardened_text_examples(1, "eval")[0]
    joined = " ".join(example.assistant_response.immediate_action).lower()
    assert "fuel smell" in joined
    assert "do not rely on boiling" in joined
