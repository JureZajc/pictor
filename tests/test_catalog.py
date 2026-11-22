import json

import pytest

from pictor import catalog
from pictor.catalog import InstalledModel, build_options, load_catalog
from pictor.config import config_dir
from pictor.hardware import Hardware

CAPS = ("completion", "vision", "tools", "thinking")


def machine(accelerator: str, budget: float) -> Hardware:
    return Hardware(
        os="macOS" if accelerator == "apple-silicon" else "Linux",
        os_version="1",
        arch="arm64" if accelerator == "apple-silicon" else "x86_64",
        cpu="Test CPU",
        cores=8,
        ram_gb=32.0,
        gpus=(),
        accelerator=accelerator,
        budget_gb=budget,
        budget_note="",
    )


def pick(accelerator: str, budget: float, installed=(), version="0.40.16") -> catalog.ModelOption:
    options = build_options(machine(accelerator, budget), list(installed), version)
    chosen = [o for o in options if o.recommended]
    assert len(chosen) == 1
    return chosen[0]


def test_bundled_catalog_is_well_formed():
    bundled = load_catalog()
    tags = [m.tag for m in bundled.models]
    assert len(tags) >= 5 and len(set(tags)) == len(tags)
    assert [m.tier for m in bundled.models] == sorted(m.tier for m in bundled.models)
    sizes = [m.size_gb for m in bundled.models]
    assert sizes == sorted(sizes), "bigger tiers should be bigger downloads"
    assert all(m.size_gb > 0 and m.name and m.note for m in bundled.models)
    assert bundled.updated and bundled.source.startswith("https://")


def test_user_catalog_replaces_the_bundled_one():
    custom = {"models": [{"tag": "tiny:1b", "size_gb": 1.0, "tier": 1}]}
    (config_dir() / "models.json").write_text(json.dumps(custom))
    assert [m.tag for m in load_catalog().models] == ["tiny:1b"]


def test_a_broken_user_catalog_is_ignored():
    (config_dir() / "models.json").write_text("{not json")
    assert len(load_catalog().models) >= 5


def test_need_and_fit():
    assert catalog.need_gb(7.7) == 11.0
    assert catalog.fit_for(11.0, 16.8) == "fits"
    assert catalog.fit_for(18.5, 16.8) == "tight"
    assert catalog.fit_for(20.6, 16.8) == "too_big"


@pytest.mark.parametrize(
    ("accelerator", "budget", "expected"),
    [
        ("apple-silicon", 16.8, "gemma4:12b-mlx"),  # 24 GB Mac
        ("apple-silicon", 5.6, "qwen3.5:2b-mlx"),  # 8 GB Mac
        ("apple-silicon", 11.2, "gemma4:12b-mlx"),  # 16 GB Mac
        ("nvidia", 22.8, "gemma4:26b"),  # 24 GB card
        ("nvidia", 30.4, "qwen3.6:35b"),  # 32 GB card
        ("nvidia", 11.4, "gemma4:12b"),  # 12 GB card
        ("nvidia", 7.6, "qwen3.5:4b"),  # 8 GB card
        ("cpu", 8.0, "qwen3.5:4b"),
        ("cpu", 4.0, "qwen3.5:0.8b"),
    ],
)
def test_recommendation_follows_the_memory_budget(accelerator, budget, expected):
    assert pick(accelerator, budget).tag == expected


def test_recommended_model_always_fits_when_anything_does():
    for budget in (3.5, 5, 8, 12, 16, 24, 32, 48, 96, 200):
        assert pick("nvidia", budget).fit == "fits"


def test_nothing_fits_recommends_the_smallest_model():
    best = pick("cpu", 1.0)
    assert best.tag == "qwen3.5:0.8b" and best.fit != "fits"


def test_mlx_builds_are_only_offered_on_apple_silicon_with_a_new_enough_ollama():
    tags = {o.tag for o in build_options(machine("nvidia", 16.8), [], "0.40.16")}
    assert not any(tag.endswith("-mlx") for tag in tags)
    assert pick("apple-silicon", 16.8, version="0.30.0").tag == "gemma4:12b"
    assert pick("apple-silicon", 16.8, version=None).tag == "gemma4:12b"


def test_standard_build_is_used_when_only_it_fits():
    # The MLX build of this model is larger than the standard one.
    options = build_options(machine("apple-silicon", 22.0), [], "0.40.16")
    gemma_26 = next(o for o in options if o.name == "Gemma 4 26B")
    assert gemma_26.tag == "gemma4:26b" and gemma_26.fit == "fits"


def test_an_installed_build_is_offered_instead_of_downloading_the_other_one():
    installed = [InstalledModel("gemma4:12b", 7.7, CAPS)]
    best = pick("apple-silicon", 16.8, installed)
    assert best.tag == "gemma4:12b" and best.installed


def test_an_installed_model_is_kept_unless_a_download_is_much_better():
    # One size class better is not worth a multi-gigabyte download.
    nearly_as_good = [InstalledModel("qwen3.5:9b", 6.6, CAPS)]
    assert pick("nvidia", 16.0, nearly_as_good).tag == "qwen3.5:9b"
    # Two classes better is.
    much_smaller = [InstalledModel("qwen3.5:4b", 3.3, CAPS)]
    assert pick("nvidia", 16.0, much_smaller).tag == "gemma4:12b"
    # An installed model that does not fit is never the pick.
    too_big = [InstalledModel("qwen3.8:27b", 18.0, CAPS)]
    assert pick("nvidia", 16.0, too_big).tag == "gemma4:12b"


def test_installed_models_outside_the_catalog_are_listed():
    installed = [
        InstalledModel("hf.co/someone/big-model:Q4_K_M", 17.7, ("completion", "vision")),
        InstalledModel("mystery-vl:14b", 9.0, CAPS, "14B"),
    ]
    options = {o.tag: o for o in build_options(machine("nvidia", 22.8), installed, "0.40.16")}
    no_tools = options["hf.co/someone/big-model:Q4_K_M"]
    assert no_tools.installed and not no_tools.usable and "tools" in no_tools.reason
    assert not no_tools.recommended
    extra = options["mystery-vl:14b"]
    assert extra.usable and extra.source == "installed" and extra.params == "14B"
    # Ranked with the catalog models of its size, one class below the best download,
    # so being installed already tips the choice.
    assert extra.score == 5 and extra.recommended


def test_a_model_that_cannot_see_is_only_recommended_as_a_last_resort():
    blind = InstalledModel("text-only:30b", 18.0, ("completion", "tools"))
    assert pick("nvidia", 22.8, [blind]).vision
    options = build_options(machine("nvidia", 22.8), [blind], "0.40.16")
    assert not next(o for o in options if o.tag == "text-only:30b").vision


def test_tags_compare_with_their_implicit_latest():
    assert catalog.normalize_tag("gemma4") == "gemma4:latest"
    assert catalog.normalize_tag("gemma4:12b") == "gemma4:12b"
    assert catalog.normalize_tag("hf.co/a/b") == "hf.co/a/b:latest"
    assert catalog.normalize_tag("localhost:5000/a/b") == "localhost:5000/a/b:latest"
    installed = [InstalledModel("gemma4:12b-mlx", 7.7, CAPS)]
    options = build_options(machine("apple-silicon", 16.8), installed, "0.40.16")
    assert next(o for o in options if o.tag == "gemma4:12b-mlx").installed


def test_version_comparison():
    assert catalog.version_at_least("0.40.16", "0.32.12")
    assert catalog.version_at_least("0.32.12", "0.32.12")
    assert not catalog.version_at_least("0.9.6", "0.32.12")
    assert catalog.version_at_least("0.41.0-rc1", "0.32.12")
    assert not catalog.version_at_least(None, "0.32.12")
    assert catalog.version_at_least(None, "")


def test_context_size_grows_with_the_budget():
    assert catalog.suggest_num_ctx(4) == 8192
    assert catalog.suggest_num_ctx(8) == 12288
    assert catalog.suggest_num_ctx(16.8) == 16384
