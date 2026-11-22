"""The model catalog and the rule that picks a model for a machine.

The catalog is a small JSON file of Ollama models that can both see pictures and call
tools, ordered by size. Local models change quickly, so the list is data, not code: a
``models.json`` in the config folder replaces the bundled one, and any tool-capable model
already installed in Ollama is offered as well.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from importlib import resources

from pictor.config import config_dir
from pictor.hardware import Hardware

#: A model slightly over budget still runs, with some layers on the CPU.
TIGHT_FACTOR = 1.15
#: An installed model is preferred unless a download would be two or more size classes
#: better: one class is not worth several gigabytes and a wait.
INSTALLED_BONUS = 1.5


@dataclass(frozen=True)
class Variant:
    tag: str
    size_gb: float


@dataclass(frozen=True)
class CatalogModel:
    tag: str
    name: str
    params: str
    size_gb: float
    tier: int
    note: str = ""
    mlx: Variant | None = None


@dataclass(frozen=True)
class Catalog:
    models: tuple[CatalogModel, ...]
    updated: str = ""
    source: str = ""
    mlx_min_ollama: str = ""


@dataclass(frozen=True)
class InstalledModel:
    """A model present in the local Ollama store, with what ``/api/show`` says it can do."""

    tag: str
    size_gb: float
    capabilities: tuple[str, ...] = ()
    params: str = ""


@dataclass
class ModelOption:
    tag: str
    name: str
    params: str
    size_gb: float
    need_gb: float
    fit: str  # "fits" | "tight" | "too_big"
    installed: bool
    usable: bool
    vision: bool
    thinking: bool
    source: str  # "catalog" | "installed"
    score: float
    note: str = ""
    reason: str = ""
    recommended: bool = False
    capabilities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- loading


def _parse(raw: dict) -> Catalog:
    models = []
    for item in raw.get("models", []):
        mlx = item.get("mlx")
        models.append(
            CatalogModel(
                tag=item["tag"],
                name=item.get("name", item["tag"]),
                params=item.get("params", ""),
                size_gb=float(item["size_gb"]),
                tier=int(item.get("tier", len(models) + 1)),
                note=item.get("note", ""),
                mlx=Variant(mlx["tag"], float(mlx["size_gb"])) if mlx else None,
            )
        )
    models.sort(key=lambda m: m.tier)
    return Catalog(
        models=tuple(models),
        updated=raw.get("updated", ""),
        source=raw.get("source", ""),
        mlx_min_ollama=raw.get("mlx_min_ollama", ""),
    )


def load_catalog() -> Catalog:
    """Load the user's ``models.json`` if there is a valid one, else the bundled list."""
    override = config_dir() / "models.json"
    if override.is_file():
        try:
            catalog = _parse(json.loads(override.read_text(encoding="utf-8")))
            if catalog.models:
                return catalog
        except (OSError, ValueError, KeyError, TypeError):
            pass
    bundled = resources.files("pictor").joinpath("data/models.json").read_text(encoding="utf-8")
    return _parse(json.loads(bundled))


# --------------------------------------------------------------------------- sizing


def need_gb(size_gb: float) -> float:
    """Memory a model takes while working: weights, vision encoder and context.

    Calibrated against a real run: a 7.7 GB download peaked at 11.1 GB in ``ollama ps``
    while editing a picture with a 16k context.
    """
    return round(size_gb * 1.2 + 1.8, 1)


def fit_for(need: float, budget: float) -> str:
    if need <= budget:
        return "fits"
    if need <= budget * TIGHT_FACTOR:
        return "tight"
    return "too_big"


def suggest_num_ctx(budget_gb: float) -> int:
    """Context window to request. Ollama's own default is only 4k on smaller machines,
    which the picture, the tool list and a few steps of history would overflow."""
    if budget_gb >= 12:
        return 16384
    return 12288 if budget_gb >= 6 else 8192


def normalize_tag(tag: str) -> str:
    """Give ``name`` its implicit ``:latest`` so tags compare the way Ollama resolves them."""
    tag = tag.strip()
    return tag if ":" in tag.rsplit("/", 1)[-1] else f"{tag}:latest"


def version_tuple(version: str | None) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", (version or "").split("-")[0])[:3])


def version_at_least(version: str | None, minimum: str | None) -> bool:
    if not minimum:
        return True
    if not version:
        return False
    return version_tuple(version) >= version_tuple(minimum)


# --------------------------------------------------------------------------- options


def _pick_variant(
    model: CatalogModel, hardware: Hardware, installed: dict[str, InstalledModel], use_mlx: bool
) -> Variant:
    """Choose which build of a catalog model to offer on this machine."""
    standard = Variant(model.tag, model.size_gb)
    variants = [standard]
    if use_mlx and model.mlx:
        variants.insert(0, model.mlx)  # MLX builds are made for Apple Silicon
    for variant in variants:
        if normalize_tag(variant.tag) in installed:
            return variant
    for variant in variants:
        if fit_for(need_gb(variant.size_gb), hardware.budget_gb) == "fits":
            return variant
    return min(variants, key=lambda v: v.size_gb)


def build_options(
    hardware: Hardware,
    installed: list[InstalledModel],
    ollama_version: str | None = None,
    catalog: Catalog | None = None,
) -> list[ModelOption]:
    """List every model the user can pick, sized against this machine, best pick flagged."""
    catalog = catalog or load_catalog()
    by_tag = {normalize_tag(m.tag): m for m in installed}
    use_mlx = hardware.accelerator == "apple-silicon" and version_at_least(
        ollama_version, catalog.mlx_min_ollama
    )

    options: list[ModelOption] = []
    claimed: set[str] = set()
    for model in catalog.models:
        variant = _pick_variant(model, hardware, by_tag, use_mlx)
        have = by_tag.get(normalize_tag(variant.tag))
        caps = list(have.capabilities) if have else ["completion", "vision", "tools", "thinking"]
        need = need_gb(variant.size_gb)
        usable = "tools" in caps
        options.append(
            ModelOption(
                tag=variant.tag,
                name=model.name,
                params=model.params,
                size_gb=variant.size_gb,
                need_gb=need,
                fit=fit_for(need, hardware.budget_gb),
                installed=have is not None,
                usable=usable,
                vision="vision" in caps,
                thinking="thinking" in caps,
                source="catalog",
                score=float(model.tier),
                note=model.note,
                reason="" if usable else "This build cannot call tools.",
                capabilities=caps,
            )
        )
        claimed.add(normalize_tag(variant.tag))

    for key, have in by_tag.items():
        if key in claimed:
            continue
        caps = list(have.capabilities)
        usable = "tools" in caps
        need = need_gb(have.size_gb)
        # Rank an unknown model alongside catalog models of about the same size.
        score = float(max((m.tier for m in catalog.models if m.size_gb <= have.size_gb), default=0))
        options.append(
            ModelOption(
                tag=have.tag,
                name=have.tag,
                params=have.params,
                size_gb=round(have.size_gb, 1),
                need_gb=need,
                fit=fit_for(need, hardware.budget_gb),
                installed=True,
                usable=usable,
                vision="vision" in caps,
                thinking="thinking" in caps,
                source="installed",
                score=score,
                note="Already installed in Ollama.",
                reason="" if usable else "This model cannot call tools.",
                capabilities=caps,
            )
        )

    best = recommend(options)
    if best:
        best.recommended = True
    return options


def recommend(options: list[ModelOption]) -> ModelOption | None:
    """Pick the most capable model that fits; fall back to the smallest if none does."""
    usable = [o for o in options if o.usable]
    if not usable:
        return None
    fitting = [o for o in usable if o.fit == "fits"]
    if not fitting:
        return min(usable, key=lambda o: (not o.vision, o.need_gb))

    def rank(option: ModelOption) -> tuple[float, float]:
        score = option.score + (INSTALLED_BONUS if option.installed else 0)
        if not option.vision:
            score -= 3  # a model that cannot see the picture is a much weaker editor
        return score, -option.need_gb

    return max(fitting, key=rank)
