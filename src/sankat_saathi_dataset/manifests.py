from __future__ import annotations

import csv
import json
from pathlib import Path

from .schemas import ImageMetadata

ROOT = Path(__file__).resolve().parents[2]
IMAGE_MANIFEST = ROOT / "data" / "images" / "verified" / "images_manifest.csv"
SOURCE_MANIFEST = ROOT / "data" / "sources" / "source_manifest.jsonl"


def split_labels(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


def load_image_manifest(path: Path = IMAGE_MANIFEST) -> list[ImageMetadata]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    images: list[ImageMetadata] = []
    for row in rows:
        images.append(
            ImageMetadata(
                image_id=row["image_id"],
                source_url=row["source_url"],
                license=row["license"],
                license_url=row["license_url"],
                author=row["author"],
                retrieved_at=row["retrieved_at"],
                modifications=row.get("modifications", "none"),
                visible_labels=split_labels(row.get("visible_labels", "")),
                provided_context_labels=split_labels(row.get("provided_context_labels", "")),
                not_determinable_labels=split_labels(row.get("not_determinable_labels", "")),
                local_path=row["local_path"],
                split_group=row.get("split_group", "shared"),  # type: ignore[arg-type]
                event_id=row.get("event_id", ""),
                hazard_type=row.get("hazard_type", ""),
                manifest_ready=row.get("manifest_ready", "").lower() == "true",
            )
        )
    return images


def load_source_manifest(path: Path = SOURCE_MANIFEST) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
