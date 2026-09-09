#!/usr/bin/env python3
"""Create a deterministic COCO subset annotation and curl download config."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3576)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--curl-config", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.annotations.read_text(encoding="utf-8"))
    images = source["images"]
    if not 0 < args.count <= len(images):
        parser.error(f"count must be between 1 and {len(images)}")

    selected_ids = set(random.Random(args.seed).sample([row["id"] for row in images], args.count))
    selected_images = sorted((row for row in images if row["id"] in selected_ids), key=lambda row: row["id"])
    selected_annotations = [row for row in source["annotations"] if row["image_id"] in selected_ids]

    subset = {
        "info": source.get("info", {}),
        "licenses": source.get("licenses", []),
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": source["categories"],
    }
    args.output_annotations.parent.mkdir(parents=True, exist_ok=True)
    args.curl_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_annotations.write_text(json.dumps(subset, separators=(",", ":")), encoding="utf-8")

    curl_lines: list[str] = []
    for row in selected_images:
        url = row.get("coco_url") or f"http://images.cocodataset.org/val2017/{row['file_name']}"
        curl_lines.extend((f'url = "{url}"', "remote-name"))
    args.curl_config.write_text("\n".join(curl_lines) + "\n", encoding="utf-8")

    metadata = {
        "source_annotations": str(args.annotations.resolve()),
        "source_annotations_sha256": sha256(args.annotations),
        "selection": "Python random.Random(seed).sample over COCO image IDs, then sorted by image ID",
        "seed": args.seed,
        "images": len(selected_images),
        "annotations": len(selected_annotations),
        "first_image_ids": [row["id"] for row in selected_images[:10]],
        "output_annotations": str(args.output_annotations.resolve()),
        "output_annotations_sha256": sha256(args.output_annotations),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
