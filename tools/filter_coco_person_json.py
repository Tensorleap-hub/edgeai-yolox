#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict


def filter_json(src: Path, dst: Path):
    with src.open("r") as f:
        data: Dict[str, Any] = json.load(f)

    keep_cid = 1  # COCO person
    anns = [a for a in data.get("annotations", []) if a.get("category_id") == keep_cid]

    # Keep all images; images without persons will just have zero annotations
    imgs = data.get("images", [])
    valid_img_ids = {img["id"] for img in imgs}

    # Drop annotations whose image_id is missing from images
    anns = [a for a in anns if a["image_id"] in valid_img_ids]

    data["categories"] = [{"supercategory": "person", "id": 1, "name": "person"}]
    data["annotations"] = anns
    data["images"] = imgs

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        json.dump(data, f)

    print(f"Wrote {dst} with {len(imgs)} images, {len(anns)} annotations")


def main():
    ap = argparse.ArgumentParser(description="Filter COCO JSON to person-only.")
    ap.add_argument("--src", type=Path, required=True, help="Source COCO JSON")
    ap.add_argument("--dst", type=Path, required=True, help="Destination JSON")
    args = ap.parse_args()
    filter_json(args.src, args.dst)


if __name__ == "__main__":
    main()
