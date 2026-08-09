"""Re-export catalog images at a usable resolution.

The canonical dataset ships 60x80 thumbnails. At any real card size that's a
4-5x upscale, and no amount of resampling puts back detail that was never
captured.

`benitomartin/fashion-product-images-small-900x1200` is a community re-export of
the same catalog - same ids, same metadata - at 900x1200. Stored at 720px, which
is 2x the widest the UI ever displays, so the browser only ever scales down.

Only image bytes change. build_catalog.py remains the source of truth for which
products are selected and for every metadata field.

Run:  python scripts/upgrade_images.py            # 900x1200 source (~6 GB once)
      python scripts/upgrade_images.py --small    # 384x512 source (~2 GB once)
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCES = {
    "large": "benitomartin/fashion-product-images-small-900x1200",
    "small": "benitomartin/fashion-product-images-small-384x512",
}

CATALOG_PATH = ROOT / "data" / "catalog.csv"
IMAGE_DIR = ROOT / "static" / "products"

# Stored width. 720 px = 2x the widest the UI ever displays (the product detail
# image at ~360 CSS px), so the browser scales DOWN even on a HiDPI screen.
# Downscaling looks clean; upscaling is what made the originals soft. Going
# higher would only add megabytes no display can show.
TARGET_WIDTH = 720
JPEG_QUALITY = 80


def data_files(repo: str) -> list[str]:
    """The parquet shards in a dataset repo, in order."""
    info = HfApi().repo_info(repo, repo_type="dataset")
    return sorted(
        s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--small", action="store_true",
        help="Use the 384x512 source (~2 GB) instead of 900x1200 (~6 GB).",
    )
    args = parser.parse_args()
    repo = SOURCES["small" if args.small else "large"]

    if not CATALOG_PATH.exists():
        raise SystemExit("Run scripts/build_catalog.py first.")

    catalog = pd.read_csv(CATALOG_PATH)
    wanted = {int(v) for v in catalog["id"]}
    print(f"Catalog needs {len(wanted)} images. Source: {repo}")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    replaced = 0
    for filename in data_files(repo):
        if not wanted:
            break
        print(f"\nFetching {filename} (cached after first run)...")
        local = hf_hub_download(repo, filename, repo_type="dataset")

        parquet = pq.ParquetFile(local)
        for group_index in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(group_index, columns=["id", "image"])
            ids = table.column("id").to_pylist()
            images = table.column("image").to_pylist()

            for product_id, blob in zip(ids, images):
                product_id = int(product_id)
                if product_id not in wanted:
                    continue

                image = Image.open(io.BytesIO(blob["bytes"])).convert("RGB")
                if image.width > TARGET_WIDTH:
                    height = round(image.height * TARGET_WIDTH / image.width)
                    image = image.resize(
                        (TARGET_WIDTH, height), resample=Image.Resampling.LANCZOS
                    )
                image.save(
                    IMAGE_DIR / f"{product_id}.jpg",
                    format="JPEG", quality=JPEG_QUALITY, optimize=True,
                    progressive=True,
                )
                wanted.discard(product_id)
                replaced += 1

            print(f"  row group {group_index + 1}/{parquet.metadata.num_row_groups}"
                  f" — {replaced} replaced, {len(wanted)} still needed")

    print(f"\nReplaced {replaced} images at up to {TARGET_WIDTH}px wide.")
    if wanted:
        print(f"{len(wanted)} ids were not present upstream; their existing "
              "images were left in place.")

    total = sum(p.stat().st_size for p in IMAGE_DIR.glob("*.jpg"))
    print(f"static/products is now {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
