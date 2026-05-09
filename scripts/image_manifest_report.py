from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.manifests import load_image_manifest


def main() -> None:
    images = load_image_manifest()
    existing = [image for image in images if (ROOT / image.local_path).exists()]
    ready = [image for image in images if image.manifest_ready]
    report = {
        "image_count": len(images),
        "manifest_ready_count": len(ready),
        "existing_file_count": len(existing),
        "split_groups": Counter(image.split_group for image in images).most_common(),
        "hazard_types": Counter(image.hazard_type for image in images).most_common(),
        "licenses": Counter(image.license for image in images).most_common(),
        "missing_files": [image.local_path for image in images if image.manifest_ready and not (ROOT / image.local_path).exists()],
        "placeholder_rows": [
            image.image_id
            for image in images
            if not image.manifest_ready
            or image.source_url.startswith("REPLACE_")
            or image.license.startswith("REPLACE_")
            or image.retrieved_at == "TBD"
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
