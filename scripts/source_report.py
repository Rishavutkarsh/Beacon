from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sankat_saathi_dataset.manifests import load_source_manifest


def main() -> None:
    rows = load_source_manifest()
    report = {
        "source_count": len(rows),
        "ready_count": sum(1 for row in rows if row.get("source_ready")),
        "organizations": Counter(row.get("organization", "") for row in rows).most_common(),
        "jurisdictions": Counter(row.get("jurisdiction", "") for row in rows).most_common(),
        "not_ready": [row.get("source_id") for row in rows if not row.get("source_ready")],
        "missing_fields": {
            row.get("source_id", f"row_{idx}"): [
                field
                for field in ["url", "title", "organization", "published_at", "accessed_at", "jurisdiction", "source_section", "usage_notes"]
                if not row.get(field)
            ]
            for idx, row in enumerate(rows)
            if any(not row.get(field) for field in ["url", "title", "organization", "published_at", "accessed_at", "jurisdiction", "source_section", "usage_notes"])
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
