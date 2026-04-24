from __future__ import annotations

from pathlib import Path
from typing import Any


def save_tradeoff_csv(path: str | Path, rows: list[dict[str, Any]], keys: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(keys)
    lines = [header]
    for r in rows:
        lines.append(",".join(str(r.get(k, "")) for k in keys))
    path.write_text("\n".join(lines), encoding="utf-8")
