"""
Scan a directory for ``sp_baseline_partially_observed_history_<n>.json`` files
and write the last hypothesis program from each run's history (final step with
a ``hypothesis`` field, same rule as the driver).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any, Optional

HISTORY_STEM = "sp_baseline_partially_observed_history"
_FILE_RE = re.compile(rf"^{re.escape(HISTORY_STEM)}_(\d+)\.json$")


def last_hypothesis_from_history(history: list[dict[str, Any]]) -> Optional[str]:
    for entry in reversed(history):
        h = entry.get("hypothesis")
        if h is not None:
            return h
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scan_dir",
        type=pathlib.Path,
        nargs="?",
        default=pathlib.Path(__file__).resolve().parent.parent,
        help="Directory to scan (default: smc/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output JSON path (default: <scan_dir>/sp_baseline_partially_observed_last_hypotheses_extracted.json)",
    )
    parser.add_argument(
        "--text-output",
        type=pathlib.Path,
        default=None,
        help="Plain-text path: last hypotheses only, separated by blank lines (default: same basename as JSON with .txt)",
    )
    args = parser.parse_args()
    scan_dir: pathlib.Path = args.scan_dir.resolve()
    out_path = args.output or (scan_dir / "sp_baseline_partially_observed_last_hypotheses_extracted.json")
    text_path = args.text_output
    if text_path is None:
        text_path = out_path.with_suffix(".txt")

    matches: list[tuple[int, pathlib.Path]] = []
    for p in scan_dir.iterdir():
        m = _FILE_RE.match(p.name)
        if m:
            matches.append((int(m.group(1)), p))
    matches.sort(key=lambda t: t[0])

    rows: list[dict[str, Any]] = []
    for file_index, p in matches:
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("meta") or {}
        run_number = meta.get("run_number", file_index)
        history = data.get("history") or []
        rows.append(
            {
                "source_file": p.name,
                "file_index": file_index,
                "run_number": run_number,
                "solved": data.get("solved"),
                "trials": data.get("trials"),
                "opened": data.get("opened"),
                "last_hypothesis": last_hypothesis_from_history(history),
            }
        )

    rows.sort(key=lambda r: (int(r["run_number"]), r["file_index"]))
    payload = {"scan_dir": str(scan_dir), "count": len(rows), "runs": rows}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path} ({len(rows)} history files)")

    hyps = [r["last_hypothesis"] for r in rows if r.get("last_hypothesis")]
    text_body = "\n\n".join(hyps)
    text_path = text_path.resolve()
    with text_path.open("w", encoding="utf-8") as f:
        f.write(text_body)
    print(f"Wrote {text_path} ({len(hyps)} hypotheses, text only)")


if __name__ == "__main__":
    main()
