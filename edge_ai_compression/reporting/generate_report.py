"""Generate a static HTML report from experiment / demo / benchmark outputs.

    python -m edge_ai_compression.reporting.generate_report \
        --demo results/demo/report.json \
        --benchmark results/benchmark.json \
        --pareto results/pareto/pareto_frontier.csv \
        --profile raspberry_pi \
        --out reports/index.html

Pure Python, no web framework. Every input is optional; the report renders
whatever is available. A JSON summary is written next to the HTML.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

from edge_ai_compression.utils.metrics_io import extract_metrics, load_json

_DISCLAIMER = (
    "Numbers measured on the synthetic <code>fake</code> dataset (random labels) "
    "verify plumbing only — they are NOT real model quality. Latency/size/RAM are "
    "real measurements of the model's compute; accuracy on synthetic data is not."
)

_CSS = """
:root { --fg:#1a1a1a; --muted:#666; --bg:#fff; --accent:#ee4c2c; --line:#e2e2e2; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       color: var(--fg); background: var(--bg); margin: 0; padding: 2rem;
       max-width: 900px; margin: 0 auto; line-height: 1.5; }
h1 { margin-bottom: 0.2rem; }
h2 { border-bottom: 2px solid var(--line); padding-bottom: 0.3rem; margin-top: 2rem; }
.sub { color: var(--muted); margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--line); }
th { background: #fafafa; }
.disclaimer { background: #fff8e1; border-left: 4px solid #f0ad4e; padding: 0.75rem 1rem;
              border-radius: 4px; margin: 1rem 0; }
.badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 10px;
         font-size: 0.8rem; font-weight: 600; }
.ok { background: #e6f4ea; color: #137333; }
.bad { background: #fce8e6; color: #c5221f; }
code { background: #f3f3f3; padding: 0.1rem 0.3rem; border-radius: 3px; }
footer { color: var(--muted); font-size: 0.85rem; margin-top: 3rem;
         border-top: 1px solid var(--line); padding-top: 1rem; }
"""


def _read_pareto_csv(path: Path, limit: int = 25) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    return rows[0], rows[1 : 1 + limit]


def collect_inputs(
    *,
    demo: Path | None = None,
    benchmark: Path | None = None,
    pareto: Path | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    """Load whichever inputs exist into a structured dict (missing → absent key)."""
    collected: dict[str, Any] = {}

    if demo and demo.is_file():
        collected["demo"] = load_json(demo)
    if benchmark and benchmark.is_file():
        collected["benchmark"] = load_json(benchmark)
    if pareto and pareto.is_file():
        header, rows = _read_pareto_csv(pareto)
        collected["pareto"] = {"header": header, "rows": rows}

    # Hardware scoring: prefer benchmark metrics, else demo metrics.
    if profile:
        metrics_source = collected.get("benchmark") or collected.get("demo")
        if metrics_source is not None:
            from edge_ai_compression.hardware.scoring import score_candidate

            metrics = extract_metrics(metrics_source)
            score_input = {
                "accuracy": metrics["accuracy"],
                "latency_ms": metrics["latency_ms"],
                "size_mb": metrics["size_mb"],
                "ram_mb": metrics["ram_mb"],
            }
            result = score_candidate(score_input, profile)
            collected["hardware"] = {"profile": profile, "result": result.to_dict()}

    return collected


def _tbl(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _fmt(val: Any, spec: str = "") -> str:
    if val is None:
        return "—"
    if spec and isinstance(val, (int, float)):
        return format(val, spec)
    return str(val)


def _demo_section(demo: dict[str, Any]) -> str:
    b = demo.get("baseline", {})
    c = demo.get("compressed", {})
    comp = demo.get("compression", {})
    rows = [
        [
            "baseline",
            _fmt(b.get("num_parameters"), ",d"),
            _fmt(b.get("size_mb"), ".3f"),
            _fmt((b.get("latency_ms") or {}).get("mean"), ".3f"),
            _fmt(b.get("synthetic_accuracy"), ".3f"),
        ],
        [
            "compressed",
            _fmt(c.get("num_parameters"), ",d"),
            _fmt(c.get("size_mb"), ".3f"),
            _fmt((c.get("latency_ms") or {}).get("mean"), ".3f"),
            _fmt(c.get("synthetic_accuracy"), ".3f"),
        ],
    ]
    table = _tbl(["Model", "Params", "Size (MB)", "Latency (ms)", "Synth. acc"], rows)
    summary = (
        f"<p>Size ratio <b>{_fmt(comp.get('size_ratio'))}×</b> &nbsp;·&nbsp; "
        f"Latency speedup <b>{_fmt(comp.get('latency_speedup'))}×</b> &nbsp;·&nbsp; "
        f"Param reduction <b>{_fmt((comp.get('param_reduction') or 0) * 100, '.1f')}%</b> "
        f"&nbsp;·&nbsp; Sparsity <b>{_fmt(comp.get('sparsity_after_pruning'), '.3f')}</b></p>"
    )
    return f"<h2>Compression demo (baseline vs compressed)</h2>{table}{summary}"


def _benchmark_section(bench: dict[str, Any]) -> str:
    lat = bench.get("latency_ms", {})
    rows = [
        ["mean", _fmt(lat.get("mean"), ".3f")],
        ["p50", _fmt(lat.get("p50"), ".3f")],
        ["p95", _fmt(lat.get("p95"), ".3f")],
        ["p99", _fmt(lat.get("p99"), ".3f")],
    ]
    lat_tbl = _tbl(["Latency (ms)", "Value"], rows)
    meta = [
        ["Parameters", _fmt(bench.get("num_parameters"), ",d")],
        ["Size (MB)", _fmt(bench.get("size_mb"), ".3f")],
        ["FLOPs (est.)", _fmt(bench.get("flops_estimate"), ",.0f")],
        ["Peak RAM (MiB)", _fmt(bench.get("peak_ram_mib"), ".1f")],
        ["Throughput (inf/s)", _fmt(bench.get("throughput_ips"), ".1f")],
    ]
    meta_tbl = _tbl(["Metric", "Value"], meta)
    src = html.escape(str(bench.get("source", "")))
    return f"<h2>Benchmark ({src})</h2>{lat_tbl}{meta_tbl}"


def _hardware_section(hw: dict[str, Any]) -> str:
    r = hw["result"]
    badge = (
        '<span class="badge ok">FEASIBLE</span>'
        if r["feasible"]
        else ('<span class="badge bad">INFEASIBLE</span>')
    )
    util = r.get("utilization", {})
    util_rows = [[k, _fmt(v * 100, ".1f") + "%"] for k, v in util.items()]
    util_tbl = _tbl(["Budget", "Utilization"], util_rows) if util_rows else ""
    violations = "".join(f"<li>{html.escape(v)}</li>" for v in r.get("violations", []))
    viol_html = f"<p>Violations:</p><ul>{violations}</ul>" if violations else ""
    return (
        f"<h2>Hardware feasibility — {html.escape(hw['profile'])}</h2>"
        f"<p>{badge} &nbsp; score <b>{_fmt(r['score'], '.4f')}</b></p>"
        f"{util_tbl}{viol_html}"
    )


def _pareto_section(pareto: dict[str, Any]) -> str:
    header, rows = pareto["header"], pareto["rows"]
    if not header:
        return "<h2>Pareto frontier</h2><p>No frontier points.</p>"
    return f"<h2>Pareto frontier ({len(rows)} points)</h2>{_tbl(header, rows)}"


def build_report_html(collected: dict[str, Any]) -> str:
    """Render the collected inputs into a full HTML document string."""
    sections: list[str] = []
    if "demo" in collected:
        sections.append(_demo_section(collected["demo"]))
    if "benchmark" in collected:
        sections.append(_benchmark_section(collected["benchmark"]))
    if "hardware" in collected:
        sections.append(_hardware_section(collected["hardware"]))
    if "pareto" in collected:
        sections.append(_pareto_section(collected["pareto"]))
    if not sections:
        sections.append(
            "<p>No inputs were provided. Pass at least one of "
            "<code>--demo</code>, <code>--benchmark</code>, or "
            "<code>--pareto</code>.</p>"
        )

    body = "\n".join(sections)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Edge AI Compression Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Edge AI Compression Report</h1>
<p class="sub">Hardware-aware neural network compression &amp; deployment</p>
<div class="disclaimer">{_DISCLAIMER}</div>
{body}
<footer>Generated by <code>edge_ai_compression.reporting.generate_report</code>.
Latency/size/RAM are real; synthetic-data accuracy is not a quality metric.</footer>
</body>
</html>
"""


def _json_summary(collected: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"sections": sorted(collected.keys())}
    metrics_source = collected.get("benchmark") or collected.get("demo")
    if metrics_source is not None:
        summary["metrics"] = extract_metrics(metrics_source)
    if "hardware" in collected:
        summary["hardware"] = collected["hardware"]
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m edge_ai_compression.reporting.generate_report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--demo", type=Path, default=None, help="Demo report.json.")
    p.add_argument("--benchmark", type=Path, default=None, help="Benchmark report JSON.")
    p.add_argument("--pareto", type=Path, default=None, help="Pareto frontier CSV.")
    p.add_argument("--profile", default=None, help="Score metrics against this hardware profile.")
    p.add_argument("--out", type=Path, default=Path("reports/index.html"), help="Output HTML path.")
    args = p.parse_args(argv)

    collected = collect_inputs(
        demo=args.demo, benchmark=args.benchmark, pareto=args.pareto, profile=args.profile
    )
    if not collected:
        print("Warning: no inputs found — writing an empty report shell.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_report_html(collected), encoding="utf-8")
    json_path = args.out.with_suffix(".summary.json")
    json_path.write_text(json.dumps(_json_summary(collected), indent=2), encoding="utf-8")

    print(f"Wrote {args.out}")
    print(f"Wrote {json_path}")
    print(f"Sections: {', '.join(sorted(collected)) or '(none)'}")


if __name__ == "__main__":
    main()
