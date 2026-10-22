"""Portable offline comparison report. No web dependencies, external assets or telemetry."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

CSS = """
:root{color-scheme:light;--ink:#202321;--muted:#5a615b;--line:#d7dcd5;--paper:#f5f6f2;--red:#a02623;--green:#25633b}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}header,main,footer{max-width:1120px;margin:auto;padding:24px 32px}header{display:flex;justify-content:space-between;border-bottom:1px solid var(--line)}header b{font-size:15px}header span,small{color:var(--muted);font-size:12px}h1{font-size:28px;letter-spacing:-.025em;line-height:1.25;margin:8px 0 12px}h2{font-size:17px;margin:28px 0 8px}h3{font-size:14px;margin:16px 0 4px}p{max-width:78ch;margin:8px 0 16px}.status{color:var(--red);font-weight:650}.pass{color:var(--green)}.identity{font-size:12px;color:var(--muted)}.identity code{overflow-wrap:anywhere}code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}code{font-size:12px}pre{padding:16px;border:1px solid var(--line);background:white;white-space:pre;overflow-x:auto;font-size:12px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}th,td{text-align:right;padding:12px 10px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-size:12px;font-weight:500}.fail td:last-child{color:var(--red);font-weight:650}summary{cursor:pointer;padding:12px 0;font-weight:550}details{border-bottom:1px solid var(--line)}a.download{display:inline-block;padding:8px 12px;border:1px solid var(--ink);color:var(--ink);text-decoration:none;font-size:12px}.case-body{padding:0 0 16px}.case-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.case-body ul{padding-left:20px;font-size:13px}.case-body code{overflow-wrap:anywhere}.note{color:var(--muted);font-size:13px}footer{color:var(--muted);font-size:12px}a:focus-visible,summary:focus-visible{outline:3px solid #406d83;outline-offset:3px}::selection{background:#c9ded1;color:var(--ink)}
@media(max-width:700px){header,main,footer{padding:20px 16px}header{gap:12px;flex-wrap:wrap}h1{font-size:24px}table.metrics thead{display:none}table.metrics tr{display:block;padding:12px 0;border-bottom:1px solid var(--line)}table.metrics td{display:flex;justify-content:space-between;border:0;padding:4px 0;gap:16px}table.metrics td:before{content:attr(data-label);color:var(--muted);font-size:12px}table.metrics td:first-child{font-weight:650}.case-grid{grid-template-columns:1fr;gap:0}.cost-wrap{overflow-x:auto}pre{max-width:100%}}
@media print{body{background:white}header,main,footer{max-width:none}.download{display:none!important}details{break-inside:avoid}pre{white-space:pre-wrap;overflow-wrap:anywhere}}
"""


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def shell(title: str, body: str) -> str:
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)} · RAG Retrieval Gate</title><style>{CSS}</style></head><body><header><b>RAG Retrieval Gate</b><span>Offline comparison · no external assets</span></header><main>{body}</main><footer>Retrieval evidence for the declared dataset and configuration. This does not evaluate generated answer quality or production safety.</footer></body></html>'


def metric_table(rows: list[dict]) -> str:
    labels = ("Metric", "Baseline", "Candidate", "Delta", "Confidence interval", "Result")
    rendered = []
    for row in rows:
        interval = f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]" if row["ci_low"] is not None else "Not estimated"
        values = (
            row["metric"],
            f"{row['baseline']:.4f}",
            f"{row['candidate']:.4f}",
            f"{row['delta']:+.4f}",
            interval,
            row["verdict"],
        )
        cells = "".join(
            f'<td data-label="{label}">{esc(value)}</td>' for label, value in zip(labels, values, strict=True)
        )
        rendered.append(f'<tr class="{"fail" if row["failed"] else "ok"}">{cells}</tr>')
    heads = "".join(f'<th scope="col">{label}</th>' for label in labels)
    return f'<table class="metrics"><thead><tr>{heads}</tr></thead><tbody>{"".join(rendered)}</tbody></table>'


def render_html(result: dict) -> str:
    failed = not result["passed"]
    uncertain = [row["metric"] for row in result["metrics"] if row["verdict"] == "not significant"]
    title = "Candidate regresses" if failed else "Inconclusive metric drops" if uncertain else "No detected regression"
    status = (
        ", ".join(result["regressions"])
        if failed
        else f"Gate passed, but drops are not statistically resolved: {', '.join(uncertain)}"
        if uncertain
        else "Pass for this dataset and gate configuration"
    )
    dataset = result["dataset"]
    identity = result["source_identity"]
    mode = (
        "Threshold only · bootstrap off"
        if not result["bootstrap"]
        else f"Paired bootstrap · {result['bootstrap']['resamples']:,} resamples · seed {result['bootstrap']['seed']} · alpha {result['bootstrap']['alpha']}"
    )
    primary_names = {"hit_at_k", "mrr_at_k", "ndcg_at_k"}
    primary = [row for row in result["metrics"] if row["metric"] in primary_names]
    secondary = [row for row in result["metrics"] if row["metric"] not in primary_names]
    unavailable = sorted(
        set(result["baseline_report"]["metrics"])
        & {"hard_negative_top1_rate", "citation_precision", "answers_fully_supported_rate"}
        - {row["metric"] for row in result["metrics"]}
    )
    source_rows = "".join(
        f"<p>{key.title()}: <code>{esc(identity[key])}</code></p>" for key in ("corpus", "queries", "labels")
    )
    cases = []
    before_by_id = {case["id"]: case for case in result["baseline_report"]["cases"]}
    passed_cases = sum(case["passed"] for case in result["candidate_report"]["cases"])
    for after in result["candidate_report"]["cases"]:
        before = before_by_id[after["id"]]
        panels = []
        for label, case in (("Baseline", before), ("Candidate", after)):
            ranking = "".join(f"<li><code>{esc(doc)}</code></li>" for doc in case["retrieved_ids"])
            panels.append(
                f"<div><h3>{label}</h3><p>MRR: {case['reciprocal_rank']:.4f} · nDCG: {case['ndcg']:.4f} · case check: {'pass' if case['passed'] else 'fail'}</p><ol>{ranking}</ol><p>Missing context: {esc(', '.join(case['missing_context_ids']) or 'None')}</p><p>Citations not retrieved: {esc(', '.join(case['citations_not_retrieved']) or 'None')}</p></div>"
            )
        cases.append(
            f'<details><summary>{esc(after["id"])} · case check {"passed" if after["passed"] else "failed"} · MRR {before["reciprocal_rank"]:.4f} → {after["reciprocal_rank"]:.4f}</summary><div class="case-body"><p>{esc(after.get("query", "Question text unavailable in this saved report."))}</p><p class="note">Labelled relevant: {esc(", ".join(after.get("relevant_ids", [])) or "Not recorded")} · Required context: {esc(", ".join(after.get("required_context_ids", [])) or "Not recorded")}</p><div class="case-grid">{"".join(panels)}</div></div></details>'
        )
    encoded = base64.b64encode(json.dumps(result, indent=2, allow_nan=False).encode("utf-8")).decode("ascii")
    perf_rows = []
    for side in ("baseline", "candidate"):
        performance = (result.get("performance") or {}).get(side)
        if performance:
            perf_rows.append(
                f"<tr><td>{esc(side)}</td><td>{performance['index_build_s']:.2f}</td><td>{performance['query_p50_ms']:.2f}</td><td>{performance['query_p95_ms']:.2f}</td><td>{esc(performance['peak_rss_mb'])}</td></tr>"
            )
    body = f"""<p class="identity">Comparable run reports · schema {result["schema_version"]} · metric schema {result["metric_schema"]} · tool {esc(result["tool_version"])}</p>
<h1>{title}</h1><p class="status {"pass" if not failed else ""}">{esc(status)}</p>
<p>{esc(result["baseline"])} → {esc(result["candidate"])} · {esc(dataset["name"])} · {dataset["documents"]:,} documents · {dataset["cases"]:,} questions · k={result["k"]}</p>
<p class="note">{esc(mode)}. Maximum absolute drop: {result["max_drop"]}. {"Small evaluation: six or fewer questions. Treat this as a smoke test, not an accuracy benchmark." if dataset["cases"] <= 6 else "A pass means no regression detected under this test; it does not prove an improvement."}</p>
<details class="identity"><summary>Matching recorded source identities</summary>{source_rows}<p>Algorithm: {esc(identity["algorithm"])}. Fingerprints bind recorded contents, not authorship: these unsigned reports must come from a trusted run.</p><p>Baseline config: <code>{esc(result["baseline_config_identity"])}</code></p><p>Candidate config: <code>{esc(result["candidate_config_identity"])}</code></p></details>
<h2>Retrieval metrics</h2>{metric_table(primary)}<details><summary>All other metrics ({len(secondary)})</summary>{metric_table(secondary)}<p class="note">Not applicable (no qualifying labels): {esc(", ".join(unavailable) or "None")}.</p></details>
<h2>Question evidence</h2><p>{passed_cases} of {len(cases)} candidate case checks passed. Ranking metrics can regress even when every case passes.</p>{"".join(cases)}
<details><summary>Measured cost · not gated</summary><p class="note">Memory is the process peak, including earlier work in the same process. Compare isolated runs for cost decisions.</p><div class="cost-wrap"><table><thead><tr><th>Run</th><th>Build seconds</th><th>p50 ms</th><th>p95 ms</th><th>Peak RSS MB</th></tr></thead><tbody>{"".join(perf_rows)}</tbody></table></div></details>
<h2>Reproduce the comparison</h2><p class="note">Run from the same working directory with the same local files and tool version. Downloads remain opt-in.</p><pre>{esc(result.get("reproduce") or "Command not recorded. Use the configurations and dataset identity in the JSON.")}</pre><a class="download" href="data:application/json;base64,{encoded}" download="rag-gate-comparison.json">Download comparison JSON</a>"""
    return shell(title, body)


def write_html(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(result), encoding="utf-8")


def write_invalid(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        shell(
            "Invalid comparison",
            f'<h1>Invalid comparison</h1><p class="status">No gate verdict</p><p>{esc(message)}</p><p>Regenerate both run reports from compatible source data and rerun. CLI exit code: 2.</p>',
        ),
        encoding="utf-8",
    )
