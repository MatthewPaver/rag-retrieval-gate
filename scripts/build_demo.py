"""Build the public evidence explorer from real offline CLI runs. No model downloads."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(output: Path) -> None:
    """Keep the browser a viewer of engine evidence, never a second gate implementation."""
    output.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "product.css", "product.js"):
        source = ROOT / "web" / name
        if source.resolve() != (output / name).resolve():
            shutil.copyfile(source, output / name)
    fixture = json.loads((ROOT / "fixtures/policies.json").read_text())
    bundle: dict = {"version": 1, "documents": fixture["corpus"], "scenarios": {}}
    for key, config, expected, stem in (
        ("regression", "candidate-chunk-12", 1, "comparison"),
        ("unchanged", "candidate-bm25-b05", 0, "unchanged"),
    ):
        html_name = "report.html" if key == "regression" else "unchanged-report.html"
        # Stable relative commands remain copyable from the repository root.
        command = [
            "gate",
            "--dataset",
            "fixtures/policies.json",
            "--k",
            "3",
            "--baseline",
            "configs/baseline.json",
            "--candidate",
            f"configs/{config}.json",
            "--no-bootstrap",
            "--output",
            f"reports/{stem}.json",
            "--html",
            f"reports/{html_name}",
        ]
        # CLI owns validation, comparison and the portable HTML, including its reproduction command.
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            shutil.copytree(ROOT / "fixtures", workspace / "fixtures")
            shutil.copytree(ROOT / "configs", workspace / "configs")
            result = subprocess.run(
                [sys.executable, "-m", "rag_gate", *command], cwd=workspace, capture_output=True, text=True
            )
            if result.returncode != expected:
                raise RuntimeError(f"{key}: expected CLI exit {expected}, got {result.returncode}: {result.stderr}")
            report = json.loads((workspace / f"reports/{stem}.json").read_text())
            shutil.copyfile(workspace / f"reports/{stem}.json", output / f"{stem}.json")
            shutil.copyfile(workspace / f"reports/{html_name}", output / html_name)
        bundle["scenarios"][key] = report
    (output / "evidence.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(ROOT / "fixtures/policies.json", output / "policies.json")
    for name in ("baseline", "candidate-chunk-12", "candidate-bm25-b05"):
        shutil.copyfile(ROOT / f"configs/{name}.json", output / f"{name}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "web")
    build(parser.parse_args().output.resolve())
