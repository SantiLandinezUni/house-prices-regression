#!/usr/bin/env python
"""Build the assignment .zip: report + notebook + best submission CSV.

Usage:
    python package_submission.py <kaggle_user_id>

Checks every file exists and that the CSV matches the format Kaggle expects
before writing <kaggle_user_id>.zip in the project root.
"""

import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def find_report():
    """The report, in whatever format it was written."""
    for pattern in ("*.pdf", "*.docx", "*.md"):
        found = sorted((ROOT / "report").glob(pattern))
        if found:
            return found[0]
    return None


def find_submission():
    """The submission CSV. If several exist, the one with the best CV score wins."""
    candidates = sorted((ROOT / "submissions").glob("*.csv"))
    if not candidates:
        return None
    scored = [(float(p.stem.split("cv")[-1]), p) for p in candidates if "cv" in p.stem]
    return min(scored)[1] if scored else candidates[0]


def check_submission(path):
    """Verify the CSV is one Kaggle will accept."""
    sub = pd.read_csv(path)
    sample = pd.read_csv(ROOT / "data" / "raw" / "sample_submission.csv")

    problems = []
    if list(sub.columns) != ["Id", "SalePrice"]:
        problems.append(f"columns are {list(sub.columns)}, expected ['Id', 'SalePrice']")
    if len(sub) != 1459:
        problems.append(f"{len(sub)} rows, expected 1459")
    if not sub["Id"].equals(sample["Id"]):
        problems.append("Id column does not match sample_submission.csv")
    if sub["SalePrice"].isna().any():
        problems.append("SalePrice contains missing values")
    if (sub["SalePrice"] <= 0).any():
        problems.append("SalePrice contains non-positive values")
    if sub["SalePrice"].max() < 1000:
        problems.append("SalePrice looks like log prices - did expm1 get skipped?")
    return problems


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python package_submission.py <kaggle_user_id>")
    kaggle_id = sys.argv[1].strip()

    report = find_report()
    notebook = ROOT / "notebooks" / "01_house_prices.ipynb"
    submission = find_submission()

    missing = []
    if report is None:
        missing.append("report/  (no .pdf, .docx or .md found)")
    if not notebook.exists():
        missing.append(str(notebook.relative_to(ROOT)))
    if submission is None:
        missing.append("submissions/  (no .csv found)")
    if missing:
        sys.exit("missing:\n  " + "\n  ".join(missing))

    problems = check_submission(submission)
    if problems:
        sys.exit("submission CSV failed validation:\n  " + "\n  ".join(problems))

    out = ROOT / f"{kaggle_id}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in (report, notebook, submission):
            zf.write(path, arcname=path.name)

    print(f"created {out.name}  ({out.stat().st_size / 1024:.0f} KB)\n")
    with zipfile.ZipFile(out) as zf:
        for info in zf.infolist():
            print(f"  {info.filename:<40} {info.file_size / 1024:>8.0f} KB")


if __name__ == "__main__":
    main()
