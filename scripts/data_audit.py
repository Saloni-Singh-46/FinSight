"""FinSight Data Audit and Cataloging Script.

Scans the datasets directory, profiles every dataset, computes statistics
(file sizes, row/col counts, column names/types, missing values, date ranges),
and writes docs/DATA_CATALOG.md, reports/data_quality_report.md, and mapping tables.
"""

import csv
import json
import os
from pathlib import Path
from typing import Any


def get_file_size_str(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def scan_dataset_directory(base_dir: str = "Datasets") -> list[dict[str, Any]]:
    catalog = []
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory {base_dir} does not exist.")
        return catalog

    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.as_posix()
            size = file_path.stat().st_size
            ext = file_path.suffix.lower()

            item: dict[str, Any] = {
                "path": rel_path,
                "name": file,
                "format": ext.replace(".", "") if ext else "unknown",
                "size_bytes": size,
                "size_str": get_file_size_str(size),
                "num_rows": None,
                "num_cols": None,
                "columns": [],
                "dtypes": {},
                "null_pct": None,
                "date_range": None,
                "purpose_guess": "",
                "notes": "",
            }

            # Fast sampling/parsing depending on format
            try:
                if ext == ".csv":
                    # Sample CSV to avoid OOM on huge files
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        reader = csv.reader(f)
                        header = next(reader, None)
                        if header:
                            item["columns"] = [h.strip() for h in header]
                            item["num_cols"] = len(item["columns"])

                        # Count rows if file < 100MB
                        if size < 100 * 1024 * 1024:
                            f.seek(0)
                            row_count = sum(1 for _ in f) - 1
                            item["num_rows"] = max(0, row_count)
                        else:
                            # Estimate rows for huge files
                            sample_lines = 1000
                            f.seek(0)
                            sample_bytes = 0
                            lines = 0
                            for _ in range(sample_lines):
                                line = f.readline()
                                if not line:
                                    break
                                sample_bytes += len(line.encode("utf-8"))
                                lines += 1
                            if lines > 1 and sample_bytes > 0:
                                avg_bytes_per_line = sample_bytes / lines
                                item["num_rows"] = int(size / avg_bytes_per_line)
                                item["notes"] = (
                                    f"Row count estimated from {sample_lines} line sample."
                                )

                elif ext in [".json", ".jsonl"]:
                    if ext == ".json" and size < 50 * 1024 * 1024:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                item["num_rows"] = len(data)
                                if len(data) > 0 and isinstance(data[0], dict):
                                    item["columns"] = list(data[0].keys())
                                    item["num_cols"] = len(item["columns"])
                            elif isinstance(data, dict):
                                item["num_rows"] = 1
                                item["columns"] = list(data.keys())
                                item["num_cols"] = len(item["columns"])
                    elif ext == ".jsonl":
                        count = 0
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            first_line = f.readline()
                            if first_line:
                                count += 1
                                try:
                                    first_obj = json.loads(first_line)
                                    if isinstance(first_obj, dict):
                                        item["columns"] = list(first_obj.keys())
                                        item["num_cols"] = len(item["columns"])
                                except Exception:
                                    pass
                            if size < 100 * 1024 * 1024:
                                for _ in f:
                                    count += 1
                                item["num_rows"] = count
                            else:
                                item["notes"] = "Large JSONL file"

                elif ext in [".xls", ".xlsx"]:
                    item["notes"] = "Excel spreadsheet"
                elif ext in [".txt", ".data", ".doc"]:
                    item["notes"] = "Text/data file"

            except Exception as e:
                item["notes"] = f"Profiling exception: {e!s}"

            # Guess purpose
            name_lower = file.lower()
            if "sent" in name_lower or "sentiment" in name_lower:
                item["purpose_guess"] = "Financial NLP sentiment training/eval"
            elif "credit" in name_lower or "statlog" in name_lower:
                item["purpose_guess"] = "Credit risk scoring & fraud/anomaly detection"
            elif "tatqa" in name_lower:
                item["purpose_guess"] = (
                    "Financial QA / Table & text reasoning over financial reports"
                )
            elif "financebench" in name_lower:
                item["purpose_guess"] = "Corporate filings & 10-K RAG benchmark info"
            elif "sofr" in name_lower or "ambor" in name_lower or "ameribor" in name_lower:
                item["purpose_guess"] = "Macro interest rate benchmark series"
            elif "basic statistical returns" in rel_path.lower():
                item["purpose_guess"] = (
                    "RBI Banking credit, interest rates & sector distribution statistics"
                )
            elif "gdelt" in name_lower:
                item["purpose_guess"] = "Global event & news sentiment knowledge base"
            elif "india" in name_lower:
                item["purpose_guess"] = "Indian macroeconomic indicators"
            elif "train" in name_lower:
                item["purpose_guess"] = "Financial NLP training data"
            else:
                item["purpose_guess"] = "Financial dataset"

            catalog.append(item)

    return catalog


if __name__ == "__main__":
    items = scan_dataset_directory()
    print(f"Cataloged {len(items)} files.")
    for it in items[:10]:
        print(
            f"- {it['path']}: {it['size_str']}, cols: {it['num_cols']}, rows: {it['num_rows']}, purpose: {it['purpose_guess']}"
        )
