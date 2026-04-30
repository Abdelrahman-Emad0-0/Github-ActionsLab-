"""Generate a markdown report for the full ML pipeline."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import toml
from pydantic import BaseModel


class ReportsConfig(BaseModel):
    validation_raw_path: str
    validation_cleaned_path: str
    cleaning_log_path: str
    feature_log_path: str
    metrics_path: str
    classification_metrics_path: str
    pipeline_report_path: str


class AppConfig(BaseModel):
    reports: ReportsConfig


def load_config(filepath: str) -> AppConfig:
    raw_config = toml.load(filepath)
    try:
        return AppConfig.model_validate(raw_config)
    except AttributeError:
        return AppConfig.parse_obj(raw_config)


def load_json(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def render_report(
    raw_validation: dict,
    cleaned_validation: dict,
    cleaning_log: dict,
    feature_log: dict,
    baseline_metrics: dict,
    classification_metrics: dict,
) -> str:
    return f"""# End-to-End ML Pipeline Report

Generated at: {datetime.utcnow().isoformat()} UTC

## 1) Validation Before Cleaning
- Rows: {raw_validation.get("row_count")}
- Duplicate rows: {raw_validation.get("duplicate_rows")}
- Missing values total: {sum(raw_validation.get("missing_by_column", {}).values())}

## 2) Cleaning Summary
- Duplicates removed: {cleaning_log.get("duplicates_removed")}
- Final cleaned rows: {cleaning_log.get("final_row_count")}

## 3) Validation After Cleaning
- Rows: {cleaned_validation.get("row_count")}
- Duplicate rows: {cleaned_validation.get("duplicate_rows")}
- Missing values total: {sum(cleaned_validation.get("missing_by_column", {}).values())}

## 4) Feature Engineering
- Input rows: {feature_log.get("input_rows")}
- Output rows: {feature_log.get("output_rows")}
- Output feature count: {feature_log.get("output_feature_count")}

## 5) Baseline Training (Random Forest)
- Accuracy: {baseline_metrics.get("accuracy")}
- F1 Score: {baseline_metrics.get("f1_score")}

## 6) Classifier Benchmark
- Best model: {classification_metrics.get("best_model")}
- Best F1 score: {classification_metrics.get("best_f1_score")}

## 7) Notes
- All artifacts are stored under `reports/` and `models/`.
- This report is generated automatically from pipeline artifacts.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.toml")
    args = parser.parse_args()

    app_config = load_config(args.config)
    r = app_config.reports

    report_text = render_report(
        load_json(r.validation_raw_path),
        load_json(r.validation_cleaned_path),
        load_json(r.cleaning_log_path),
        load_json(r.feature_log_path),
        load_json(r.metrics_path),
        load_json(r.classification_metrics_path),
    )
    Path(r.pipeline_report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(r.pipeline_report_path).write_text(report_text, encoding="utf-8")
    print(f"Pipeline report saved -> {r.pipeline_report_path}")
