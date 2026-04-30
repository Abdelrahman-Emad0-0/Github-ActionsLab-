"""Train a baseline Random Forest model."""

import argparse
import json
import os
import pickle
from pathlib import Path

import pandas as pd
import toml
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


class DataConfig(BaseModel):
    cleaned_data_path: str
    train_feature_columns: list[str]
    target_column: str
    test_size: float
    random_state: int


class ModelConfig(BaseModel):
    n_estimators: int
    max_depth: int
    model_output_path: str


class ReportsConfig(BaseModel):
    metrics_path: str


class AppConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    reports: ReportsConfig


def load_config(filepath: str) -> AppConfig:
    raw_config = toml.load(filepath)
    try:
        return AppConfig.model_validate(raw_config)
    except AttributeError:
        return AppConfig.parse_obj(raw_config)


def load_data(filepath: str, feature_columns: list[str], target_column: str):
    df = pd.read_csv(filepath)
    X = df[feature_columns]
    y = df[target_column]
    return X, y


def train_model(X_train, y_train, config: AppConfig) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=config.model.n_estimators,
        max_depth=config.model.max_depth,
        random_state=config.data.random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }


def save_model(model, filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved -> {filepath}")


def save_metrics(metrics: dict, filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {filepath}")
    print(f"  accuracy : {metrics['accuracy']}")
    print(f"  f1_score : {metrics['f1_score']}")


if __name__ == "__main__":
    model_version = os.getenv("MODEL_VERSION", "v1")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.toml")
    args = parser.parse_args()

    config = load_config(args.config)

    X, y = load_data(
        config.data.cleaned_data_path,
        config.data.train_feature_columns,
        config.data.target_column,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.data.test_size, random_state=config.data.random_state
    )

    model = train_model(X_train, y_train, config)
    metrics = evaluate_model(model, X_test, y_test)

    model_dir = config.model.model_output_path.rsplit("/", 1)[0]
    model_path = f"{model_dir}/model_{model_version}.pkl"
    save_model(model, model_path)
    save_metrics(metrics, config.reports.metrics_path)
