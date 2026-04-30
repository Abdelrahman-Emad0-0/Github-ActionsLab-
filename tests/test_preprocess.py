import pandas as pd
import pytest

from src.data.preprocess import clean_data


class _Cfg:
    target_column = "depression_label"
    numeric_columns = ["age", "stress_level"]
    categorical_columns = ["gender"]


def test_clean_data_removes_duplicates():
    df = pd.DataFrame({
        "age": [20, 20, 30],
        "stress_level": [5, 5, 3],
        "gender": ["male", "male", "female"],
        "depression_label": [0, 0, 1],
    })
    cleaned, log = clean_data(df, _Cfg())
    assert log["duplicates_removed"] == 1
    assert len(cleaned) == 2


def test_clean_data_imputes_numeric_missing():
    df = pd.DataFrame({
        "age": [20.0, None, 30.0],
        "stress_level": [5.0, 5.0, 3.0],
        "gender": ["male", "female", "female"],
        "depression_label": [0, 1, 1],
    })
    cleaned, log = clean_data(df, _Cfg())
    assert cleaned["age"].isna().sum() == 0
    assert log["age_missing_imputed"] == 1


def test_clean_data_imputes_categorical_missing():
    df = pd.DataFrame({
        "age": [20.0, 25.0, 30.0],
        "stress_level": [5.0, 4.0, 3.0],
        "gender": ["male", None, "female"],
        "depression_label": [0, 1, 1],
    })
    cleaned, log = clean_data(df, _Cfg())
    assert cleaned["gender"].isna().sum() == 0
    assert log["gender_missing_imputed"] == 1


def test_clean_data_final_row_count_in_log():
    df = pd.DataFrame({
        "age": [20.0, 25.0],
        "stress_level": [5.0, 3.0],
        "gender": ["male", "female"],
        "depression_label": [0, 1],
    })
    cleaned, log = clean_data(df, _Cfg())
    assert log["final_row_count"] == len(cleaned)
