# src/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Создаёт препроцессор:
    - стандартизация числовых признаков
    - категориальные признаки можно добавить позже
    """

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)
        ]
    )

    return preprocessor


def split_data(df: pd.DataFrame, target_col: str, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
