import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score

DEFAULT_TARGET = "Montant"  # fallback if user does not specify


def get_target_column(df: pd.DataFrame, target: str = None) -> str:
    """Return the target column name. If `target` is None, try to infer.
    """
    if target and target in df.columns:
        return target
    # infer common numeric target candidates
    candidates = [col for col in df.select_dtypes(include=[np.number]).columns if col != "Date"]
    if DEFAULT_TARGET in candidates:
        return DEFAULT_TARGET
    if candidates:
        return candidates[0]
    raise ValueError("No suitable target column found in dataset")


def split_data(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target])
    y = df[target]
    # Encode categorical columns using one‑hot (simple approach)
    X = pd.get_dummies(X, drop_first=True)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(df: pd.DataFrame, target: str = None, model_type: str = "auto"):
    """Train a RandomForest model. `model_type` can be "regression", "classification" or "auto".
    Returns the fitted model and the test data for later evaluation.
    """
    target_col = get_target_column(df, target)
    X_train, X_test, y_train, y_test = split_data(df, target_col)

    if model_type == "auto":
        # decide based on dtype of target
        if pd.api.types.is_numeric_dtype(y_train):
            model_type = "regression"
        else:
            model_type = "classification"
    if model_type == "regression":
        model = RandomForestRegressor(n_estimators=200, random_state=0)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)
    return model, (X_test, y_test)


def evaluate_model(model, X_test, y_test, task: str = "regression"):
    """Return a dict of common performance metrics.
    """
    y_pred = model.predict(X_test)
    if task == "regression":
        return {
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        }
    else:
        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred, average="weighted"),
        }
