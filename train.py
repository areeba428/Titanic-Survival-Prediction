"""Train the Titanic survival model and register it in MLflow.

Mirrors the notebook: mean Age, mode Embarked, encoded Sex and Embarked,
Family = SibSp + Parch, RandomForestClassifier(n_estimators=100, random_state=42).
"""

import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_PATH = os.getenv("TITANIC_DATA_PATH", "Datasets/Titanic-Dataset.csv")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "titanic-survival-random-forest")
MIN_TEST_ACCURACY = float(os.getenv("MIN_TEST_ACCURACY", "0.75"))


def load_features(path):
    data = pd.read_csv(path)
    data = data.drop(columns="Cabin")
    data["Age"] = data["Age"].fillna(data["Age"].mean())
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    data.replace(
        {"Sex": {"male": 0, "female": 1}, "Embarked": {"S": 0, "C": 1, "Q": 2}},
        inplace=True,
    )
    data["Family"] = data["Parch"] + data["SibSp"]
    features = data.drop(
        columns=["PassengerId", "Name", "Ticket", "Survived"],
    )
    return features, data["Survived"]


def main():
    X, y = load_features(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {accuracy:.4f}")

    if accuracy < MIN_TEST_ACCURACY:
        print(
            f"Accuracy {accuracy:.4f} is below MIN_TEST_ACCURACY={MIN_TEST_ACCURACY}",
            file=sys.stderr,
        )
        return 1

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("Titanic Survival Prediction")
    signature = infer_signature(X_test, predictions)

    with mlflow.start_run(run_name="random-forest"):
        mlflow.log_params(model.get_params())
        mlflow.log_metric("test_accuracy", accuracy)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_test.head(5),
            registered_model_name=MODEL_NAME,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        version = model_info.registered_model_version
        mlflow.MlflowClient().set_registered_model_alias(
            MODEL_NAME, "production", version
        )

    print(f"Registered {MODEL_NAME} version {version} at {TRACKING_URI}")
    print(f"Model URI: {model_info.model_uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
