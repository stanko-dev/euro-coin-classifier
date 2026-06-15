"""Навчання та оцінювання класичних моделей багатокласової класифікації."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ModelEvaluation:
    """Результати навчання й тестування однієї моделі."""

    model_name: str
    model: BaseEstimator
    accuracy: float
    training_time: float
    predictions: np.ndarray
    classification_report: pd.DataFrame
    confusion_matrix: np.ndarray


def create_models(random_state: int = 42) -> OrderedDict[str, BaseEstimator]:
    """Створює десять моделей із помірними параметрами для звичайного ноутбука."""

    return OrderedDict(
        [
            (
                "Логістична регресія",
                LogisticRegression(max_iter=2000, random_state=random_state),
            ),
            ("Метод k-найближчих сусідів", KNeighborsClassifier(n_neighbors=5)),
            (
                "Метод опорних векторів",
                SVC(kernel="rbf", random_state=random_state),
            ),
            (
                "Дерево рішень",
                DecisionTreeClassifier(random_state=random_state),
            ),
            (
                "Випадковий ліс",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
            (
                "Надзвичайно рандомізовані дерева",
                ExtraTreesClassifier(
                    n_estimators=200,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
            (
                "Градієнтний бустинг",
                GradientBoostingClassifier(
                    n_estimators=50,
                    random_state=random_state,
                ),
            ),
            (
                "AdaBoost",
                AdaBoostClassifier(n_estimators=100, random_state=random_state),
            ),
            ("Гаусівський наївний Баєс", GaussianNB()),
            (
                "Багатошаровий перцептрон",
                MLPClassifier(
                    hidden_layer_sizes=(64,),
                    max_iter=500,
                    early_stopping=True,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[BaseEstimator, float]:
    """Навчає модель і повертає тривалість навчання в секундах."""

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start_time
    return model, training_time


def build_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    """Формує звіт sklearn і перекладає службові назви українською."""

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).transpose()
    report_frame = report_frame.rename(
        index={
            "accuracy": "загальна точність",
            "macro avg": "макросереднє",
            "weighted avg": "зважене середнє",
        },
        columns={
            "precision": "точність",
            "recall": "повнота",
            "f1-score": "F1-міра",
            "support": "кількість",
        },
    )
    return report_frame


def evaluate_model(
    model_name: str,
    model: BaseEstimator,
    training_time: float,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> ModelEvaluation:
    """Обчислює прогнози, accuracy, classification report і confusion matrix."""

    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    report = build_classification_report(y_test, predictions, class_names)
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(class_names)),
    )
    return ModelEvaluation(
        model_name=model_name,
        model=model,
        accuracy=accuracy,
        training_time=training_time,
        predictions=predictions,
        classification_report=report,
        confusion_matrix=matrix,
    )


def train_and_evaluate_model(
    model_name: str,
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> ModelEvaluation:
    """Послідовно навчає та повністю оцінює одну модель."""

    trained_model, training_time = train_model(model, X_train, y_train)
    return evaluate_model(
        model_name=model_name,
        model=trained_model,
        training_time=training_time,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
    )


def result_row(evaluation: ModelEvaluation) -> dict[str, Any]:
    """Перетворює оцінку моделі на рядок підсумкової таблиці."""

    return {
        "model name": evaluation.model_name,
        "accuracy": evaluation.accuracy,
        "training time": evaluation.training_time,
    }
