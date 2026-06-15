"""Фінальний аналіз результатів і двовимірне t-SNE представлення."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def choose_safe_perplexity(sample_count: int) -> int:
    """Обирає perplexity, яка гарантовано менша за кількість об'єктів."""

    if sample_count < 3:
        raise ValueError("Для t-SNE потрібно щонайменше три тестові об'єкти.")

    safe_perplexity = min(30, max(2, (sample_count - 1) // 3))
    return min(safe_perplexity, sample_count - 1)


def calculate_tsne_embedding(
    X_test: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, int]:
    """Зменшує багатовимірні тестові ознаки до двох координат t-SNE."""

    if X_test.ndim != 2:
        raise ValueError("Матриця тестових ознак повинна бути двовимірною.")

    safe_perplexity = choose_safe_perplexity(len(X_test))
    embedding = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=safe_perplexity,
    ).fit_transform(X_test)
    return embedding, safe_perplexity


def build_final_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Створює фінальну таблицю й сортує моделі за тестовою точністю."""

    required_columns = {"model name", "accuracy", "training time"}
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        raise ValueError(
            "У таблиці результатів відсутні стовпці: "
            + ", ".join(sorted(missing_columns))
        )

    return (
        results.loc[:, ["model name", "accuracy", "training time"]]
        .rename(columns={"accuracy": "test accuracy"})
        .sort_values("test accuracy", ascending=False, ignore_index=True)
    )


def select_accuracy_time_balance(
    final_results: pd.DataFrame,
    allowed_accuracy_drop: float = 0.10,
) -> pd.Series:
    """Серед достатньо точних моделей обирає модель із найменшим часом fit."""

    if final_results.empty:
        raise ValueError("Таблиця результатів не може бути порожньою.")

    best_accuracy = float(final_results["test accuracy"].max())
    candidates = final_results[
        final_results["test accuracy"] >= best_accuracy - allowed_accuracy_drop
    ]
    return candidates.sort_values(
        ["training time", "test accuracy"],
        ascending=[True, False],
    ).iloc[0]

