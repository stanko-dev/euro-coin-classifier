"""Візуалізації для первинного аналізу кропів монет."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .data_preparation import (
    CLASS_NAMES,
    CLASS_NAMES_UA,
    CoinSample,
    class_distribution,
)


def plot_class_distribution(
    samples: Iterable[CoinSample],
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Будує стовпчикову діаграму кількості монет кожного номіналу."""

    sample_list = list(samples)
    distribution = class_distribution(sample_list)
    class_ids = list(CLASS_NAMES)
    labels = [CLASS_NAMES_UA[class_id] for class_id in class_ids]
    values = [distribution[class_id] for class_id in class_ids]

    fig, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(labels, values, color="#4C78A8", edgecolor="#2F4B6C")
    axis.set_title("Розподіл вирізаних монет за номіналами")
    axis.set_xlabel("Номінал монети")
    axis.set_ylabel("Кількість об'єктів")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.bar_label(bars, padding=3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_crop_grid(
    samples: Iterable[CoinSample],
    examples_per_class: int = 3,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Показує по кілька вирізаних прикладів кожного класу."""

    if examples_per_class <= 0:
        raise ValueError("Кількість прикладів для класу має бути додатною.")

    grouped_samples: dict[int, list[CoinSample]] = defaultdict(list)
    for sample in samples:
        if len(grouped_samples[sample.class_id]) < examples_per_class:
            grouped_samples[sample.class_id].append(sample)

    fig, axes = plt.subplots(
        len(CLASS_NAMES),
        examples_per_class,
        figsize=(3.2 * examples_per_class, 2.55 * len(CLASS_NAMES)),
        squeeze=False,
    )
    fig.suptitle("Приклади кропів монет для кожного класу", fontsize=15, y=0.995)

    for row, class_id in enumerate(CLASS_NAMES):
        class_name = CLASS_NAMES_UA[class_id]
        class_samples = grouped_samples[class_id]
        for column in range(examples_per_class):
            axis = axes[row, column]
            axis.axis("off")
            if column < len(class_samples):
                sample = class_samples[column]
                axis.imshow(sample.image)
                axis.set_title(f"{class_name}\n{sample.source_image}", fontsize=10)
            else:
                axis.set_title(f"{class_name}\nнемає прикладу", fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.985))
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_hog_example(
    grayscale_image: np.ndarray,
    hog_image: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Порівнює grayscale-кроп із наочним зображенням його HOG-контурів."""

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(grayscale_image, cmap="gray")
    axes[0].set_title("Монета у відтінках сірого")
    axes[1].imshow(hog_image, cmap="gray")
    axes[1].set_title("Візуалізація HOG-ознак")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    model_name: str,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Будує теплову карту матриці помилок із назвами номіналів."""

    fig, axis = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Кількість прогнозів"},
        ax=axis,
    )
    axis.set_title(f"Матриця помилок: {model_name}")
    axis.set_xlabel("Передбачений клас")
    axis.set_ylabel("Справжній клас")
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", rotation=0)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_model_comparison(
    results,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Порівнює точність і час навчання всіх успішних моделей."""

    ordered = results.sort_values("accuracy", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].barh(ordered["model name"], ordered["accuracy"], color="#4C78A8")
    axes[0].set_title("Порівняння точності моделей")
    axes[0].set_xlabel("Точність на тестовій вибірці")
    axes[0].set_ylabel("Модель")
    axes[0].set_xlim(0, 1)
    axes[0].grid(axis="x", linestyle="--", alpha=0.35)

    axes[1].barh(
        ordered["model name"],
        ordered["training time"],
        color="#F58518",
    )
    axes[1].set_title("Порівняння часу навчання")
    axes[1].set_xlabel("Час навчання, секунд")
    axes[1].set_ylabel("Модель")
    axes[1].grid(axis="x", linestyle="--", alpha=0.35)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_accuracy_comparison(
    results,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Будує окрему стовпчикову діаграму тестової точності моделей."""

    ordered = results.sort_values("test accuracy", ascending=True)
    fig, axis = plt.subplots(figsize=(10, 6.5))
    bars = axis.barh(
        ordered["model name"],
        ordered["test accuracy"],
        color="#4C78A8",
    )
    axis.set_title("Порівняння точності моделей на тестовій вибірці")
    axis.set_xlabel("Тестова точність")
    axis.set_ylabel("Модель")
    axis.set_xlim(0, 1)
    axis.grid(axis="x", linestyle="--", alpha=0.35)
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in bars.datavalues], padding=3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_training_time_comparison(
    results,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Будує окрему стовпчикову діаграму часу навчання моделей."""

    ordered = results.sort_values("training time", ascending=False)
    fig, axis = plt.subplots(figsize=(10, 6.5))
    bars = axis.barh(
        ordered["model name"],
        ordered["training time"],
        color="#F58518",
    )
    axis.set_title("Порівняння часу навчання моделей")
    axis.set_xlabel("Час навчання, секунд")
    axis.set_ylabel("Модель")
    axis.grid(axis="x", linestyle="--", alpha=0.35)
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in bars.datavalues], padding=3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_tsne_classes(
    embedding: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Показує двовимірні t-SNE точки, зафарбовані відповідно до класу."""

    if embedding.ndim != 2 or embedding.shape[1] != 2:
        raise ValueError("t-SNE представлення повинно мати два стовпці.")
    if len(embedding) != len(labels):
        raise ValueError("Кількість t-SNE точок і міток повинна збігатися.")

    fig, axis = plt.subplots(figsize=(9, 7))
    colors = sns.color_palette("tab10", n_colors=len(class_names))
    for class_id, class_name in enumerate(class_names):
        mask = labels == class_id
        axis.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=58,
            alpha=0.8,
            color=colors[class_id],
            edgecolor="white",
            linewidth=0.5,
            label=class_name,
        )

    axis.set_title(title)
    axis.set_xlabel("Компонента t-SNE 1")
    axis.set_ylabel("Компонента t-SNE 2")
    axis.grid(linestyle="--", alpha=0.2)
    axis.legend(title="Номінал монети", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
