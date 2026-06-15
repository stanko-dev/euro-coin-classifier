"""Виділення числових ознак і підготовка даних до класифікації."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skimage.color import rgb2gray
from skimage.feature import hog

from .data_preparation import CLASS_NAMES, CLASS_NAMES_UA, CoinSample


@dataclass(frozen=True)
class FeatureSet:
    """Числові представлення всіх кропів та їхні цільові класи."""

    grayscale_images: np.ndarray
    flattened_pixels: np.ndarray
    hog_features: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class FeatureChecks:
    """Результати базової перевірки матриці ознак і цільового вектора."""

    x_shape: tuple[int, int]
    y_shape: tuple[int, ...]
    missing_values_x: int
    missing_values_y: int
    duplicate_feature_rows: int
    class_distribution: dict[int, int]


@dataclass(frozen=True)
class PreparedData:
    """Масштабовані ознаки, кодувальник і навчально-тестовий поділ."""

    X_scaled: np.ndarray
    y_encoded: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder
    scaler: StandardScaler


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Перетворює RGB-кроп на одноканальне зображення у діапазоні [0, 1]."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Очікується RGB-зображення з трьома каналами.")
    return rgb2gray(image).astype(np.float32)


def extract_hog_features(
    grayscale_image: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
) -> np.ndarray:
    """Обчислює HOG-дескриптор напрямків контурів одного зображення."""

    if grayscale_image.ndim != 2:
        raise ValueError("Для HOG потрібне двовимірне зображення у відтінках сірого.")

    return hog(
        grayscale_image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    ).astype(np.float32)


def extract_feature_matrices(samples: Iterable[CoinSample]) -> FeatureSet:
    """Формує grayscale, flattened-pixel і HOG представлення всіх кропів."""

    sample_list = list(samples)
    if not sample_list:
        raise ValueError("Неможливо виділити ознаки з порожнього списку кропів.")

    grayscale_images = np.stack(
        [convert_to_grayscale(sample.image) for sample in sample_list]
    )
    flattened_pixels = grayscale_images.reshape(len(sample_list), -1)
    hog_features = np.stack(
        [extract_hog_features(image) for image in grayscale_images]
    )
    targets = np.asarray([sample.class_id for sample in sample_list], dtype=np.int64)

    return FeatureSet(
        grayscale_images=grayscale_images,
        flattened_pixels=flattened_pixels.astype(np.float32),
        hog_features=hog_features,
        targets=targets,
    )


def check_feature_data(X: np.ndarray, y: np.ndarray) -> FeatureChecks:
    """Перевіряє розміри, пропуски, дублікати та розподіл класів."""

    if X.ndim != 2:
        raise ValueError("Матриця X повинна бути двовимірною.")
    if y.ndim != 1:
        raise ValueError("Цільовий вектор y повинен бути одновимірним.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("Кількість рядків X не збігається з довжиною y.")

    unique_rows = np.unique(X, axis=0).shape[0]
    counts = Counter(int(value) for value in y)
    distribution = {class_id: counts.get(class_id, 0) for class_id in CLASS_NAMES}

    return FeatureChecks(
        x_shape=X.shape,
        y_shape=y.shape,
        missing_values_x=int(np.isnan(X).sum()),
        missing_values_y=int(np.isnan(y.astype(float)).sum()),
        duplicate_feature_rows=int(X.shape[0] - unique_rows),
        class_distribution=distribution,
    )


def print_feature_checks(checks: FeatureChecks) -> None:
    """Друкує результати перевірки ознак українською мовою."""

    print(f"Розмір матриці ознак X: {checks.x_shape}")
    print(f"Розмір цільового вектора y: {checks.y_shape}")
    print(f"Пропущених значень у X: {checks.missing_values_x}")
    print(f"Пропущених значень у y: {checks.missing_values_y}")
    print(f"Повністю однакових рядків ознак: {checks.duplicate_feature_rows}")
    print("Розподіл класів до поділу:")
    for class_id, count in checks.class_distribution.items():
        print(f"  {class_id} ({CLASS_NAMES_UA[class_id]}): {count}")


def extract_hog_visualization(grayscale_image: np.ndarray) -> np.ndarray:
    """Створює наочне зображення градієнтів HOG для пояснення методу."""

    if grayscale_image.ndim != 2:
        raise ValueError("Для HOG потрібне двовимірне зображення у відтінках сірого.")

    _, hog_image = hog(
        grayscale_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        visualize=True,
        feature_vector=True,
    )
    return hog_image


def prepare_train_test_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreparedData:
    """Кодує класи, масштабує ознаки та виконує стратифікований поділ."""

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    return PreparedData(
        X_scaled=X_scaled,
        y_encoded=y_encoded,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        label_encoder=label_encoder,
        scaler=scaler,
    )
