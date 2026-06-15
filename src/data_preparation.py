"""Завантаження YOLO-розмітки та формування набору кропів монет."""

from __future__ import annotations

import argparse
import math
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError


CLASS_NAMES = {
    0: "1 cent",
    1: "2 cent",
    2: "5 cent",
    3: "10 cent",
    4: "20 cent",
    5: "50 cent",
    6: "1 euro",
    7: "2 euro",
}

CLASS_NAMES_UA = {
    0: "1 цент",
    1: "2 центи",
    2: "5 центів",
    3: "10 центів",
    4: "20 центів",
    5: "50 центів",
    6: "1 євро",
    7: "2 євро",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


class DatasetWarning(UserWarning):
    """Попередження про окремий проблемний файл або об'єкт датасету."""


@dataclass(frozen=True)
class CoinSample:
    """Один об'єкт монети, перетворений на приклад для класифікації."""

    image: np.ndarray
    class_id: int
    class_name: str
    source_image: str
    bounding_box: tuple[int, int, int, int]


@dataclass
class PreparationReport:
    """Лічильники успішних операцій і виявлених проблем."""

    total_source_images: int = 0
    total_annotation_files: int = 0
    successful_crops: int = 0
    missing_label_files: int = 0
    empty_label_files: int = 0
    invalid_annotation_lines: int = 0
    unknown_class_ids: int = 0
    unreadable_images: int = 0
    invalid_bounding_boxes: int = 0
    clipped_bounding_boxes: int = 0


@dataclass(frozen=True)
class YoloAnnotation:
    """Перевірений рядок YOLO-розмітки."""

    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float
    line_number: int


def find_image_files(images_dir: Path) -> list[Path]:
    """Повертає підтримувані файли зображень у стабільному порядку."""

    if not images_dir.exists():
        raise FileNotFoundError(f"Каталог із зображеннями не знайдено: {images_dir}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Шлях до зображень не є каталогом: {images_dir}")

    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_yolo_annotations(
    label_path: Path,
    report: PreparationReport,
) -> list[YoloAnnotation]:
    """Зчитує та перевіряє всі непорожні рядки одного YOLO-файлу."""

    try:
        text = label_path.read_text(encoding="utf-8")
    except OSError as error:
        report.invalid_annotation_lines += 1
        warnings.warn(
            f"Не вдалося прочитати файл розмітки {label_path}: {error}",
            DatasetWarning,
        )
        return []

    non_empty_lines = [
        (line_number, line.strip())
        for line_number, line in enumerate(text.splitlines(), start=1)
        if line.strip()
    ]
    if not non_empty_lines:
        report.empty_label_files += 1
        warnings.warn(f"Файл розмітки порожній: {label_path}", DatasetWarning)
        return []

    annotations: list[YoloAnnotation] = []
    for line_number, line in non_empty_lines:
        parts = line.split()
        if len(parts) != 5:
            report.invalid_annotation_lines += 1
            warnings.warn(
                f"Некоректний рядок {line_number} у {label_path}: очікується 5 значень.",
                DatasetWarning,
            )
            continue

        try:
            class_id = int(parts[0])
            center_x, center_y, width, height = map(float, parts[1:])
        except ValueError:
            report.invalid_annotation_lines += 1
            warnings.warn(
                f"Некоректні числові значення в рядку {line_number} файлу {label_path}.",
                DatasetWarning,
            )
            continue

        if class_id not in CLASS_NAMES:
            report.unknown_class_ids += 1
            warnings.warn(
                f"Невідомий class_id={class_id} у рядку {line_number} файлу {label_path}.",
                DatasetWarning,
            )
            continue

        values = (center_x, center_y, width, height)
        if not all(math.isfinite(value) for value in values):
            report.invalid_annotation_lines += 1
            warnings.warn(
                f"Нескінченне або відсутнє числове значення в рядку {line_number} файлу {label_path}.",
                DatasetWarning,
            )
            continue

        if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
            report.invalid_annotation_lines += 1
            warnings.warn(
                f"Центр рамки поза межами [0, 1] у рядку {line_number} файлу {label_path}.",
                DatasetWarning,
            )
            continue

        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            report.invalid_bounding_boxes += 1
            warnings.warn(
                f"Некоректна ширина або висота рамки в рядку {line_number} файлу {label_path}.",
                DatasetWarning,
            )
            continue

        annotations.append(
            YoloAnnotation(
                class_id=class_id,
                center_x=center_x,
                center_y=center_y,
                width=width,
                height=height,
                line_number=line_number,
            )
        )

    return annotations


def yolo_to_pixel_box(
    annotation: YoloAnnotation,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Перетворює нормалізовані координати YOLO на піксельну рамку."""

    x_center_px = annotation.center_x * image_width
    y_center_px = annotation.center_y * image_height
    box_width_px = annotation.width * image_width
    box_height_px = annotation.height * image_height

    x_min = math.floor(x_center_px - box_width_px / 2)
    y_min = math.floor(y_center_px - box_height_px / 2)
    x_max = math.ceil(x_center_px + box_width_px / 2)
    y_max = math.ceil(y_center_px + box_height_px / 2)
    return x_min, y_min, x_max, y_max


def clip_pixel_box(
    bounding_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[tuple[int, int, int, int], bool]:
    """Обрізає рамку за межами зображення та повідомляє, чи була зміна."""

    x_min, y_min, x_max, y_max = bounding_box
    clipped_box = (
        max(0, min(image_width, x_min)),
        max(0, min(image_height, y_min)),
        max(0, min(image_width, x_max)),
        max(0, min(image_height, y_max)),
    )
    return clipped_box, clipped_box != bounding_box


def load_coin_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    target_size: tuple[int, int] = (64, 64),
) -> tuple[list[CoinSample], PreparationReport]:
    """Створює класифікаційний набір, вирізаючи всі розмічені монети."""

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(f"Каталог із розміткою не знайдено: {labels_dir}")
    if not labels_dir.is_dir():
        raise NotADirectoryError(f"Шлях до розмітки не є каталогом: {labels_dir}")
    if target_size[0] <= 0 or target_size[1] <= 0:
        raise ValueError("Розмір кропу має складатися з додатних чисел.")

    image_files = find_image_files(images_dir)
    report = PreparationReport(
        total_source_images=len(image_files),
        total_annotation_files=sum(1 for path in labels_dir.glob("*.txt") if path.is_file()),
    )
    samples: list[CoinSample] = []

    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            report.missing_label_files += 1
            warnings.warn(
                f"Для зображення {image_path.name} не знайдено файл розмітки.",
                DatasetWarning,
            )
            continue

        annotations = read_yolo_annotations(label_path, report)
        if not annotations:
            continue

        try:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            report.unreadable_images += 1
            warnings.warn(
                f"Не вдалося відкрити зображення {image_path}: {error}",
                DatasetWarning,
            )
            continue

        image_width, image_height = image.size
        for annotation in annotations:
            raw_box = yolo_to_pixel_box(annotation, image_width, image_height)
            bounding_box, was_clipped = clip_pixel_box(
                raw_box,
                image_width,
                image_height,
            )
            if was_clipped:
                report.clipped_bounding_boxes += 1
                warnings.warn(
                    f"Рамку в рядку {annotation.line_number} файлу {label_path.name} "
                    "обрізано до меж зображення.",
                    DatasetWarning,
                )

            x_min, y_min, x_max, y_max = bounding_box
            if x_min >= x_max or y_min >= y_max:
                report.invalid_bounding_boxes += 1
                warnings.warn(
                    f"Порожня рамка в рядку {annotation.line_number} файлу {label_path.name}.",
                    DatasetWarning,
                )
                continue

            crop = image.crop(bounding_box)
            if crop.width == 0 or crop.height == 0:
                report.invalid_bounding_boxes += 1
                warnings.warn(
                    f"Не вдалося отримати кроп із рядка {annotation.line_number} "
                    f"файлу {label_path.name}.",
                    DatasetWarning,
                )
                continue

            resized_crop = crop.resize(target_size, LANCZOS)
            samples.append(
                CoinSample(
                    image=np.asarray(resized_crop, dtype=np.uint8),
                    class_id=annotation.class_id,
                    class_name=CLASS_NAMES[annotation.class_id],
                    source_image=image_path.name,
                    bounding_box=bounding_box,
                )
            )
            report.successful_crops += 1

    return samples, report


def class_distribution(samples: Iterable[CoinSample]) -> dict[int, int]:
    """Рахує кількість кропів для кожного класу, включно з порожніми."""

    counts = Counter(sample.class_id for sample in samples)
    return {class_id: counts.get(class_id, 0) for class_id in CLASS_NAMES}


def format_class_distribution(samples: Iterable[CoinSample]) -> str:
    """Формує текстову таблицю розподілу класів."""

    distribution = class_distribution(samples)
    rows = [
        f"{'ID класу':<10} {'Номінал':<12} {'Кількість':>10}",
        "-" * 34,
    ]
    for class_id, class_name in CLASS_NAMES.items():
        rows.append(f"{class_id:<10} {class_name:<12} {distribution[class_id]:>10}")
    return "\n".join(rows)


def print_dataset_summary(samples: list[CoinSample], report: PreparationReport) -> None:
    """Друкує основні розміри датасету, таблицю класів і проблеми."""

    print(f"Загальна кількість вихідних зображень: {report.total_source_images}")
    print(f"Загальна кількість файлів розмітки: {report.total_annotation_files}")
    print(f"Успішно вирізано об'єктів монет: {report.successful_crops}")
    print(f"Кількість класів: {len(CLASS_NAMES)}")
    print("\nРозподіл об'єктів за класами:")
    print(format_class_distribution(samples))
    print("\nРезультати перевірок:")
    print(f"  Відсутні файли розмітки: {report.missing_label_files}")
    print(f"  Порожні файли розмітки: {report.empty_label_files}")
    print(f"  Некоректні рядки анотацій: {report.invalid_annotation_lines}")
    print(f"  Невідомі класи: {report.unknown_class_ids}")
    print(f"  Зображення, які не вдалося прочитати: {report.unreadable_images}")
    print(f"  Некоректні рамки: {report.invalid_bounding_boxes}")
    print(f"  Рамки, обрізані до меж зображення: {report.clipped_bounding_boxes}")


def save_crop_examples(
    samples: Iterable[CoinSample],
    output_dir: str | Path,
    examples_per_class: int = 3,
) -> list[Path]:
    """Зберігає кілька кропів кожного класу для ручної перевірки."""

    if examples_per_class <= 0:
        raise ValueError("Кількість прикладів для класу має бути додатною.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    saved_counts = Counter()

    for sample in samples:
        if saved_counts[sample.class_id] >= examples_per_class:
            continue

        number = saved_counts[sample.class_id] + 1
        output_path = output_dir / f"class_{sample.class_id}_{number}.png"
        Image.fromarray(sample.image).save(output_path)
        saved_paths.append(output_path)
        saved_counts[sample.class_id] += 1

        if all(saved_counts[class_id] >= examples_per_class for class_id in CLASS_NAMES):
            break

    return saved_paths


def run_stage_one(
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    examples_per_class: int = 3,
) -> tuple[list[CoinSample], PreparationReport]:
    """Виконує повний перший етап і зберігає контрольні візуалізації."""

    from .visualization import plot_class_distribution, plot_crop_grid

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    samples, report = load_coin_dataset(
        data_dir / "images",
        data_dir / "labels",
    )
    print_dataset_summary(samples, report)

    saved_crops = save_crop_examples(
        samples,
        output_dir / "crop_examples",
        examples_per_class=examples_per_class,
    )
    plot_class_distribution(
        samples,
        save_path=output_dir / "class_distribution.png",
    )
    plot_crop_grid(
        samples,
        examples_per_class=examples_per_class,
        save_path=output_dir / "crop_grid.png",
    )
    print(f"\nЗбережено контрольних кропів: {len(saved_crops)}")
    print(f"Результати записано до: {output_dir.resolve()}")
    return samples, report


def parse_arguments() -> argparse.Namespace:
    """Зчитує параметри командного запуску першого етапу."""

    parser = argparse.ArgumentParser(
        description="Підготовка кропів монет євро з YOLO-розмітки."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--examples-per-class", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    run_stage_one(
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        examples_per_class=arguments.examples_per_class,
    )
