"""Сувора технічна та візуальна перевірка першого етапу проєкту."""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preparation import (
    CLASS_NAMES,
    CLASS_NAMES_UA,
    LANCZOS,
    CoinSample,
    load_coin_dataset,
)


IMAGES_DIR = PROJECT_ROOT / "data" / "images"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Зображення підібрані так, щоб перевірити різні номінали, кількість і положення монет.
VALIDATION_IMAGE_NAMES = ["001.jpg", "079.jpg", "085.jpg", "096.jpg", "118.jpg"]
EXAMPLES_PER_CLASS = 8


@dataclass(frozen=True)
class ValidationSummary:
    """Короткі підсумки ручної та програмної перевірки кропів."""

    images_checked: int
    crops_checked: int
    total_images_scanned: int
    total_crops_scanned: int
    suspicious_crops: int
    missing_labels: int
    invalid_boxes: int
    saved_examples_checked: int
    coordinate_mismatches: int
    crop_mismatches: int


def read_raw_annotations(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Незалежно читає коректні числові рядки YOLO для контрольного розрахунку."""

    annotations = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id_text, center_x, center_y, width, height = line.split()
        annotations.append(
            (
                int(class_id_text),
                float(center_x),
                float(center_y),
                float(width),
                float(height),
            )
        )
    return annotations


def independent_pixel_box(
    annotation: tuple[int, float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Повторно обчислює піксельну рамку без виклику основної функції проєкту."""

    _, center_x, center_y, width, height = annotation
    x_center_px = center_x * image_width
    y_center_px = center_y * image_height
    box_width_px = width * image_width
    box_height_px = height * image_height

    x_min = math.floor(x_center_px - box_width_px / 2)
    y_min = math.floor(y_center_px - box_height_px / 2)
    x_max = math.ceil(x_center_px + box_width_px / 2)
    y_max = math.ceil(y_center_px + box_height_px / 2)
    return x_min, y_min, x_max, y_max


def group_samples_by_source(samples: list[CoinSample]) -> dict[str, list[CoinSample]]:
    """Групує кропи в тому самому порядку, у якому записані анотації."""

    grouped: dict[str, list[CoinSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.source_image].append(sample)
    return grouped


def choose_crop_examples(
    samples: list[CoinSample],
    examples_per_class: int = EXAMPLES_PER_CLASS,
) -> list[CoinSample]:
    """Вибирає перші збережені та додаткові рівномірні приклади кожного класу."""

    selected: list[CoinSample] = []
    for class_id in CLASS_NAMES:
        class_samples = [sample for sample in samples if sample.class_id == class_id]
        if len(class_samples) <= examples_per_class:
            selected.extend(class_samples)
            continue

        # Перші три відповідають файлам outputs/crop_examples, решта охоплюють весь клас.
        selected_indexes = [0, 1, 2]
        remaining_indexes = np.linspace(
            3,
            len(class_samples) - 1,
            examples_per_class - 3,
            dtype=int,
        ).tolist()
        selected_indexes.extend(remaining_indexes)
        selected.extend(class_samples[index] for index in selected_indexes)
    return selected


def find_suspicious_crops(samples: list[CoinSample]) -> list[tuple[CoinSample, str]]:
    """Шукає технічно порожні, пошкоджені або явно нетипові кропи."""

    suspicious = []
    for sample in samples:
        reasons = []
        if sample.image.shape != (64, 64, 3):
            reasons.append("неправильна форма масиву")
        if sample.image.dtype != np.uint8:
            reasons.append("неправильний тип даних")
        if sample.image.size == 0:
            reasons.append("порожній масив")
        elif float(sample.image.std()) < 10 or int(np.ptp(sample.image)) < 30:
            reasons.append("майже однорідне зображення")

        x_min, y_min, x_max, y_max = sample.bounding_box
        width = x_max - x_min
        height = y_max - y_min
        if width <= 0 or height <= 0:
            reasons.append("порожня рамка")
        elif not 0.65 <= width / height <= 1.50:
            reasons.append("нетипове співвідношення сторін рамки")

        if reasons:
            suspicious.append((sample, ", ".join(reasons)))
    return suspicious


def verify_coordinates_and_crops(
    samples: list[CoinSample],
) -> tuple[int, int, int]:
    """Звіряє всі рамки та масиви кропів із незалежним повторним вирізанням."""

    grouped = group_samples_by_source(samples)
    coordinate_mismatches = 0
    crop_mismatches = 0
    invalid_boxes = 0

    for image_path in sorted(IMAGES_DIR.glob("*")):
        if not image_path.is_file():
            continue
        label_path = LABELS_DIR / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        image_width, image_height = image.size
        annotations = read_raw_annotations(label_path)
        image_samples = grouped[image_path.name]

        if len(annotations) != len(image_samples):
            crop_mismatches += abs(len(annotations) - len(image_samples))

        for annotation, sample in zip(annotations, image_samples):
            expected_box = independent_pixel_box(annotation, image_width, image_height)
            x_min, y_min, x_max, y_max = expected_box
            if not (
                0 <= x_min < x_max <= image_width
                and 0 <= y_min < y_max <= image_height
            ):
                invalid_boxes += 1
                continue

            if sample.bounding_box != expected_box or sample.class_id != annotation[0]:
                coordinate_mismatches += 1

            expected_crop = np.asarray(
                image.crop(expected_box).resize((64, 64), LANCZOS),
                dtype=np.uint8,
            )
            if not np.array_equal(sample.image, expected_crop):
                crop_mismatches += 1

    return coordinate_mismatches, crop_mismatches, invalid_boxes


def validate_saved_crop_examples(samples: list[CoinSample]) -> int:
    """Перевіряє, що всі раніше збережені PNG відкриваються і відповідають кропам."""

    sample_by_class: dict[int, list[CoinSample]] = defaultdict(list)
    for sample in samples:
        sample_by_class[sample.class_id].append(sample)

    checked = 0
    for crop_path in sorted((OUTPUT_DIR / "crop_examples").glob("class_*_*.png")):
        _, class_id_text, number_text = crop_path.stem.split("_")
        class_id = int(class_id_text)
        sample_number = int(number_text) - 1
        with Image.open(crop_path) as crop_image:
            crop_array = np.asarray(crop_image.convert("RGB"), dtype=np.uint8)
        expected = sample_by_class[class_id][sample_number].image
        if crop_array.shape != (64, 64, 3) or not np.array_equal(crop_array, expected):
            raise ValueError(f"Збережений кроп не відповідає даним: {crop_path}")
        checked += 1
    return checked


def create_bbox_validation_figure(
    samples: list[CoinSample],
    image_names: list[str] = VALIDATION_IMAGE_NAMES,
) -> Path:
    """Показує п'ять оригінальних фотографій із рамками та назвами класів."""

    grouped = group_samples_by_source(samples)
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(2, 3, figsize=(15, 16))
    axes = axes.ravel()
    fig.suptitle(
        "Перевірка перетворення YOLO-координат на оригінальних зображеннях",
        fontsize=17,
    )

    for axis, image_name in zip(axes, image_names):
        image_path = IMAGES_DIR / image_name
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        axis.imshow(image)
        axis.set_title(f"Зображення {image_name}")
        axis.axis("off")

        for sample in grouped[image_name]:
            x_min, y_min, x_max, y_max = sample.bounding_box
            color = colors[sample.class_id]
            axis.add_patch(
                Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill=False,
                    edgecolor=color,
                    linewidth=2.5,
                )
            )
            text_y = max(5, y_min - 8)
            axis.text(
                x_min,
                text_y,
                CLASS_NAMES_UA[sample.class_id],
                color="white",
                fontsize=8,
                bbox={"facecolor": color, "alpha": 0.9, "pad": 2},
            )

    axes[-1].axis("off")
    axes[-1].text(
        0.5,
        0.5,
        "Рамки мають охоплювати монети,\nа підписи — відповідати номіналам.",
        ha="center",
        va="center",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = OUTPUT_DIR / "bbox_validation.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_crop_validation_grid(
    selected_samples: list[CoinSample],
    examples_per_class: int = EXAMPLES_PER_CLASS,
) -> Path:
    """Створює розширену сітку з вісьмома кропами кожного номіналу."""

    grouped: dict[int, list[CoinSample]] = defaultdict(list)
    for sample in selected_samples:
        grouped[sample.class_id].append(sample)

    fig, axes = plt.subplots(
        len(CLASS_NAMES),
        examples_per_class,
        figsize=(18, 19),
        squeeze=False,
    )
    fig.suptitle(
        "Розширена перевірка кропів монет за класами",
        fontsize=18,
    )

    for row, class_id in enumerate(CLASS_NAMES):
        for column, sample in enumerate(grouped[class_id]):
            axis = axes[row, column]
            axis.imshow(sample.image)
            axis.set_title(sample.source_image, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(
                    CLASS_NAMES_UA[class_id],
                    fontsize=12,
                    rotation=0,
                    labelpad=42,
                    va="center",
                )

    fig.tight_layout(rect=(0.03, 0, 1, 0.975))
    output_path = OUTPUT_DIR / "crop_validation_grid.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def print_validation_summary(summary: ValidationSummary) -> None:
    """Друкує запитаний підсумок валідації українською мовою."""

    print("Підсумок валідації Stage 1:")
    print(f"  Кількість зображень, перевірених програмно: {summary.total_images_scanned}")
    print(f"  Кількість зображень, перевірених візуально: {summary.images_checked}")
    print(f"  Кількість кропів, перевірених програмно: {summary.total_crops_scanned}")
    print(f"  Кількість кропів, перевірених візуально: {summary.crops_checked}")
    print(f"  Кількість підозрілих кропів: {summary.suspicious_crops}")
    print(f"  Кількість відсутніх файлів розмітки: {summary.missing_labels}")
    print(f"  Кількість некоректних рамок: {summary.invalid_boxes}")
    print(f"  Перевірено збережених PNG-кропів: {summary.saved_examples_checked}")
    print(f"  Розбіжності незалежного розрахунку координат: {summary.coordinate_mismatches}")
    print(f"  Розбіжності повторно сформованих кропів: {summary.crop_mismatches}")


def main() -> ValidationSummary:
    """Запускає всі технічні перевірки та створює контрольні фігури."""

    samples, report = load_coin_dataset(IMAGES_DIR, LABELS_DIR)
    coordinate_mismatches, crop_mismatches, independently_invalid_boxes = (
        verify_coordinates_and_crops(samples)
    )
    suspicious = find_suspicious_crops(samples)
    saved_examples_checked = validate_saved_crop_examples(samples)
    selected_samples = choose_crop_examples(samples)

    bbox_path = create_bbox_validation_figure(samples)
    grid_path = create_crop_validation_grid(selected_samples)

    invalid_boxes = report.invalid_bounding_boxes + independently_invalid_boxes
    summary = ValidationSummary(
        images_checked=len(VALIDATION_IMAGE_NAMES),
        crops_checked=len(selected_samples),
        total_images_scanned=report.total_source_images,
        total_crops_scanned=len(samples),
        suspicious_crops=len(suspicious),
        missing_labels=report.missing_label_files,
        invalid_boxes=invalid_boxes,
        saved_examples_checked=saved_examples_checked,
        coordinate_mismatches=coordinate_mismatches,
        crop_mismatches=crop_mismatches,
    )
    print_validation_summary(summary)
    print(f"  Фігура з рамками: {bbox_path.relative_to(PROJECT_ROOT)}")
    print(f"  Розширена сітка кропів: {grid_path.relative_to(PROJECT_ROOT)}")

    if suspicious:
        print("\nКропи, які потребують ручної уваги:")
        for sample, reason in suspicious:
            print(f"  {sample.source_image}, {CLASS_NAMES_UA[sample.class_id]}: {reason}")

    return summary


if __name__ == "__main__":
    main()
