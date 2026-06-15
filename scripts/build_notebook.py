"""Оновлення навчального ноутбука комірками другого і третього етапів."""

from __future__ import annotations

import json
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "euro_coin_classification.ipynb"
STAGE_TWO_MARKER = "# Етап 2. Виділення ознак"


def markdown(text: str) -> dict:
    """Створює Markdown-комірку стандартного формату Jupyter."""

    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    """Створює порожню кодову комірку стандартного формату Jupyter."""

    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

# Повторний запуск скрипту замінює попередні комірки етапів 2-3, а не дублює їх.
base_cells = []
for cell in notebook["cells"]:
    source = "".join(cell.get("source", []))
    if STAGE_TWO_MARKER in source:
        break
    base_cells.append(cell)

# Оновлюємо імпорти першого етапу, щоб підписи класів у ноутбуці були українськими.
for cell in base_cells:
    source = "".join(cell.get("source", []))
    if "from src.data_preparation import (" in source:
        source = source.replace(
            "    CLASS_NAMES,\n",
            "    CLASS_NAMES,\n    CLASS_NAMES_UA,\n",
        )
        cell["source"] = source.splitlines(keepends=True)
    if "Клас першого кропу" in source:
        source = source.replace(
            "first_sample.class_name",
            "CLASS_NAMES_UA[first_sample.class_id]",
        )
        cell["source"] = source.splitlines(keepends=True)


stage_cells = [
    markdown(
        """
# Етап 2. Виділення ознак і підготовка даних

Класичні алгоритми машинного навчання не працюють безпосередньо із зображеннями як із картинками. Для них кожен кроп потрібно подати як набір чисел, тобто **ознак**. Ознака описує певну властивість об'єкта: яскравість окремого пікселя, напрямок краю, форму рельєфу або іншу вимірювану характеристику. На цьому етапі будуть побудовані два числові представлення: розгорнуті grayscale-пікселі та HOG-дескриптори.
"""
    ),
    markdown(
        """
## 7. Перетворення кропів у відтінки сірого

Колір монети залежить не лише від номіналу, а й від освітлення, камери та стану поверхні. Перехід до відтінків сірого зменшує кількість каналів із трьох до одного й дозволяє зосередитися на формі, цифрах та рельєфі. Значення кожного пікселя після перетворення лежить у діапазоні від 0 до 1. Очікується масив форми `(кількість монет, 64, 64)`.
"""
    ),
    code(
        """
from src.features import (
    check_feature_data,
    extract_feature_matrices,
    extract_hog_visualization,
    prepare_train_test_data,
    print_feature_checks,
)
from src.models import create_models, result_row, train_and_evaluate_model
from src.visualization import (
    plot_confusion_matrix,
    plot_hog_example,
    plot_model_comparison,
)

# Одним проходом формуємо grayscale-зображення, raw pixels, HOG і цільові класи.
feature_set = extract_feature_matrices(samples)
print(f"Форма grayscale-масиву: {feature_set.grayscale_images.shape}")
print(f"Мінімальна яскравість: {feature_set.grayscale_images.min():.3f}")
print(f"Максимальна яскравість: {feature_set.grayscale_images.max():.3f}")
"""
    ),
    markdown(
        """
## 8. Розгорнуті пікселі

Найпростіше числове представлення зображення — записати всі його пікселі в один довгий рядок. Для кропу `64x64` отримуємо `4096` ознак: спочатку значення першого рядка зображення, потім другого і так далі. Такий підхід зберігає всю інформацію про яскравість, але чутливий до невеликих зсувів, поворотів та змін освітлення. Він підготовлений для порівняння, проте основним представленням для моделей буде HOG.
"""
    ),
    code(
        """
print(f"Форма матриці розгорнутих пікселів: {feature_set.flattened_pixels.shape}")
print(f"Кількість числових ознак одного кропу: {feature_set.flattened_pixels.shape[1]}")
"""
    ),
    markdown(
        """
## 9. HOG-ознаки

HOG (Histogram of Oriented Gradients) ділить зображення на невеликі клітинки й підраховує, у яких напрямках у них найчастіше змінюється яскравість. Фактично дескриптор описує контури та напрямки країв, а не точні значення кожного пікселя. Для монет це корисно, тому що цифри номіналу, обідок і елементи карбування утворюють характерні лінії. HOG зазвичай стійкіший за raw pixels до помірних змін освітлення та невеликих локальних відмінностей.
"""
    ),
    code(
        """
hog_image = extract_hog_visualization(feature_set.grayscale_images[0])
plot_hog_example(
    feature_set.grayscale_images[0],
    hog_image,
    save_path=OUTPUT_DIR / "hog_example.png",
    show=True,
);
print(f"Кількість HOG-ознак для однієї монети: {feature_set.hog_features.shape[1]}")
"""
    ),
    markdown(
        """
## 10. Матриця ознак `X` і цільовий вектор `y`

Рядок матриці `X` відповідає одній монеті, а кожен стовпець — одному числовому HOG-показнику. Вектор `y` містить правильний клас для кожного рядка. Перед навчанням перевіряються розміри, пропущені значення, повністю однакові рядки ознак і початковий розподіл класів. Це допомагає знайти технічні проблеми до того, як вони вплинуть на моделі.
"""
    ),
    code(
        """
# HOG обираємо основною матрицею ознак для всіх десяти моделей.
X = feature_set.hog_features
y = feature_set.targets

feature_checks = check_feature_data(X, y)
print_feature_checks(feature_checks)
"""
    ),
    markdown(
        """
## 11. Кодування, нормалізація та поділ вибірки

`LabelEncoder` переводить мітки класів у послідовні цілі числа, зрозумілі алгоритмам. `StandardScaler` для кожної HOG-ознаки віднімає середнє та ділить на стандартне відхилення, тому ознаки отримують зіставний масштаб.

Нормалізація особливо важлива для **KNN**, бо він порівнює відстані між об'єктами; для **SVM**, бо масштаб впливає на положення розділювальної межі; для **логістичної регресії**, бо полегшує чисельну оптимізацію; для **MLPClassifier**, бо стабілізує та прискорює градієнтне навчання. Далі 20% об'єктів відводяться для тестування. `stratify=y_encoded` зберігає частки всіх восьми номіналів, а `random_state=42` робить поділ відтворюваним.
"""
    ),
    code(
        """
prepared_data = prepare_train_test_data(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("Відповідність початкових і закодованих класів:")
for encoded_value, original_class in enumerate(prepared_data.label_encoder.classes_):
    print(f"  {CLASS_NAMES_UA[int(original_class)]}: {encoded_value}")

print(f"Форма навчальної матриці: {prepared_data.X_train.shape}")
print(f"Форма тестової матриці: {prepared_data.X_test.shape}")
print(f"Кількість навчальних міток: {prepared_data.y_train.shape[0]}")
print(f"Кількість тестових міток: {prepared_data.y_test.shape[0]}")
"""
    ),
    markdown(
        """
# Етап 3. Навчання та оцінювання моделей

Для об'єктивного порівняння всі десять алгоритмів отримують однакові HOG-ознаки та той самий train/test split. Час вимірюється тільки для `fit`, а якість визначається на відкладеній тестовій вибірці. Крім загальної accuracy, для кожного класу виводяться precision, recall, F1-міра та матриця помилок.
"""
    ),
    markdown(
        """
## 12. Спільні функції експерименту

У цьому блоці створюються моделі та контейнери результатів. Допоміжна функція запускає навчання, зберігає прогноз у словнику `predictions`, додає рядок до майбутньої таблиці та будує heatmap. Обробка `try/except` потрібна, щоб помилка одного алгоритму не зупинила решту експерименту.
"""
    ),
    code(
        """
import numpy as np
import pandas as pd
from IPython.display import display

models = create_models(random_state=42)
class_names_ua = [
    CLASS_NAMES_UA[int(class_id)]
    for class_id in prepared_data.label_encoder.classes_
]

results_rows = []
predictions = {}
trained_models = {}
model_evaluations = {}
model_errors = {}


def run_notebook_model(model_name):
    # Навчаємо одну модель, показуємо метрики та зберігаємо результат.

    print(f"Початок навчання: {model_name}")
    try:
        evaluation = train_and_evaluate_model(
            model_name=model_name,
            model=models[model_name],
            X_train=prepared_data.X_train,
            y_train=prepared_data.y_train,
            X_test=prepared_data.X_test,
            y_test=prepared_data.y_test,
            class_names=class_names_ua,
        )
    except Exception as error:
        model_errors[model_name] = str(error)
        print(f"Помилка під час навчання моделі «{model_name}»: {error}")
        return None

    predictions[model_name] = evaluation.predictions
    trained_models[model_name] = evaluation.model
    model_evaluations[model_name] = evaluation
    results_rows.append(result_row(evaluation))

    print(f"Час навчання: {evaluation.training_time:.4f} с")
    print(f"Точність на тестовій вибірці: {evaluation.accuracy:.4f}")
    print("Класифікаційний звіт:")
    display(evaluation.classification_report.round(3))

    model_number = list(models).index(model_name) + 1
    plot_confusion_matrix(
        evaluation.confusion_matrix,
        class_names_ua,
        model_name,
        save_path=OUTPUT_DIR / "confusion_matrices" / f"model_{model_number:02d}.png",
        show=True,
    )
    return evaluation


print(f"Підготовлено моделей: {len(models)}")
"""
    ),
]


model_sections = [
    (
        "Логістична регресія",
        """
## 13. LogisticRegression — логістична регресія

Логістична регресія будує лінійні межі між класами та перетворює їхні оцінки на ймовірності. Для багатокласової задачі модель одночасно порівнює всі номінали. Це хороший базовий метод: він швидкий, відтворюваний і показує, наскільки класи вже розділяються лінійною комбінацією HOG-ознак. Збільшений `max_iter` дає оптимізатору достатньо кроків для збіжності.
""",
    ),
    (
        "Метод k-найближчих сусідів",
        """
## 14. KNeighborsClassifier — метод k-найближчих сусідів

KNN не будує явної математичної моделі, а шукає у навчальній вибірці п'ять найближчих об'єктів. Прогноз визначається голосуванням їхніх класів. Метод добре працює, коли монети одного номіналу утворюють компактні групи в просторі HOG-ознак, але сильно залежить від правильного масштабування відстаней.
""",
    ),
    (
        "Метод опорних векторів",
        """
## 15. SVC — метод опорних векторів

SVC шукає межі з якомога більшим відступом між класами. Радіально-базисне RBF-ядро дозволяє будувати нелінійні межі, що корисно для складних відмінностей у рельєфі монет. На невеликому наборі даних цей метод часто показує сильний результат, хоча навчання може бути повільнішим за лінійні алгоритми.
""",
    ),
    (
        "Дерево рішень",
        """
## 16. DecisionTreeClassifier — дерево рішень

Дерево рішень послідовно ділить простір ознак за умовами на окремі гілки. Кожен поділ обирається так, щоб краще відокремити класи. Модель легко інтерпретувати, але одне глибоке дерево може пристосуватися до випадкових особливостей навчальної вибірки й гірше узагальнювати нові фотографії.
""",
    ),
    (
        "Випадковий ліс",
        """
## 17. RandomForestClassifier — випадковий ліс

Випадковий ліс навчає багато дерев на різних підвибірках об'єктів та ознак, а потім об'єднує їхні голоси. Усереднення зменшує перенавчання окремого дерева й робить прогноз стабільнішим. Фіксований `random_state` забезпечує повторюваність результату.
""",
    ),
    (
        "Надзвичайно рандомізовані дерева",
        """
## 18. ExtraTreesClassifier — надзвичайно рандомізовані дерева

Extra Trees також є ансамблем дерев, але пороги поділу вибираються більш випадково, ніж у випадковому лісі. Це додатково зменшує схожість між деревами та часто прискорює навчання. Порівняння з Random Forest покаже, чи корисна така сильніша рандомізація для HOG-описів монет.
""",
    ),
    (
        "Градієнтний бустинг",
        """
## 19. GradientBoostingClassifier — градієнтний бустинг

Градієнтний бустинг будує дерева послідовно: кожне наступне намагається виправити помилки вже створеного ансамблю. Метод може знаходити складні нелінійні залежності, але потребує більше часу, бо дерева не можна навчити незалежно. Кількість оцінювачів обмежена, щоб експеримент залишався придатним для звичайного ноутбука.
""",
    ),
    (
        "AdaBoost",
        """
## 20. AdaBoostClassifier — адаптивний бустинг

AdaBoost послідовно додає слабкі класифікатори та збільшує увагу до об'єктів, які попередні кроки визначили неправильно. Підсумковий клас отримується зваженим голосуванням. Алгоритм дає змогу перевірити, чи можна покращити прості правила, концентруючись на складних прикладах монет.
""",
    ),
    (
        "Гаусівський наївний Баєс",
        """
## 21. GaussianNB — гаусівський наївний Баєс

GaussianNB оцінює для кожного класу розподіл значень ознак як нормальний і припускає їхню умовну незалежність. Для HOG це припущення спрощене, бо сусідні компоненти пов'язані, зате модель навчається дуже швидко. Вона є корисною контрольною точкою для порівняння складніших методів.
""",
    ),
    (
        "Багатошаровий перцептрон",
        """
## 22. MLPClassifier — багатошаровий перцептрон

MLPClassifier є невеликою класичною нейронною мережею з одним прихованим шаром. Вона комбінує HOG-ознаки через нелінійні перетворення та навчається методом зворотного поширення помилки. Помірний розмір шару, `early_stopping` і фіксований `random_state` обмежують час та ризик перенавчання; TensorFlow або PyTorch тут не використовуються.
""",
    ),
]

for model_name, explanation in model_sections:
    stage_cells.append(markdown(explanation))
    stage_cells.append(code(f'_ = run_notebook_model("{model_name}")'))


stage_cells.extend(
    [
        markdown(
            """
## 23. Підсумкове порівняння моделей

Результати всіх успішно навчених моделей об'єднуються в одну таблицю та сортуються за accuracy. Окремо зберігається час виконання `fit`, тому можна оцінити не лише якість, а й обчислювальну вартість методу. Внутрішній DataFrame має запитані стовпці `model name`, `accuracy`, `training time`, а для показу в ноутбуці вони перекладаються українською.
"""
        ),
        code(
            """
results_df = pd.DataFrame(
    results_rows,
    columns=["model name", "accuracy", "training time"],
).sort_values("accuracy", ascending=False, ignore_index=True)

results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)
np.savez(OUTPUT_DIR / "model_predictions.npz", **predictions)

comparison_table = results_df.rename(
    columns={
        "model name": "Модель",
        "accuracy": "Точність",
        "training time": "Час навчання, с",
    }
)
display(
    comparison_table.style.format(
        {"Точність": "{:.4f}", "Час навчання, с": "{:.4f}"}
    )
)

plot_model_comparison(
    results_df,
    save_path=OUTPUT_DIR / "model_comparison.png",
    show=True,
);

print(f"Збережено прогнозів моделей у словнику: {len(predictions)}")
print("Прогнози також записано до outputs/model_predictions.npz")
print(f"Моделей із помилками: {len(model_errors)}")
"""
        ),
        markdown(
            """
## Висновок етапів 2–3

Кропи монет перетворено на HOG-вектори, ознаки нормалізовано, а вибірку стратифіковано поділено на навчальну й тестову частини. Десять класичних алгоритмів порівнюються на однакових даних, тому різниця в accuracy відображає поведінку самих методів, а не інший випадковий поділ. Прогнози збережені у словнику `predictions` і надалі можуть бути використані для t-SNE та аналізу помилок.
"""
        ),
        code(
            """
if not results_df.empty:
    best_result = results_df.iloc[0]
    print(
        f"Найвищу точність показала модель «{best_result['model name']}»: "
        f"{best_result['accuracy']:.4f}."
    )
    print(
        "Під час інтерпретації результату слід враховувати невеликий розмір "
        "тестової вибірки та дисбаланс між номіналами."
    )
else:
    print("Жодну модель не вдалося навчити; потрібно переглянути повідомлення про помилки.")
"""
        ),
        markdown(
            """
# Етап 4. Фінальна візуалізація та t-SNE аналіз

На завершальному етапі результати впорядковуються в єдину таблицю, а точність і час навчання показуються на окремих діаграмах. Додатково багатовимірні HOG-ознаки тестових монет зменшуються до двох координат методом t-SNE. Це дає змогу візуально оцінити, наскільки добре номінали відокремлюються один від одного та де виникають помилки моделей.
"""
        ),
        markdown(
            """
## 24. Фінальна таблиця результатів

Таблиця містить назву моделі, accuracy на тестовій вибірці та час виконання навчання. Рядки сортуються за тестовою точністю у спадному порядку, тому найкращий за основною метрикою алгоритм розташований першим. Час залежить від комп'ютера, але дає корисне відносне порівняння складності методів у межах одного запуску.
"""
        ),
        code(
            """
from IPython.display import Markdown

from src.final_analysis import (
    build_final_comparison,
    calculate_tsne_embedding,
    select_accuracy_time_balance,
)
from src.visualization import (
    plot_accuracy_comparison,
    plot_training_time_comparison,
    plot_tsne_classes,
)

final_results_df = build_final_comparison(results_df)
final_results_df.to_csv(OUTPUT_DIR / "final_model_comparison.csv", index=False)

final_table_ua = final_results_df.rename(
    columns={
        "model name": "Модель",
        "test accuracy": "Тестова точність",
        "training time": "Час навчання, с",
    }
)
display(
    final_table_ua.style.format(
        {"Тестова точність": "{:.4f}", "Час навчання, с": "{:.4f}"}
    )
)
"""
        ),
        markdown(
            """
## 25. Порівняння точності та часу навчання

Перша діаграма показує частку правильно класифікованих тестових монет: довший стовпчик означає кращу accuracy. Друга діаграма відображає лише час `fit`: коротший стовпчик відповідає швидшому навчанню. Графіки розділені, тому велике значення часу градієнтного бустингу не заважає читати різницю в точності.
"""
        ),
        code(
            """
plot_accuracy_comparison(
    final_results_df,
    save_path=OUTPUT_DIR / "final_accuracy_comparison.png",
    show=True,
);

plot_training_time_comparison(
    final_results_df,
    save_path=OUTPUT_DIR / "final_training_time_comparison.png",
    show=True,
);
"""
        ),
        markdown(
            """
## 26. Що показує t-SNE

t-SNE — це метод нелінійного зменшення розмірності, який намагається розташувати схожі об'єкти поруч, а несхожі — далі один від одного. Він потрібен тому, що HOG-вектор має 1764 ознаки й людина не може безпосередньо побачити такий простір. Зменшення до двох вимірів дозволяє намалювати кожну тестову монету однією точкою на площині.

На scatter plot колір відповідає класу, а відстань між точками приблизно відображає локальну схожість HOG-описів. Окремі компактні кольорові групи означають, що класи добре розрізняються. Якщо кольори перекриваються, моделі можуть плутати ці номінали. Для монет перекриття природне: різні номінали мають круглу форму, подібні краї та текстури, а освітлення, поворот, фон і сторона монети змінюють вигляд навіть у межах одного класу. Глобальні відстані між далекими групами t-SNE не слід трактувати як точну математичну міру.
"""
        ),
        code(
            """
tsne_embedding, safe_perplexity = calculate_tsne_embedding(
    prepared_data.X_test,
    random_state=42,
)
np.save(OUTPUT_DIR / "tsne_embedding.npy", tsne_embedding)

print(f"Кількість тестових об'єктів: {len(prepared_data.X_test)}")
print(f"Автоматично обрана безпечна perplexity: {safe_perplexity}")
print(f"Форма двовимірного t-SNE представлення: {tsne_embedding.shape}")
"""
        ),
        markdown(
            """
## 27. t-SNE для справжніх класів

Спочатку точки зафарбовуються за правильними мітками з тестової вибірки. Цей графік показує реальну структуру класів у просторі HOG-ознак після зменшення розмірності. Він є еталоном для наступних графіків із прогнозами моделей.
"""
        ),
        code(
            """
plot_tsne_classes(
    tsne_embedding,
    prepared_data.y_test,
    class_names_ua,
    title="t-SNE тестових монет: справжні класи",
    save_path=OUTPUT_DIR / "tsne" / "true_classes.png",
    show=True,
);
"""
        ),
        markdown(
            """
## 28. t-SNE для прогнозів усіх моделей

Координати точок залишаються такими самими, але колір тепер задається прогнозом конкретної моделі. Якщо кольорова структура близька до графіка справжніх класів, модель добре відтворює реальний поділ. Точки зі зміненим кольором відповідають помилкам класифікації. Окремий scatter plot будується для кожної з десяти навчених моделей.
"""
        ),
        code(
            """
for model_number, model_name in enumerate(models, start=1):
    if model_name not in predictions:
        print(f"Для моделі «{model_name}» немає прогнозів, тому графік пропущено.")
        continue

    plot_tsne_classes(
        tsne_embedding,
        predictions[model_name],
        class_names_ua,
        title=f"t-SNE прогнозованих класів: {model_name}",
        save_path=OUTPUT_DIR / "tsne" / f"model_{model_number:02d}.png",
        show=True,
    )
"""
        ),
        markdown(
            """
## 29. Фінальні висновки

Найвищу тестову точність у проведеному експерименті показала **логістична регресія** — приблизно 85,3%. Найменший час власне навчання мав **KNN**, оскільки під час `fit` цей метод переважно запам'ятовує навчальні дані, а основні обчислення переносить на прогнозування. Найкращим практичним балансом за прийнятим правилом став **SVC**: серед моделей, що відстають від лідера не більше ніж на 10 відсоткових пунктів, він навчався найшвидше.

Окреме дерево рішень та AdaBoost спрацювали гірше, оскільки прості деревоподібні правила недостатньо стабільно описують велику кількість пов'язаних HOG-ознак. GaussianNB також обмежений припущенням про незалежність ознак, яке для сусідніх градієнтів не виконується. Результат не є ідеальним через невеликий набір із 336 монет, нерівномірну кількість класів, різне освітлення, положення, повороти та фони зображень.

HOG і класичні алгоритми простіші за CNN та глибоке навчання: вони потребують менше даних, обчислень і часу на налаштування. Водночас HOG добре описує контури цифр, обідка та рельєфу монет, тому такий підхід достатньо обґрунтований для університетського дослідження й дозволяє чесно порівняти багато методів на одному наборі ознак.
"""
        ),
        code(
            """
best_model = final_results_df.iloc[0]
fastest_model = final_results_df.sort_values("training time").iloc[0]
balanced_model = select_accuracy_time_balance(final_results_df)

display(
    Markdown(
        f"**Перевірка висновків за поточним запуском.** "
        f"Найкраща модель — **{best_model['model name']}** "
        f"із точністю **{best_model['test accuracy']:.4f}**. "
        f"Найшвидше навчалася модель **{fastest_model['model name']}** "
        f"за **{fastest_model['training time']:.4f} с**. "
        f"Найкращий баланс точності й часу за описаним правилом має "
        f"**{balanced_model['model name']}**."
    )
)
"""
        ),
        markdown(
            """
# Коротко для захисту

1. У роботі використано набір EURO coins із фотографіями восьми номіналів монет євро.
2. Початково це був набір для детекції об'єктів, тому одна фотографія могла містити кілька монет.
3. Для багатокласової класифікації кожну розмічену монету потрібно було перетворити на окремий приклад.
4. YOLO-анотації містили номер класу та нормалізовані координати центра, ширини й висоти рамки.
5. Координати переводилися в пікселі за фактичним розміром фотографії, після чого монети вирізалися й масштабувалися до `64x64`.
6. Кропи переводилися у відтінки сірого, а з них виділялися розгорнуті пікселі та HOG-ознаки.
7. Основними були HOG-ознаки, бо вони описують напрямки країв, цифри, обідок і рельєф монети.
8. Ознаки нормалізувалися методом `StandardScaler`, щоб їхній масштаб не спотворював KNN, SVM, логістичну регресію та MLP.
9. Дані стратифіковано поділили на 80% для навчання і 20% для тестування.
10. Було навчено десять моделей: Logistic Regression, KNN, SVC, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, GaussianNB та MLPClassifier.
11. Якість оцінювалася за accuracy, precision, recall і F1-мірою, а також враховувався час навчання.
12. Матриця помилок показує, скільки монет кожного справжнього класу модель віднесла до кожного прогнозованого класу.
13. t-SNE зменшує 1764 HOG-ознаки до двох координат і показує, які номінали утворюють окремі групи, а які перекриваються.
14. Найкращу точність у цьому експерименті показала логістична регресія — близько 85,3%.
15. Підсумковий висновок полягає в тому, що HOG разом із класичними моделями дає добрий і зрозумілий результат для невеликого університетського набору, хоча більший датасет і CNN могли б підвищити точність.
"""
        ),
    ]
)

notebook["cells"] = base_cells + stage_cells
NOTEBOOK_PATH.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(f"Ноутбук оновлено: {NOTEBOOK_PATH}")
