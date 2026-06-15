# Euro Coin Classifier

Дослідження різних методів багатокласової класифікації для визначення номіналу монети євро на основі набору даних [EURO coins dataset](https://www.kaggle.com/datasets/janstaffa/euro-coins-dataset).

## Опис

- 336 кропів монет з 150 фотографій (8 класів: 1, 2, 5, 10, 20, 50 центів, 1, 2 євро)
- Ознаки: HOG-дескриптори (1764 компоненти)
- 10 методів класифікації: Logistic Regression, KNN, SVC, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, GaussianNB, MLP
- Найкращий результат: Логістична регресія — 0.8529

## Структура

- `notebooks/` — Jupyter notebook з повним аналізом
- `outputs/` — графіки, матриці помилок, t-SNE, CSV з результатами
- `report/` — звіт PDF
- `src/` — Python модулі

## Датасет

Завантажити з Kaggle: https://www.kaggle.com/datasets/janstaffa/euro-coins-dataset та розпакувати в папку `data/`

## Запуск

```bash
pip install -r requirements.txt
jupyter notebook notebooks/euro_coin_classification.ipynb
```
