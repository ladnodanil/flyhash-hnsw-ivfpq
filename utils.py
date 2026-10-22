
import pickle

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def _load_dataset(dataset_name: str) -> np.ndarray:
    try:
        if dataset_name == "mnist_784":
            X, _ = fetch_openml(data_id=554, return_X_y=True, as_frame=False)
        else:
            X, _ = fetch_openml(dataset_name, version=1, return_X_y=True, as_frame=False)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Не удалось загрузить датасет {dataset_name!r} из OpenML.") from exc

    try:
        return (np.asarray(X) / 255.0).astype(np.float32)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Данные датасета {dataset_name!r} имеют неподдерживаемый формат.") from exc


def _split_dataset(
    X_full: np.ndarray,
    train_size: int,
    test_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if X_full.ndim != 2:
        raise ValueError(f"Ожидался двумерный массив, получена форма {X_full.shape}.")
    if train_size <= 0 or test_size <= 0:
        raise ValueError("Размеры обучающей и тестовой выборок должны быть положительными.")
    if train_size + test_size > len(X_full):
        raise ValueError(
            f"Нельзя выбрать {train_size} обучающих и {test_size} тестовых объектов "
            f"из {len(X_full)} доступных."
        )

    try:
        X_train, X_remaining = train_test_split(
            X_full,
            train_size=train_size,
            random_state=random_state,
            shuffle=True,
        )

        X_test, _ = train_test_split(
            X_remaining,
            train_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
    except ValueError as exc:
        raise ValueError(
            f"Не удалось разделить датасет при размерах train={train_size}, test={test_size}."
        ) from exc

    return X_train, X_test


def _load_baseline_results(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as file:
            baseline_data = pickle.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Файл baseline-результатов {file_path!r} не найден. "
            "Сначала запустите baseline.py."
        ) from exc
    except OSError as exc:
        raise OSError(f"Не удалось прочитать файл baseline-результатов {file_path!r}.") from exc
    except (pickle.UnpicklingError, EOFError) as exc:
        raise ValueError(f"Файл baseline-результатов {file_path!r} повреждён.") from exc

    if not isinstance(baseline_data, dict) or "indices" not in baseline_data:
        raise ValueError(
            f"Файл {file_path!r} не содержит словарь baseline-результатов с ключом 'indices'."
        )
    return baseline_data


def recall_k(true_ids, pred_ids, k):
    if k <= 0:
        raise ValueError(f"Параметр k должен быть положительным, получено {k}.")
    if len(true_ids) != len(pred_ids):
        raise ValueError("Число эталонных и предсказанных наборов соседей не совпадает.")
    if len(true_ids) == 0:
        raise ValueError("Невозможно вычислить Recall для пустых наборов соседей.")
    if any(len(ids) < k for ids in true_ids) or any(len(ids) < k for ids in pred_ids):
        raise ValueError(f"Каждый набор соседей должен содержать не менее {k} элементов.")

    final_recall_k = np.mean([
        len(set(t[:k]).intersection(set(p[:k]))) / k
        for t, p in zip(true_ids, pred_ids)
    ])
    return float(final_recall_k)
