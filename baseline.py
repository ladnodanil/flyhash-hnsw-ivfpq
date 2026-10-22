import os
import pickle
import tempfile
import time

import psutil
from sklearn.neighbors import NearestNeighbors

from config_lab import Config
from utils import _load_dataset, _split_dataset


def baseline(config: Config | None = None):
    config = config or Config()

    X_full = _load_dataset(config.name_dataset)

    X_train, X_test = _split_dataset(
        X_full=X_full,
        train_size=config.train_size,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    process = psutil.Process()

    # ----------------- ПОСТРОЕНИЕ ИНДЕКСА -----------------
    nn_model = NearestNeighbors(
        metric=config.nn_metric,
        n_neighbors=config.n_neighbors,
        algorithm=config.algorithm,
    )

    mem_before = process.memory_info().rss
    start_time = time.perf_counter()

    nn_model.fit(X_train)

    build_time = time.perf_counter() - start_time
    mem_after = process.memory_info().rss

    index_memory_mb = (mem_after - mem_before) / (1024 ** 2)

    # ----------------- СЕРИАЛИЗАЦИЯ ИНДЕКСА -----------------
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "baseline.pkl")

        with open(index_path, "wb") as f:
            pickle.dump(nn_model, f, protocol=pickle.HIGHEST_PROTOCOL)

        serialized_index_mb = os.path.getsize(index_path) / (1024 ** 2)

    # ----------------- ПОИСК -----------------
    query_start = time.perf_counter()

    distances, indices = nn_model.kneighbors(X_test)

    query_time = time.perf_counter() - query_start
    query_time_per_item = query_time / len(X_test)

    # ----------------- РЕЗУЛЬТАТЫ -----------------
    baseline_results = {
        "indices": indices,
        "distances": distances,
        "build_time": build_time,
        "query_time": query_time_per_item,
        "index_memory_mb": index_memory_mb,
        "serialized_index_mb": serialized_index_mb,
    }

    with open(config.output_file, "wb") as f:
        pickle.dump(baseline_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n--- (Этап 1: Baseline) ---")
    print(f"Время построения индекса:      {build_time:.4f} сек")
    print(f"Среднее время поиска:          {query_time_per_item * 1000:.4f} мс")
    print(f"Прирост RSS:                   {index_memory_mb:.2f} MiB")
    print(f"Размер сериализованного индекса: {serialized_index_mb:.2f} MiB")
    print(f"Сохранено в: {config.output_file}")

    return baseline_results


if __name__ == "__main__":
    baseline()