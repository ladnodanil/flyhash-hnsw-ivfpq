
import os
import pickle
import tempfile
import time

import numpy as np
import psutil
from flyhash import FlyHash

from config_lab import Config
from utils import _load_baseline_results, _load_dataset, _split_dataset, recall_k


def flyhash_flow(config: Config | None = None):
    config = config or Config()

    baseline_data = _load_baseline_results(config.output_file)

    X_full = _load_dataset(config.name_dataset)
    X_train, X_test = _split_dataset(
        X_full=X_full,
        train_size=config.train_size,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    # ----------------- ЭТАП ОБУЧЕНИЯ / ХЭШИРОВАНИЯ -----------------
    process = psutil.Process()

    mem_before = process.memory_info().rss
    start_train_time = time.perf_counter()

    hasher = FlyHash(
        input_dim=X_train.shape[1],
        hash_dim=config.hash_dimension,
        sparsity=config.sparsity_ratio,
        seed=config.random_state,
    )

    train_hashes = hasher(X_train)

    build_time = time.perf_counter() - start_train_time
    mem_after = process.memory_info().rss

    index_memory_mb = (mem_after - mem_before) / (1024 ** 2)

    # ----------------- СЕРИАЛИЗАЦИЯ ИНДЕКСА -----------------
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "flyhash.pkl")

        with open(index_path, "wb") as f:
            pickle.dump(
                {
                    "hasher": hasher,
                    "train_hashes": train_hashes,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        serialized_index_mb = os.path.getsize(index_path) / (1024 ** 2)

    # ----------------- ЭТАП ПОИСКА -----------------
    start_query_time = time.perf_counter()

    test_hashes = hasher(X_test)
    total_test = len(X_test)

    pred_indices = np.zeros(
        (total_test, config.n_neighbors),
        dtype=np.int64,
    )

    for i in range(total_test):
        current_test_hash = test_hashes[i]

        distances = np.count_nonzero(
            train_hashes != current_test_hash,
            axis=1,
        )

        top_k = np.argpartition(
            distances,
            config.n_neighbors - 1,
        )[:config.n_neighbors]

        pred_indices[i] = top_k[np.argsort(distances[top_k])]

    query_time = time.perf_counter() - start_query_time
    query_time_per_item = query_time / total_test

    # ----------------- ВЫЧИСЛЕНИЕ RECALL -----------------
    recall_1 = recall_k(baseline_data["indices"], pred_indices, 1)
    recall_5 = recall_k(baseline_data["indices"], pred_indices, 5)
    recall_10 = recall_k(baseline_data["indices"], pred_indices, 10)

    # ----------------- ВЫВОД РЕЗУЛЬТАТОВ -----------------
    print("\n" + "=" * 55)
    print("        (Этап 2: FlyHash)        ")
    print("=" * 55)

    print(f"Recall@1:                         {recall_1:.4f}")
    print(f"Recall@5:                         {recall_5:.4f}")
    print(f"Recall@10:                        {recall_10:.4f}")

    print("-" * 55)

    print(f"Время генерации хэшей (Train):    {build_time:.4f} сек")
    print(f"Общее время поиска ({total_test} точек): {query_time:.4f} сек")
    print(f"Среднее время поиска на запрос:   {query_time_per_item * 1000:.4f} мс")
    print(f"Прирост RSS:                      {index_memory_mb:.2f} MiB")
    print(f"Размер сериализованного индекса:  {serialized_index_mb:.2f} MiB")

    print("=" * 55)

    return {
        "recall@1": recall_1,
        "recall@5": recall_5,
        "recall@10": recall_10,
        "build_time": build_time,
        "query_time": query_time_per_item,
        "index_memory_mb": index_memory_mb,
        "serialized_index_mb": serialized_index_mb,
    }


if __name__ == "__main__":
    flyhash_flow()
