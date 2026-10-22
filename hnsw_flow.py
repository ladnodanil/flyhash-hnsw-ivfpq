import os
import tempfile
import time

import psutil
from hnswlib import Index

from config_lab import Config
from utils import _load_baseline_results, _load_dataset, _split_dataset, recall_k


def hnsw_flow(config: Config | None = None):
    config = config or Config()

    baseline_data = _load_baseline_results(config.output_file)

    X_full = _load_dataset(config.name_dataset)
    X_train, X_test = _split_dataset(
        X_full=X_full,
        train_size=config.train_size,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    # ----------------- ПОСТРОЕНИЕ ИНДЕКСА -----------------
    process = psutil.Process()

    mem_before = process.memory_info().rss
    build_start = time.perf_counter()

    hnsw = Index(space=config.space, dim=X_train.shape[1])

    hnsw.init_index(
        max_elements=len(X_train),
        ef_construction=config.ef_construction,
        random_seed=config.random_state,
    )

    hnsw.add_items(X_train)

    build_time = time.perf_counter() - build_start
    mem_after = process.memory_info().rss

    index_memory_mb = (mem_after - mem_before) / (1024**2)

    # ----------------- СЕРИАЛИЗАЦИЯ ИНДЕКСА -----------------
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "hnsw.bin")

        hnsw.save_index(index_path)

        serialized_index_mb = os.path.getsize(index_path) / (1024**2)

    # ----------------- ПОИСК -----------------
    hnsw.set_ef(config.ef)

    query_start = time.perf_counter()

    pred_indices, _ = hnsw.knn_query(
        X_test,
        k=config.n_neighbors,
    )

    query_time = time.perf_counter() - query_start
    query_time_per_item = query_time / len(X_test)

    # ----------------- ВЫЧИСЛЕНИЕ RECALL -----------------
    recall_1 = recall_k(baseline_data["indices"], pred_indices, 1)
    recall_5 = recall_k(baseline_data["indices"], pred_indices, 5)
    recall_10 = recall_k(baseline_data["indices"], pred_indices, 10)

    # ----------------- ВЫВОД РЕЗУЛЬТАТОВ -----------------
    print("\n" + "=" * 55)
    print("         (Этап 3: HNSW)         ")
    print("=" * 55)

    print(f"Recall@1:                         {recall_1:.4f}")
    print(f"Recall@5:                         {recall_5:.4f}")
    print(f"Recall@10:                        {recall_10:.4f}")

    print("-" * 55)

    print(f"Время построения индекса:         {build_time:.4f} сек")
    print(f"Общее время поиска ({len(X_test)} точек): {query_time:.4f} сек")
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
    hnsw_flow()
