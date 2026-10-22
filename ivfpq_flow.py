import os
import tempfile
import time

import faiss
import psutil

from config_lab import Config
from utils import _load_baseline_results, _load_dataset, _split_dataset, recall_k


def ivfpq_flow(config: Config | None = None):
    config = config or Config()

    baseline_data = _load_baseline_results(config.output_file)

    X_full = _load_dataset(config.name_dataset)
    X_train, X_test = _split_dataset(
        X_full=X_full,
        train_size=config.train_size,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    # ----------------- ОБУЧЕНИЕ И ПОСТРОЕНИЕ ИНДЕКСА -----------------
    process = psutil.Process()

    mem_before = process.memory_info().rss
    build_start = time.perf_counter()

    dimension = X_train.shape[1]
    quantizer = faiss.IndexFlatL2(dimension)

    index = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        config.nlist,
        config.m_subquantizers,
        config.bits_per_code,
    )

    index.train(X_train)
    index.add(X_train)

    build_time = time.perf_counter() - build_start
    mem_after = process.memory_info().rss

    index_memory_mb = (mem_after - mem_before) / (1024**2)

    # ----------------- СЕРИАЛИЗАЦИЯ ИНДЕКСА -----------------
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "ivfpq.index")

        faiss.write_index(index, index_path)

        serialized_index_mb = os.path.getsize(index_path) / (1024**2)

    # ----------------- ПОИСК -----------------
    index.nprobe = config.nprobe

    query_start = time.perf_counter()

    _, pred_indices = index.search(
        X_test,
        config.n_neighbors,
    )

    query_time = time.perf_counter() - query_start
    query_time_per_item = query_time / len(X_test)

    # ----------------- ВЫЧИСЛЕНИЕ RECALL -----------------
    recall_1 = recall_k(baseline_data["indices"], pred_indices, 1)
    recall_5 = recall_k(baseline_data["indices"], pred_indices, 5)
    recall_10 = recall_k(baseline_data["indices"], pred_indices, 10)

    # ----------------- ВЫВОД РЕЗУЛЬТАТОВ -----------------
    print("\n" + "=" * 55)
    print("         (Этап 4: IVFPQ)        ")
    print("=" * 55)

    print(f"Recall@1:                         {recall_1:.4f}")
    print(f"Recall@5:                         {recall_5:.4f}")
    print(f"Recall@10:                        {recall_10:.4f}")

    print("-" * 55)

    print(f"Время обучения и построения:      {build_time:.4f} сек")
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
    ivfpq_flow()
