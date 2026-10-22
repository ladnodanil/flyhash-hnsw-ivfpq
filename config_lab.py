
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    name_dataset: str = "mnist_784"
    # baseline parameters
    nn_metric: str = "euclidean"
    n_neighbors: int = 10
    algorithm: str = "brute"
    train_size: int = 10_000
    test_size: int = 500
    random_state: int = 42
    output_file: str = "baseline_bruteforce_knn_result.PKL"
    # flyhash parameters
    hash_dimension: int = 7840
    sparsity_ratio: float = 0.05
    flyhash_output_file: str = "flyhash_result.PKL"
    encoding_batch_size: int = 128
    # hnsw parameters
    space: str = 'l2'
    ef_construction: int = 200
    ef: int = 50
    # ivfpq parameters
    nlist: int = 100
    m_subquantizers: int = 28 # 784/28= 28
    bits_per_code: int = 8
    nprobe: int = 10
    


