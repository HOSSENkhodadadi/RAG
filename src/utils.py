import torch
from typing import Any
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import faiss
# import scann
from src.config import Config

class Transformer:
    def __init__(self, model_name: str, batch_size: int = 128, normalize: bool = False):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name).to(self.device)

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=self.normalize
        )
        return embeddings
def build_index(vectors: np.ndarray, db_type: Config.VectorDBNames, similarity: Config.SimilarityMetricNames):
    if db_type == Config.VectorDBNames.FAISS:
        if similarity == Config.SimilarityMetricNames.L2:
            index = faiss.IndexFlatL2(vectors.shape[1])
        elif similarity == Config.SimilarityMetricNames.IP:
            index = faiss.IndexFlatIP(vectors.shape[1])
        elif similarity == Config.SimilarityMetricNames.CS:
            faiss.normalize_L2(vectors)
            index = faiss.IndexFlatIP(vectors.shape[1])
        else:
            raise ValueError("Unsupported similarity metric for FAISS")

        index.add(vectors)
        return index

    elif db_type == Config.VectorDBNames.SCANN:
        searcher = scann.scann_ops_pybind.builder(vectors, 10, "dot_product").build()
        return searcher

    else:
        raise ValueError("Unsupported vector DB type")
        


def calculate_similarity(query_vecs: np.ndarray, passage_vecs: np.ndarray, similarity: Config.SimilarityMetricNames):
    if similarity == Config.SimilarityMetricNames.CS:
        query_vecs = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
        passage_vecs = passage_vecs / np.linalg.norm(passage_vecs, axis=1, keepdims=True)
        return np.matmul(query_vecs, passage_vecs.T)
    
    elif similarity == Config.SimilarityMetricNames.L2:
        return -np.linalg.norm(query_vecs[:, None] - passage_vecs, axis=2)

    elif similarity == Config.SimilarityMetricNames.IP:
        return np.matmul(query_vecs, passage_vecs.T)

    else:
        raise ValueError("Unknown similarity metric")

def plot_metrics(metric_dict: dict[str, float], title: str = "Metric"):
    keys = list(metric_dict.keys())
    values = list(metric_dict.values())

    plt.figure(figsize=(10, 5))
    plt.bar(keys, values)
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.title(title)
    plt.grid(True)
    plt.show()
