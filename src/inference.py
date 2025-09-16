import numpy as np
import faiss
# import scann
from src.config import Config

class SemanticSearcher:
    def __init__(self, db_type: Config.VectorDBNames, similarity: Config.SimilarityMetricNames, dim: int, top_k: int = 50):
        self.db_type = db_type
        self.similarity = similarity
        self.dim = dim
        self.top_k = top_k
        self.engine = self._build_engine()

    def _build_engine(self):
        if self.db_type == Config.VectorDBNames.FAISS:
            if self.similarity == Config.SimilarityMetricNames.L2:
                return faiss.IndexFlatL2(self.dim)
            elif self.similarity in [Config.SimilarityMetricNames.IP, Config.SimilarityMetricNames.CS]:
                return faiss.IndexFlatIP(self.dim)
            else:
                raise ValueError("Unsupported FAISS similarity metric")

        elif self.db_type == Config.VectorDBNames.SCANN:
            # Placeholder — ScaNN is built during indexing
            return None

        else:
            raise ValueError("Unsupported vector DB type")

    def index(self, embeddings: np.ndarray):
        if self.db_type == Config.VectorDBNames.FAISS:
            if self.similarity == Config.SimilarityMetricNames.CS:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.engine.add(embeddings)

        elif self.db_type == Config.VectorDBNames.SCANN:
            if self.similarity == Config.SimilarityMetricNames.L2:
                self.engine = scann.scann_ops_pybind.builder(
                    embeddings, self.top_k, "squared_l2"
                ).score_brute_force().build()

            elif self.similarity in [Config.SimilarityMetricNames.IP, Config.SimilarityMetricNames.CS]:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                self.engine = scann.scann_ops_pybind.builder(
                    embeddings, self.top_k, "dot_product"
                ).score_brute_force().build()

            else:
                raise ValueError("Unsupported ScaNN similarity metric")

    def search(self, query_embeddings: np.ndarray) -> np.ndarray:
        if self.db_type == Config.VectorDBNames.FAISS:
            if self.similarity == Config.SimilarityMetricNames.CS:
                query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            _, indices = self.engine.search(query_embeddings, self.top_k)
            return indices

        elif self.db_type == Config.VectorDBNames.SCANN:
            results = []
            for q in query_embeddings:
                _, ids = self.engine.search(q)
                results.append(ids)
            return np.array(results)

        else:
            raise ValueError("Unsupported vector DB type")
