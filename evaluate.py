from src.inference import SemanticSearcher
from src.config import VectorDBNames, SimilarityMetricNames

searcher = SemanticSearcher(VectorDBNames.FAISS, SimilarityMetricNames.CS, dim=768)
searcher.index(passage_embeddings)
retrieved = searcher.search(query_embeddings)
