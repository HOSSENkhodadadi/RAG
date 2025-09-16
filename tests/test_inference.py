import unittest
from src.inference import SemanticSearcher
from src.config import Config

class TestSemanticSearcher(unittest.TestCase):
    def test_faiss_init(self):
        searcher = SemanticSearcher(Config.VectorDB.FAISS, Config.SimilarityMetrics.CS)
        self.assertIsNotNone(searcher)

if __name__ == "__main__":
    unittest.main()
