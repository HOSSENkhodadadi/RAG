# import enum

# class Config:
#     DATASET_ROOT = 'data/'
#     DEVICE = 'cuda'  # This will be overridden in train.py using torch

#     class DatasetNames(enum.Enum):
#         MS_MARCO = 'ms-marco'
#         HOTPOT_QA = 'hotpot-qa'

#     class TransformerModels(enum.Enum):
#         ALL_MPNET_BASE_V2 = 'all-mpnet-base-v2'

#     class VectorDB(enum.Enum):
#         FAISS = 'faiss'
#         SCANN = 'scann'

#     class SimilarityMetrics(enum.Enum):
#         L2 = 'l2'
#         IP = 'ip'
#         CS = 'cs'

import enum
import os
import torch
class Config:
    class Device:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    class Paths:
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        DATASET_ROOT = os.path.join(ROOT_DIR, 'data')
        MODEL_DIR = os.path.join(ROOT_DIR, 'models')
        SCHEMA_DIR = os.path.join(DATASET_ROOT, 'schemas')

    class Retrieval:
        RETRIEVAL_CAPACITY = 50
        EMBEDDING_DIM = 768

    class DatasetNames(enum.Enum):
        MS_MARCO = 'ms-marco'
        HOTPOT_QA = 'hotpot-qa'

    class TransformerModelNames(enum.Enum):
        ALL_MPNET_BASE_V2 = 'all-mpnet-base-v2'
        MULTI_QA_MPNET_BASE_DOT_V1 = 'multi-qa-mpnet-base-dot-v1'
        ALL_DISTILROBERTA_V1 = 'all-distilroberta-v1'

    class VectorDBNames(enum.Enum):
        FAISS = 'faiss'
        SCANN = 'scann'

    class SimilarityMetricNames(enum.Enum):
        L2 = 'l2'
        IP = 'ip'
        CS = 'cs'

    class AggregatorNames(enum.Enum):
        MIN_MIN_0 = 'min-min-0'
        MIN_AVG_0 = 'min-avg-0'
        MIN_MAX_0 = 'min-max-0'
        AVG_MIN_0 = 'avg-min-0'
        AVG_AVG_0 = 'avg-avg-0'
        AVG_MAX_0 = 'avg-max-0'
        MAX_MIN_0 = 'max-min-0'
        MAX_AVG_0 = 'max-avg-0'
        MAX_MAX_0 = 'max-max-0'
        MIN_MIN_1 = 'min-min-1'
        MIN_AVG_1 = 'min-avg-1'
        MIN_MAX_1 = 'min-max-1'
        AVG_MIN_1 = 'avg-min-1'
        AVG_AVG_1 = 'avg-avg-1'
        AVG_MAX_1 = 'avg-max-1'
        MAX_MIN_1 = 'max-min-1'
        MAX_AVG_1 = 'max-avg-1'
        MAX_MAX_1 = 'max-max-1'

    class TokenizerNames(enum.Enum):
        SIMPLE = 'simple'
        LEMMA = 'lemmatization'
