from src.config import Config
from src.utils import Transformer
from src.preprocess import Dataset
from src.inference import SemanticSearcher
from src.train import train
from src.train import Mapper
import torch
# from src.inference import run_inference

# def main(mode="train"):
#     # Load datasets
#     ms_marco_dataset = Dataset('ms-marco')
#     hotpot_dataset = Dataset('hotpot-qa')

#     # Initialize transformer
#     transformer = Transformer(Config.TRANSFORMER_MODEL_NAMES.ALL_MPNET_BASE_V2)
#     # Initialize semantic searcher
#     semantic_searcher = SemanticSearcher(Config.VECTOR_DB_NAMES.FAISS, Config.SIMILARITY_METRIC_NAMES.CS)

#     if mode == "train":
#         train_mapper(ms_marco_dataset, transformer, semantic_searcher)
#     elif mode == "inference":
#         run_inference(ms_marco_dataset, transformer, semantic_searcher)

# if __name__ == "__main__":
#     main(mode="train")


# src/main.py

def main():
    # 1️⃣ Initialize device
    device = Config.Device.DEVICE
    # Load datasets
    ms_marco_dataset = Dataset('ms-marco')
    hotpot_dataset = Dataset('hotpot-qa')

    # 2️⃣ Load dataset
    query_embs = ms_marco_dataset.query_embeddings  # numpy array or tensor
    passage_embs = ms_marco_dataset.passage_embeddings
    baseline_indices = ms_marco_dataset.baseline_retrieved  # list of retrieved indices

    # 3️⃣ Initialize mapper
    mapper = Mapper(dim=query_embs.shape[1]).to(device)

    # 4️⃣ Train mapper
    trained_mapper = train(
        mapper,
        ms_marco_dataset,
        query_embs,
        passage_embs,
        baseline_indices,
        epochs=50,
        patience=3,
        batch_size=512,
        lr=1e-3,
        total=4,
        pos_ratio=0.75,
        mode="worst-worst",
        margin=0.2,
        norm=2
    )

    # 5️⃣ Save trained mapper
    torch.save(trained_mapper.state_dict(), "trained_mapper.pt")
    print("Training completed and model saved!")

if __name__ == "__main__":
    main()
