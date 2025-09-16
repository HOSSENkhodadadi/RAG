# RAG
# Improving Retrieval Mechanism in Retrieval-Augmented Generation Architecture

This project performs semantic search using a custom-trained mapper and FAISS/ScaNN.

## Features

- FAISS/ScaNN vector search
- SentenceTransformer embeddings
- Custom trainable mapper network
- HotpotQA + MS MARCO datasets

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
