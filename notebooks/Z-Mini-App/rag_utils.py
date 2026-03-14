import torch
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(embed_model_id, device=device)

index_data = joblib.load("../Y-Mini-RAG/rag_index.pkl")
all_chunks = index_data["chunks"]
chunk_embeddings = index_data["embeddings"]

def encode(texts, batch_size=32):
    embeddings = embed_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings.cpu()


def retrieve(query, top_k=2):
    query_emb = encode([query]).numpy()[0]
    scores = np.dot(chunk_embeddings, query_emb)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "chunk_id": int(idx),
            "score": float(scores[idx]),
            "text": all_chunks[idx]
        })

    return results


def rag_pipeline(query):
    retrieved = retrieve(query, top_k=1)
    best = retrieved[0]

    return {
        "type": "rag",
        "query": query,
        "answer": best["text"],
        "score": best["score"],
        "retrieved_chunks": retrieved
    }