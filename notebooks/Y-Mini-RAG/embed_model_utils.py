from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(embed_model_id, device=device)

def encode(texts, batch_size=32):
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings.cpu()