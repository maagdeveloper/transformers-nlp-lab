# Mini Transformer NLP Lab

Small, notebook-first NLP project for training and fine-tuning transformer models from scratch.

## What is inside

- `notebooks/Mini-Decoder-LLM/`: decoder-only pipeline (`LlamaForCausalLM`)
  - data prep -> tokenizer -> pretraining
  - assistant SFT
  - downstream classification
  - inference notebooks

- `notebooks/Mini-Encoder-LLM/`: encoder pipeline (`BertForMaskedLM`)
  - data prep -> tokenizer -> MLM pretraining
  - downstream classification, NER, QA
  - contrastive embedding fine-tuning (`embed_model`)
  - export embeddings for projector

- `embedding-projector-standalone/`: local TensorFlow Embedding Projector app
  - `index.html`
  - `oss_data/oss_demo_projector_config.json` (expects `embeddings.tsv` + `metadata.tsv`)

- `notebooks/smol135-instruct.ipynb`: quick reference notebook for SmolLM2 instruct usage.

## Main outputs

Training artifacts are saved inside notebook folders, including:
- `model/`
- `assistant/`
- `classifier/`
- `ner/`
- `qa/`
- `embed_model/`

## Run order (recommended)

1. Run notebooks in numeric order inside each track.
2. For embeddings visualization, run:
   - `Mini-Encoder-LLM/6. fine-tune -> contrastive-similarity-embeddings ...`
   - `Mini-Encoder-LLM/7. export embeddings for embeddings-projector.ipynb`
3. Put exported `embeddings.tsv` and `metadata.tsv` under `embedding-projector-standalone/oss_data/`, then open `embedding-projector-standalone/index.html`.

## Dependencies

Install from `requirements.txt`.
GPU is strongly recommended for training/fine-tuning.
