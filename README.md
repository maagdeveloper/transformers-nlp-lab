# Mini LLM Notebooks

This repository is a notebook-first playground for building and fine-tuning small language models from scratch.

It contains two end-to-end tracks under `notebooks/`:
- `Mini-Decoder-LLM`: a GPT-style decoder model (`LlamaForCausalLM`) with optional SFT assistant tuning, plus downstream classification fine-tuning.
- `Mini-Encoder-LLM`: a BERT-style encoder model (`BertForMaskedLM`) with downstream classification, NER, and QA fine-tuning.

## Project Structure

`notebooks/Mini-Decoder-LLM/`
- `0. prepare data.ipynb`: builds `data.txt` from FineWeb and cleans/filters text.
- `1. train tokenizer.ipynb`: trains a ByteLevel BPE tokenizer (`vocab_size=32000`).
- `2. pretrain model from scratch.ipynb`: pretrains a small decoder LM.
- `3. prepare tokenizer for chat_template.ipynb`: adds chat template for assistant-style SFT.
- `4. sft fine-tune assistant.ipynb`: instruction tuning using Alpaca + Dolly (+ formatted sentiment examples).
- `5. fine-tune -> Classification.ipynb`: sentiment classification on `tweet_eval`.
- `6. fine-tune -> NER.ipynb`: token classification on CoNLL-2003.
- `inference.ipynb` and `zz.ipynb`: generation and task-specific inference experiments.

`notebooks/Mini-Encoder-LLM/`
- `1. train tokenizer.ipynb`: trains a WordPiece tokenizer (`vocab_size=30000`).
- `2. pretrain model MLM from scratch.ipynb`: pretrains a compact BERT MLM.
- `3. fine-tune -> Classification.ipynb`: sentiment classification on `tweet_eval`.
- `4. fine-tune -> NER.ipynb`: token classification on CoNLL-2003.
- `4. fine-tune -> QA.ipynb`: question answering on SQuAD.
- `inference.ipynb`: fill-mask and classifier inference checks.

Other:
- `notebooks/smol135-instruct.ipynb`: reference experiments with Hugging Face SmolLM2 instruction models.

## Saved Artifacts

Each track stores local training outputs next to notebooks:
- `model/`: base pretrained model + tokenizer
- `classifier/`, `ner/`, `qa/`, `assistant/`: task-specific fine-tuned checkpoints
- `*.safetensors`, `tokenizer.json`, `config.json`, and trainer/checkpoint files

## Typical Workflow

Run notebooks sequentially inside each track:
1. Prepare data
2. Train tokenizer
3. Pretrain base model
4. Fine-tune for downstream tasks
5. Validate via inference notebooks

## Requirements (from notebooks)

Core libraries used:
- `transformers`
- `datasets`
- `tokenizers`
- `trl` (for SFT)
- `torch`
- `evaluate`
- `matplotlib`

GPU is expected for practical training/fine-tuning.
