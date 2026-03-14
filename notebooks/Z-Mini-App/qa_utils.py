from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
qa_model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_id)
qa_model = AutoModelForCausalLM.from_pretrained(qa_model_id).to(device).eval()

def generate_qa_answer(query):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful QA assistant. Answer the user's question clearly, briefly, and factually. If you are not sure, say you are not sure."
        },
        {"role": "user", "content": query},
    ]

    chat_input = qa_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = qa_tokenizer(chat_input, return_tensors="pt").to(qa_model.device)

    output = qa_model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        eos_token_id=qa_tokenizer.eos_token_id,
        pad_token_id=qa_tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
    )

    input_len = inputs["input_ids"].shape[-1]
    generated_tokens = output[0][input_len:]

    answer = qa_tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    ).strip()

    answer = "\n".join(line.strip() for line in answer.splitlines() if line.strip())

    return answer


def qa_pipeline(query):
    return {
        "type": "qa",
        "query": query,
        "answer": generate_qa_answer(query)
    }