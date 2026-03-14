from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
chat_model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_id)
chat_model = AutoModelForCausalLM.from_pretrained(chat_model_id).to(device).eval()


def generate_chat_answer(query):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer clearly and simply."
        },
        {
            "role": "user",
            "content": query
        },
    ]

    chat_input = chat_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = chat_tokenizer(chat_input, return_tensors="pt").to(chat_model.device)

    output = chat_model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        top_k=50,
        eos_token_id=chat_tokenizer.eos_token_id,
        pad_token_id=chat_tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )

    input_len = inputs["input_ids"].shape[-1]
    generated_tokens = output[0][input_len:]

    answer = chat_tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    ).strip()

    answer = "\n".join(line.strip() for line in answer.splitlines() if line.strip())
    return answer


def chat_pipeline(query):
    return {
        "type": "chat",
        "query": query,
        "answer": generate_chat_answer(query)
    }