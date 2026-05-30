# -*- coding: utf-8 -*-
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

print(f"Loading model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("Model loaded")


def generate_poem(prompt, max_new_tokens=300):
    messages = [
        {"role": "system", "content": "你是一位精通中国古典诗词的AI诗人。请根据用户的要求创作古诗。"},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def compute_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return math.exp(loss.item())


# Acrostic poem
print("=" * 60)
acrostic_prompt = '请创作一首藏头诗，每句的第一个字依次是"深""度""学""习"。要求是五言或七言绝句，意境优美，符合古诗格律。'
acrostic_poem = generate_poem(acrostic_prompt)
print(acrostic_poem)
print()

# Continuation poem
print("=" * 60)
continuation_prompt = '请以"大漠孤烟照高阁"为第一句，续写一首完整的七言律诗或绝句。要求意境连贯，风格统一，符合古诗格律。'
continuation_poem = generate_poem(continuation_prompt)
print(continuation_poem)
print()

# Perplexity
print("=" * 60)
ppl1 = compute_perplexity(acrostic_poem)
print(f"Acrostic poem PPL: {ppl1:.2f}")
ppl2 = compute_perplexity(continuation_poem)
print(f"Continuation poem PPL: {ppl2:.2f}")
print("Done")
