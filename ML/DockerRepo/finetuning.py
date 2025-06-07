from huggingface_hub import login
import os
import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from dotenv import load_dotenv

load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
login(token=huggingface_token)

# finetune_lora_mistral.py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mistral-7B-v0.1"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,  
    device_map="auto"
)

device = 0 if torch.cuda.is_available() else -1 
model.to(device if device != -1 else "cpu") 
print(True if device == 0 else False)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
# out = generator("赤ずきんは", max_new_tokens=200, num_return_sequences=2)
# for content in out:
#     text = content["generated_text"]
#     print("生成されたテキストは:\n" + text)

input_text = "赤ずきんは"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

# テキスト生成
output_ids = model.generate(
    input_ids,
    max_new_tokens=200,
    num_return_sequences=2,
    do_sample=True,
    temperature=0.9,
    top_p=0.95,
)

# 出力をデコードして表示
for i, output in enumerate(output_ids):
    text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"[{i+1}] 生成されたテキスト:\n{text}\n")