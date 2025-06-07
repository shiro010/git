from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
import torch


tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium", legacy=False)
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

device = 0 if torch.cuda.is_available() else -1 
model.to(device if device != -1 else "cpu") 
print(True if device == 0 else False)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
set_seed(42)
out = generator("赤ずきんは", max_new_tokens=200, num_return_sequences=2)
for content in out:
    text = content["generated_text"]
    print("生成されたテキストは:\n" + text)
