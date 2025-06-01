from huggingface_hub import login
import os
import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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

tokenizer = AutoTokenizer.from_pretrained(model_id)