# Description: This script loads a model from a checkpoint and saves it as a trained model.
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

path = '/work/ohmygod0193/data-science/lab3'

# Model name and directory
model_name = "t5-base-news-headline-generation-v1/checkpoint-16000"
model_dir = f"{path}/Model/{model_name}"

# Load tokenizer and model from the checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# Ensure model is in evaluation mode
model.eval()

# Save the loaded model as a trained model
model.save_pretrained(f"{path}/Trained_Model")

# Optionally, save the tokenizer as well
tokenizer.save_pretrained(f"{path}/Trained_Model")