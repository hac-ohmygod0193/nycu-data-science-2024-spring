# Description: This script generates headlines for the test data using the Finetuned T5 model and saves them to a JSON file.
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm

# Function to generate headlines based on given bodies
def generate_headlines_from_bodies(bodies, headlines, model, tokenizer, start_index=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model.to(device)
    for i in tqdm(range(start_index, len(bodies))):
        body = bodies[i]
        inputs = tokenizer.encode("summarize: " + body, return_tensors="pt", max_length=1024, truncation=True).to(device)
        outputs = model.generate(inputs, max_length=64, num_beams=4, length_penalty=2.0, early_stopping=True)
        headline = tokenizer.decode(outputs[0], skip_special_tokens=True)
        headlines.append(headline)
        json_item = {'headline': headline}
        with open(f"{path}/ongoing.json",'a') as json_file:
            json.dump(json_item, json_file)
            json_file.write('\n')
    return headlines

# Function to save headlines to a JSON file
def save_submission_file(headlines, file_path):
    with open(file_path, 'w') as file:
        for headline in headlines:
            submission = {"headline": headline}
            json.dump(submission, file)
            file.write('\n')

path = '.'
# Load the saved model and tokenizer
model_dir = f"{path}/finetuned_t5_base"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

all_data = pd.read_json(path+'/DS_hw3/test.json', lines=True)
# Example list of bodies
bodies = all_data['body']

# Read the last line of generated_submission.json to determine the start index
try:
    submission = pd.read_json(path+'/ongoing.json', lines=True)
    start_index = len(submission['headline'])
    headlines = submission['headline'].to_list()
except FileNotFoundError:
    start_index = 0
    headlines=[]
print("Start=",start_index)
# Generate headlines from the start_index, and save them to a JSON file as soon as they are generated 
generated_headlines = generate_headlines_from_bodies(bodies, headlines, model, tokenizer, start_index=start_index)

# Save all headlines to a JSON file
submission_file_path = f"{path}/111511198-1.json"
save_submission_file(generated_headlines, submission_file_path)
