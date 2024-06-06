# Description: This file is used to train the model with the dataset and save the model to the Model folder
import torch
import json
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer
from datasets import Dataset
import nltk
import string
import numpy as np
import evaluate
import os

path = '/work/ohmygod0193/data-science/lab3'
model_checkpoint = "t5-base"
# Read the JSON file
all_data = pd.read_json(path+'/train.json', lines=True)
train_data = Dataset.from_pandas(all_data[:80000])
test_data = Dataset.from_pandas(all_data[80000:])
print(train_data)
print(test_data)
# Load the model and tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

nltk.download('punkt')


prefix = "summarize: "
max_input_length = 1024
max_target_length = 64

def clean_text(text):
  sentences = nltk.sent_tokenize(text.strip())
  sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
  sentences_cleaned_no_titles = [sent for sent in sentences_cleaned
                                 if len(sent) > 0 and
                                 sent[-1] in string.punctuation]
  text_cleaned = "\n".join(sentences_cleaned_no_titles)
  return text_cleaned

def preprocess_data(examples):
  texts_cleaned = [clean_text(text) for text in examples["body"]]
  inputs = [prefix + text for text in texts_cleaned]
  model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

  # Setup the tokenizer for targets
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(examples["headline"], max_length=max_target_length,
                       truncation=True)

  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

tokenized_train_data = train_data.map(preprocess_data, batched=True)
tokenized_test_data = test_data.map(preprocess_data, batched=True)

batch_size = 8
model_name = "finetuned_t5_base"
model_dir = f"{path}/Model/{model_name}"


args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=1000,
    logging_strategy="steps",
    logging_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    #fp16=True, #only when using tpu can use fp16, or training loss will be 0.0
    load_best_model_at_end=True,
    metric_for_best_model="rouge2",
    report_to="wandb"
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

os.environ["WANDB_PROJECT"] = f"News-headline-generation-{model_name}"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


metric_rouge = evaluate.load("rouge", rouge_types=["rouge1", "rouge2", "rougeL"])
metric_bertscore = evaluate.load("bertscore")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                      for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                      for label in decoded_labels]
    
    # Compute ROUGE scores
    rouge = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    bertscore = metric_bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    
    # Extract ROUGE f1 scores
    result = {key: value * 100 for key, value in rouge.items()}

    # Extract BERTScore f1 score
    result["bertscore_f1"] = bertscore["f1"][0] * 100

    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}
# Function that returns an untrained model to be trained
def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_test_data,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(model_dir)