import warnings
warnings.filterwarnings('ignore')

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
prefix = "summarize: "


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

from transformers import AutoTokenizer
TK_ckpt = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(TK_ckpt)  # use tokeniozer from Hugging Face

from transformers import DataCollatorForSeq2Seq
checkpoint = "t5small_TextSummarization/" # released full model path
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

import evaluate

rouge = evaluate.load("rouge")

import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

from datasets import load_dataset
billsum_test = load_dataset("billsum", split="test")
tokenized_billsum_test = billsum_test.map(preprocess_function, batched=True)

def show_param_ratio(model):
    num_param = 0
    for param in model.parameters():
        num_param += param.numel()
    num_mask = 0
    for name, param in model.named_buffers():
        if "mask" in name:
            num_mask += int((param == 0).sum())
    print((num_param - num_mask) / num_param)
def load_pruned_model(model_name):
    from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
    pruned_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    import torch
    import torch.nn.utils.prune as prune
    import torch.nn.utils.prune as prune
    from transformers.models.t5.modeling_t5 import T5LayerSelfAttention, T5LayerCrossAttention, T5LayerFF
    # Apply prune.identity to the layers that were pruned
    for module in pruned_model.modules():
        if isinstance(module, torch.nn.Linear):  # Check the layer type as per your model's pruned layers
            prune.identity(module, 'weight')
    pruned_model.load_state_dict(torch.load(model_name+'/model_state_dict.pth'))
    return pruned_model

model_name = 'pruned_model_V6'
max_length = 128
pruned_model = load_pruned_model(model_name)

pruned_model.to(device)
pruned_training_args = Seq2SeqTrainingArguments(
    output_dir="pruned_billsum_model_V6",
    evaluation_strategy="no",
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20,
    lr_scheduler_type="linear",
    seed=42,
    fp16=True,
    logging_steps=10000,
    predict_with_generate=True,
    save_steps=5000,
    do_eval=False
)
pruned_trainer = Seq2SeqTrainer(
    model=pruned_model,
    args=pruned_training_args,
    train_dataset=tokenized_billsum_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
print('Model Name:',model_name)
print('Max Length:',max_length)
show_param_ratio(pruned_model)

results = pruned_trainer.predict(tokenized_billsum_test,max_length=max_length)
predictions = np.where(results[0] != -100, results[0], tokenizer.pad_token_id)
decoded_prediction = tokenizer.batch_decode(predictions, skip_special_tokens=True)

import pandas as pd
import csv

df_results = pd.DataFrame(columns=['ID','Predict'])

for i, prediction in enumerate(decoded_prediction):
    # Escape quotes by replacing "," with "."
    summary_escaped = prediction.replace(',', '.')
    
    # Create a new row DataFrame and append it
    new_row = pd.DataFrame({'ID': [i], 'Predict': [summary_escaped]})
    df_results = pd.concat([df_results, new_row], ignore_index=True)

# Print the resulting DataFrame
print(df_results)

# Function to escape double quotes and handle newlines
def escape_special_characters(text):
    return text.replace('"', '""').replace('\n', ' ')

# Apply escaping to the 'Summary' column
df_results['Predict'] = df_results['Predict'].apply(escape_special_characters)
df_results.to_csv(model_name+'_sample_submission_'+str(max_length)+'.csv', index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')

def calculate_lcs(X, Y):
    """
    Helper function to calculate the longest common subsequence of sequences X and Y.
    """
    m, n = len(X), len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Computes the ROUGE-Lsum score based on the longest common subsequence summed over all sentences in the summaries.
    
    Args:
    solution (pd.DataFrame): The DataFrame containing the correct summaries.
    submission (pd.DataFrame): The DataFrame containing participant's predicted summaries.
    row_id_column_name (str): The column name for the row ID in both DataFrames.

    Returns:
    float: The mean ROUGE-Lsum score across all predictions.
    """
    # Ensure indices for proper alignment
    solution.set_index(row_id_column_name, inplace=True)
    submission.set_index(row_id_column_name, inplace=True)

    total_score = 0

    for idx in solution.index:
        if idx not in submission.index:
            raise ParticipantVisibleError(f"Missing prediction for ID {idx}.")

        ref_summary = solution.loc[idx, 'Label']
        pred_summary = submission.loc[idx, 'Predict']

        # Tokenize sentences
        ref_sentences = ref_summary.split('.')
        pred_sentences = pred_summary.split('.')

        # Calculate LCS for each sentence pair
        lcs_sum = 0
        for ref_sent in ref_sentences:
            ref_tokens = ref_sent.strip().lower().split()
            best_lcs = 0
            for pred_sent in pred_sentences:
                pred_tokens = pred_sent.strip().lower().split()
                lcs_length = calculate_lcs(ref_tokens, pred_tokens)
                best_lcs = max(best_lcs, lcs_length)
            lcs_sum += best_lcs

        # Calculate ROUGE-L for the current pair of summaries
        ref_length = sum(len(sent.strip().split()) for sent in ref_sentences)
        if ref_length > 0:
            rouge_l = lcs_sum / ref_length
        else:
            rouge_l = 0
        total_score += rouge_l

    # Compute the average ROUGE-L score across all submissions
    mean_rouge_lsum = total_score / len(solution)

    return mean_rouge_lsum

df_label = pd.DataFrame(columns=['ID','Label'])

for i, label in enumerate(billsum_test):
    # Escape quotes by replacing "," with "."
    label_escaped = label['summary'].replace(',', '.')
    
    # Create a new row DataFrame and append it
    new_row = pd.DataFrame({'ID': [i], 'Label': [label_escaped]})
    df_label = pd.concat([df_label, new_row], ignore_index=True)
print('Final Score =',score(df_label, df_results, 'ID'))
