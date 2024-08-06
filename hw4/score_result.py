from datasets import load_dataset
billsum_test = load_dataset("billsum", split="test")

import pandas as pd
model_name = 'pruned_model_V6'
max_length = 128
df_results = pd.read_csv(model_name+'_sample_submission_'+str(max_length)+'.csv',encoding='utf-8')
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
print('Model Name:',model_name)
print('Max Length:',max_length)
print('Final Score =',score(df_label, df_results, 'ID'))
