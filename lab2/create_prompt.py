from groq import Groq
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas
import time
import re
from tqdm import tqdm
import random
import json

data = pandas.read_csv("mmlu_sample.csv")
input_data = data["input"]
A = data["A"]
B = data["B"]
C = data["C"]
D = data["D"]
tasks = data["task"]
answers = data["target"]

prompt = """
Topic:
{task}

QUESTION:
{QUESTION}

Options:
{ANSWER_OPTIONS}

Answer:{answer}
"""
task_dict = {}
start_index = 0
for i, (question, a, b, c, d, task, answer) in tqdm(
    enumerate(
        zip(
            input_data[start_index:],
            A[start_index:],
            B[start_index:],
            C[start_index:],
            D[start_index:],
            tasks[start_index:],
            answers[start_index:],
        )
    )
):
    if task not in task_dict:
        task_dict[task] = []
    llm_prompt = prompt.format(
        task=task,
        QUESTION=question,
        ANSWER_OPTIONS=f"A) {a}\nB) {b}\nC) {c}\nD) {d}",
        answer=answer,
    )
    print(llm_prompt)
    task_dict[task].append(llm_prompt)
    # Convert task_dict to JSON
    json_data = json.dumps(task_dict)

    # Save JSON data to file
    with open("fewshots_db.json", "w") as file:
        file.write(json_data)
