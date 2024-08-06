import sys
import json

version = 1
# Mapping of model numbers to model names
model_mapping = {
    "0": "llama2-70b-4096",
    "1": "mixtral-8x7b-32768",
    "2": "gemma-7b-it",
    "3": "gemini-pro",
}

# Check if the user has provided a model number
if len(sys.argv) < 2:
    print("Please provide a model number (0, 1, or 2) as a command-line argument.")
    print(str(model_mapping))
    sys.exit(1)


# Check if the provided model number is valid
if str(sys.argv[1]) not in model_mapping:
    print("Invalid model number. Please provide a valid model number (0, 1, or 2).")
    print(str(model_mapping))
    sys.exit(1)
selected_model = model_mapping[str(sys.argv[1])]

from groq import Groq
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas
import time
import re
from tqdm import tqdm
import random
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

data = pandas.read_csv("submit/submit.csv")
index = data.iloc[:, 0]
input_data = data["input"]
A = data["A"]
B = data["B"]
C = data["C"]
D = data["D"]
tasks = data["task"]

load_dotenv()

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
genai.configure()
# Randomly choose GROQ API key
groq_api_keys = [
    os.environ.get("GROQ_API_KEY"),
    os.environ.get("GROQ_API_KEY_1"),
    os.environ.get("GROQ_API_KEY_2"),
]
selected_api_key = random.choice(groq_api_keys)

if selected_api_key is None:
    print("No GROQ API key found.")
    sys.exit(1)
if selected_model == "gemini-pro":
    gemini = True
    gemini_pro = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.1,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        },
    )
else:
    gemini = False
client = Groq(
    api_key=selected_api_key,
)

model = selected_model

prompt = """
You are an expert on {task}.
Answer the following questions about the {task}.
Choose the best answer from the Options provided. Only one answer is correct.

Here are five example pairs of question and answer about the {task}:
<example>
Ex1. 
{example_1}
Ex2. 
{example_2}
Ex3. 
{example_3}
Ex4. 
{example_4}
Ex5.
{example_5}
</example>

QUESTION:
{QUESTION}

Options:

{ANSWER_OPTIONS}

"""
print("Model =", model)
print(len(input_data))
target = []

try:
    with open(model + f"-answer-{version}.txt", "r") as f:
        content = f.read()
        target = eval(content)
except:
    print(model + f"-answer-{version}.txt Not Found. Creating a new file.")
start_index = len(target)
print(start_index)

# Read JSON data from file
with open("fewshots_db.json", "r") as file:
    json_data = file.read()

    # Convert JSON data to dictionary
    task_dict = json.loads(json_data)

for i, (question, a, b, c, d, task) in tqdm(
    enumerate(
        zip(
            input_data[start_index:],
            A[start_index:],
            B[start_index:],
            C[start_index:],
            D[start_index:],
            tasks[start_index:],
        )
    )
):
    print("ID =", start_index + i)
    try:
        examples = random.sample(task_dict[task], 5)
        example_1 = examples[0]
        example_2 = examples[1]
        example_3 = examples[2]
        example_4 = examples[3]
        example_5 = examples[4]
    except:
        print("No examples found for", task)
        answer = random.choice(["A", "B", "C", "D"])
        print("Randomly chosen answer:", answer)
        target.append(answer)
        with open(model + f"-answer-{version}.txt", "w") as f:
            f.write(str(target))
        continue
    llm_prompt = prompt.format(
        task=task,
        example_1=example_1,
        example_2=example_2,
        example_3=example_3,
        example_4=example_4,
        example_5=example_5,
        QUESTION=question,
        ANSWER_OPTIONS=f"A) {a}\nB) {b}\nC) {c}\nD) {d}",
    )
    if gemini:
        response = gemini_pro.invoke(llm_prompt).content
        pattern = r"([A-Z]\))"
        matches = re.findall(pattern, response)
        answer = matches[0][0] if matches else response[0]
        print(response)
        if matches:
            print("Correct answer letter:", matches[0][0])
        else:
            match = re.search(r"Answer:[A-D]", response)
            if match:
                answer = response[-1] if match else random.choice(["A", "B", "C", "D"])
                print("The answer is:", answer)
            else:
                if "A" <= answer <= "D":
                    print("The answer is:", answer)
                else:
                    answer = random.choice(["A", "B", "C", "D"])
                    print("No answer found. Randomly chosen answer:", answer)
        
    else:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": str(llm_prompt),
                }
            ],
            model=model,
        )
        response = chat_completion.choices[0].message.content
        print(chat_completion.choices[0].message.content)
        pattern = r"([A-Z]\))"
        matches = re.findall(pattern, response)
        answer = matches[0][0] if matches else response[0]
        if matches:
            print("Correct answer letter:", matches[0][0])
        else:
            match = re.search(r"The answer is ([A-Z])\.", response)
            if match:
                answer = match.group(1)
                print("The answer is:", answer)
            else:
                if "A" <= answer <= "D":
                    print("The answer is:", answer)
                else:
                    answer = random.choice(["A", "B", "C", "D"])
                    print("No answer found. Randomly chosen answer:", answer)
    target.append(answer)
    with open(model + f"-answer-{version}.txt", "w") as f:
        f.write(str(target))

    time.sleep(1)

# Create a DataFrame with id and target columns
output_data = pandas.DataFrame({"ID": list(index), "target": target})
# Save the DataFrame to a CSV file
output_data.to_csv(model + f"-output-{version}.csv", index=False)
