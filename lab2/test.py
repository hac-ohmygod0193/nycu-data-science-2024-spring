from groq import Groq
import os
from dotenv import load_dotenv
import google.generativeai as genai
from llama_index.llms import Gemini
import pandas
import time
import re
from tqdm import tqdm

data = pandas.read_csv("submit/submit.csv")

input = data["input"]
A = data["A"]
B = data["B"]
C = data["C"]
D = data["D"]
tasks = data["task"]

load_dotenv()

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
genai.configure()
# model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

llm = Gemini(model="models/gemini-pro")

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY_1"),
)


prompt = """
You are a professional high school european history teacher.
Answer the following questions about the {task}.
Choose the best answer from the Options provided. Only one answer is correct.
You must only output the letter of the correct answer.
For example, if the correct answer is A, you can only output "A".

QUESTION:
{QUESTION}

Options:

{ANSWER_OPTIONS}

"""

print(len(input))
target = []
# Read the answers from the file
with open("answer.txt", "r") as f:
    # Read the content of the file
    content = f.read()
    # Convert the string representation of the list back to a list
    target = eval(content)
start_index = len(target)
print(start_index)
# Iterate over the lists starting from the starting index
for i, (question, a, b, c, d, task) in tqdm(
    enumerate(
        zip(
            input[start_index:],
            A[start_index:],
            B[start_index:],
            C[start_index:],
            D[start_index:],
            tasks[start_index:],
        )
    )
):
    llm_prompt = prompt.format(
        task=task, QUESTION=question, ANSWER_OPTIONS=f"A) {a}\nB) {b}\nC) {c}\nD) {d}"
    )
    # print(llm_prompt)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": str(llm_prompt),
            }
        ],
        model="mixtral-8x7b-32768",
    )
    response = chat_completion.choices[0].message.content
    print(chat_completion.choices[0].message.content)
    # Define the regular expression pattern to match answer choices
    pattern = r"([A-Z]\))"

    # Search for the pattern in the text
    matches = re.findall(pattern, response)

    # Print the first match (which corresponds to the correct answer letter)
    answer = matches[0][0] if matches else response[0]
    if matches:
        print("Correct answer letter:", matches[0][0])
    else:
        print("No answer found.")
    target.append(answer)
    with open("answer.txt", "w") as f:
        f.write(str(target))

    time.sleep(1)


# Create a DataFrame with id and target columns
output_data = pandas.DataFrame({"id": range(len(target)), "target": target})
# Save the DataFrame to a CSV file
output_data.to_csv("output.csv", index=False)
