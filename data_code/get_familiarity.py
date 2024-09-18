import openai
import json
import os
import jsonlines
def gen_score(occu, des, model_name):
    client =  openai.OpenAI()
    response =client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an AI and Labor Market expert. Please answer the following questionnaire:"},
            {"role": "user", "content":f"How likely do you think a person with the following occupation is familiar with artificial intelligence? (Please use a scale from 1 to 7 where '1' means 'Not at all familiar' and '7' means 'Extremely familiar').\n Directly answer with the score in the format 'Score: your assessment' without explanation. \n Occupation: {occu} \n Description: {des}"}
            ]
        )
    return find_score(response.choices[0].message.content )


def find_score(text):
    index_abstract = text.lower().find("score:")

    if index_abstract != -1:
        return text[index_abstract + len("score:"):].strip()

    return -1

import pandas as pd

# 载入Excel文件
file_path = './data/Occupation Data.xlsx'
df = pd.read_excel(file_path, header=0)
# df = df.drop([0, 1])

model_name = "gpt-4-turbo"
MAX_RETRY = 10
out = []
print(df.columns.tolist())
for index, row in df.iterrows():
    o_id = row['O*NET-SOC Code']
    occu = row['Title']
    des = row['Description']
    retry_time = 0
    

    while retry_time < MAX_RETRY:
        score = gen_score(occu, des, model_name)
        if score != -1: break
    out.append(
        {
            "o_id":  o_id,
            "occu": occu,
            "score": score,
        })
    with jsonlines.open("./data/AI_familiarity.json", "w") as writer:
        writer.write_all(out)
    # print(f'Task ID: {task_id}, Task: {task}')
    
    