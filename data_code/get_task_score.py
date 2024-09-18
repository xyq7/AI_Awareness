import openai
import json
import os
os.environ['OPENAI_API_KEY'] = "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"
import jsonlines
def gen_score(year, task, model_name):
    client =  openai.OpenAI()
    response =client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an AI expert. Please answer the following questionnaire:"},
            {"role": "user", "content":f"Over the next {year}, how likely do you think it is that the following occupational task will be replaced by artificial intelligence? (Please use a scale from 1 to 7 where '1' means 'Not at all likely' and '7' means 'Extremely likely').\n Directly answer with the score in the format 'Score: your assessment' without explanation. Task: {task}"}
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
file_path = '/home/yueqi/SocAI/TaskStatements.xlsx'
df = pd.read_excel(file_path)
years = ["1 year", "5 years", "10 years", "20 years"]
# 假设Excel文件中的列名为 "task id" 和 "task"
model_name = "gpt-4-turbo"
MAX_RETRY = 10
out = []
for index, row in df.iterrows():
    task_id = row['Task ID']
    task = row['Task']
    retry_time = 0
    
    for year in years:
        while retry_time < MAX_RETRY:
            score = gen_score(year, task, model_name)
            if score != -1: break
        out.append(
            {
                "Task ID":  task_id,
                "Task": task,
                "year": year,
                "score": score,
            })
    with jsonlines.open("./task_score.json", "w") as writer:
        writer.write_all(out)
    # print(f'Task ID: {task_id}, Task: {task}')
    
    