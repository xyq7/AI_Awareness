import json
import pandas as pd


file_path = 'data/Task Statements.xlsx'
df = pd.read_excel(file_path)
years = ["1 year", "5 years", "10 years", "20 years"]

json_file_path = 'data/task_score.json'
json_df = pd.read_json(json_file_path, lines=True)

pivot_json_df = json_df.pivot_table(index=["Task ID", "Task"], columns="year", values="score", aggfunc='first').reset_index()

result_df = pd.merge(df, pivot_json_df, on=["Task ID", "Task"], how="left")

print(result_df.head())

df = result_df

score_columns = ['1 year', '5 years', '10 years', '20 years']
for col in score_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  

grouped = df.groupby(['O*NET-SOC Code', 'Title']).agg({
    '1 year': 'mean',
    '5 years': 'mean',
    '10 years': 'mean',
    '20 years': 'mean'
}).reset_index()

grouped = grouped[['O*NET-SOC Code', 'Title', '1 year', '5 years', '10 years', '20 years']]

print(grouped.head())

new_file_path = 'data/Objective_score.xlsx'
grouped.to_excel(new_file_path, index=False)
