# AI_Awareness

## Contents
- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [Installation Guide](#installation-guide)
- [Data Process](#data-process)
- [Data Analysis](#data-analysis)
- [License](./LICENSE)

## Overview
Recent advancements in artificial intelligence (AI) are reshaping the labor market. 
Existing research has primarily focused on the objective likelihood of AI-driven occupation
replacement and its potential societal impacts. 
However, individuals’ awareness of job replacement by AI also influences their preparedness, leading to long-term societal effects.
This study investigates the disparities in awareness of AI-driven occupation replacement across 
different racial/ethnic and educational groups.
We conduct an online survey in November 2023 (n=4816) to analyze people’s subjective assessment of being replaced by AI and thereby calculate the group-level correlation with objective assessments as an indicator of awareness.
Our findings reveal that traditionally underrepresented racial/ethnic groups and those with lower educational attainment are less aware of AI’s potential to replace jobs, which may further exacerbate long-term inequalities. Interventions to improve public awareness of AI could enhance preparedness and help ameliorate these disparities.


## Repo Contents
- [code](./code): source code to analyze result data.
- [data_code](./data_code): source code to process the data.
- [figure](./figure): Figures involved in README.

## Installation Guide
```
pip install jsonlines
pip install openai
pip install openpyxl
pip install pandas
pip install scipy
pip install matplotlib
```
Since this package requires access to the OpenAI API, you will need to register an account and obtain your OPENAI_API_KEY. Please follow the instructions provided in the OpenAI documentation for registration and obtaining the API keys: OpenAI Documentation. The code has been test with OpenAI Services. Setup the your OpenAI API key:

```bash
export OPENAI_API_KEY='yourkey'
```
## Data Process
The processed data are available at available at https://github.com/xyq7/AI_Awareness_Data. We can directly download and save at ```/data/```.
Following are the procedure in obtaining AI Familiarity and Objective Assessment.
### Download External Data
```bash
mkdir data
cd data
wget https://www.onetcenter.org/dl_files/database/db_29_0_excel/Occupation%20Data.xlsx
wget https://www.onetcenter.org/dl_files/database/db_29_0_excel/Task%20Statements.xlsx
wget https://www.bls.gov/soc/2018/soc_2010_to_2018_crosswalk.xlsx
wget https://www.onetcenter.org/taxonomy/2019/soc/2019_to_SOC_Crosswalk.xlsx
cd ..

```
### Obtain AI Familiarity


```bash
# estimate occupation AI Familiarity with GPT4
python data_code/get_familiarity.py
```

### Objective Assessment


```bash
# obtain task-level occupation task replacement score
python data_code/get_task_score.py

## aggregate to occupations
python data_code/get_obj_score.py
```

## Data Analysis 

### Overall Statistics for All Groups
```bash
python code/statistics.py
```

### Figures for Groups
```bash
python code/analysis.py
```


