# AI_Awareness

## Contents
- [Overview](#overview)
- [Repo Contents](#repo-contents)
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
- [src](./src): source code to analyze result data.
- [script](./script): scripts for data analysis.
- [data_code](./data_code): source code to process the data.


## Data Process Procedure

### AI Familiarity
```bash
cd data_code

# estimate occupation AI Familiarity with GPT4
python get_familiarity.py
```
The processed familiarity is available at https://github.com/xyq7/AI_Awareness_Data.


### Subjective Assessment
The subjective assessment is available at https://github.com/xyq7/AI_Awareness_Data.



### Objective Assessment


```bash
cd data_code

# obtain task-level occupation
```
The objective assessment is available at https://github.com/xyq7/AI_Awareness_Data.


## Data Analysis Procedure

