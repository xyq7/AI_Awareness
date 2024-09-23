import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Load the filtered data
file_path_filtered_data = 'data/Survey_data.csv'
filtered_data = pd.read_csv(file_path_filtered_data)

# Ensure the relevant columns are float
filtered_data['Self-Objective'] = filtered_data['Self-Objective'].astype(float)
filtered_data['Self-Subjective'] = filtered_data['Self-Subjective'].astype(float)

# Mapping functions
def remap_education(value):
    if value <= 6:
        return 'High school or less'
    elif 7 <= value <= 10:
        return 'Some college or\nassociate degree'
    elif value == 11:
        return 'Bachelor’s degree'
    else:
        return 'Master\'s degree\nor higher'

def remap_race(value):
    if value == 1:
        return 'American Indian\nor Alaska Native'
    elif value == 2:
        return 'Asian'
    elif value == 3:
        return 'Black or African\nAmerican'
    elif value == 4:
        return 'Native Hawaiian,\nother Pacific Islander,\nother race, or multiracial'
    elif value == 6:
        return 'Hispanic or Latino'
    else:
        return 'White'

# Apply mappings
filtered_data['Education'] = filtered_data['Education'].apply(remap_education)
filtered_data['Race'] = filtered_data['Race'].apply(remap_race)

# Function to compute statistics and correlation for a given group
def compute_stats_and_corr(group):
    mean_self_obj = group['Self-Objective'].mean()
    std_self_obj = group['Self-Objective'].std()
    mean_self_subj = group['Self-Subjective'].mean()
    std_self_subj = group['Self-Subjective'].std()
    count = len(group)
    corr, _ = pearsonr(group['Self-Objective'], group['Self-Subjective'])
    se = np.sqrt((1 - corr**2) / (count - 2))
    # fisher_z = 0.5 * np.log((1 + corr) / (1 - corr))
    return pd.Series({
        'mean_self_objective': mean_self_obj,
        'std_self_objective': std_self_obj,
        'mean_self_subjective': mean_self_subj,
        'std_self_subjective': std_self_subj,
        'count': count,
        'correlation': corr,
        'se': se
    })

# Apply the function to each group by Race
race_stats = filtered_data.groupby('Race').apply(compute_stats_and_corr).reset_index()
print("Statistics grouped by Race:")
print(race_stats)
print("-" * 50)

# Apply the function to each group by Gender
gender_stats = filtered_data.groupby('Gender').apply(compute_stats_and_corr).reset_index()
print("Statistics grouped by Gender:")
print(gender_stats)
print("-" * 50)


# Apply the function to each group by Education
education_stats = filtered_data.groupby('Education').apply(compute_stats_and_corr).reset_index()
print("Statistics grouped by Education:")
print(education_stats)
print("-" * 50)


education_stats = filtered_data.groupby(['Race','Education']).apply(compute_stats_and_corr).reset_index()
print("Statistics grouped by Education+Race:")
print(education_stats)
print("-" * 50)
education_stats.to_csv('./education_stats.csv', index=False)