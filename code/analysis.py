import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load the filtered data
file_path_filtered_data = 'data/Survey_data.csv'
filtered_data = pd.read_csv(file_path_filtered_data)

# Ensure the relevant columns are float
filtered_data['Self-Objective'] = filtered_data['Self-Objective'].astype(float)
filtered_data['Self-Subjective'] = filtered_data['Self-Subjective'].astype(float)
education_order = ['High school or less', 'Some college or\nassociate degree', 'Bachelor’s degree', 'Master\'s degree\nor higher']
race_order = ['Black or African\nAmerican','Hispanic\nor Latino','Asian', 'White']

# Remapping Functions
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
    if value ==1:
        return 'American Indian\nor Alaska Native'
    elif value ==2:
        return 'Asian'
    elif value == 3:
        return 'Black or African\nAmerican'
    elif value == 4:
        return 'Native Hawaiian,\nother Pacific Islander,\nother race, or multiracial'
    elif value == 6:
        return 'Hispanic\nor Latino'
    else:
        return 'White'
filtered_data['Education'] = filtered_data['Education'].apply(remap_education)
filtered_data['Education-Mother'] = filtered_data['Education-Mother'].apply(remap_education)
filtered_data['Education-Father'] = filtered_data['Education-Father'].apply(remap_education)
filtered_data['Race'] = filtered_data['Race'].apply(remap_race)

# Analysis Functions
def calculate_mean_std(grouped_data,group_name, variable_name):
    results_self = {}


    for name, group in grouped_data:
        group_size = len(group)
    
        mean_self_subjective = group[variable_name].mean()
        std_self_subjective = group[variable_name].std()
        results_self[name] = {
            "group_size": group_size,
            "mean": mean_self_subjective,
            "ci_lower":  mean_self_subjective-std_self_subjective,
            "ci_upper":  mean_self_subjective+std_self_subjective
        }
        print(name, mean_self_subjective, std_self_subjective)
    return results_self
def calculate_percentage(grouped_data,group_name, variable_name):
    results_self = {}
    results_other = {}

    for name, group in grouped_data:
        group_size = len(group)
        proportion_greater_than_4 = (group[variable_name] >4).mean()
        results_self[name] = {
            "group_size": group_size,
            "percentage": proportion_greater_than_4
        }

    return results_self
def calculate_correlations(grouped_data, group_name):
    results_self = {}

    for name, group in grouped_data:
        group_size = len(group)
        correlation_self, _ = pearsonr(group['Self-Subjective'], group['Self-Objective'])
    
        mean_self_subjective = group['Self-Subjective'].mean()
        std_self_subjective = group['Self-Subjective'].std()
        mean_self_objective = group['Self-Objective'].mean()
        std_self_objective = group['Self-Objective'].std()

        se_z = np.sqrt((1 - correlation_self**2) / (group_size - 2))

        results_self[name] = {
            "group_size": group_size,
            "correlation": correlation_self,
            "ci_lower":  correlation_self-se_z,
            "ci_upper":  correlation_self+se_z,
            "mean_sub": mean_self_subjective,
            "std_sub":  std_self_subjective,
            "mean_obj": mean_self_objective,
            "std_obj":  std_self_objective,
            "se": se_z
        }

    return results_self
def calculate_correlations_by_race(data, group_name):
    race_groups = data.groupby('Race')
    return_results_self = {}

    for race, group in race_groups:
        if race in race_order:
            education_groups = group.groupby(group_name)
            results_self_education = calculate_correlations(education_groups, group_name)
            return_results_self[race] = results_self_education

    return return_results_self
def calculate_mean_std_by_race(data, group_name,variable):
    race_groups = data.groupby('Race')
    return_results_self = {}

    for race, group in race_groups:
        if race in race_order:
            education_groups = group.groupby(group_name)
            results_self_education= calculate_mean_std(education_groups, group_name,variable)
            return_results_self[race] = results_self_education

    return return_results_self
def calculate_percentage_by_race(data, group_name,variable):
    race_groups = data.groupby('Race')
    return_results_self = {}
    return_results_other = {}

    for race, group in race_groups:
        if race in race_order:
            education_groups = group.groupby(group_name)
            results_self_education= calculate_percentage(education_groups, group_name,variable)
            return_results_self[race] = results_self_education

    return return_results_self

# Ploting Functions
def plot_percentage_and_save(results, overall, group_name, plot_type, order):
    labels = order
    corrs = [results[label]["percentage"] for label in labels]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    group_sizes = [results[label]["percentage"]  for i, label in enumerate(labels)]
    x = np.arange(len(labels))  # the label locations
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for j, label in enumerate(order):
        ax.bar(x, corrs, capsize=5, color=colors)

        ax.text(x[j],0, f'{group_sizes[j]}', ha='center', va='bottom', fontsize=14)
        

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i in range(len(labels)):
        ax.text(x[i],-0.09, f'{group_sizes[i]}', ha='center', va='bottom', fontsize=14)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Education", fontsize=12, title_fontsize=14, loc='lower right', bbox_to_anchor=(1.26, 0.06))
    ax.set_ylabel("Portion Familar with AI", fontsize=16)
    ax.set_xticks([])
    ax.yaxis.set_tick_params(labelsize=16)
    fig.tight_layout()
    plt.savefig(f"./figure/{group_name}_{plot_type}_fams.png")
    plt.savefig(f"./figure/{group_name}_{plot_type}_fams.pdf")   
def plot_correlations_and_save(results, overall, group_name, plot_type, order):
    labels = order
    corrs = [results[label]["correlation"] for label in labels]
    ci_lowers = [corrs[i] - results[label]["ci_lower"] for i, label in enumerate(labels)]
    ci_uppers = [results[label]["ci_upper"] - corrs[i] for i, label in enumerate(labels)]
    group_sizes = [results[label]["group_size"]  for i, label in enumerate(labels)]
    x = np.arange(len(labels))  # the label locations


    fig, ax = plt.subplots(figsize=(7, 6))

    for i in range(len(labels)):
        ax.errorbar(x[i], corrs[i], yerr=[[ci_lowers[i]], [ci_uppers[i]]], fmt='o', capsize=5, label=f'{labels[i]},n={results[labels[i]]["group_size"]}')
        ax.text(x[i],-0.1, f'{group_sizes[i]}', ha='center', va='bottom', fontsize=14)

    # Add overall reference
    ax.axhline(y=overall['correlation'], color='blue', linestyle='--', linewidth=1, label=f'Overall')
    ax.fill_between(x, overall['ci_lower'], overall['ci_upper'], color='blue', alpha=0.2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Sub.-Obj. Correlation', fontsize=16)
    # ax.set_title(f'Correlations vs. {group_name}', fontsize=18)
    ax.set_xticks([])
    ax.set_ylim(-0.1, 0.4)
    ax.yaxis.set_tick_params(labelsize=14)
    # ax.set_xticklabels(labels, ha='center',rotation=30, fontsize=14)
    for i, race_name in enumerate(order):
        x = np.mean(i)
        ax.text(x, 0.41, race_name, ha='center', va='bottom', fontsize=14, fontweight='bold')
    fig.tight_layout()

    plt.savefig(f"./figure/{group_name}_{plot_type}_correlations.png")
    plt.savefig(f"./figure/{group_name}_{plot_type}_correlations.pdf")
def plot_correlations_and_save_stratified(results, overall, group_name, plot_type, order):
    race_names = race_order 
    num_races = len(race_names)
    fig, ax = plt.subplots(figsize=(14, 6))
    markers = ['o', 's', 'D', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    for i, race_name in enumerate(race_names):
        race_results = results[race_name]
        corrs = [race_results[label]["correlation"] for label in order]
        ci_lowers = [corr - race_results[label]["ci_lower"] for corr, label in zip(corrs, order)]
        ci_uppers = [race_results[label]["ci_upper"] - corr for corr, label in zip(corrs, order)]
        group_sizes = [race_results[label]["group_size"] for label in order]
        
        x = np.arange(len(order)) + i * (len(order) + 1)  # Adjusted to space out different races

        for j, label in enumerate(order):
            ax.errorbar(x[j], corrs[j], yerr=[[ci_lowers[j]], [ci_uppers[j]]], fmt=markers[j], color=colors[j], capsize=5, label=label if i == 0 else "")
            ax.text(x[j], -0.2, f'{group_sizes[j]}', ha='center', va='bottom', fontsize=14)
        
        if i < num_races - 1:
            ax.axvline(x=x[-1] + 1, color='gray', linestyle='--')
    for i, race_name in enumerate(race_order):
        x = np.mean(np.arange(len(order)) + i * (len(order) + 1))
        ax.text(x, 0.41, race_name, ha='center', va='bottom', fontsize=14, fontweight='bold')
    # Customize axes
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys(), title="Education", fontsize=12, title_fontsize=14)
    ax.legend(by_label.values(), by_label.keys(), title="Education", fontsize=12, title_fontsize=14, loc='lower right', bbox_to_anchor=(1.26, 0.06))
    ax.set_ylabel('Sub.-Obj. Correlation', fontsize=16)
    # ax.set_xticks(np.arange(len(order) * num_races) + (len(order) - 1) / 2)
    # ax.set_xticklabels([f"{label}\n({race_name})" for race_name in race_names for label in order], rotation=45, ha='right', fontsize=10)
    ax.set_ylim(-0.2, 0.4)
    ax.set_xticks([])
    # ax.set_title(f'Correlations vs. {group_name}', fontsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    fig.tight_layout()
    plt.savefig(f"./figure/race/{group_name}_{plot_type}_correlations.png")
    plt.savefig(f"./figure/race/{group_name}_{plot_type}_correlations.pdf")    
def plot_meanstd_and_save(results, overall, group_name, plot_type, order, var):
    labels = order
    map_var = {"Self-Objective": "Objective Assessment", "Self-Subjective": "Subjective Assessment"}
    corrs = [results[label]["mean"] for label in labels]
    ci_lowers = [corrs[i] - results[label]["ci_lower"] for i, label in enumerate(labels)]
    ci_uppers = [results[label]["ci_upper"] - corrs[i] for i, label in enumerate(labels)]
    group_sizes = [results[label]["group_size"]  for i, label in enumerate(labels)]
    x = np.arange(len(labels))  # the label locations


    fig, ax = plt.subplots(figsize=(7, 6))

    for i in range(len(labels)):
        ax.errorbar(x[i], corrs[i], yerr=[[ci_lowers[i]], [ci_uppers[i]]], fmt='o', capsize=5, label=f'{labels[i]},n={results[labels[i]]["group_size"]}')
        ax.text(x[i],-1.6, f'{group_sizes[i]}', ha='center', va='bottom', fontsize=14)

    for i, race_name in enumerate(order):
        x = np.mean(i)
        ax.text(x, 1.55, race_name, ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(map_var[var], fontsize=16)
    # ax.set_title(f'Correlations vs. {group_name}', fontsize=18)
    ax.set_xticks([])
    ax.set_ylim(-1.6, 1.5)
    ax.yaxis.set_tick_params(labelsize=16)
    # ax.set_xticklabels(labels, ha='center',rotation=30, fontsize=14, fontweight='bold')

    fig.tight_layout()

    plt.savefig(f"./figure/{group_name}_{plot_type}_{var}.png")
    plt.savefig(f"./figure/{group_name}_{plot_type}_{var}.pdf")

def plot_meanstd_and_save_stratified(results, overall, group_name, plot_type, order,var):
    race_names = race_order 
    num_races = len(race_names)
    map_var = {"Self-Objective": "Objective Assessment", "Self-Subjective": "Subjective Assessment"}
    fig, ax = plt.subplots(figsize=(14, 6))
    markers = ['o', 's', 'D', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    for i, race_name in enumerate(race_names):
        race_results = results[race_name]
        corrs = [race_results[label]["mean"] for label in order]
        ci_lowers = [corr - race_results[label]["ci_lower"] for corr, label in zip(corrs, order)]
        ci_uppers = [race_results[label]["ci_upper"] - corr for corr, label in zip(corrs, order)]
        group_sizes = [race_results[label]["group_size"] for label in order]
        
        x = np.arange(len(order)) + i * (len(order) + 1)  # Adjusted to space out different races

        for j, label in enumerate(order):
            ax.errorbar(x[j], corrs[j], yerr=[[ci_lowers[j]], [ci_uppers[j]]], fmt=markers[j], color=colors[j], capsize=5, label=label if i == 0 else "")
            ax.text(x[j],-1.6, f'{group_sizes[j]}', ha='center', va='bottom', fontsize=14)
        
        if i < num_races - 1:
            ax.axvline(x=x[-1] + 1, color='gray', linestyle='--')

    for i, race_name in enumerate(race_order):
        x = np.mean(np.arange(len(order)) + i * (len(order) + 1))
        ax.text(x, 1.55, race_name, ha='center', va='bottom', fontsize=14, fontweight='bold')
    # Customize axes
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys(), title="Education", fontsize=12, title_fontsize=14)
    ax.legend(by_label.values(), by_label.keys(), title="Education", fontsize=12, title_fontsize=14, loc='lower right', bbox_to_anchor=(1.26, 0.06))
    ax.set_ylabel(map_var[var], fontsize=16)
    # ax.set_xticks(np.arange(len(order) * num_races) + (len(order) - 1) / 2)
    # ax.set_xticklabels([f"{label}\n({race_name})" for race_name in race_names for label in order], rotation=45, ha='right', fontsize=10)
    ax.set_ylim(-1.6, 1.5)
    ax.set_xticks([])
    # ax.set_title(f'Correlations vs. {group_name}', fontsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    fig.tight_layout()
    plt.savefig(f"./figure/race/{group_name}_{plot_type}_{var}.png")
    plt.savefig(f"./figure/race/{group_name}_{plot_type}_{var}.pdf")
def plot_percentage_and_save_stratified(results, overall, group_name, plot_type, order, var):
    race_names = race_order
    num_races = len(race_names)
    map_var = {"Self-Objective": "Objective Assessment", "Self-Subjective": "Subjective Assessment"}
    fig, ax = plt.subplots(figsize=(16, 6))
    markers = ['o', 's', 'D', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    hatches = ['/', '\\', 'x', '|']
    
    for i, race_name in enumerate(race_names):
        race_results = results[race_name]
        corrs = [race_results[label]["percentage"]*100 for label in order]
        group_sizes = [race_results[label]["group_size"] for label in order]
        
        x = np.arange(len(order)) + i * (len(order) + 1)  # Adjusted to space out different races
        # import pdb; pdb.set_trace()
        for j, label in enumerate(order):
            bar = ax.bar(x[j], corrs[j], color=colors[j], hatch=hatches[j], edgecolor='black')

            ax.text( x[j], corrs[j]+0.5,f'{corrs[j]:.1f}%', ha='center', va='center', fontsize=14)
            ax.text( x[j],-1.2 ,f'{group_sizes[j]}', ha='center', va='center', fontsize=14)
        
        if i < num_races - 1:
            ax.axvline(x=x[-1] + 1, color='gray', linestyle='--')

    for i, race_name in enumerate(race_names):
        x = np.mean(np.arange(len(order)) + i * (len(order) + 1))
        ax.text(x, 31, race_name, ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Customize axes
    handles = [mpatches.Patch(facecolor=colors[i], hatch=hatches[i], label=label) for i, label in enumerate(order)]
    ax.legend(handles=handles, title="Education", fontsize=12, title_fontsize=14, loc='lower center', bbox_to_anchor=(1.1, 0.06), ncol=1)
    ax.set_ylabel("Occupational AI Familiarity Percentage %", fontsize=16)
    ax.set_xticks([])
    ax.set_ylim(0, 30)
    ax.yaxis.set_tick_params(labelsize=16)
    fig.tight_layout()
    plt.savefig(f"./figure/race/{group_name}_{plot_type}_{var}.png")
    plt.savefig(f"./figure/race/{group_name}_{plot_type}_{var}.pdf")

overall_self = calculate_correlations([('Overall', filtered_data)], 'Overall')['Overall']

# plot figures

race_groups = filtered_data.groupby('Race')
results_self_race = calculate_correlations(race_groups, 'Race')
plot_correlations_and_save(results_self_race, overall_self, 'Race', 'Self-Evaluation',race_order)





self_percentage= calculate_percentage(race_groups, 'Race',"familarity")
plot_percentage_and_save(self_percentage, overall_self,'Race', 'Self', race_order)

self_race_education_results = calculate_correlations_by_race(filtered_data, 'Education')
plot_correlations_and_save_stratified(self_race_education_results, overall_self, 'Education', 'Self', education_order)


filtered_data['Self-Subjective'] = (filtered_data['Self-Subjective'] - filtered_data['Self-Subjective'].mean()) / filtered_data['Self-Subjective'].std()
filtered_data['Self-Objective'] = (filtered_data['Self-Objective'] - filtered_data['Self-Objective'].mean()) / filtered_data['Self-Objective'].std()
for var in ['Self-Subjective','Self-Objective']:
    race_groups = filtered_data.groupby('Race')
    self_race_results = calculate_mean_std(race_groups, 'Race',var)
    plot_meanstd_and_save(self_race_results, overall_self, 'Race', 'Self', race_order,var)

for var in ['Self-Subjective','Self-Objective']:
    self_race_education_results = calculate_mean_std_by_race(filtered_data, 'Education',var)
    plot_meanstd_and_save_stratified(self_race_education_results, overall_self, 'Education', 'Self', education_order,var)

self_race_education_results = calculate_correlations_by_race(filtered_data, 'Education')

for var in ['familarity']:
    self_race_education_results = calculate_percentage_by_race(filtered_data, 'Education',var)
    plot_percentage_and_save_stratified(self_race_education_results, overall_self, 'Education', 'Self', education_order,var)


# Draw 
# grouped = filtered_data.groupby('Gender')[['Self-Subjective', 'Self-Objective']].agg(['mean', 'std'])

# group_sizes = filtered_data['Gender'].value_counts().reindex(['Male', 'Female'])

# # 分组计算均值和标准差，确保按照指定顺序
# grouped = filtered_data.groupby('Gender')[['Self-Subjective', 'Self-Objective']].agg(['mean', 'std']).reindex(['Male', 'Female'])
# positions = range(len(grouped.index))
# # 绘图
# fig, ax = plt.subplots(figsize=(8, 6))

# # 绘制 Self-Subjective
# ax.errorbar(positions, grouped[('Self-Subjective', 'mean')], yerr=grouped[('Self-Subjective', 'std')],
#              fmt='o', capsize=5, label='Self-Subjective', linestyle='-', color='blue')

# # 绘制 Self-Objective
# ax.errorbar(positions, grouped[('Self-Objective', 'mean')], yerr=grouped[('Self-Objective', 'std')],
#              fmt='^', capsize=5, label='Self-Objective', linestyle='--', color='green')

# # 设置图表细节
# ax.set_xticks(positions)
# # ax.set_xticklabels(grouped.index)
# ax.set_title('Mean ± STD of Self-Subjective and Self-Objective by Gender')
# # for i, (gender, count) in enumerate(group_sizes.items()):
# #     ax.text(i, grouped.loc[gender][('Self-Subjective', 'mean')], f'n={count}', ha='center', va='bottom')
# ax.set_xticklabels([f'{gender}\nn={count}' for gender, count in group_sizes.items()])

# ax.set_xlabel('Gender')
# ax.set_ylabel('Scores')
# ax.legend()

# plt.tight_layout()
# plt.savefig("./figure/gender.png")
# exit()
# exit()




# Function to plot correlations with confidence intervals, including overall reference, and save the plots
# def plot_correlations_and_save_stratified(results, overall, group_name, plot_type, order, race_name):
#     # Ensure the labels follow the specified order
#     labels = order
#     corrs = [results[label]["correlation"] for label in labels]
#     ci_lowers = [corrs[i] - results[label]["ci_lower"] for i, label in enumerate(labels)]
#     ci_uppers = [results[label]["ci_upper"] - corrs[i] for i, label in enumerate(labels)]
#     group_sizes = [results[label]["group_size"]  for i, label in enumerate(labels)]
    
#     x = np.arange(len(labels))  # the label locations

#     fig, ax = plt.subplots(figsize=(6.5,5.5))

#     # Plotting with different colors for error bars
#     for i in range(len(labels)):
#         ax.errorbar(x[i], corrs[i], yerr=[[ci_lowers[i]], [ci_uppers[i]]], fmt='o', capsize=5, label=f'{labels[i]},n={results[labels[i]]["group_size"]}')
#         ax.text(x[i],-0.2, f'{group_sizes[i]}', ha='center', va='bottom', fontsize=14)

#     # Add overall reference
#     ax.axhline(y=overall['correlation'], color='black', linestyle='--', linewidth=1, label=f'Overall {plot_type} vs Objective')
#     ax.fill_between(x, overall['ci_lower'], overall['ci_upper'], color='blue', alpha=0.2)



#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     # ax.set_xlabel('Group', fontsize=16, labelpad=20)
#     ax.set_ylabel('Sub.-Obj. Correlation', fontsize=16)
#     ax.set_title(f'Correlations vs. {group_name} ({race_name})', fontsize=18)
#     ax.set_xticks(x)
#     ax.set_ylim(-0.2, 0.4)
#     ax.yaxis.set_tick_params(labelsize=14)
#     ax.set_xticklabels(labels, ha='center',rotation=30, fontsize=14)
#     # if len(labels) ==2:
#     #     ax.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, frameon=True)
#     # else:
#     #     ax.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=2, frameon=True)
    
#     fig.tight_layout()
#     # plt.subplots_adjust(top=0.75)  # Adjust the top to make room for the legend
#     plt.savefig(f"./figure/race/{group_name}_{plot_type}_{race_name}_correlations.png")
#     plt.savefig(f"./figure/race/{group_name}_{plot_type}_{race_name}_correlations.pdf")
    # plt.show()

# Example using Race group for Education correlation
# for race_name, results in self_race_education_results.items():
# self_race_income_results, other_race_income_results = calculate_correlations_by_race(filtered_data, 'Income')

# import pdb; pdb.set_trace()
# plot_correlations_and_save_stratified(self_race_income_results, overall_self, 'Income', 'Self',income_order)
# for race_name, results in other_race_education_results.items():
#     plot_correlations_and_save_stratified(results, overall_self, 'Education', 'Anchor', education_order, race_name)



# filtered_data['Self-Subjective'] = (filtered_data['Self-Subjective'] - filtered_data['Self-Subjective'].mean()) / filtered_data['Self-Subjective'].std()
# filtered_data['Self-Objective'] = (filtered_data['Self-Objective'] - filtered_data['Self-Objective'].mean()) / filtered_data['Self-Objective'].std()
# filtered_data['familarity'] = (filtered_data['familarity'] - filtered_data['familarity'].mean()) / filtered_data['familarity'].std()





# import pdb; pdb.set_trace()

# for var in ['familarity']:
#     self_race_income_results = calculate_percentage_by_race(filtered_data, 'Income',var)
#     plot_percentage_and_save_stratified(self_race_income_results, overall_self, 'Income', 'Self', income_order,var)
  
# # self_race_gender_results, other_race_gender_results = calculate_correlations_by_race(filtered_data, 'Gender')
# # Example using Race group for Education correlation
# for race_name, results in self_race_gender_results.items():
#     plot_correlations_and_save_stratified(results, overall_self, 'Gender', 'Self', gender_order, race_name)
# for race_name, results in other_race_gender_results.items():
#     plot_correlations_and_save_stratified(results, overall_self, 'Gender', 'Anchor',gender_order, race_name)
    