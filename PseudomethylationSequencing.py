# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# Define helper functions
def hunter_combiner(file_list, foresplit, backsplit):
    """
    Combines methylation or pseudomethylation data from multiple files into a single DataFrame.
    """
    combined_df = pd.DataFrame()
    for f in file_list:
        # Extract sample name from file name
        sample_name = f.split(foresplit)[1].split(backsplit)[0]
        temp_df = pd.read_csv(f, sep='\t', header=None)
        temp_df['pos'] = temp_df.iloc[:, 0] + '.' + temp_df.iloc[:, 1].astype(str)
        # Calculate methylation ratio
        temp_df[sample_name] = temp_df.iloc[:, 3] / (temp_df.iloc[:, 3] + temp_df.iloc[:, 4])
        temp_df = temp_df[['pos', sample_name]]
        temp_df.set_index('pos', inplace=True)
        if combined_df.empty:
            combined_df = temp_df
        else:
            combined_df = combined_df.join(temp_df, how='outer')
    return combined_df

# 1. Load and combine real methylation data over all sites
# Define the file paths to your methylation data
methylation_files = glob.glob('../../dissertation/methyl_mutations/seqalign/wgss_bismark_output/*_final.cov')

# Combine methylation data across samples
methylation_df = hunter_combiner(
    methylation_files,
    foresplit='output/',
    backsplit='_final.cov'
)

# Save the methylation DataFrame if needed
# methylation_df.to_csv('methylation_matrix.csv')

# 2. Load and combine pseudomethylation data over all sites
# Define the file paths to your pseudomethylation data (from WGS data aligned with bismark in bedgraph format)
pseudomethylation_files = glob.glob('../../dissertation/methyl_mutations/seqalign/wgss_bismark_output/*.bedgraph')

# Combine pseudomethylation data across samples
pseudomethylation_df = hunter_combiner(
    pseudomethylation_files,
    foresplit='output/',
    backsplit='.bedgraph'
)

# Save the pseudomethylation DataFrame if needed
# pseudomethylation_df.to_csv('pseudomethylation_matrix.csv')

# 3. Intersect methylation and pseudomethylation data with VCF files to get mutated sites
# Note: The intersection was performed using bedtools intersect of the VCFs against the bismark bedgraphs
# Assuming the intersected files are available as *_intersected.bed

# Define the file paths to your intersected methylation data
intersected_methylation_files = glob.glob('../../dissertation/methyl_mutations/seqalign/wgss_bismark_output/*_intersected_meth.bed')

# Combine intersected methylation data
intersected_methylation_df = hunter_combiner(
    intersected_methylation_files,
    foresplit='output/',
    backsplit='_intersected_meth.bed'
)

# Define the file paths to your intersected pseudomethylation data
intersected_pseudomethylation_files = glob.glob('../../dissertation/methyl_mutations/seqalign/wgss_bismark_output/*_intersected.bed')

# Combine intersected pseudomethylation data
intersected_pseudomethylation_df = hunter_combiner(
    intersected_pseudomethylation_files,
    foresplit='output/',
    backsplit='_intersected.bed'
)

# 4. Generate the four required data objects
# 4.1 Real methylation data over all common sites
# 4.2 Pseudomethylation data over all common sites
# Find common positions between methylation and pseudomethylation data
common_positions = methylation_df.index.intersection(pseudomethylation_df.index)

# Filter dataframes to include only common positions
methylation_common = methylation_df.loc[common_positions]
pseudomethylation_common = pseudomethylation_df.loc[common_positions]

# 4.3 Real methylation data over intersected mutants
# 4.4 Pseudomethylation data over intersected mutants
# Find common positions among intersected data
mutated_positions = intersected_methylation_df.index.intersection(intersected_pseudomethylation_df.index)

# Filter dataframes to include only mutated positions
methylation_mutated = intersected_methylation_df.loc[mutated_positions]
pseudomethylation_mutated = intersected_pseudomethylation_df.loc[mutated_positions]

# 5. Calculate mean methylation and pseudomethylation levels
# For all common sites
mean_methylation_all = methylation_common.mean(axis=1)
mean_pseudomethylation_all = pseudomethylation_common.mean(axis=1)

# For mutated sites
mean_methylation_mutated = methylation_mutated.mean(axis=1)
mean_pseudomethylation_mutated = pseudomethylation_mutated.mean(axis=1)

# 6. Calculate meth_mutscaled for all sites and mutated sites
epsilon = 1e-8  # Small constant to prevent division by zero

# For all sites
meth_mutscaled_all = (mean_methylation_all / (mean_pseudomethylation_all + epsilon)).clip(upper=1)

# For mutated sites
meth_mutscaled_mutated = (mean_methylation_mutated / (mean_pseudomethylation_mutated + epsilon)).clip(upper=1)

# 7. Calculate changes in apparent methylation level
# For all sites
change_in_methylation_all = meth_mutscaled_all - mean_methylation_all

# For mutated sites
change_in_methylation_mutated = meth_mutscaled_mutated - mean_methylation_mutated

# 8. Visualization

# 8.1 Violin plots

# Violin plot for all common sites
violin_data_all = pd.DataFrame({
    'Methylation': mean_methylation_all,
    'Pseudomethylation': mean_pseudomethylation_all
}).melt(var_name='Type', value_name='Level')

plt.figure(figsize=(8,6))
sns.violinplot(x='Type', y='Level', data=violin_data_all, inner='quartile')
plt.title('Violin Plot of Methylation and Pseudomethylation Levels (All Common Sites)')
plt.ylabel('Level')
plt.savefig('violin_plot_all_common_sites.png', dpi=300)
plt.show()

# Violin plot for mutated sites
violin_data_mutated = pd.DataFrame({
    'Methylation': mean_methylation_mutated,
    'Pseudomethylation': mean_pseudomethylation_mutated
}).melt(var_name='Type', value_name='Level')

plt.figure(figsize=(8,6))
sns.violinplot(x='Type', y='Level', data=violin_data_mutated, inner='quartile')
plt.title('Violin Plot of Methylation and Pseudomethylation Levels (Mutated Sites)')
plt.ylabel('Level')
plt.savefig('violin_plot_mutated_sites.png', dpi=300)
plt.show()

# 8.2 Joint plots

# Joint plot for all common sites
sns.jointplot(
    x=mean_methylation_all,
    y=mean_pseudomethylation_all,
    kind='scatter',
    s=10,
    alpha=0.5
)
plt.xlabel('Mean Methylation Level (All Sites)')
plt.ylabel('Mean Pseudomethylation Level (All Sites)')
plt.suptitle('Joint Plot of Methylation vs. Pseudomethylation (All Sites)', y=1.02)
plt.savefig('joint_plot_all_common_sites.png', dpi=300)
plt.show()

# Joint plot for mutated sites
sns.jointplot(
    x=mean_methylation_mutated,
    y=mean_pseudomethylation_mutated,
    kind='scatter',
    s=10,
    alpha=0.5
)
plt.xlabel('Mean Methylation Level (Mutated Sites)')
plt.ylabel('Mean Pseudomethylation Level (Mutated Sites)')
plt.suptitle('Joint Plot of Methylation vs. Pseudomethylation (Mutated Sites)', y=1.02)
plt.savefig('joint_plot_mutated_sites.png', dpi=300)
plt.show()

# 8.3 Joint plot of methylation vs meth_mutscaled

# For all sites
sns.jointplot(
    x=mean_methylation_all,
    y=meth_mutscaled_all,
    kind='scatter',
    s=10,
    alpha=0.5
)
plt.xlabel('Mean Methylation Level (All Sites)')
plt.ylabel('meth_mutscaled (All Sites)')
plt.suptitle('Joint Plot of Methylation vs. meth_mutscaled (All Sites)', y=1.02)
plt.savefig('joint_plot_methylation_vs_meth_mutscaled_all_sites.png', dpi=300)
plt.show()

# For mutated sites
sns.jointplot(
    x=mean_methylation_mutated,
    y=meth_mutscaled_mutated,
    kind='scatter',
    s=10,
    alpha=0.5
)
plt.xlabel('Mean Methylation Level (Mutated Sites)')
plt.ylabel('meth_mutscaled (Mutated Sites)')
plt.suptitle('Joint Plot of Methylation vs. meth_mutscaled (Mutated Sites)', y=1.02)
plt.savefig('joint_plot_methylation_vs_meth_mutscaled_mutated_sites.png', dpi=300)
plt.show()

# 8.4 Density plots of changes in apparent methylation level

# For all sites
plt.figure(figsize=(8,6))
sns.kdeplot(change_in_methylation_all, shade=True)
plt.xlabel('Change in Apparent Methylation Level (All Sites)')
plt.title('Density Plot of Changes in Apparent Methylation Level (All Sites)')
plt.savefig('density_plot_change_in_methylation_all_sites.png', dpi=300)
plt.show()

# For mutated sites
plt.figure(figsize=(8,6))
sns.kdeplot(change_in_methylation_mutated, shade=True)
plt.xlabel('Change in Apparent Methylation Level (Mutated Sites)')
plt.title('Density Plot of Changes in Apparent Methylation Level (Mutated Sites)')
plt.savefig('density_plot_change_in_methylation_mutated_sites.png', dpi=300)
plt.show()

