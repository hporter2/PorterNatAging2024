import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.dummy import DummyRegressor
import pickle
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

## loading data downloaded directly from GEO web portal
df = pd.read_csv('./geoPulls/GSE71443_control+case_Brain_white_crlmm_1.8.11.GEO.snp.tab.txt', sep = '\t', index_col = 0)

##splitting the data into signal and allele dataframes
sigdf = df.loc[:,df.columns.str.contains('Signal')]
alleles = df.loc[:,~df.columns.str.contains('Signal')]
sigdf.columns = alleles.columns

## generating sample metdata matrix using output from ALE (Giles, et al. 2017).
annotdf = pd.read_csv('./6801_age_metadata.csv')
annotdf = annotdf[annotdf.ExperimentID == 71443]
annotdf['GSM'] = 'GSM' + annotdf.iloc[:,1].astype(str)
annotdf.index = annotdf.GSM

## aligning sample annotations to preserve sample names and disease status
sample_scrape = pd.read_csv('./geoPulls/sample_copy_from_web.txt', sep = '\t', index_col = 0, header=None)
sample_scrape.columns = ['nameOriginal']
sample_scrape['nameData'] = sample_scrape.nameOriginal.str.split('_', expand = True).iloc[:,2]

## removing duplicated samples and generating final metadata object
sampwdat = sample_scrape[sample_scrape.nameData.isin(sigdf.columns)]
sampwdat = sampwdat.merge(pd.DataFrame(annotdf.Age), left_index = True, right_index = True)
sampwdat = sampwdat[~sampwdat.nameData.duplicated()]


## converting allele calls into 0 for homozygous A calls, 2 for homozygous B, and 1 for either heterozygous

alleles[alleles=='AA'] = 0
alleles[alleles=='BB'] = 2
alleles[alleles=='BA'] = 1
alleles[alleles=='AB'] = 1

## Clock and dummy clock training functions

def trainAndStoreClocks(traindat, target, n_models=100, outdir = '', outstr = 'test', startseed = 0):
    """
    Train N ElasticNet models, and save the scores and MAE to disk as well as the model coefficients.
    
    Parameters:
    traindat (DataFrame): A matrix of samples x features.
    target (Series): A vector of target ages, length n_samples.
    n_models (int): The number of models to train.
    
    Outputs:
    Two files are saved to disk:
    - 'model_scores.tsv': Contains train set model score and MAE, test set model score and MAE.
    - 'model_coefficients.tsv': Contains a matrix of dimensions features x models with the model coefficients.
    """
    # Lists to hold score and coefficient data
    scores_data = []
    coefficients_data = []
    
    # Loop to train n_models ElasticNet models
    for i in range(n_models):
        model = ElasticNetCV(cv=5, random_state=(startseed + i), n_jobs=6, l1_ratio=0.5, max_iter=5000, n_alphas = 100)
        
        # Splitting the data into train and test sets
        train = traindat.sample(frac=0.75, random_state=(startseed + i)) # 75% for training
        test = traindat.drop(train.index)                   # the rest for testing
        
        # Getting the target values for the train and test sets
        trainage = target.loc[train.index]
        testage = target.loc[test.index]
        
        # Fit the model
        model.fit(train, trainage)
        pickle.dump(model, open(('./' + outdir + outstr + '_' + str(i) + '_clockmodel.sav'), 'wb'))
        # Store the scores and MAE for both train and test sets
        scores_data.append([
            model.score(train, trainage),
            MAE(trainage, model.predict(train)),
            model.score(test, testage),
            MAE(testage, model.predict(test))
        ])
        
        # Store coefficients
        coefficients_data.append(model.coef_)
        
        # Plot distributions of age in train and test sets for the first model only
        if i == 0:
            sns.distplot(trainage)
            sns.distplot(testage)
            plt.show()
        
    # Convert scores and coefficients to DataFrames and save
    scores_df = pd.DataFrame(scores_data, columns=[
        'Train_Score', 'Train_MAE', 'Test_Score', 'Test_MAE'
    ])
    coefficients_df = pd.DataFrame(coefficients_data).T  # Transpose to get features as rows
    coefficients_df.index = traindat.columns
    scores_df.to_csv('./' + outdir + outstr + 'model_scores.tsv', sep='\t', index=False)
    coefficients_df.to_csv('./' + outdir + outstr + 'model_coefficients.tsv', sep='\t')
    
    return None


def train_dummy_models(traindat, target, n_models=1, outstr = 'test'):
    """
    Train N ElasticNet models, and save the scores and MAE to disk as well as the model coefficients.
    
    Parameters:
    traindat (DataFrame): A matrix of samples x features.
    target (Series): A vector of target ages, length n_samples.
    n_models (int): The number of models to train.
    
    Outputs:
    Two files are saved to disk:
    - 'model_scores.tsv': Contains train set model score and MAE, test set model score and MAE.
    - 'model_coefficients.tsv': Contains a matrix of dimensions features x models with the model coefficients.
    """
    # Lists to hold score and coefficient data
    scores_data = []
    coefficients_data = []
    
    # Loop to train n_models ElasticNet models
    for i in range(n_models):
        model = DummyRegressor(strategy = 'mean')
        
        # Splitting the data into train and test sets
        train = traindat.sample(frac=0.75, random_state=i) # 75% for training
        test = traindat.drop(train.index)                   # the rest for testing
        
        # Getting the target values for the train and test sets
        trainage = target.loc[train.index]
        testage = target.loc[test.index]
        
        # Fit the model
        model.fit(train, trainage)
        
        # Store the scores and MAE for both train and test sets
        scores_data.append([
            model.score(None, trainage),
            MAE(trainage, model.predict(train)),
            model.score(None, testage),
            MAE(testage, model.predict(test))
        ])
        
        # Store coefficients
#         coefficients_data.append(model.coef_)
        
        # Plot distributions of age in train and test sets for the first model only
        if i == 0:
            sns.distplot(trainage)
            sns.distplot(testage)
            plt.show()
            print(model.constant_)
            sns.regplot(trainage, model.predict(train))
            print(np.corrcoef(trainage, model.predict(train)))
            plt.show()
        
    # Convert scores and coefficients to DataFrames and save
    scores_df = pd.DataFrame(scores_data, columns=[
        'Train_Score', 'Train_MAE', 'Test_Score', 'Test_MAE'
    ])
#     coefficients_df = pd.DataFrame(coefficients_data).T  # Transpose to get features as rows
#     coefficients_df.index = traindat.columns
    scores_df.to_csv('./' + outstr + 'model_scores.tsv', sep='\t', index=False)
#     coefficients_df.to_csv('./' + outstr + 'model_coefficients.tsv', sep='\t')
    
    return None
    
### run model training

target = sampwdat.Age
target.index = sampwdat.nameData

trainAndStoreClocks(sigdf.T, target, n_models = 20, outdir = 'SignalSaves/', outstr = 'signalsOnly')
trainAndStoreClocks(alleles.T, target, n_models = 20, outdir = 'AlleleSaves/', outstr = 'allelesOnly')
train_dummy_models(sigdf.T, target, 20, outstr = 'sigDummy')
train_dummy_models(alleles.T, target, 20, outstr = 'alleleDummy')

## functions for plotting and downstream statistics
def read_and_pivot_data(filepaths, group_labels, value_vars=['Test_Score', 'Test_MAE']):
    """
    Read multiple TSV files containing model scores, add group labels,
    and pivot the data to have groups with test scores and test MAE.

    Parameters:
    filepaths (list): A list of file paths to the TSV files.
    group_labels (list): A list of labels describing the group each model was trained on.
    value_vars (list): A list of columns to pivot on. Defaults to ['Test_Score', 'Test_MAE'].

    Returns:
    combined_data (DataFrame): A DataFrame containing the pivoted data.
    """
    all_scores = []

    # Read each file and add the group label
    for filepath, label_store in zip(filepaths, group_labels):
        scores = pd.read_csv(filepath, sep='\t')
        scores['Group'] = label_store
        all_scores.append(scores)

    # Concatenate all scores into a single DataFrame
    combined_scores = pd.concat(all_scores, ignore_index=True)

    # Pivot the DataFrame to have groups with test scores and test MAE
    combined_data = pd.melt(combined_scores, id_vars=['Group'], value_vars=value_vars,
                            var_name='Metric', value_name='Value')

    return combined_data

def plot_kde_score(data, group_labels, savefig=None):
    """
    Plot a KDE distribution for the 'Test_Score' metric from the data and optionally save the figure.

    Parameters:
    data (DataFrame): The data containing the scores and groups.
    group_labels (list): A list of labels describing the group each model was trained on.
    savefig (str, optional): The filename to save the figure. If None, the figure is not saved.
    """
    # Filter data for the 'Test_Score' metric
    score_data = data[data['Metric'] == 'Test_Score']

    # Plotting KDE plot for 'Test_Score'
    plt.figure(figsize=(10, 6))
    score_plot = sns.kdeplot(data=score_data, x='Value', hue='Group', common_norm=False)
    
    # Setting custom ticks on the x-axis
    score_plot.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    for label in score_plot.xaxis.get_majorticklabels()[1::2]:
        label.set_visible(False)
    
    # Adding legend and title
    plt.title('Distribution of Test Scores')
    plt.xlabel('Model Score ($R^{2}$)')
    plt.ylabel('KDE')
    # Saving the figure if a filename is provided
    if savefig:
        plt.savefig(savefig, format='eps')

    plt.show()

def plot_kde_mae(data, group_labels, savefig=None):
    """
    Plot a KDE distribution for the 'Test_MAE' metric from the data and optionally save the figure.

    Parameters:
    data (DataFrame): The data containing the MAE and groups.
    group_labels (list): A list of labels describing the group each model was trained on.
    savefig (str, optional): The filename to save the figure. If None, the figure is not saved.
    """
    # Filter data for the 'Test_MAE' metric
    mae_data = data[data['Metric'] == 'Test_MAE']

    # Plotting KDE plot for 'Test_MAE'
    plt.figure(figsize=(10, 6))
    mae_plot = sns.kdeplot(data=mae_data, x='Value', hue='Group', common_norm=False)
    
    # Setting custom ticks on the x-axis
    mae_plot.xaxis.set_major_locator(plt.MultipleLocator(1))
    for label in mae_plot.xaxis.get_majorticklabels()[1::2]:
        label.set_visible(False)

    # Adding legend and title
    plt.title('Distribution of Test MAE')
    plt.xlabel('Mean Absolute Error')
    plt.ylabel('KDE')
    # Saving the figure if a filename is provided
    if savefig:
        plt.savefig(savefig, format='eps')

    plt.show()

def OneWayANOVA(df, group_col, value_col):
    group_data = {}
    for group in df[group_col].unique():
        groupval = df[df[group_col] == group][value_col].values
        group_data[group] = groupval
    fstat, pval = f_oneway(*group_data.values())
    return fstat, pval

def TukeyPostHoc(df, group_col, value_col):
    tukey = pairwise_tukeyhsd(endog=df[value_col], groups = df[group_col], alpha = 0.05)
    return tukey.summary()

## plotting calls
combdata = read_and_pivot_data(['./SignalSaves/signalsOnlymodel_scores.tsv',
                               './AlleleSaves/allelesOnlymodel_scores.tsv',
                               './alleleDummymodel_scores.tsv'],
                    ['ArraySignal', 'AlleleCall', 'Dummy'])
plot_kde_score(combdata, ['ArraySignal', 'AlleleCall', 'Dummy'], 'snpclock_score.eps' )
plot_kde_mae(combdata, ['ArraySignal', 'AlleleCall', 'Dummy'], 'snpclock_mae.eps')

## one-way ANOVA for group differences and post-hoc tests
metaf, metap = OneWayANOVA(combdata[combdata.Metric == 'Test_Score'], 'Group', 'Value')
print(f"F-statistic: {metaf}, P-value: {metap}")
tukeyres = TukeyPostHoc(combdata[combdata.Metric == 'Test_Score'], 'Group', 'Value')
print(tukeyres)

metaf, metap = OneWayANOVA(combdata[combdata.Metric == 'Test_MAE'], 'Group', 'Value')
print(f"F-statistic: {metaf}, P-value: {metap}")
tukeyres = TukeyPostHoc(combdata[combdata.Metric == 'Test_MAE'], 'Group', 'Value')
print(tukeyres)


