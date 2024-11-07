import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.dummy import DummyRegressor
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as ss
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

## read in data and sample annotations
with open('Freeman mouse methylation arrays 20220610/beta_values.pkl', 'rb') as pickle_file:
    content = pickle.load(pickle_file)

labels = pd.read_excel('./Freeman mouse methylation arrays 20220610/Mouse Methylation plate 4 (Freeman VicA mm plate 1) sample locations 2022-06-10.xlsx')
labels['ids'] = labels['Array ID'].astype(str) + '_' + labels['Array Location'].astype(str)
labels.index = labels.ids
labels.columns = ['well', 'method', 'name', 'age', 'id', 'arrayid', 'loc', 'ids']
labels.name = labels.name.str.replace('Sample ', '')
labels.age = labels.age.str.replace(' months', '')
labels['id_meth'] = labels.name.astype(str) + '_' + labels.method.astype(str)
labels.index = labels.id_meth

## remove duplicate samples to exclude technical replicates from further analysis
labels = labels.loc[~labels.index.duplicated()]
content = content.loc[:,~content.columns.duplicated()]

## split data into three frames for each data type
conv = content.loc[:,labels[labels['method'] == 'Converted'].id_meth]
uncon = content.loc[:,labels[labels['method'] == 'Unconverted'].id_meth]

## append annotations in case matrices are combined later
conv.columns = conv.columns.str.replace('_Converted', '')
uncon.columns = uncon.columns.str.replace('_Unconverted', '')

## set up imputation
imputer = sklearn.impute.KNNImputer(n_neighbors=3)

def knnimpute(df):
    arl = np.array_split(df, 10)
    for i in range(0, len(arl)):
        arl[i] = pd.DataFrame(imputer.fit_transform(arl[i]))
    outdf = pd.concat(arl)
    outdf.columns = df.columns
    outdf.index = df.index
    return outdf

convi = knnimpute(conv)  
unconi = knnimpute(uncon)

## supplemental figures of data distributions
sns.jointplot(x = uncon.mean(axis=1), y = conv.mean(axis=1), s = 1)
plt.show()

sns.jointplot(x = uncon.fillna(1).mean(axis = 1), y = conv.fillna(1).mean(axis = 1), s = 1,
             linewidth=0)
plt.show()

def train_elastic_net_models_mifilt(traindat, target, n_models=100, outstr='test', existing_results=False):
    """
    Train N ElasticNet models with mutual information filtering, save results, or read existing results.

    Parameters:
    traindat (DataFrame): Training data (samples x features).
    target (Series): Target variable.
    n_models (int): Number of models to train.
    outstr (str): Output filename prefix.
    existing_results (bool): If True, read existing results from disk instead of recomputing.

    Outputs:
    - '<outstr>_model_scores.tsv': Contains model scores and MAE.
    - '<outstr>_model_coefficients.tsv': Contains model coefficients.
    - Models are saved as '<outstr>_<i>_model.sav'.
    """
    scores_filename = f'./{outstr}_model_scores.tsv'
    coefficients_filename = f'./{outstr}_model_coefficients.tsv'

    if existing_results and os.path.exists(scores_filename) and os.path.exists(coefficients_filename):
        print(f"Reading existing results from {scores_filename} and {coefficients_filename}")
        scores_df = pd.read_csv(scores_filename, sep='\t')
        coefficients_df = pd.read_csv(coefficients_filename, sep='\t', index_col=0)
        return scores_df, coefficients_df

    # Lists to hold score and coefficient data
    scores_data = []
    coefficients_data = []

    for i in range(n_models):
        model_filename = f'./{outstr}_{i}_model.sav'

        if existing_results and os.path.exists(model_filename):
            print(f"Loading existing model from {model_filename}")
            model = pickle.load(open(model_filename, 'rb'))
        else:
            # Initialize model
            model = ElasticNetCV(cv=2, random_state=i+10, n_jobs=6, l1_ratio=0.5, max_iter=15000, n_alphas=100)

            # Splitting the data into train and test sets
            train = traindat.sample(frac=0.75, random_state=i+10)
            test = traindat.drop(train.index)

            # Getting the target values for the train and test sets
            trainage = target.loc[train.index]
            testage = target.loc[test.index]

            # Mutual information filtering
            trainmi = mutual_info_regression(train, trainage)
            trainmi /= np.max(trainmi)
            trainmi = pd.Series(trainmi, index=train.columns)
            sites = trainmi[trainmi > 0.2].index

            # Filter features
            train = train[sites]
            test = test[sites]

            # Fit the model
            model.fit(train, trainage)

            # Save the model
            pickle.dump(model, open(model_filename, 'wb'))

        # Evaluate the model
        train_score = model.score(train, trainage)
        train_mae = MAE(trainage, model.predict(train))
        test_score = model.score(test, testage)
        test_mae = MAE(testage, model.predict(test))

        scores_data.append([train_score, train_mae, test_score, test_mae])

        # Store coefficients
        coefficients = pd.Series(model.coef_, index=train.columns)
        coefficients_data.append(coefficients)

        # Plot distributions of age in train and test sets for the first model only
        if i == 0:
            plt.figure()
            sns.distplot(trainage, label='Train Age Distribution')
            sns.distplot(testage, label='Test Age Distribution')
            plt.legend()
            plt.show()

    # Convert scores and coefficients to DataFrames and save
    scores_df = pd.DataFrame(scores_data, columns=['Train_Score', 'Train_MAE', 'Test_Score', 'Test_MAE'])
    coefficients_df = pd.DataFrame(coefficients_data).T  # Transpose to get features as rows
    coefficients_df.columns = [f'Model_{i}' for i in range(n_models)]
    scores_df.to_csv(scores_filename, sep='\t', index=False)
    coefficients_df.to_csv(coefficients_filename, sep='\t')

    return scores_df, coefficients_df

def train_elastic_net_models(traindat, target, n_models=1, outstr = 'test'):
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
        model = ElasticNetCV(cv=5, random_state=i, n_jobs=6, l1_ratio=0.5, max_iter=5000, n_alphas = 100)
        
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
    scores_df.to_csv('./' + outstr + 'model_scores.tsv', sep='\t', index=False)
    coefficients_df.to_csv('./' + outstr + 'model_coefficients.tsv', sep='\t')
    
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

## fitting actual clocks, dummy models and permuted models
target = labels[labels.method == 'Converted'].age[0:39]
target.index = pd.Series(target.index).str.split('_', expand = True).iloc[:,0]
traindat = traindat.drop('38')
target = target.drop('38')
train_elastic_net_models_mifilt(convi, target.astype(int), 100, 'internal_MI_converted_100iter')
train_dummy_models(convi, target.astype(int), 100, 'Converted_mean')
## permuting age labels 
target = target.sample(frac=1, ignore_index=True)
target.index = traindat.index
train_elastic_net_models(convi, target, 100, 'Converted_permuted')


target = labels[labels.method == 'Unconverted'].age[0:39]
target.index = pd.Series(target.index).str.split('_', expand = True).iloc[:,0]
traindat = traindat.drop('38')
target = target.drop('38')
train_elastic_net_models_mifilt(unconi, target.astype(int), 100, 'internal_MI_unconverted_100iter')
train_dummy_models(unconi, target.astype(int), 100, 'Unconverted_mean')
## permuting age labels  for unconverted
target = target.sample(frac=1, ignore_index=True)
target.index = traindat.index
train_elastic_net_models(unconi, target, 100, 'Unconverted_permuted')


## generating prediction from combined matrix
bothdf = convi.merge(unconi, left_index = True, right_index = True)
train_elastic_net_models_mifilt(bothdf, target.astype(int), 100, 'internal_MI_both')


def clockslice(matrix, sites, annotmat):
    '''
    Slice either data matrix to rows over the given features and draw regression plots 
    against annotmat given (in this case age).
    '''
    tmpmat = matrix.loc[sites].dropna()
    target = annotmat
    target.index = pd.Series(target.index).str.split('_', expand = True).iloc[:,0]
    for f in tmpmat.index:
        sns.regplot(target, tmpmat.loc[f])
        plt.show()
    return None
    
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
    plt.xlabel('Mean Absolute Error')
    plt.ylabel('KDE')

    # Saving the figure if a filename is provided
    if savefig:
        plt.savefig(savefig, format='eps')

    plt.show()
    
## calls for generating plots
combdata = read_and_pivot_data(['internal_MI_unconverted_100iter_model_scores.tsv',
                                'internal_MI_converted_100iter_model_scores.tsv',
                                'internal_MI_both.tsv',
                               'Converted_meanmodel_scores.tsv', 'Unconverted_permutedmodel_scores.tsv',
                               'Converted_permutedmodel_scores.tsv'],
                    ['Unconverted', 'Converted', 'Both', 'Dummy', 'PermuteUnc', 'PermuteCon'])
plot_kde_score(combdata, ['Unconverted', 'Converted', 'Both', 'Dummy', 'PermuteUnc', 'PermuteCon'], '600clocks_score.eps' )
plot_kde_mae(combdata, ['Unconverted', 'Converted', 'Both', 'Dummy', 'PermuteUnc', 'PermuteCon'], '600clocks_mae.eps')

## statistical analysis
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

metaf, metap = OneWayANOVA(combdata[combdata.Metric == 'Test_Score'], 'Group', 'Value')
print(f"F-statistic: {metaf}, P-value: {metap}")
tukeyres = TukeyPostHoc(combdata[combdata.Metric == 'Test_Score'], 'Group', 'Value')
print(tukeyres)

metaf, metap = OneWayANOVA(combdata[combdata.Metric == 'Test_MAE'], 'Group', 'Value')
print(f"F-statistic: {metaf}, P-value: {metap}")
tukeyres = TukeyPostHoc(combdata[combdata.Metric == 'Test_MAE'], 'Group', 'Value')
print(tukeyres)

## plotting age acceleration from clocks of each type
def compareClocks(model1, model2, model1data, model2data, model1age, model2age, plttitle):
    model1pred = model1.predict(model1data)
    model2pred = model2.predict(model2data)
    mod1delta = model1pred - model1age
    mod2delta = model2pred - model2age
    sns.regplot(x = mod1delta, y = mod2delta)
    plt.savefig(plttitle, format = 'eps')
    print('CorrelationCoefficient = ')
    return None

conv_clock19 = pickle.load('./converted_allsites_19_clockmodel.sav')
unc_clock19 = pickle.load('./unconverted_allsites_19_clockmodel.sav')


convagevec = labels[labels.method == 'Converted'].age[0:39]
convagevec.index = pd.Series(convagevec.index).str.split('_', expand = True).iloc[:,0]
convagevec = convagevec.drop('38')

uncagevec = labels[labels.method == 'Unconverted'].age[0:39]
uncagevec.index = pd.Series(uncagevec.index).str.split('_', expand = True).iloc[:,0]
uncagevec = uncagevec.drop('38')

convi_proc = convi.T.drop('38')
convi_proc.columns = (pd.Series(convi_proc.columns) + '_conv')

unconi_proc = unconi.T.drop('38')
unconi_proc.columns = (pd.Series(unconi_proc.columns) + '_uncon')

unctrain19 = unconi_proc.sample(frac=0.75, random_state=19)
unctest19 = unconi_proc.drop(unctrain19.index) 
unctestage = uncagevec.loc[unctest19.index]
contrain19 = convi_proc.sample(frac=0.75, random_state=19)
contest19 = convi_proc.drop(contrain19.index)
contestage = convagevec.loc[contest19.index]

compareClocks(conv_clock19, unc_clock19,
              contest19, unctest19,
              contestage.astype(int), unctestage.astype(int),
             'Imputed_ageaccel_regrplot_testonly.eps')
             
## plotting mutual information for each data type
mi = mutual_info_regression(convi.T[~convi.T.index.duplicated(keep='first')].drop_duplicates(),
                            labels[labels.method == 'Converted'].age[0:39])
mi /= np.max(mi)
mi = pd.Series(mi)
mi2 = mutual_info_regression(unconi.T[~unconi.T.index.duplicated(keep='first')].drop_duplicates(),
                            labels[labels.method == 'Unconverted'].age[0:39])
mi2 /= np.max(mi2)
mi2 = pd.Series(mi2)
mimat = pd.DataFrame([mi, mi2]).T
mimat.columns = ['Converted', 'Unconverted']
sns.jointplot(data = mimat[(mimat.Converted > .2 ) | (mimat.Unconverted > .2)],
              x = 'Converted', y = 'Unconverted', s = 1)
plt.savefig('MI_distributions.png', format = 'png')

## plot specific clock models for comparison
## was performed before refactor for multiple clocks


def train_and_predict(traindat, target, random_state=0):
    """
    Train an ElasticNet model and return the model and predictions for both train and test sets.

    Parameters:
    traindat (DataFrame): A matrix of samples x features.
    target (Series): A vector of target ages, length n_samples.
    random_state (int): The seed for the random number generator.

    Returns:
    model: Trained ElasticNetCV model.
    predictions (dict): A dictionary containing the actual and predicted ages for both train and test sets.
    """
    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(traindat, target, test_size=0.25, random_state=random_state)
    
    # Initialize and fit the ElasticNetCV model
    model = ElasticNetCV(cv=2, random_state=random_state, n_jobs=6, l1_ratio=0.5, max_iter=15000)
    model.fit(X_train, y_train)
    
    
    # Get predictions for both the train and test sets
    predictions = {
        'y_train': y_train,
        'y_train_pred': model.predict(X_train),
        'y_test': y_test,
        'y_test_pred': model.predict(X_test)
    }
    print(model.score(X_test, y_test))
    return model, predictions

def plot_actual_vs_predicted(y_actual, y_predicted, title, filename):
    """
    Plot actual vs. predicted ages and save the plot as an EPS file.

    Parameters:
    y_actual (array-like): Actual ages.
    y_predicted (array-like): Predicted ages.
    title (str): The title of the plot.
    filename (str): The filename for saving the plot.
    """
    plt.figure(figsize=(8, 8))
    sns.regplot(x=y_actual, y=y_predicted)
    plt.plot([0, 30], [0, 30], '--', color='grey')  # Add a y=x reference line
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.title(title)
    plt.savefig(filename, format='eps')
    plt.close() 

def select_common_features(convi, unconi, mi, mi2, labels, 
                           method_converted='Converted', 
                           method_unconverted='Unconverted', 
                           sample_to_remove='38'):
    """
    Select features based on the outer join of primary feature selections from two series 
    and remove a specified sample from the train data and targets.

    Parameters:
    convi (DataFrame): Converted data with all features.
    unconi (DataFrame): Unconverted data with all features.
    mi (Series): Primary feature selection for converted data.
    mi2 (Series): Primary feature selection for unconverted data.
    labels (DataFrame): DataFrame with target labels.
    method_converted (str): Label for converted data in the labels DataFrame.
    method_unconverted (str): Label for unconverted data in the labels DataFrame.
    sample_to_remove (str): Sample label to remove from the data and targets.

    Returns:
    ctraindat, utraindat (DataFrame): Dataframes with common features selected.
    ctarget, utarget (Series): Target series with the specified sample removed.
    """
    # Get common features
    common_features = mi.index.union(mi2.index)
    
    # Select the features from both dataframes that are in the union of primary feature selections
    ctraindat = convi.loc[common_features].T
    utraindat = unconi.loc[common_features].T

    # Get the target labels for both converted and unconverted data
    ctarget = labels[labels.method == method_converted].age.astype(int)
    utarget = labels[labels.method == method_unconverted].age.astype(int)

    # Adjust the index to remove the method part and any specified samples
    ctarget.index = ctarget.index.str.split('_', expand=True).get_level_values(0)
    utarget.index = utarget.index.str.split('_', expand=True).get_level_values(0)

    # Drop the specified sample
    ctraindat.drop(index=sample_to_remove, errors='ignore', inplace=True)
    utraindat.drop(index=sample_to_remove, errors='ignore', inplace=True)
    ctarget.drop(index=sample_to_remove, errors='ignore', inplace=True)
    utarget.drop(index=sample_to_remove, errors='ignore', inplace=True)

    return ctraindat, utraindat, ctarget, utarget

ctraindat, utraindat, ctarget, utarget = select_common_features(convi, unconi,
                                                                mi[mi > 0.2], mi2[mi2 > 0.2],
                                                                labels)


model_converted, predictions_converted = train_and_predict(, ctarget, 8675309)
model_unconverted, predictions_unconverted = train_and_predict(utraindat, utarget,  8675309)

# Plot and save the four comparisons
plot_actual_vs_predicted(
    predictions_converted['y_test'],
    predictions_converted['y_test_pred'],
    'Converted Model on Converted Test Set',
    'converted_on_converted.eps'
)

plot_actual_vs_predicted(
    predictions_unconverted['y_test'],
    predictions_unconverted['y_test_pred'],
    'Unconverted Model on Unconverted Test Set',
    'unconverted_on_unconverted.eps'
)

### Enrichment testing for clock sites
### feature tables were downloaded manually from UCSC table browser for each specified table
### each bed file was then stored as <taxon_id>_<feature>.bed for mm10

def genome(taxon_id):
    background_template = ('{taxonid}_{feature}.bed')
    genes = pb.BedTool(background_template.format(taxonid = taxon_id, feature = 'genes'))
    exons = pb.BedTool(background_template.format(taxonid = taxon_id, feature = 'exons'))
    introns = pb.BedTool(background_template.format(taxonid = taxon_id, feature = 'introns'))
    promoters = pb.BedTool(background_template.format(taxonid = taxon_id, feature = 'promoters'))
    islands = pb.BedTool(background_template.format(taxonid = taxon_id, feature = 'islands'))
    shores = pb.BedTool(background_template.format(taxonid = taxon_id, feature = 'shores'))
    shelves = pb.BedTool(background_template.format(taxonid = taxon_id, feature = 'shelves'))
    regels = 'nan'
    eqtls = 'nan'
    dnase = 'nan'
    enhancers = 'nan'
        # return genes, exo
    return genes, exons, introns, promoters, islands, shores, shelves, regels, eqtls, dnase, enhancers

def genemap (input_files, taxon_id = 9606):
    sample_template = ('{sampleid}.bed')
    out_df = pd.DataFrame()
    genes, exons, introns, promoters, islands, shores, shelves ,regels, eqtls, dnase, enhancers = genome(taxon_id)
#     out_df.index = ['Input', 'Genes', 'Exons', 'Introns', 'Promoters', 'Intergenic']
    for i in input_files:
        print(i)
        sample = pb.BedTool(sample_template.format(sampleid = i))
        input_count = sample.count()
        print(input_count)
        gene_map = sample.intersect(genes, u = True)
        gene_count = gene_map.count()
        ex_map = sample.intersect(exons, u = True)
        ex_count = ex_map.count()
        in_map = sample.intersect(introns, u = True)
        in_count = in_map.count()
        prom_map = sample.intersect(promoters, u = True)
        prom_count = prom_map.count()
        inter_count = (input_count - (prom_count + gene_count))
        if inter_count < 0:
            inter_count = 0
        count_ser = pd.Series([input_count, gene_count, ex_count,
                               in_count, prom_count, inter_count])
        count_ser.index = ['Input', 'Genes', 'Exons', 'Introns', 'Promoters', 'Intergenic']
        out_df[i] = count_ser
    return out_df

def islemap (input_files, taxon_id = 9606):
    sample_template = ('{sampleid}.bed')
    out_df = pd.DataFrame()
    genes, exons, introns, promoters, islands, shores, shelves ,regels, eqtls, dnase, enhancers = genome(taxon_id)
#     out_df.index = ['Input', 'Genes', 'Exons', 'Introns', 'Promoters', 'Intergenic']
    for i in input_files:
        print(i)
        sample = pb.BedTool(sample_template.format(sampleid = i))
        input_count = sample.count()
        isle_count = sample.intersect(islands, u = True).count()
        shore_count = sample.intersect(shores, u = True).count()
        shelf_count = sample.intersect(shelves, u = True).count()
#         sea_count = sample.intersect(b = [islands, shores, shelves], v = True).count()
        sea_count = (input_count - (isle_count + shore_count + shelf_count))
        count_ser = pd.Series([input_count, isle_count, shore_count,
                              shelf_count, sea_count])
        count_ser.index = ['Input', 'Island', 'Shore', 'Shelf', 'Sea']
        out_df[i] = count_ser
    return out_df

def bars2(exp_vector_list, bg_vector, outfile):
    exdf = pd.DataFrame()
    enrich_df = pd.DataFrame()
    top_df = pd.DataFrame()
    bot_df = pd.DataFrame()
    listnames = str
    import scipy.stats as ss
    for i in exp_vector_list:
        featurevec = []
        citops = []
        cibots = []
        for j in i.index[1:]:
            exp_in_feat = i.loc[j]
            exp_notin = i[0] - exp_in_feat 
            bck_in_feat = bg_vector.loc[j]
            bck_notin = bg_vector[0] - bg_vector.loc[j]
            table = np.asarray([[exp_in_feat, bck_in_feat], [exp_notin, bck_notin]])
            tempt = contingency_tables.Table2x2(table)
            oddsrat = tempt.log_oddsratio
            pval = tempt.log_oddsratio_pvalue()
            CI = tempt.log_oddsratio_confint()
            citops.append(CI[0])
            cibots.append(CI[1])
            print([pval, j, i.name, CI])
            featurevec.append(oddsrat)
            
    #         enrich_df[i.name] = pd.Series(featurevec)
        enrich_df[i.name] = pd.Series(featurevec)
        top_df[i.name] = pd.Series(citops)
        bot_df[i.name] = pd.Series(cibots)
    enrich_df.index = i.index[1:]
    enrich_df['features'] = enrich_df.index
    top_df.index = i.index[1:]
    bot_df.index = i.index[1:]
    top_df['features'] = enrich_df.index
    bot_df['features'] = enrich_df.index
    topdf = pd.melt(top_df, id_vars = 'features', var_name = 'group', value_name = 'top_CI')
    botdf = pd.melt(bot_df, id_vars = 'features', var_name = 'group', value_name = 'bot_CI')
    plotdf = pd.melt(enrich_df, id_vars='features', var_name="group", value_name="Log Odds Ratio")
    yerr = [plotdf['Log Odds Ratio'] - botdf.bot_CI, topdf.top_CI - plotdf['Log Odds Ratio']]
    fig, ax = plt.subplots()
    ax = sns.barplot(data = plotdf, x = 'features', y = 'Log Odds Ratio', hue = 'group')
    sns.boxplot(data = topdf, x = 'features', y = 'top_CI', hue = 'group')
    sns.boxplot(data = botdf, x = 'features', y = 'bot_CI', hue = 'group')
#     plt.errorbar(x=range(len(plotdf['features'])), y= plotdf['Log Odds Ratio'], yerr=yerr,
#                  fmt='none', c= 'r')
#     plt.xticks(range(len(plotdf['features'])), plotdf['features'])
    plt.ylim(-2, 2)
    ax.legend_.remove()
    plt.savefig(outfile, format = 'eps')
    plt.show()

### loci from coefs.tsv were flattened into a list of all sites ever included then intersected with the manifest file from Illumina to generate bed files for converted, unconverted, and background sites

def arrayCGtoPosition(newsites_list, illumina_annot = annot, target_columns = ['CHR', 'MAPINFO']):
    '''
    Intersects lists of sites from 
    methylation array and converts to positions
    for comparison to sequencing-based datasets. 
    '''
    tempdf = illumina_annot.loc[illumina_annot.index.intersection(newsites_list),target_columns]
    print('Array dimensions after intersection')
    display(tempdf.shape)
    tempdf.iloc[:,0] = tempdf.iloc[:,0].astype(int)
    tempdf.iloc[:,1] = tempdf.iloc[:,1].astype(int)
    tempdf.columns = ['chr', 'start']
    tempdf['chr'] = ('chr' + tempdf['chr'].astype(str))
    tempdf['pos'] = tempdf['chr'] + '.' + tempdf['start'].astype(str)
    tempdf['name'] = tempdf.index
    tempdf.index = tempdf['pos']
    tempdf.pop('pos')
    return tempdf

def positionToBED(position_df, bed_file):
    '''
    Converts position df to .bed format for other comparisons. 
    '''
#     assert 
    tempdf = position_df.copy()
    tempdf['end'] = tempdf['start'] + 1
    tempdf = tempdf.loc[:,['chr', 'start', 'end', 'name']]
    tempdf.to_csv(bed_file, sep = '\t', header = None, index = None)
    return None



    
isledf = islemap(['MouseSitesConverted', 'MouseSitesUnconverted',
                  'MouseSitesBackground'])
genedf = genemap(['MouseSitesConverted', 'MouseSitesUnconverted',
                  'MouseSitesBackground'])

bars2([genedf['MouseSitesConverted'], genedf['MouseSitesUnconverted']],
      genedf['MouseSitesBackground'],
     'mouseclock_gene_errorbars.eps')
bars2([isledf['MouseSitesConverted'], isledf['MouseSitesUnconverted']],
      isledf['MouseSitesBackground'],
     'mouseclock_isle_errorbars.eps')


