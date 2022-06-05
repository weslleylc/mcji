import numpy as np
import pandas as pd
import sys
from scipy.stats import rankdata, ks_2samp, mannwhitneyu, ttest_ind, ttest_rel
from scipy.stats import t
from simpledb import SimpleDB


from corrected_dependent_ttest import corrected_dependent_ttest
from combined_ftest_5x2cv import combined_ftest_5x2cv
import scipy.stats






def calculate_df(df, description, f=combined_ftest_5x2cv):
    df['value'] = df['value'].astype(float)
    df['it'] = df['it'].astype(int)
    df['n_features'] = df['n_features'].astype(int)
    df['elapsed_time'] = df['elapsed_time'].astype(float)
    df['enddatetime'] = pd.to_datetime(df['enddatetime'])
    df['startdatetime'] = pd.to_datetime(df['startdatetime'])

    for dataset_name in df['dataset'].unique():
        current_df = df.loc[df['dataset'] == dataset_name]
        n1 = int(current_df['n1'].iloc[0])
        n2 = int(current_df['n2'].iloc[0])
        experiments = current_df.groupby(['classifier'])['value'].apply(list)
        winner = experiments.index[experiments.apply(np.mean).argmax()]
        ties = False
        for classifier in current_df['classifier'].unique():
            if classifier != winner:
                # statistic, pvalue = f(experiments[winner], experiments[classifier])
                # pvalue = modified_t_student(experiments[winner], experiments[classifier], n1, n2)
                _, _, _, pvalue = corrected_dependent_ttest(experiments[winner], experiments[classifier], n1, n2, alpha=0.05)
                # current_df.loc[current_df['classifier'] == classifier, 'statistic'] = statistic
                current_df.loc[current_df['classifier'] == classifier, 'pvalue'] = pvalue
                if pvalue >= 0.05:
                    ties = True
                    current_df.loc[current_df['classifier'] == classifier, 'evaluation'] = 0
                else:
                    current_df.loc[current_df['classifier'] == classifier, 'evaluation'] = -1
        if ties:
            current_df.loc[current_df['classifier'] == winner, 'evaluation'] = 0
        else:
            current_df.loc[current_df['classifier'] == winner, 'evaluation'] = 1
        # df.loc[current_df.index, ['statistic', 'pvalue', 'evaluation']] = current_df[
        #     ['statistic', 'pvalue', 'evaluation']]
        df.loc[current_df.index, ['pvalue', 'evaluation']] = current_df[['pvalue', 'evaluation']]

    df = df.groupby(["dataset", "classifier"])[['n_features', 'value', 'evaluation']].mean().reset_index()
    df = df.merge(description, how="left", left_on="dataset", right_on="Dataset")
    return df

description = pd.read_csv('data_description.csv')
description['Class distribution'] = description['Class distribution'].apply(lambda x: [int(y) for y in x.split(";")])
description['Normalized STD'] = description['Class distribution'].apply(lambda x: np.std(x)/np.sum(x))

#
# columns = {'metric': 'text', 'value': 'text', 'classifier': 'text', 'n_features': 'text',
#            'elapsed_time': 'text', 'it': 'text', 'dataset': 'text', 'filename': 'text'}
# db = SimpleDB(columns=columns, audit_db_name="testdb")
columns = {'metric': 'text', 'value': 'text', 'classifier': 'text', 'n_features': 'text',
           'elapsed_time': 'text', 'it': 'text', 'fold': 'text', 'n1': 'text', 'n2': 'text',
           'dataset': 'text', 'filename': 'text'}
n_repeats = 10
n_splits = 10
db = SimpleDB(columns=columns, audit_db_name="{}x{}".format(n_repeats, n_splits))
db_df = db.get_data()
# db_df = db_df.loc[db_df['classifier'] != 'log']
finished = db_df.groupby(['dataset'])['it'].count()
finished = finished.loc[finished == n_repeats * n_splits * db_df['classifier'].nunique()].index
db_df = db_df.loc[db_df['dataset'].isin(finished)]
# db_df = db_df.loc[db_df['dataset'] == 'thyroid-disease-dis']

df = calculate_df(db_df, description)
results = df.replace({'evaluation': {0: 'ties', 1: 'won', -1: 'lost'}})

print(df['dataset'].nunique())


results = results.groupby(['classifier', 'evaluation'])['evaluation'].count()
results.name = 'value'
results = results.reset_index()
results = results.pivot_table(values=['value'], columns=['evaluation'], index=['classifier'])
print(results.fillna(0))
# results.columns = ['lost', 'ties', 'won']
# print(results.fillna(0))