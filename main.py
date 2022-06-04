import os.path
import time
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from MCJITransformer import MCJITransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from evaluetor.model import Model
import multiprocessing as mp
from simpledb import SimpleDB

num_workers = int(mp.cpu_count() * 0.8)

n_splits=10
metrics = [accuracy_score]
internal_gss = StratifiedKFold(n_splits=2)
gss = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=10, random_state=42)

train_size = 0.8
folder = "/home/weslleylc/PycharmProjects/UCI/processed_data/classification/"
features_range_percentil = np.linspace(0.5, 0.9, 5)
n = 5
max_iter = [1000]
version = 1
func_type = 'maximize'
func_type = 'minimize'
solver = ["saga"]

def get_pipeline(cls, selector=None, memory="./cache"):
    if selector is not None:
        return Pipeline([
            ('scaler', preprocessing.MinMaxScaler()),
            ('selector', selector),
            ('classifier', cls)],
            memory=memory)
    else:
        return Pipeline([
            ('scaler', preprocessing.MinMaxScaler()),
            ('classifier', cls)],
            memory=memory)


def get_models(features_range, C=[0.01, 0.1, 1, 10, 100]):

    ####################################LogisticRegression##############################################

    params = {
        'classifier__C': C,
        'classifier__max_iter': max_iter,
        # 'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': solver,
    }

    logr = get_pipeline(LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))

    model_logr = Model(cls=logr, metrics=metrics, params=params, n_jobs=num_workers,
                       gss=internal_gss, name='log')

    ####################################LogisticRegression##############################################
    selector = MCJITransformer(k=10, func_type=func_type)

    params = {
        'selector__k': features_range,
        'classifier__C': C,
        'classifier__max_iter': max_iter,
        # 'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': solver,
    }

    logr = get_pipeline(LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
                        selector=selector)

    model_logr_mcji = Model(cls=logr, metrics=metrics, params=params, n_jobs=num_workers,
                            gss=internal_gss, name='mcji')

    ####################################LogisticRegression##############################################
    selector = SelectKBest(mutual_info_classif, k=10)

    params = {
        'selector__k': features_range,
        'classifier__C': C,
        'classifier__max_iter': max_iter,
        # 'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': solver,
    }

    logr = get_pipeline(LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
                        selector=selector)

    model_logr_mic = Model(cls=logr, metrics=metrics, params=params, n_jobs=num_workers,
                           gss=internal_gss, name="mic")

    return [model_logr, model_logr_mcji, model_logr_mic]


def train_test_eq_split(X, y, n_per_class, random_state=None):
    if random_state:
        np.random.seed(random_state)
    sampled = X.groupby(y, sort=False).apply(
        lambda frame: frame.sample(n_per_class))
    mask = sampled.index.get_level_values(1)

    X_train = X.drop(mask).values
    X_test = X.loc[mask].values
    y_train = y.drop(mask).values
    y_test = y.loc[mask].values

    return X_train, X_test, y_train, y_test

rows = []
it = 0
description = pd.read_csv('data_description.csv')
description['Class distribution'] = description['Class distribution'].apply(lambda x: [int(y) for y in x.split(";")])
description['Normalized STD'] = description['Class distribution'].apply(lambda x: np.std(x)/np.sum(x))
# description = description.loc[description['Size'] < 5000]
# description = description.loc[description['Normalized STD'] < 0.3]
# description = description.loc[description['Attributes'] < 50]
# description = description.loc[description['Size'] > 1000]
data_names = description['Dataset'].unique()

datasets = [f for f in listdir(folder) if isfile(join(folder, f)) and f.replace(".data", "") in data_names]
columns = {'metric': 'text', 'value': 'text', 'classifier': 'text', 'n_features': 'text',
           'elapsed_time': 'text', 'it': 'text', 'fold': 'text', 'n1': 'text', 'n2': 'text',
           'dataset': 'text', 'filename': 'text'}


db = SimpleDB(columns=columns, audit_db_name="10x10")
db.init_db()


def process_file(filename, db):
    audit = db.get_file_process_status(filename)
    if (audit is None or (audit is not None and audit[3] != 'SUCCESS')):
        try:
            db.start_file_process(filename)
            print (f"Now processing text for {filename}")
            db.finalize_file_process(filename)
        except:
            db.finalize_file_process(filename, 'ERROR')


for dataset_name in datasets:
    df = pd.read_csv(os.path.join(folder, dataset_name), header=None, sep=" ").values

    dataset_name = dataset_name.replace(".data", "")
    print(dataset_name)
    X, Y = df[:, :-1], df[:, -1]
    n_features = X.shape[1]

    classifiers = get_models(features_range=[int(x * n_features) for x in features_range_percentil])
    for index, model in enumerate(classifiers):
        for it, (train_index, test_index) in enumerate(gss.split(X, Y)):
            filename = "{}-{}-{}".format(dataset_name, model.name, it)
            audit = db.get_file_process_status(filename)

            if audit is None or (audit is not None and audit[-1] != 'SUCCESS'):
                try:
                    print(f"Now processing text for {filename}")
                    X_train, X_test = X[train_index], X[test_index]
                    Y_train, Y_test = Y[train_index], Y[test_index]
                    start = time.time()
                    result, predict, search = model.eval(X_train, X_test, Y_train, Y_test)
                    end = time.time()
                    if "selector" not in search.best_estimator_.get_params():
                        result['n_features'] = n_features
                    else:
                        result['n_features'] = search.best_estimator_.get_params()['selector'].get_params()['k']
                    result['elapsed_time'] = end - start
                    result['it'] = it
                    result['fold'] = (it + 1) % n_splits
                    result['n1'] = len(Y_train)
                    result['n2'] = len(Y_test)
                    result['dataset'] = dataset_name
                    result['filename'] = filename
                    result.columns = ['metric', 'value', 'classifier', 'n_features', 'elapsed_time',
                                      'it', 'fold', 'n1', 'n2', 'dataset',  'filename']
                    result = result.loc[result['metric'] == 'accuracy_score_test']
                    db.start_file_process(filename, result.to_dict('rows')[0])
                    db.finalize_file_process(filename)
                except:
                    db.finalize_file_process(filename, 'ERROR')