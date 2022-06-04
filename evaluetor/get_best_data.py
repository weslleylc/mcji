import numpy as np
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv("./data/chagas_processado.csv")

data = data.loc[data["situacao_sinal"].isin(["LEGIVEL", "FIBRILAÇÃO VENTRICULAR OU TAQUICARDIA VENTRICULAR"])]

# ruins = [262, 26, 36, 76, 81, 168, 65, 85, 286,303, 306, 290, 265, 262, 24, 80, 158, 119, 355, 369, 324, 328, 78]
# data = data.loc[np.logical_not(data['filename'].isin(ruins))]

data = data.drop_duplicates(subset='nome', keep='last')
data = data.fillna(0)

data = data[['cancer', 'has', 'dm2', 'cardiopatia_outra', 'marcapasso', 'sincope',
               'fibrilacao/flutter_atrial', 'i_r_cronica', 'dlp', 'coronariopatia',
               'embolia_pulmonar', 'ins_cardiaca', 'avc', 'dvp', 'tsh', 'tabagismo',
               'alcoolismo', 'sedentarismo', "idade", "obito", "filename"]]

columns = ['cancer', 'has', 'dm2', 'cardiopatia_outra', 'marcapasso', 'sincope',
               'fibrilacao/flutter_atrial', 'i_r_cronica', 'dlp', 'coronariopatia',
               'embolia_pulmonar', 'ins_cardiaca', 'avc', 'dvp', 'tsh', 'tabagismo',
               'alcoolismo', 'sedentarismo', "idade"]

# data = data[['sincope', "obito", "filename"]]

columns = ['sincope']

data['label'] = data['obito']
data = data.drop(['obito'], axis=1)



for column in columns:
    data[column] = data[column].apply(lambda x: str(x).replace(",", ".")).astype(float)
# data = data.dropna(how='all')
data = data.fillna(0.0).reset_index().drop(["index"], axis=1)
positive_data = data.loc[data['label'] == 1]
negative_data = data.loc[data['label'] == 0]
#############################################FUNCTION TO EXTRACT FEATURES##########################################################

Y = data["label"]
clinical_data = data[columns]

mm_scaler = preprocessing.MinMaxScaler()
X = mm_scaler.fit_transform(clinical_data)

negative_data_index = Y.loc[Y == 0].index.values
negative_data = X[negative_data_index]
positive_data_index = Y.loc[Y == 1].index.values
positive_data = X[positive_data_index]



list_best_idx = negative_data_index
count = 0
for x in positive_data:
    values = abs(negative_data - x.reshape(1, -1))
    values = values.sum(axis=1)
    best_idx = np.argmin(values)
    negative_data = np.delete(negative_data, best_idx, 0)
    negative_data_index = np.delete(negative_data_index, best_idx, 0)

list_best_idx = list(set(list_best_idx) - set(negative_data_index))
list_best_idx.extend(positive_data_index)
best_signals = np.sort(data.iloc[list_best_idx]["filename"].values.reshape(-1))



