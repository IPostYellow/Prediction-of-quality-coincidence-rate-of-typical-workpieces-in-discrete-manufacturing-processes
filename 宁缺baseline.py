import numpy as np
import pandas as pd
import catboost as cbt
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import gc
import math
import time
from tqdm import tqdm
import datetime
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
pd.options.display.max_rows = None
train = pd.read_csv('first_round_training_data.csv')
test = pd.read_csv('first_round_testing_data.csv')
submit = pd.read_csv('submit_example.csv')
data = train.append(test).reset_index(drop=True)
dit = {'Excellent': 0, 'Good': 1, 'Pass': 2, 'Fail': 3}
data['label'] = data['Quality_label'].map(dit)
train['label'] = train['Quality_label'].map(dit)
feature_name = ['Parameter{0}'.format(i) for i in range(5, 11)]
tr_index = ~data['label'].isnull()
X_train = data[tr_index][feature_name].reset_index(drop=True)
y = data[tr_index]['label'].reset_index(drop=True).astype(int)
X_test = data[~tr_index][feature_name].reset_index(drop=True)
print(X_train.shape, X_test.shape)
oof = np.zeros((X_train.shape[0], 4))
prediction = np.zeros((X_test.shape[0], 4))
seeds = [19970412, 2019 * 2 + 1024, 4096, 2048, 1024]
num_model_seed = 5
for model_seed in range(num_model_seed):
    print(model_seed + 1)
oof_cat = np.zeros((X_train.shape[0], 4))
prediction_cat = np.zeros((X_test.shape[0], 4))
skf = StratifiedKFold(n_splits=5, random_state=seeds[model_seed], shuffle=True)
for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
    print(index)
train_x, test_x, train_y, test_y = X_train.iloc[train_index], X_train.iloc[test_index], y.iloc[train_index], y.iloc[
    test_index]
gc.collect()
cbt_model = cbt.CatBoostClassifier(iterations=800, learning_rate=0.01, verbose=300,
                                   early_stopping_rounds=200, loss_function='MultiClass')
cbt_model.fit(train_x, train_y, eval_set=(train_x, train_y))
oof_cat[test_index] += cbt_model.predict_proba(test_x)
prediction_cat += cbt_model.predict_proba(X_test) / 5
gc.collect()
oof += oof_cat / num_model_seed
prediction += prediction_cat / num_model_seed
print('logloss', log_loss(pd.get_dummies(y).values, oof_cat))
print('ac', accuracy_score(y, np.argmax(oof_cat, axis=1)))
print('mae', 1 / (1 + np.sum(np.absolute(np.eye(4)[y] - oof_cat)) / 480))
print('logloss', log_loss(pd.get_dummies(y).values, oof))
print('ac', accuracy_score(y, np.argmax(oof, axis=1)))
print('mae', 1 / (1 + np.sum(np.absolute(np.eye(4)[y] - oof)) / 480))
sub = test[['Group']]
prob_cols = [i for i in submit.columns if i not in ['Group']]
for i, f in enumerate(prob_cols):
    sub[f] = prediction[:, i]
for i in prob_cols:
    sub[i] = sub.groupby('Group')[i].transform('mean')
sub = sub.drop_duplicates()
sub.to_csv("submission.csv", index=False)
