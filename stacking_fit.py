from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingClassifier
from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
import catboost as cbt
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
# Initializing models
inputfile = 'training_data.csv'
inputfile_test = 'testing_data.csv'

data_train = pd.read_csv('first_round_training_data.csv', header=None)
data_test = pd.read_csv(inputfile_test, header=None)
X = data_train.iloc[1:, 5:10]
# print(X)
# print('-'*80)
X = np.array(X)
data_train = pd.read_csv(inputfile, header=None)
y = data_train.iloc[1:, 10]
y = np.array(y)
X_test = data_test.iloc[1:, 5:10]
# print(X_test)
X_test = np.array(X_test)

# clf1 = cbt.CatBoostClassifier(iterations=1000,task_type='GPU',loss_function='MultiClass')
# clf2 = lgb.LGBMClassifier(num_leaves=31,bagging_fraction=0.5,feature_fraction=0.8,max_depth=10,n_estimators=200)
clf2=RandomForestClassifier()
clf3 = xgb.XGBClassifier(n_estimators=500)
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf2, clf3],
                          meta_classifier=lr)

sclf.fit(X, y)

y_predict = sclf.predict_proba(X_test)
# print(y_predict[0])
y_predcit_50 = [y_predict[i:i + 50] for i in range(0, y_predict.shape[0], 50)]  # 120组，每组50

A = []
# 把每组的列加起来
for i in y_predcit_50:
    a = np.sum(i, axis=0) / 50
    A.append(a)
A = pd.DataFrame(A)
A.columns = ['Excellent ratio', 'Good ratio', 'Pass ratio', 'Fail ratio']
A.to_csv('submission_stacking.csv', index=True, index_label='Group')

# B = pd.DataFrame()
# B.insert(0, 'Excellent ratio', A['0'])
# B.insert(1, 'Good ratio', A['1'])
# B.insert(2, 'Pass ratio', A['2'])
# B.insert(3, 'Fail ratio', A['3'])
# B.to_csv('Submission_rf5.csv', header=True, index_label='Group', index=True)
