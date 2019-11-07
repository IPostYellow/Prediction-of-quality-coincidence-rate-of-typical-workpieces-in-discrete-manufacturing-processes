import numpy as np
import pandas as pd
import catboost as cbt
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

inputfile = '../training_data.csv'
inputfile_test = '../testing_data.csv'
# data_train = pd.read_csv(inputfile, header=None)
data_train = pd.read_csv('../first_round_training_data.csv', header=None)
data_test = pd.read_csv(inputfile_test, header=None)
X = data_train.iloc[1:, 0:10]
X=np.array(X)
y = data_train.iloc[1:, 13:16]
y=np.array(y)
print(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=19931028)  # 划分训练集和验证集

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,RandomForestRegressor

# param_nestimators = {"n_estimators": range(1, 201, 10)}
# param_maxfeatures = {"max_features": range(1, 11, 1)}
# param_test=[param_nestimators,param_maxfeatures]
# gsearch = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_test)
# gsearch.fit(X,y)
# print(gsearch.best_params_)
# print(gsearch.best_score_)

rf_clf = RandomForestRegressor(n_estimators=151)  # 随机森林配置
rf_clf.fit(X_train, y_train)  # 随机森林训练
y_predict = rf_clf.predict(X_train)  # 随机森林预测训练集预测值
y_val_predict = rf_clf.predict(X_val)  # 随机森林预测验证集预测值

y_predict = np.array(y_predict)
y_predict_int = y_predict.astype(float)

y_val_predict = np.array(y_val_predict)
y_val_predict_int = y_val_predict.astype(float)

y_int = np.array(y_train)
y_int = y_int.astype(float)
mae = (np.sum(np.absolute(y_predict_int - y_int))) / len(y_int)
temp = y_predict_int - y_int
acc = np.sum(temp == 0) / len(y_int)
print('训练集的mae:' + str(mae))
print('训练集准确度:' + str(acc))

y_val_int = np.array(y_val)
y_val_int = y_val_int.astype(float)
mae = (np.sum(np.absolute(y_val_predict_int - y_val_int))) / len(y_val_int)
temp = y_val_predict_int - y_val_int
acc = np.sum(temp == 0) / len(y_val_int)
print('验证集的mae:' + str(mae))
print('验证集准确度:' + str(acc))