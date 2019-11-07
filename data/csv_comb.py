import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

inputfile = '../first_round_training_data.csv'  # 训练样本
inputfile_test = '../testing_data.csv'  # 测试样本

data = pd.read_csv(inputfile, header=None)
data_test = pd.read_csv(inputfile_test, header=None)

sc = StandardScaler()
sc_test = StandardScaler()

X = data.iloc[1:, 0:20]
X = sc.fit_transform(X)  # 将训练集数据标准化

X_test_raw = data_test.iloc[1:, 0:10]  # 读取测试数据的前十个特征
X_test_raw = sc_test.fit_transform(X_test_raw)  # 将测试集数据标准化

X_train_all = X[:, 0:10]  # 训练集数据
y_train_all = X[:, 19]  # 训练集标签

# 属性1参数
# X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.5, random_state=666)
# rf_clf = RandomForestRegressor(random_state=256, n_estimators=50)
# 属性1结束
# 属性2参数
# param_nestimators = {"n_estimators": range(1, 201, 10)}
# param_maxfeatures = {"max_features": range(1, 11, 1)}
# param_test = [param_nestimators, param_maxfeatures]
# gsearch = GridSearchCV(estimator=RandomForestRegressor(n_estimators=31), param_grid=param_maxfeatures, cv=10)
# gsearch.fit(X_train_all, y_train_all)
# print(gsearch.best_params_)
# print(gsearch.best_score_)

# X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.5, random_state=666)
# rf_clf = RandomForestRegressor(random_state=256, n_estimators=50)
# 属性2结束
# 属性3参数
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.5, random_state=666)
rf_clf = RandomForestRegressor(random_state=256, n_estimators=50)
# 属性3结束
# 线性回归
from sklearn.linear_model import LinearRegression

# X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.5, random_state=666)
# lin = LinearRegression()
# lin.fit(X_train, y_train)
# lin_predict = lin.predict(X_val)
# lin_predict_train = lin.predict(X_train)
rf_clf.fit(X_train, y_train)
rf_predict = rf_clf.predict(X_val)
rf_predict_train = rf_clf.predict(X_train)

print("训练集均方差：" + str(mean_squared_error(y_train, rf_predict_train)))
print("验证集均方差：" + str(mean_squared_error(y_val, rf_predict)))

rf_predict_test = rf_clf.predict(X_test_raw)
rf_predict_test_raw = pd.DataFrame(rf_predict_test)
rf_predict_test_raw.to_csv('属性10.csv', index=False, header=False)
