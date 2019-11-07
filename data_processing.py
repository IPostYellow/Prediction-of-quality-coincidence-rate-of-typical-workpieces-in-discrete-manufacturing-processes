import numpy as np
import pandas as pd
import catboost as cbt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def turn_y_to_rate(data, m):
    '''
    :param data:数据
    :param m: 间隔
    :return: 统计后的值
    '''
    data50 = [data[i:i + 50] for i in range(0, len(data), m)]
    final_list = []
    group_dict = {}
    for i in data50:
        zero = 0
        one = 0
        two = 0
        three = 0
        for j in i:
            if j == '0':
                zero += 1
            elif j == '1':
                one += 1
            elif j == '2':
                two += 1
            elif j == '3':
                three += 1
        group_dict['zero'] = zero / 50.0
        group_dict['one'] = one / 50.0
        group_dict['two'] = two / 50.0
        group_dict['three'] = three / 50.0
        final_list.append(group_dict)
        group_dict = {}
    return final_list


inputfile = 'training_data.csv'
inputfile_test = './data/总属性1.csv'
# data_train = pd.read_csv(inputfile, header=None)
data_train = pd.read_csv('first_round_training_data.csv', header=None)
data_test = pd.read_csv(inputfile_test, header=None)
X = data_train.iloc[1:, 5:11]
# sc = StandardScaler()
# X = sc.fit_transform(X)
# sc_test = StandardScaler()

X_test = data_test.iloc[0:, 5:11]
# X_test = sc_test.fit_transform(X_test)
data_train = pd.read_csv(inputfile, header=None)
y = data_train.iloc[1:, 10]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=19931028)  # 划分训练集和验证集

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

# ada_clf = AdaBoostClassifier(random_state=666)#集成500个分类器
# ada_clf.fit(X,y)

# y_predict = ada_clf.predict(X_test)
cbt_model = cbt.CatBoostClassifier(iterations=1200,learning_rate=0.05,verbose=300,
early_stopping_rounds=1000,task_type='GPU',
loss_function='MultiClass')
cbt_model.fit(X, y ,eval_set=(X,y)) #0.292

# svm=LinearSVC()
# svm.fit(X,y)
# knn_clf = KNeighborsClassifier()
# knn_clf.fit(X, y)
# y_predict = knn_clf.predict(X_test)
# y_predict = svm.predict(X_test)

# param_nestimators = {"n_estimators": range(1, 201, 10)}
# param_maxfeatures = {"max_features": range(1, 11, 1)}
# param_test=[param_nestimators,param_maxfeatures]
# gsearch = GridSearchCV(estimator=RandomForestClassifier(n_estimators=151), param_grid=param_maxfeatures, cv=10)
# gsearch.fit(X,y)
# print(gsearch.best_params_)
# print(gsearch.best_score_)

# rf_clf = RandomForestClassifier(n_estimators=151)  # 随机森林配置
# rf_clf.fit(X_train, y_train)  # 随机森林训练
# y_predict = rf_clf.predict(X_train)  # 随机森林预测训练集预测值
# y_val_predict = rf_clf.predict(X_val)  # 随机森林预测验证集预测值

# print(rf_clf.feature_importances_)
# print('-' * 80)
y_predict = cbt_model.predict(X)
# y_predict=[i[0] for i in y_predict]
# print(y_predict)

# gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)
# X_train=np.array(X_train)
# X_val=np.array(X_val)
# gbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='l1', early_stopping_rounds=5)
#
# y_predict = gbm.predict(X_train, num_iteration=gbm.best_iteration_)
#
# y_val_predict = gbm.predict(X_val, num_iteration=gbm.best_iteration_)

y_predict = np.array(y_predict)
y_predict_int = y_predict.astype(int)

# y_val_predict = np.array(y_val_predict)
# y_val_predict_int = y_val_predict.astype(int)

# y_int = np.array(y_train)
# y_int = y_int.astype(int)
# mae = (np.sum(np.absolute(y_predict_int - y_int))) / len(y_int)
# temp = y_predict_int - y_int
# acc = np.sum(temp == 0) / len(y_int)
# print('训练集的mae:' + str(mae))
# print('训练集准确度:' + str(acc))
#
# y_val_int = np.array(y_val)
# y_val_int = y_val_int.astype(int)
# mae = (np.sum(np.absolute(y_val_predict_int - y_val_int))) / len(y_val_int)
# temp = y_val_predict_int - y_val_int
# acc = np.sum(temp == 0) / len(y_val_int)
# print('验证集的mae:' + str(mae))
# print('验证集准确度:' + str(acc))

# y_test_predict = rf_clf.predict(X_test)
y_predcit_50 = [y_test_predict[i:i + 50] for i in range(0, len(y_test_predict), 50)]

# dict = turn_y_to_rate(y_test_predict, 50)
# zero = []
# one = []
# two = []
# three = []
# for i in dict:
#     zero.append(i['zero'])
#     one.append(i['one'])
#     two.append(i['two'])
#     three.append(i['three'])
#
# B = pd.DataFrame()
# B.insert(0, 'Excellent ratio', zero)
# B.insert(1, 'Good ratio', one)
# B.insert(2, 'Pass ratio', two)
# B.insert(3, 'Fail ratio', three)
# B.to_csv('Submission_rf5.csv', header=True, index_label='Group', index=True)
A = []
for i in y_predcit_50:
    a = pd.value_counts(i) / 50
    A.append(a)
A = pd.DataFrame(A)
A = A.where(A.notnull(), 0)
B = pd.DataFrame()
B.insert(0, 'Excellent ratio', A['0'])
B.insert(1, 'Good ratio', A['1'])
B.insert(2, 'Pass ratio', A['2'])
B.insert(3, 'Fail ratio', A['3'])
B.to_csv('Submission_rf5.csv', header=True, index_label='Group', index=True)
