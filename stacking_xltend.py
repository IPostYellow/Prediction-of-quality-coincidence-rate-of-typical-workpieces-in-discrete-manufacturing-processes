from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingClassifier
import pandas as pd
import numpy as np
import warnings
import catboost as cbt
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
# Initializing models
inputfile = 'training_data.csv'
inputfile_test = 'testing_data.csv'

data_train = pd.read_csv('first_round_training_data.csv', header=None)
data_test = pd.read_csv(inputfile_test, header=None)
X = data_train.iloc[1:, 5:11]
X = np.array(X)
data_train = pd.read_csv(inputfile, header=None)
y = data_train.iloc[1:, 10]
y = np.array(y)

# clf1 = cbt.CatBoostClassifier(task_type='GPU',loss_function='MultiClass')
# clf2 = lgb.LGBMClassifier(num_leaves=31)
# clf3 = xgb.XGBClassifier()
# lr = LogisticRegression()
# sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
#                           meta_classifier=lr)

clf1 = RandomForestClassifier(random_state=22)
clf2 = xgb.XGBClassifier(n_estimators=500, random_state=22)
lr = LogisticRegression(random_state=22)
sclf = StackingClassifier(classifiers=[clf1, clf2],
                          meta_classifier=lr)

params = {'randomforestclassifier__n_estimators': range(200, 300, 10),
          'xgbclassifier__n_estimatores': range(300, 400, 10),
          }

grid = GridSearchCV(estimator=sclf,
                    param_grid=params,
                    cv=5,
                    refit=True)
grid.fit(X, y)

cv_keys = ('mean_test_score', 'std_test_score', 'params')

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_[cv_keys[0]][r],
             grid.cv_results_[cv_keys[1]][r] / 2.0,
             grid.cv_results_[cv_keys[2]][r]))

print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)
