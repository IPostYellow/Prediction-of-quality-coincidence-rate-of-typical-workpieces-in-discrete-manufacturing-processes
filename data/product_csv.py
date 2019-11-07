import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

inputfile = '../testing_data.csv'  # 训练样本
data = pd.read_csv(inputfile, header=None)
sc_test = StandardScaler()
X_test_raw = data.iloc[1:, 0:10]  # 读取测试数据的前十个特征
data_0 = sc_test.fit_transform(X_test_raw)
data_0=pd.DataFrame(data_0)
data_1 = pd.read_csv('./属性1.csv', header=None, sep=',')
data_2 = pd.read_csv('./属性2.csv', header=None, sep=',')
data_3 = pd.read_csv('./属性3.csv', header=None, sep=',')
data_4 = pd.read_csv('./属性4.csv', header=None, sep=',')
data_5 = pd.read_csv('./属性5.csv', header=None, sep=',')
data_6 = pd.read_csv('./属性6.csv', header=None, sep=',')
data_7 = pd.read_csv('./属性7.csv', header=None, sep=',')
data_8 = pd.read_csv('./属性8.csv', header=None, sep=',')
data_9 = pd.read_csv('./属性9.csv', header=None, sep=',')
data_10 = pd.read_csv('./属性10.csv', header=None, sep=',')
data_0.insert(loc=10,column=None,value=data_1)
data_0.insert(loc=11,column=None,value=data_2)
data_0.insert(loc=12,column=None,value=data_3)
data_0.insert(loc=13,column=None,value=data_4)
data_0.insert(loc=14,column=None,value=data_5)
data_0.insert(loc=15,column=None,value=data_6)
data_0.insert(loc=16,column=None,value=data_7)
data_0.insert(loc=17,column=None,value=data_8)
data_0.insert(loc=18,column=None,value=data_9)
data_0.insert(loc=19,column=None,value=data_10)


# data_1.insert(loc=1,column=None,value=data_2)
# data_1.insert(loc=2,column=None,value=data_3)
# data_1.insert(loc=3,column=None,value=data_4)
# data_1.insert(loc=4,column=None,value=data_5)
# data_1.insert(loc=5,column=None,value=data_6)
# data_1.insert(loc=6,column=None,value=data_7)
# data_1.insert(loc=7,column=None,value=data_8)
# data_1.insert(loc=8,column=None,value=data_9)
# data_1.insert(loc=9,column=None,value=data_10)

data_0.to_csv('总属性1.csv',header=False,index=False)