from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


def pca_deal(data, n):
    '''
    :param data:需要处理的数据
    :param n:保留几个主成分
    :return: 经过PCA降维后的数据
    '''
    pca = PCA(n_components=n)
    redata = pca.fit_transform(data)

    return redata


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
        result = pd.value_counts(i) / 50  # 统计i中元素的个数
        group_dict['zero'] = result[0]
        group_dict['one'] = result[1]
        group_dict['two'] = result[2]
        group_dict['three'] = result[3]
        final_list.append(group_dict)
        group_dict = {}
    return final_list


# y = pd.read_csv('.\data\submission_knn.csv')
# y = np.array(y)
# y = y.tolist()
# # y_predcit = [int(j) for i in y for j in i]
# y_predcit=y

# y_predcit_50 = [y_predict[i:i + 50] for i in range(0, len(y_predict), 50)]
# A=[]
# for i in y_predcit_50:
#     a=pd.value_counts(i)/50
#     A.append(a)
# A=pd.DataFrame(A)
# A.to_csv('submission_A.csv',index=True)
# list = turn_y_to_rate(list, 50)
#
# for i in list:
#     print(i)

