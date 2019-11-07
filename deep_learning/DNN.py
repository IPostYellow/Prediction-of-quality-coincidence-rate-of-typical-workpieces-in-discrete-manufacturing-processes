import pandas as pd
from keras import Sequential, regularizers
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers import SimpleRNN, LSTM, MaxPooling2D, Conv2D, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
from data_process import pca_deal

inputfile = '../first_round_training_data.csv'
inputfile_test = '../testing_data.csv'
data_train = pd.read_csv(inputfile, header=None)
data_test = pd.read_csv(inputfile_test, header=None)
X = data_train.iloc[1:, 0:10]
y = data_train.iloc[1:, 13:16]
X=np.array(X)
y=np.array(y)
print(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=19931028)  # 划分训练集和验证集

checkpoint = ModelCheckpoint('../check_point/best_weights.h5', monitor='val_acc', verbose=1,
                             save_best_only=True,
                             mode='max')

ealystop=EarlyStopping(monitor='val_loss',patience=10,verbose=0,mode='auto')
callbacks_list = [checkpoint,ealystop]

model = Sequential()
model.add(Dense(10,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(40,activation='relu'))
model.add(Dense(3))
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['acc'])

model.fit(X, y, validation_split=0.1, epochs=30, batch_size=128, verbose=1, callbacks=callbacks_list,
          shuffle=True)

model = load_model('../check_point/best_weights.h5')

y_predict = model.predict(X_train)
y_val_predict = model.predict(X_val)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_predict,y_train))
print(mean_squared_error(y_val_predict,y_val))

# y_test_predict = model.predict(X_test)
# y_test_predict = np.argmax(y_test_predict, axis=1)
#
# y_predcit_50 = [y_test_predict[i:i + 50] for i in range(0, len(y_test_predict), 50)]
#
# A = []
# for i in y_predcit_50:
#     a = pd.value_counts(i) / 50
#     A.append(a)
# A = pd.DataFrame(A)
# A = A.where(A.notnull(), 0)
# B = pd.DataFrame()
# B.insert(0, 'Excellent ratio', A['0'])
# B.insert(1, 'Good ratio', A['1'])
# B.insert(2, 'Pass ratio', A['2'])
# B.insert(3, 'Fail ratio', A['3'])
# B.to_csv('Submission_dnn.csv', header=True, index_label='Group', index=True)
