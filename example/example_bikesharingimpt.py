import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from src.gcForest import gcForest
import matplotlib.pyplot as plt
from src.utils import set_fixed_random_seed


def print_coef(coef, column_names):
    idx = np.argsort(-abs(coef))
    for i in range(len(idx)):
        id = idx[i]
        if coef[id]==0:
            break
        print(i+1,column_names[id],coef[id])

raw_data = pd.read_csv("../Bike-Sharing-Dataset/day.csv")
print(raw_data)
df = pd.DataFrame()

df['yr']=raw_data['yr']
df['workingday']=raw_data['workingday']

tmp_dict = {1:1,2:0,3:0,4:0}
df['clearday']=raw_data['weathersit'].map(tmp_dict)

df['temp']=raw_data['temp']
df['hum']=raw_data['hum']
df['windspeed']=raw_data['windspeed']
print(df)
data = df.to_numpy()
label = raw_data['cnt'].to_numpy()
set_fixed_random_seed(42)

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33)
print(X_train)
print(X_test)
reg = RandomForestRegressor()
reg.fit(X_train,y_train)
coef = reg.feature_importances_
print_coef(coef,df.columns.values)
y_pred = reg.predict(X_test)
MSE = mean_squared_error(y_test,y_pred,squared=False)
MAE = mean_absolute_error(y_test,y_pred)
R2 = r2_score(y_test,y_pred)
print('RMSE={},MAE={},R2={}'.format(MSE,MAE,R2))
print(type(list(df.columns.values)))
column_names = list(df.columns.values)

# print(X_train)
# idx = X_train[:,3]*41<15
# print(idx)
# X_test=X_train[idx,:]
# print(len(X_test))
# y_test=y_train[idx]
X_test = X_train
y_test = y_train

n_estimators = 50
n_forests = 4
max_layer = 31
n_classes = 1
set_fixed_random_seed(42)
gc = gcForest(n_estimators, n_forests, n_classes, max_layer=max_layer, max_depth=8, n_fold=3, compute_FI=True)
best_test_prob, best_layer_test_contribution, test_err, best_layer_index, val_FI_list, test_FI_list = gc.train_and_predict(
    X_train, y_train,
    X_test, y_test)
print('best layer:', best_layer_index)

print(val_FI_list[best_layer_index])
tmp_val_FI = val_FI_list[best_layer_index]
print(test_FI_list[best_layer_index])
tmp_test_FI = test_FI_list[best_layer_index]
tmp_val_FI[tmp_val_FI<0]=0
tmp_test_FI[tmp_test_FI<0]=0
tmp_val_FI = tmp_val_FI/sum(tmp_val_FI)
tmp_test_FI = tmp_test_FI/sum(tmp_test_FI)

column_names = ['Year','isWorkingDay','isSunnyDay','Temperature','Humidity','WindSpeed']

n_features=6
x = list(np.arange(1, n_features+1))
total_width = 0.7

fig, ax = plt.subplots(figsize=(7, 2.5))
plt.bar(x, tmp_val_FI, width=total_width, fill=False, hatch = '')
# plt.bar(x, tmp_val_FI, width=width, label='Global MDI', fill=False, hatch = '')

# for i in range(len(x)):
#     x[i] = x[i] + width
# # plt.bar(x, MDAtime_list, width=width, label='MDA(DF)', color=cmap(2))
# plt.bar(x, tmp_test_FI, width=width, label='Local MDI (rainy days)', fill=False, hatch = '////')

plt.xticks(np.arange(1, n_features+1),column_names)
ax.set_xlabel('Feature')
ax.set_ylabel('Feature importance')
# plt.legend()
plt.tight_layout()
plt.savefig('../record/bikesharingimpt.pdf', dpi=150)
plt.show()