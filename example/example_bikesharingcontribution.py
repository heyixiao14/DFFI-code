import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from src.gcForest import gcForest
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
df['dteday']=raw_data['dteday']
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
reg.fit(X_train[:,1:],y_train)
coef = reg.feature_importances_
print_coef(coef,df.columns.values[1:])
y_pred = reg.predict(X_test[:,1:])
MSE = mean_squared_error(y_test,y_pred,squared=False)
MAE = mean_absolute_error(y_test,y_pred)
R2 = r2_score(y_test,y_pred)
print('RMSE={},MAE={},R2={}'.format(MSE,MAE,R2))
print(type(list(df.columns.values)))

n_estimators = 50
n_forests = 4
max_layer = 31
n_classes = 1
set_fixed_random_seed(42)
gc = gcForest(n_estimators, n_forests, n_classes, max_layer=max_layer, max_depth=8, n_fold=3, compute_FI=True)
best_test_prob, best_layer_test_contribution, test_acc, best_layer_index, val_FI_list, test_FI_list = gc.train_and_predict(
    X_train[:,1:], y_train,
    X_test[:,1:], y_test)
print('best layer:', best_layer_index)
RMSE = mean_squared_error(y_test,best_test_prob,squared=False)
print('RMSE =',RMSE)
n_features = X_train.shape[1]-1

bias = np.mean(y_train)
record_file = pd.DataFrame(columns=['id','date','label','pred','sum->','bias']+list(df.columns.values)[1:])
for id in range(len(y_test)):
    tmp_list = [id,X_test[id,0],y_test[id],y_pred[id],'','']#+list(X_test[id,:])
    for iterf in range(1,n_features+1):
        if isinstance(X_test[id,iterf],int):
            tmp_list.append(X_test[id,iterf])
        else:
            tmp_list.append('{:.2f}'.format(X_test[id,iterf]))
    record_file.loc[id*2+1]=tmp_list
    tmp = np.sum(best_layer_test_contribution[id,:])+bias
    tmp_list = [id,X_test[id,0],y_test[id],y_pred[id],tmp,bias]#+list(best_layer_test_contribution[id,:,0])
    for iterf in range(n_features):
        tmp_list.append('{:.2f}'.format(best_layer_test_contribution[id,iterf,0]))
    record_file.loc[id*2+2]=tmp_list
print(record_file)
record_file.to_csv('../record/bikesharingcontribution.csv', sep=',')



