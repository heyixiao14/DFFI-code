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

# replace the following with YOUR data and label
from src.syntheticdata import generate_3class_dataset
n_samples = 200
irrelevant_dimension = 100
set_fixed_random_seed(42)
data, label = generate_3class_dataset(n_samples, irrelevant_dim=irrelevant_dimension)

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33)

max_layer = 31 # total layers of the cascade
n_estimators = 50  # number of trees in each forest
n_forests = 4 # number of forests in each cascade layer
max_depth = 8 # maximum tree depth

# n_classes should be set, if regression then set n_classes = 1
n_classes = 3

# control randomness:
set_fixed_random_seed(42)

gc = gcForest(n_estimators, n_forests, n_classes, max_layer=max_layer, max_depth=max_depth, n_fold=3, compute_FI=True)
best_test_prob, best_layer_test_contribution, test_err, best_layer_index, val_FI_list, test_FI_list = gc.train_and_predict(
    X_train, y_train,
    X_test, y_test)

'''feature contribution'''
print(best_layer_test_contribution)

'''feature importance'''
# print(val_FI_list[best_layer_index])
tmp_val_FI = val_FI_list[best_layer_index]
tmp_val_FI[tmp_val_FI<0]=0
tmp_val_FI = tmp_val_FI/sum(tmp_val_FI)

print(tmp_val_FI,tmp_val_FI.shape)

print('best layer:', best_layer_index)
print('test err:',test_err[best_layer_index])