import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.gcForest import gcForest
from sklearn.datasets import load_svmlight_file
from src.utils import *
import time
import argparse
import random
from sklearn.metrics import accuracy_score
from src.syntheticdata import generate_3class_dataset


def main():
    print("**********", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),"**********")

    max_layer=31
    n_estimators = 50  # number of trees in each forest
    n_forests = 4
    fixed_random_seed = 42
    print("fixed_random_seed =", fixed_random_seed)
    np.random.seed(fixed_random_seed)
    random.seed(fixed_random_seed)

    # X_train, y_train, X_test, y_test = load_letter_dataset()
    n_samples = 200
    effective_dimension = 2
    irrelevant_dimension = 100
    X_train, y_train = generate_3class_dataset(n_samples,irrelevant_dim=irrelevant_dimension)
    # X_test, y_test = generate_3class_dataset(600,irrelevant_dim=irrelevant_dimension)

    print(np.bincount(y_train)/n_samples)
    prior = np.bincount(y_train)/n_samples

    n_classes = int(np.max(y_train) + 1)

    folder = '../record/'

    # '''plot test point to show contributions'''
    # print(np.where(y_test==2))
    # display_list = [6, 13, 49, 22]
    # label_list = [1, 2, 3, 3]
    # # color_list = ['#1f77b4','#ff7f0e','#2ca02c','#2ca02c']
    # plt.figure(figsize=(5, 4))
    # plt.plot([0.5, 0.5], [0, 1], color='k', linewidth=0.5, linestyle='--')
    # plt.plot([0.5, 1], [0.5, 0.5], color='k', linewidth=0.5, linestyle='--')
    # for i in range(len(display_list)):
    #     idx = display_list[i]
    #     plt.scatter(X_test[idx, 0], X_test[idx, 1], label='point{}: class{}'.format(i+1, label_list[i]))
    #     plt.xlim(0, 1)
    #     plt.ylim(0, 1)
    # plt.legend(loc='upper left')
    # plt.xlabel('$X_1$')
    # plt.ylabel('$X_2$')
    # plt.savefig('example_data.pdf',dpi=150)
    # plt.show()
    # print(y_test[display_list])

    '''plot train data'''
    plt.figure(figsize=(4.2,3))
    plt.subplots_adjust(bottom=0.2, left=0.16, top=0.9, right=0.7)
    plt.plot([0.5, 0.5], [0, 1], color='k', linewidth=0.5, linestyle='--')
    plt.plot([0.5, 1], [0.5, 0.5], color='k', linewidth=0.5, linestyle='--')
    for i in range(n_classes):
        idx = np.where(y_train==i)[0]
        Xi = X_train[idx]
        plt.scatter(Xi[:,0],Xi[:,1],label='Class{}'.format(i+1))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend(bbox_to_anchor=(1.05,0.65),borderaxespad = 0.)
    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')
    if max_layer==1:
        plt.savefig(folder+'3classdata.pdf',dpi=150)
    # plt.show()

    print("n_estimators = {}, num_forests = {}".format(n_estimators, n_forests))

    xx = np.arange(0, 1, 0.02)
    yy = np.arange(0, 1, 0.02)
    X, Y = np.meshgrid(xx, yy)
    X_test = np.concatenate((X.reshape(-1,1), Y.reshape(-1,1),np.random.rand(2500,irrelevant_dimension)),axis=1)
    X=X_test
    Y = np.zeros((len(X),), dtype=int)
    for i in range(len(X)):
        x1=X[i,0]
        x2=X[i,1]
        if x1<0.5:
            Y[i]=0
        elif x2 >0.5:
            Y[i]=1
        else:
            Y[i] = 2
    y_test = Y

    # X_test = np.array([[0,1],[1,1],[1,0]])
    # X_test = np.concatenate((X_test, np.zeros((3,100))),axis=1)
    # y_test = np.array([0,1,2])
    # print(X_test.shape)

    gc = gcForest(n_estimators, n_forests, n_classes, max_layer=max_layer, max_depth=5, n_fold=3,compute_FI=True)
    best_test_prob, best_layer_test_contribution, test_err, best_layer_index, val_FI_list, test_FI_list = gc.train_and_predict(X_train, y_train, X_test, y_test)
    # print(val_acc)
    print('best layer:', best_layer_index)
    print(test_err[best_layer_index])

    # FI(feature importance)
    print(val_FI_list[best_layer_index])
    tmp_val_FI = val_FI_list[best_layer_index]
    print(test_FI_list[best_layer_index])
    tmp_test_FI = test_FI_list[best_layer_index]
    tmp_val_FI[tmp_val_FI < 0] = 0
    tmp_test_FI[tmp_test_FI < 0] = 0
    tmp_val_FI = tmp_val_FI / sum(tmp_val_FI)
    tmp_test_FI = tmp_test_FI / sum(tmp_test_FI)
    # print(sum(X_train[:, 1]))

    plt.figure(figsize=(4,2.9))
    plt.subplots_adjust(left=0.16, bottom=0.17,right=0.85)
    # for iterc in range(3):
    # plt.subplot(1,3,iterc+1)
    val_FI = val_FI_list
    # val_FI = test_FI_list
    cmap = plt.get_cmap('Dark2')
    # plt.figure(figsize=(4,3))
    # plt.subplots_adjust(left=0.15,bottom=0.15)
    val_FI=np.maximum(val_FI, 0)
    marker_list = ['o','^','p']
    for iterf in range(2):
        # plt.plot(val_FI[:, iterf][:best_layer_index+1], color=cmap(2-iterf),marker='.',label='$X_{}$'.format(iterf+1))
        plt.plot(val_FI[:, iterf][:best_layer_index+1], color='black',marker=marker_list[iterf],label='$X_{}$'.format(iterf+1))

    # plt.plot(np.sum(val_FI[:,2:][:best_layer_index+1],axis=1)/irrelevant_dimension,color=cmap(0),marker='.', label='Irrelevant')
    plt.plot(np.sum(val_FI[:,2:][:best_layer_index+1],axis=1)/irrelevant_dimension,color='black',marker=marker_list[2],linestyle='-', label='Irrelevant')

    # if iterc==2:
    # plt.legend(bbox_to_anchor=(1.05,0.65),borderaxespad = 0.)
    plt.legend()
    plt.xlabel('Layer')
    plt.xticks(np.arange(best_layer_index+1),np.arange(1,best_layer_index+2))
    # if iterc==0:
    plt.ylabel('MDI')
        # plt.title('Class{}'.format(iterc+1))
    plt.savefig(folder + '3classimpt.pdf', dpi=150)
    plt.show()

if __name__ == '__main__':
    main()