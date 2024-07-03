from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from .feature_contribution import feature_contribution, compute_importance

class Layer:
    def __init__(self, num_forests, n_estimators, num_classes, n_features,
                 n_fold, layer_index, max_depth=100, min_samples_leaf=1, compute_FI=False, feature_indicator=None, last_layer_contribution=None):
        self.num_forests = num_forests  # number of forests
        self.n_estimators = n_estimators  # number of trees in each forest
        self.num_classes = num_classes
        self.n_features = n_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_fold = n_fold
        self.layer_index = layer_index
        self.forest_list = []
        self.compute_FI = compute_FI
        self.feature_indicator = feature_indicator
        self.last_layer_contribution = last_layer_contribution

    def train_and_predict(self, train_data, train_label, test_data,test_label=None):
        val_prob = np.zeros((self.num_forests, train_data.shape[0], self.num_classes), dtype=np.float64)
        predict_prob = np.zeros((self.num_forests, test_data.shape[0], self.num_classes), dtype=np.float64)
        val_feature_importance = np.zeros((self.num_forests, self.n_features), dtype=np.float64)
        test_feature_importance = np.zeros((self.num_forests, self.n_features), dtype=np.float64)

        n_dim = train_data.shape[1]
        contributions_list = []

        test_contribution = np.zeros((test_data.shape[0], self.n_features, self.num_classes))

        for forest_index in range(self.num_forests):
            val_prob_forest = np.zeros((train_data.shape[0], self.num_classes))
            predict_prob_forest = np.zeros([test_data.shape[0], self.num_classes])

            if self.compute_FI:
                contributions_forest = np.zeros((train_data.shape[0],self.n_features,self.num_classes))
                test_contribution_forest = np.zeros((test_data.shape[0], self.n_features, self.num_classes))

            if self.num_classes == 1:
                skf = KFold(n_splits=self.n_fold, shuffle=True)
            else:
                skf = StratifiedKFold(n_splits=self.n_fold, shuffle=True)
            kfold = 0
            # kfold_list = []
            for train_index, val_index in skf.split(train_data,train_label):
                kfold += 1
                if self.num_classes == 1: # regression
                    if forest_index % 2 == 0:
                        # print('rf regression')
                        clf = RandomForestRegressor(n_estimators=self.n_estimators,
                                                     n_jobs=-1, max_features="sqrt",
                                                     max_depth=self.max_depth,
                                                     min_samples_leaf=self.min_samples_leaf,
                                                     min_impurity_decrease=np.finfo(np.float32).eps)
                    else:
                        # print('erf regression')
                        clf = ExtraTreesRegressor(n_estimators=self.n_estimators,
                                                   n_jobs=-1, max_features="sqrt",
                                                   max_depth=self.max_depth,
                                                   min_samples_leaf=self.min_samples_leaf,
                                                   min_impurity_decrease=np.finfo(np.float32).eps)
                    X_train = train_data[train_index, :]
                    y_train = train_label[train_index]
                    clf.fit(X_train, y_train)

                    X_val = train_data[val_index, :]
                    val_p = clf.predict(X_val)
                    val_prob_forest[val_index, :] = val_p.reshape(-1,1)

                    predict_p = clf.predict(test_data)
                    predict_prob_forest += predict_p.reshape(-1,1)
                else: # classification
                    if forest_index % 2 == 0:
                        clf = RandomForestClassifier(n_estimators=self.n_estimators,
                                                 n_jobs=-1, max_features="sqrt",
                                                 max_depth=self.max_depth,
                                                 min_samples_leaf=self.min_samples_leaf,
                                                 min_impurity_decrease=np.finfo(np.float32).eps)
                    else:
                        clf = ExtraTreesClassifier(n_estimators=self.n_estimators,
                                                   n_jobs=-1, max_features="sqrt",
                                                   max_depth=self.max_depth,
                                                   min_samples_leaf=self.min_samples_leaf,
                                                   min_impurity_decrease=np.finfo(np.float32).eps)
                    X_train = train_data[train_index, :]
                    y_train = train_label[train_index]
                    clf.fit(X_train, y_train)

                    X_val = train_data[val_index, :]
                    val_p = clf.predict_proba(X_val)
                    val_prob_forest[val_index, :] = val_p

                    predict_p = clf.predict_proba(test_data)
                    predict_prob_forest += predict_p

                if self.compute_FI:
                    train_contributions = []
                    if self.feature_indicator is not None:
                        for tmp_contributions in self.last_layer_contribution:
                            train_contributions.append(tmp_contributions[train_index,:,:])
                    val_mean_contribution, test_mean_contribution = feature_contribution(forest=clf,X_train=X_train,X_val=X_val,
                                                                                         feature_indicator=self.feature_indicator,
                                                                                         last_layer_contribution=train_contributions,
                                                                                         X_test=test_data)
                    contributions_forest[val_index,:,:] = val_mean_contribution
                    test_contribution_forest+=test_mean_contribution

            val_prob[forest_index, :] = val_prob_forest
            predict_prob_forest /= self.n_fold
            predict_prob[forest_index, :] = predict_prob_forest

            if self.compute_FI:
                test_contribution_forest/= self.n_fold
            # self.forest_list.append(kfold_list)

            if self.compute_FI:
                contributions_list.append(contributions_forest)
                if self.num_classes==1:
                    mu = np.mean(train_label)
                    val_feature_importance[forest_index, :] = compute_importance(contributions_forest, train_label-mu,
                                                                                 self.num_classes)
                    test_feature_importance[forest_index, :] = compute_importance(test_contribution_forest, test_label-mu,
                                                                                  self.num_classes)
                else:
                    val_feature_importance[forest_index,:] = compute_importance(contributions_forest, train_label, self.num_classes)
                    test_feature_importance[forest_index,:] = compute_importance(test_contribution_forest, test_label, self.num_classes)
                test_contribution+=test_contribution_forest

        val_avg = np.sum(val_prob, axis=0)
        val_avg /= self.num_forests
        val_concatenate = val_prob.transpose((1, 0, 2))
        val_concatenate = val_concatenate.reshape(val_concatenate.shape[0], -1)

        predict_avg = np.sum(predict_prob, axis=0)
        predict_avg /= self.num_forests
        predict_concatenate = predict_prob.transpose((1, 0, 2))
        predict_concatenate = predict_concatenate.reshape(predict_concatenate.shape[0], -1)

        if self.compute_FI:
            # feature_importance = np.sum(val_feature_importance, axis=0)
            # feature_importance /= self.num_forests
            val_feature_importance = np.mean(val_feature_importance, axis=0)
            test_feature_importance = np.mean(test_feature_importance, axis=0)
            test_contribution/=self.num_forests

        if self.compute_FI:
            return [val_avg, val_concatenate, predict_avg, predict_concatenate, contributions_list, test_contribution,
                    val_feature_importance, test_feature_importance]
        else:
            return [val_avg, val_concatenate, predict_avg, predict_concatenate]