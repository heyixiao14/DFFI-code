from sklearn.model_selection import KFold
from .layer import *
from .utils import *

class gcForest:
    def __init__(self, num_estimator, num_forests, num_classes=1, max_layer=100,
                 min_samples_leaf=1,max_depth=None, n_fold=5, compute_FI=False):
        self.num_estimator = num_estimator
        self.num_forests = num_forests
        self.n_fold = n_fold
        self.max_depth = max_depth
        self.max_layer = max_layer
        self.min_samples_leaf = min_samples_leaf
        self.num_classes = num_classes # default num_classes=1, means regression
        self.layer_list = []
        self.number_of_layers = max_layer
        self.best_layer = -1
        self.compute_FI = compute_FI # whether to compute feature importance

    def train_and_predict(self, train_data, train_label, test_data, test_label):
        # basis information of dataset
        num_samples, num_features = train_data.shape

        # basis process
        train_data_raw = train_data.copy()
        test_data_raw = test_data.copy()

        # return value
        val_p = []
        val_err = []
        test_p = []
        test_err = []
        val_FI_list, test_FI_list = [], []
        # test_err_byclass = [[],[],[]]
        # val_err_byclass = [[], [], []]

        best_train_err = 100000000
        layer_index = 0
        best_layer_index = 0
        bad = 0

        n_features = train_data.shape[1]
        feature_indicator = None # 0 the original feature, 1 the augmented feature
        contribution_list = None # contribution_list is to be input to the next layer
        # contribution_forest的shape是(train_data_n_samples, n_features, n_classes)
        # contribution_list是contribution_forest的一个列表，对应各个森林

        while layer_index < self.max_layer:
            print("layer " + str(layer_index))
            layer = Layer(num_forests=self.num_forests, n_estimators=self.num_estimator, num_classes=self.num_classes, n_features=n_features,
                          n_fold=self.n_fold, layer_index=layer_index, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                          compute_FI=self.compute_FI, feature_indicator=feature_indicator,last_layer_contribution=contribution_list)

            if self.compute_FI:
                val_prob, val_stack, test_prob, test_stack, contribution_list, test_contribution, val_feature_importance, test_feature_importance = \
                    layer.train_and_predict(train_data, train_label, test_data, test_label)
                val_FI_list.append(val_feature_importance[np.newaxis,:])
                test_FI_list.append(test_feature_importance[np.newaxis,:])
                # print(val_feature_importance)
                # print(test_feature_importance)
            else:
                val_prob, val_stack, test_prob, test_stack = layer.train_and_predict(train_data, train_label, test_data, test_label)

            # self.layer_list.append(layer)

            feature_indicator = np.zeros(train_data_raw.shape[1],dtype=np.int8)
            for forest_id in range(self.num_forests):
                feature_indicator = np.concatenate((feature_indicator,np.ones(self.num_classes,dtype=np.int8)*(forest_id+1)))

            train_data = np.concatenate([train_data_raw, val_stack], axis=1)
            test_data = np.concatenate([test_data_raw, test_stack], axis=1)
            train_data = np.float16(train_data)
            test_data = np.float16(test_data)
            train_data = np.float32(train_data)
            test_data = np.float32(test_data)

            if self.num_classes>1:
                temp_val_acc = compute_accuracy(train_label, val_prob)
                temp_val_err = 1-temp_val_acc
                temp_test_acc = compute_accuracy(test_label, test_prob)
                temp_test_err = 1-temp_test_acc

            elif self.num_classes==1:
                temp_val_err = mean_squared_error(train_label, val_prob, squared=False)
                temp_test_err = mean_squared_error(test_label, test_prob, squared=False)

            # val_p.append(val_prob)
            # test_p.append(test_prob)

            test_err.append(temp_test_err)
            val_err.append(temp_val_err)
            # if self.num_classes==1:
            #     pass
            # else:
            #     for iterc in range(3):
            #         idxc = np.where(test_label==iterc)[0]
            #         temp_test_label = test_label[idxc]
            #         temp_test_prob = test_prob[idxc]
            #         test_err_byclass[iterc].append(1-compute_accuracy(temp_test_label,temp_test_prob))
            #
            #         idxc = np.where(train_label==iterc)[0]
            #         temp_train_label = train_label[idxc]
            #         temp_train_prob = val_prob[idxc]
            #         val_err_byclass[iterc].append(1-compute_accuracy(temp_train_label,temp_train_prob))

            print('val_err={:.3f}, test_err={:.3f}'.format(temp_val_err,temp_test_err))

            if best_train_err <= temp_val_err:
                bad += 1
            else:
                bad = 0
                best_train_err = temp_val_err
                best_layer_index = layer_index
                best_layer_test_contribution = test_contribution
                best_test_prob = test_prob
            if bad >= 3:
                self.number_of_layers = layer_index+1
                break

            self.best_layer = best_layer_index

            if layer_index==0:
                first_layer_test_contribution = test_contribution
                first_test_prob = test_prob

            layer_index = layer_index + 1

        val_FI_list = np.concatenate(val_FI_list, axis=0)
        test_FI_list = np.concatenate(test_FI_list, axis=0)

        return [best_test_prob, best_layer_test_contribution, test_err, best_layer_index, val_FI_list, test_FI_list]
        # return [first_layer_test_contribution, first_test_prob, best_layer_test_contribution, best_test_prob,
        #         val_err, val_err_byclass, test_err, test_err_byclass, best_layer_index, val_FI_list, test_FI_list]


    def predict(self, test_data, compute_FI=True, train_data=None):
        test_data_raw = test_data.copy()
        layer_index = 0
        n_features = test_data.shape[1]
        while layer_index <= self.best_layer:
            layer = self.layer_list[layer_index]
            predict_prob = np.zeros((self.num_forests, test_data.shape[0], self.num_classes), dtype=np.float64)
            n_dim = test_data.shape[1]
            for forest_index in range(self.num_forests):
                predict_prob_forest = np.zeros([test_data.shape[0], self.num_classes])
                for kfold in range(self.n_fold):
                    clf = layer.forest_list[forest_index][kfold]
                    predict_p = clf.predict_proba(test_data)
                    if compute_FI:
                        mean_pred, mean_bias, mean_contribution = feature_contribution(forest=clf, X_train=train_data,
                                                                                       X_val=test_data,
                                                                                       feature_indicator=self.feature_indicator,
                                                                                       last_layer_contribution=layer.last_layer_contribution)
                    predict_prob_forest += predict_p
                predict_prob_forest /= self.n_fold
                predict_prob[forest_index, :] = predict_prob_forest
            predict_avg = np.sum(predict_prob, axis=0)
            predict_avg /= self.num_forests
            predict_concatenate = predict_prob.transpose((1, 0, 2))
            predict_concatenate = predict_concatenate.reshape(predict_concatenate.shape[0], -1)

            test_prob, test_stack = predict_avg, predict_concatenate

            test_data = np.concatenate([test_data_raw, test_stack], axis=1)
            test_data = np.float16(test_data)
            test_data = np.float64(test_data)
            layer_index+=1

        return test_prob