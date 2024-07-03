from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, _tree
import numpy as np
import csv


def last_layer_contribution_by_node(node_index_list, last_layer_contribution):
    node_contribution = np.zeros_like(last_layer_contribution[0])
    if len(node_index_list)==0:
        return node_contribution
    for sample_index in node_index_list:
        node_contribution += last_layer_contribution[sample_index]
    node_contribution = node_contribution/len(node_index_list)
    # if np.isnan(node_contribution).any():
    #     print(node_index_list,len(node_index_list))
    return node_contribution

class tree_node:
    def __init__(self, node_id, data_idx, contribs):
        self.node_id = node_id
        self.data_idx = data_idx
        self.contribs = contribs
        self.last_contribs = None

def get_child_node(parent_node,child_id,feature_id,child_data_idx,values_list,feature_indicator,last_layer_contribution,record_file=None):
    parent_id = parent_node.node_id
    contribs = parent_node.contribs.copy()
    if (feature_indicator is not None) and (feature_indicator[feature_id]>0):
        # compute the forest_id corresponding to the splitting feature
        forest_id = feature_indicator[feature_id] - 1

        # data index back to last_layer_contribution
        if parent_node.last_contribs is None:
            parent_contribution = last_layer_contribution_by_node(parent_node.data_idx, last_layer_contribution[forest_id])
            # save it to avoid repeated computation in the right node
            parent_node.last_contribs = parent_contribution
        else:
            parent_contribution = parent_node.last_contribs

        # parent_contribution = last_layer_contribution_by_node(parent_node.data_idx, last_layer_contribution[forest_id])

        child_contribution = last_layer_contribution_by_node(child_data_idx, last_layer_contribution[forest_id])
        feature_contribution_mat = child_contribution - parent_contribution # shape (n_features, n_classes)

        # do calibration
        calibration = True
        if calibration:
            old_feature_contribs = np.sum(feature_contribution_mat,axis=0) # shape (n_classes,)
            new_feature_contribs = values_list[child_id] - values_list[parent_id]
            if len(old_feature_contribs) == 1: # regression
                temp_old = old_feature_contribs[0]
                temp_new = new_feature_contribs
                feature_contribution_mat = feature_contribution_mat.reshape(-1)
                if abs(temp_new)<np.finfo(np.float32).eps or abs(temp_old)<np.finfo(np.float32).eps:
                    feature_contribution_mat *=0
                else:
                    # feature_contribution_mat = feature_contribution_mat.reshape(-1)
                    if temp_new>0:
                        calib_idx = np.where(feature_contribution_mat > 0)[0]
                        if len(calib_idx)==0:
                            calib_idx = np.where(feature_contribution_mat < 0)[0]
                    else:
                        calib_idx = np.where(feature_contribution_mat < 0)[0]
                        if len(calib_idx)==0:
                            calib_idx = np.where(feature_contribution_mat > 0)[0]
                    temp_sum = sum(feature_contribution_mat[calib_idx])
                    temp_delta = temp_new - temp_old
                    for iter_idx in range(len(calib_idx)):
                        temp_idx = calib_idx[iter_idx]
                        feature_contribution_mat[temp_idx] = (temp_delta/temp_sum+1)*feature_contribution_mat[temp_idx]
                feature_contribution_mat = feature_contribution_mat.reshape(-1,1)
            else: # classification
                for itery in range(len(old_feature_contribs)):
                    temp_old = old_feature_contribs[itery]
                    temp_new = new_feature_contribs[itery]
                    temp_feature_contribution_mat = feature_contribution_mat[:,itery]
                    if abs(temp_new) < np.finfo(np.float32).eps or abs(temp_old) < np.finfo(np.float32).eps:
                        temp_feature_contribution_mat*=0
                    else:
                        if temp_new > 0:
                            calib_idx = np.where(temp_feature_contribution_mat > 0)[0]
                            if len(calib_idx) == 0:
                                calib_idx = np.where(feature_contribution_mat < 0)[0]
                        else:
                            calib_idx = np.where(temp_feature_contribution_mat < 0)[0]
                            if len(calib_idx) == 0:
                                calib_idx = np.where(feature_contribution_mat > 0)[0]
                        temp_sum = sum(temp_feature_contribution_mat[calib_idx])
                        temp_delta = temp_new - temp_old
                        for iter_idx in range(len(calib_idx)):
                            temp_idx = calib_idx[iter_idx]
                            temp_feature_contribution_mat[temp_idx] = (temp_delta / temp_sum + 1) * temp_feature_contribution_mat[
                                temp_idx]
                    feature_contribution_mat[:,itery] = temp_feature_contribution_mat
            if record_file is not None:
                writer = csv.writer(record_file)
                writer.writerow([temp_old,temp_new])

        # add contribution to each feature
        contribs = contribs + feature_contribution_mat
    else:
        # for the fist layer feature_indicator is None
        # or split on an original feature
        contrib = values_list[child_id] - values_list[parent_id]
        contribs[feature_id]+=contrib

    child_node = tree_node(node_id=child_id, data_idx=child_data_idx, contribs=contribs)
    return child_node

def get_leaves_contributions(model, X_train, feature_indicator=None, last_layer_contribution=None):
    n_nodes = model.tree_.node_count
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right
    feature = model.tree_.feature
    threshold = model.tree_.threshold

    # remove the single-dimensional inner arrays
    values = model.tree_.value.squeeze(axis=1)
    # reshape if squeezed into a single float
    if len(values.shape) == 0:
        values = np.array([values])
    if isinstance(model, DecisionTreeRegressor):
        # we require the values to be the same shape as the biases
        values = values.squeeze(axis=1)
        if feature_indicator is None:
            line_shape = (X_train.shape[1], 1)  # each feature, each class
        else:
            line_shape = (np.bincount(feature_indicator)[0], 1)
    elif isinstance(model, DecisionTreeClassifier):
        # scikit stores category counts, we turn them into probabilities
        normalizer = values.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        values /= normalizer

        if feature_indicator is None:
            line_shape = (X_train.shape[1], model.n_classes_) # each feature, each class
        else:
            line_shape = (np.bincount(feature_indicator)[0], model.n_classes_)

    # make into python list, accessing values will be faster
    values_list = list(values)

    leaves_contributions = {}

    record_file = None

    root_node = tree_node(node_id=0, data_idx=np.arange(len(X_train)), contribs=np.zeros(line_shape))
    stack = [root_node]  # start with the root node
    while len(stack) > 0:
        node = stack.pop()
        node_id = node.node_id
        # If the left and right child of a node is not the same we have a split node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack` so we can loop through them
        if is_split_node:
            feature_id = feature[node_id]

            # left child
            child_id = children_left[node_id]
            child_data_idx = node.data_idx[X_train[node.data_idx][:,feature_id]<=threshold[node_id]] # if the last part is multiple, then list, if single, then still list?
            # there is risk that the new data doesn't fall in the cell, then nan occurs
            stack.append(get_child_node(parent_node=node,child_id=child_id,feature_id=feature_id,
                                        child_data_idx=child_data_idx,values_list=values_list,
                                        feature_indicator=feature_indicator,
                                        last_layer_contribution=last_layer_contribution,record_file=record_file))

            # right child
            child_id = children_right[node_id]
            child_data_idx = node.data_idx[X_train[node.data_idx][:,feature_id]>threshold[node_id]]
            stack.append(get_child_node(parent_node=node,child_id=child_id,feature_id=feature_id,
                                        child_data_idx=child_data_idx,values_list=values_list,
                                        feature_indicator=feature_indicator,
                                        last_layer_contribution=last_layer_contribution,record_file=record_file))
        else:
            # if not a split node, record with the index of leaves
            leaves_contributions[node_id] = node.contribs

    if record_file is not None:
        record_file.close()

    return leaves_contributions

def get_val_contributions(model, leaves_contributions, X_val):
    leaves = model.apply(X_val)
    contributions = []
    for row, leaf in enumerate(leaves):
        contributions.append(leaves_contributions[leaf])
    return np.array(contributions)

def _iterative_mean(iter, current_mean, x):
    """
    :param iter: non-negative integer, iteration
    :param current_mean: numpy array, current value of mean
    :param x: numpy array, new value to be added to mean
    :return: numpy array, updated mean
    """
    return current_mean + ((x - current_mean) / (iter + 1))


def feature_contribution(forest, X_train, X_val, feature_indicator=None, last_layer_contribution=None, X_test=None):
    val_mean_contribution = None
    test_mean_contribution = None

    for i, tree in enumerate(forest.estimators_):
        leaves_contributions = get_leaves_contributions(model=tree,X_train=X_train, feature_indicator=feature_indicator,
                                                 last_layer_contribution=last_layer_contribution)
        val_contribution = get_val_contributions(model=tree, leaves_contributions=leaves_contributions, X_val=X_val)

        if i < 1:  # first iteration
            val_mean_contribution = val_contribution
        else:
            val_mean_contribution = _iterative_mean(i, val_mean_contribution, val_contribution)

        if X_test is not None:
            test_contribution = get_val_contributions(model=tree, leaves_contributions=leaves_contributions, X_val=X_test)
            if i < 1:  # first iteration
                test_mean_contribution = test_contribution
            else:
                test_mean_contribution = _iterative_mean(i, test_mean_contribution, test_contribution)

    return val_mean_contribution, test_mean_contribution


def compute_importance(contribution, label, n_classes):
    mean_importance = None
    if n_classes>1:
        for itery in range(n_classes):
            # print(itery)
            class_idx = np.where(label==itery)[0]
            class_contribution = contribution[class_idx] # (idx, feature, class)
            # print(class_contribution[0])
            # print(sum(class_contribution[0,:,itery]))
            tmp = np.mean(class_contribution[:,:,itery],axis=0) #shape(6,)
            # print(tmp,sum(tmp))
            if itery < 1:
                mean_importance = tmp
            else:
                mean_importance = _iterative_mean(itery, mean_importance, tmp)
    else:
        # contribution is of shape (nsamples, n_features, 1)
        n_samples, n_features, _ = contribution.shape
        mean_importance = np.zeros((n_features,))
        for iterf in range(n_features):
            tmp = contribution[:,iterf,:] # of shape (nsamples, 1)
            mean_importance[iterf] = np.dot(tmp.reshape(-1),label)/n_samples
    return mean_importance
