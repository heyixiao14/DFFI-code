import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def generate_3class_dataset(nsamples = 200,irrelevant_dim=0):
    ndim = 2
    X = np.random.rand(nsamples,ndim+irrelevant_dim)
    Y = np.zeros((nsamples,),dtype=int)
    for i in range(nsamples):
        x1=X[i,0]
        x2=X[i,1]
        if x1<0.5:
            Y[i]=0
        elif x2 >0.5:
            Y[i]=1
        else:
            Y[i] = 2
    return X, Y

if __name__ == '__main__':
    X_train,y_train = generate_3class_dataset()
    n_classes = len(np.bincount(y_train))

    # clf = DecisionTreeClassifier()
    clf = RandomForestClassifier()
    clf.fit(X_train,y_train)
    fi=clf.feature_importances_
    print(fi)
    # print(clf.decision_path(np.array([[0.1,0.1]])))

    for i in range(n_classes):
        idx = np.where(y_train==i)[0]
        Xi = X_train[idx]
        plt.scatter(Xi[:,0],Xi[:,1],label='class {}'.format(i))
    plt.show()