import numpy as np
import metrics
import linear_model
import tree
import neighbors
import model_selection
from sklearn.tree import DecisionTreeClassifier as sklearnDecisionTreeClassifier

def load_dataset(path: str):
    data = np.loadtxt(path,delimiter=',')
    X_train,X_test,y_train,y_test = model_selection.train_test_split(data)
    return X_train,X_test,y_train,y_test
 


def test_model(model,features,labels,test_features):
    clf = model()
    clf.fit(features,labels)
    y_pred = clf.predict()

class Test:
    def __init__(self,model,benchmark_model,datasets):
        self.model = model
        self.benchmark = benchmark_model
        self.datasets = datasets

    def _compare_predictions(self):
        pass

    def _load_dataset(self, path: str):
        data = np.loadtxt(path,delimiter=',')
        X_train,X_test,y_train,y_test = model_selection.train_test_split(data)
        return X_train,X_test,y_train,y_test
        
    



def main():
    models = [
        (tree.DecisionTreeClassifier,sklearnDecisionTreeClassifier)
    ]

    regression_datasets=[
        'data/boston.csv',
        'data/diabetes.csv'
    ]

    classification_datasets = [
        'data/iris.csv',
        'data/cancer.csv',
        'data/wine.csv'
    ]


   

if __name__ == '__main__':
    main()