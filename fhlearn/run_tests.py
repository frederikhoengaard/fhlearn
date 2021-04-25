import numpy as np
import metrics
import linear_model
import tree
import neighbors
import model_selection
from sklearn.tree import DecisionTreeClassifier as sklearnDecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as sklearnKNeighborsClassifier
import preprocessing



class Test:
    def __init__(
            self,
            model,
            benchmark_model,
            dataset,
            random_state=42
        ):
        self.model = model
        self.benchmark = benchmark_model
        self.dataset = dataset
        self.random_state = random_state

    def _load_dataset(self, path: str):
        data = np.loadtxt(path,delimiter=',')
        X,y = data[:,:-1],data[:,-1]
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X,targets=y,random_state=self.random_state)
        return X_train, X_test, y_train, y_test

    def _get_prediction(self,model,train_features,train_labels,test_features):
        if 'random_state' in dir(model):
            clf = model(random_state=self.random_state)
        else:
            clf = model()
        clf.fit(train_features,train_labels)
        return clf.predict(test_features)
        
    def _compare_predictions(self,pred_1,pred_2):
        return np.array_equal(pred_1,pred_2)
        
    def run_experiments(self):
        bad = []
        good = []

        X_train,X_test,y_train,y_test = self._load_dataset(self.dataset)
        y_pred = self._get_prediction(self.model,X_train, y_train, X_test)
        y_pred_benchmark = self._get_prediction(self.benchmark, X_train, y_train, X_test)
        result = self._compare_predictions(y_pred, y_pred_benchmark)
        if not result:
            bad.append((self.model,self.dataset,np.size(X_test,axis=0),(y_pred == y_pred_benchmark).sum()))
        else:
            good.append((self.model,self.dataset))
          #  print(result)
        return (good, bad)



def main():
    regression_datasets=[
        'data/boston.csv',
        'data/diabetes.csv'
    ]

    classification_datasets = [
        'data/iris.csv',
        'data/wine.csv'
      #  'data/cancer.csv',
      
    ]

    experiments = [
        (tree.DecisionTreeClassifier,sklearnDecisionTreeClassifier,'data/iris.csv',42),
        (tree.DecisionTreeClassifier,sklearnDecisionTreeClassifier,'data/wine.csv',42),
      #  (neighbors.KNeighborsClassifier,sklearnKNeighborsClassifier,classification_datasets)
    ]

    

    
    successful_tests = []
    failed_tests = []
    for experiment in experiments:
        a,b,datasets,rs = experiment
        test_module = Test(a,b,datasets,random_state=rs)
        results = test_module.run_experiments()
        successful_tests.extend(results[0])
        failed_tests.extend(results[1])

    print(successful_tests)
    if failed_tests:
        print('BAD!')
    print(failed_tests)
        
            
   

if __name__ == '__main__':
    main()