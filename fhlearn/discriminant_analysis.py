# %%

import numpy as np
from model_selection import train_test_split
from operator import itemgetter


class LinearDiscriminantAnalysis:
    def __init__(self,solver='eigen'):
        self.solver = solver
        self.n_features: int = None
        self.n_classes: int = None
        self.X_lda = None
        self.w_matrix = None
        self.is_fitted = None

    def _get_n_classes(self, labels: np.array):
        return len(np.unique(labels))

    def _get_n_obs(self, data: np.array) -> int:
        return np.size(data, axis=0)

    def _class_feature_means(
            self, 
            features: np.array, 
            labels: np.array
        ) -> np.array:
        mean_vectors = []
        for class_ in range(self.n_classes):
            mean = np.mean(features[labels == class_], axis=0)
            mean_vectors.append(np.reshape(mean, (-1,1)))
        return np.concatenate(mean_vectors, axis=1)

    def _get_within_scatter_matrix(
            self,
            features: np.array,
            labels: np.array
        ) -> np.array:
        Sw = np.zeros((self.n_features,self.n_features))
        class_means = self._class_feature_means(features,labels)
        for class_ in range(self.n_classes):
            mean_vector = np.reshape(class_means[:,class_], (-1,1))
            x = features[labels == class_]
            for i in range(len(x)):
                x_i = np.reshape(x[i], (-1,1))
                Sw += (x_i-mean_vector) @ ((x_i-mean_vector).T)
        return Sw

    def _get_between_scatter_matrix(
            self,
            features: np.array,
            labels: np.array
        ) -> np.array:
        Sb = np.zeros((self.n_features,self.n_features))
        class_means = self._class_feature_means(features,labels)
        feature_means = np.mean(features, axis=0)
        for class_ in range(self.n_classes):
            data = features[labels == class_]
            n = self._get_n_obs(data)
            mc, m = np.reshape(class_means[:,class_], (-1,1)), np.reshape(feature_means, (-1,1))
            Sb += n * (mc - m) @ ((mc - m).T)
        return Sb

    def _get_eigen(
            self, 
            within_class_scatter_matrix: np.array, 
            between_class_scatter_matrix: np.array
        ):
        Sw_inv = np.linalg.inv(within_class_scatter_matrix)
        problem = Sw_inv @ between_class_scatter_matrix
        return np.linalg.eig(problem)

    def _discriminant_function(
            self, 
            x, 
            class_probabilities, 
            covariance_matrix, 
            class_mean
        ):
        a_k = (2 * np.log(class_probabilities) - np.log(np.linalg.det(covariance_matrix)) 
              - class_mean.T @ np.linalg.inv(covariance_matrix) @ class_mean)
        b_k = 2 * class_mean.T @ np.linalg.inv(covariance_matrix) 
        return a_k + b_k.T @ x

    def explained_variance_ratio(self,n_components):
        pass

    def fit(self,features: np.array, labels: np.array):
        self.n_features = np.size(features, axis=1)
        self.n_classes = self._get_n_classes(labels)
        Sw = self._get_within_scatter_matrix(features,labels)
        Sb = self._get_between_scatter_matrix(features, labels)
        eigen_values, eigen_vectors = self._get_eigen(Sw,Sb)
        pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
        pairs = sorted(pairs, key=itemgetter(0),reverse=True)
        self.w_matrix = np.hstack((pairs[0][1].reshape(self.n_features,1), pairs[1][1].reshape(self.n_features,1))).real
        self.is_fitted = True

    def transform(self, features):
        if not self.is_fitted:
            raise ValueError()
        return features @ self.w_matrix

    def fit_transform(self, features, labels):
        self.fit(features,labels)
        X_lda = self.transform(features)
        return X_lda

    def predict(self, features):
        pass





class QuadraticDiscriminantAnalysis:
    def __init__(self,solver='eigen'):
        self.solver = solver
        self.n_features: int = None
        self.n_classes: int = None
        self.X_lda = None
        self.w_matrix = None

    def _discriminant_function(
            self, 
            x, 
            class_probabilities, 
            covariance_matrix, 
            class_mean
        ):
        a_k = (2 * np.log(class_probabilities) - np.log(np.linalg.det(covariance_matrix)) 
              - class_mean.T @ np.linalg.inv(covariance_matrix) @ class_mean)
        b_k = 2 * class_mean.T @ np.linalg.inv(covariance_matrix) 
        c_k = - np.linalg.inv(covariance_matrix)

        return a_k + b_k.T @ x + x.T @ c_k @ x

    def transform(self):
        pass

    def fit_transform(self):
        pass

    def predict(self):
        pass

# %%


data = np.loadtxt('data/wine.csv',delimiter=',')
#X_train,X_test,y_train,y_test = train_test_split(data)
X_train = data[:,:-1]
y_train = data[:,-1]

lda = LinearDiscriminantAnalysis()

lda.fit(X_train,y_train)
print(lda.transform(X_train))
# %%
