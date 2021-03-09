# %%

import numpy as np
import neighbors
import model_selection

# %%

data = np.loadtxt('data/iris.csv',delimiter=',')
X,y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

# %%
