import numpy as np


def _validate_individual_ratio(ratio: float) -> bool:
    if ratio <= 0 or ratio >= 1:
        return False
    return True


def _validate_ratios(ratios: list) -> bool:
    if sum(ratios) != 1:
        return False
    for ratio in ratios:
        if not _validate_individual_ratio(ratio):
            return False
    return True


def train_test_split(
        data: np.array, 
        targets: np.array = None, 
        test_size: float = None, 
        train_size: float = None, 
        random_state: int = None, 
        shuffle: bool = True, 
        tratify: bool = False
    ) -> list:
    if targets is not None:
        if np.shape(data)[0] == np.shape(targets)[0]:
            data = np.c_[data,targets]
        else:
            raise ValueError('Feature and target arrays not containing equal number of samples.')

    if random_state:
        np.random.seed(random_state)

    if train_size and not test_size:
        if _validate_individual_ratio(train_size):
            test_size = 1 - train_size
        else:
            raise ValueError('Invalid value passed for train_size parameter. Must be between 0 and 1')
    elif test_size and not train_size:
        if not _validate_individual_ratio(test_size):
            raise ValueError('Invalid value passed for test_size parameter. Must be between 0 and 1')
    elif test_size and train_size:
        if not _validate_ratios([test_size,train_size]):
            raise ValueError('Invalid values passed for test_size and train_size parameters. Must be between 0 and 1 and sum to 1')
    else: 
        test_size = 0.2

    indices = np.arange(len(data))

    if shuffle:
        indices = np.random.permutation(len(data)) 

    test_set_size = int(len(data) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    X_train, y_train = data[train_indices][:,:-1], data[train_indices][:,-1] 
    X_test, y_test = data[test_indices][:,:-1], data[test_indices][:,-1]
    return (X_train,X_test,y_train,y_test)


