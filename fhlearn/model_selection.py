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


def train_test_split(data: np.array, test_size: float = None, train_size: float = None, random_state: int = None):
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


    """if not test_size:
        if not train_size:
            test_size = 0.2
        else:

    
    shuffled_indices = np.random.permutation(len(data)) 
    test_set_size = int(len(data) * test_ratio) 
    test_indices = shuffled_indices[:test_set_size] 
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]"""


train_test_split('whatevs', train_size=0.8)
#print(_validate_ratios([0.2,0.8]))