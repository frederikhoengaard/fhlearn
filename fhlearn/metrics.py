import numpy as np


def accuracy_score(
        true_labels,
        predicted_labels,
        normalize: bool = True,
        ) -> float:   
    
    n_samples_true, n_samples_predicted = len(true_labels), len(predicted_labels)

    if n_samples_true != n_samples_predicted:
        raise ValueError()

    n_correct_predictions = 0

    for i in range(n_samples_true):
        if  true_labels[i] == predicted_labels[i]:
            n_correct_predictions += 1

    if normalize:
        return n_correct_predictions / n_samples_true
    else:
        return n_correct_predictions












lst1 = [1,2,3,4]

lst2 = [1,2,3,4]

lst3 = [2,3]

lst4 = [[1,2],[3,4]]

lst5 = [1,5,2,4]

print(accuracy_score(lst1,lst5))