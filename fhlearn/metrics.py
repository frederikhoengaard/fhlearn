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

    for i in range(len(true_labels)):
        if  true_labels[i] == predicted_labels[i]:
            n_correct_predictions += 1

    if normalize:
        return n_correct_predictions / n_samples_true
    else:
        return n_correct_predictions



def _get_class_weights(labels) -> dict:

    classes = set(labels)
    class_weights = {}

    class_counts = {}

    for value in labels:
        if value not in class_counts:
            class_counts[value] = 1
        else:
            class_counts[value] += 1

    for class_ in classes:
        class_weights[class_] = class_counts[class_] / len(labels)

    return class_weights



def _compute_positives_and_negatives(
        true_labels,
        predicted_labels
    ) -> dict:

    classes = set(true_labels)
    records = {}
    weights = _get_class_weights(true_labels)

    for class_ in classes:
        true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0
        for i in range(len(true_labels)):
            if predicted_labels[i] == class_:
                if true_labels[i] == class_:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if true_labels[i] != class_:
                    true_negatives += 1
                else:
                    false_negatives += 1
        records[class_] = {'tp': true_positives, 'fp': false_positives, 'tn': true_negatives, 'fn': false_negatives, 'class_weight': weights[class_]}
    return records



def precision_score(
        true_labels,
        predicted_labels,
        average=None,
        zero_divison=0
    ):

    n_samples_true, n_samples_predicted = len(true_labels), len(predicted_labels)

    if n_samples_true != n_samples_predicted:
        raise ValueError()

    records = _compute_positives_and_negatives(true_labels,predicted_labels)
    classes = records.keys()
    precision_dict = {}

    for class_ in classes:
        tp, fp = records[class_]['tp'], records[class_]['fp']
        sum_positives = tp + fp
        if sum_positives == 0: # for zero divison
            precision_dict[class_] = 0
        else:
            precision_dict[class_] = tp / (sum_positives)
    
    if not average:
        return precision_dict # class-specific precision
    else:
        if average == 'macro':
            return sum(precision_dict.values())/len(classes)
        elif average == 'micro':    # same as accuracy
            total_tp, total_fp = 0, 0
            for class_ in records.keys():
                total_tp += records[class_]['tp']
                total_fp += records[class_]['fp']
            return total_tp / (total_tp + total_fp)
        elif average == 'weighted':     # sum of class-specific precision scores weigthed by frequency in true labels
            sum_precision = 0
            for class_ in precision_dict.keys():
                sum_precision += precision_dict[class_] * records[class_]['class_weight']
            return sum_precision
        else:
            raise ValueError('Invalid argument for the "average" keyword parameter')




y_true = [1,2,3,4]

y_pred = [2,2,3,4]

print('fhlearn')
#print(precision_score(y_true,y_pred))
print(precision_score(y_true,y_pred,average='weighted'))

from sklearn.metrics import precision_score as preci
print('sklearn')
#print(preci(y_true,y_pred,average='weighted',zero_division=0))
print(preci(y_true,y_pred,average='weighted',zero_division=0))