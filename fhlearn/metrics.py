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


def confusion_matrix(
        true_labels,
        predicted_labels
    ) -> np.array:
    """
    Returns confusion matrix with rows as predicted labels and columns as true labels.

    Currently requires that n classes be encoded as 0,n-1. Otherwise it will break.
    """

    n_samples_true, n_samples_predicted = len(true_labels), len(predicted_labels)

    if n_samples_true != n_samples_predicted:
        raise ValueError()

    n_classes = len(set(true_labels))
    matrix = np.zeros((n_classes,n_classes))
    
    for i in range(len(true_labels)):
        true_label = true_labels[i]
        predicted_label = predicted_labels[i]
        matrix[predicted_label-1][true_label-1] += 1
    return matrix


def precision_score(
        true_labels,
        predicted_labels,
        average=None,
        zero_division=0
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
        if sum_positives == 0: # for handling zero-divison problem if no positives
            precision_dict[class_] = zero_division
        else:
            precision_dict[class_] = tp / sum_positives
    
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


def recall_score(
        true_labels,
        predicted_labels,
        average=None,
        zero_division=0
    ):

    n_samples_true, n_samples_predicted = len(true_labels), len(predicted_labels)

    if n_samples_true != n_samples_predicted:
        raise ValueError()

    records = _compute_positives_and_negatives(true_labels,predicted_labels)
    classes = records.keys()
    recall_dict = {}

    for class_ in classes:
        tp, fn = records[class_]['tp'], records[class_]['fn']
        potential_positives = tp + fn
        if potential_positives == 0: # for handling zero-divison problem if no positives
            recall_dict[class_] = zero_division
        else:
            recall_dict[class_] = tp / potential_positives

    if not average:
        return recall_dict
    else:
        if average == 'macro':
            return sum(recall_dict.values())/len(classes)
        elif average == 'micro':
            total_tp, total_fn = 0, 0
            for class_ in records.keys():
                total_tp += records[class_]['tp']
                total_fn += records[class_]['fn']
            return total_tp / (total_tp + total_fn)
        elif average == 'weighted':     # sum of class-specific recall scores weigthed by frequency in true labels
            sum_recall = 0
            for class_ in recall_dict.keys():
                sum_recall += recall_dict[class_] * records[class_]['class_weight']
            return sum_recall
        else:
            raise ValueError('Invalid argument for the "average" keyword parameter')


def f1_score(
        true_labels,
        predicted_labels,
        average=None,
        zero_division=0
    ):

    n_samples_true, n_samples_predicted = len(true_labels), len(predicted_labels)

    if n_samples_true != n_samples_predicted:
        raise ValueError()
    
    precision = precision_score(true_labels,predicted_labels,zero_division=zero_division)
    recall = recall_score(true_labels,predicted_labels,zero_division=zero_division)
    weights = _get_class_weights(true_labels)
    f1_dict = {}  
    
    for class_ in precision.keys():
        sum_recall_precision = precision[class_] + recall[class_]
        if sum_recall_precision == 0:
            f1_dict[class_] = zero_division
        else:
            f1_dict[class_] = 2 * (precision[class_] * recall[class_]) / sum_recall_precision
    
    if not average:
        return f1_dict
    else:
        if average == 'macro':
            sum_scores = 0
            for score in f1_dict.values():
                sum_scores += score
            return sum_scores / len(f1_dict)
        elif average == 'micro':
            precision, recall = precision_score(true_labels,predicted_labels,average=average), recall_score(true_labels,predicted_labels,average=average)
            return 2 * (precision * recall) / (precision + recall)
        elif average == 'weighted':
            sum_scores = 0
            for class_ in f1_dict.keys():
                sum_scores += f1_dict[class_] * weights[class_]
            return sum_scores
        else:
            raise ValueError('Invalid argument for the "average" keyword parameter')


def mean_squared_error(true_targets: np.array, predicted_targets: np.array, squared=True):
    if not np.shape(true_targets) == np.shape(predicted_targets):
        raise ValueError('Input arrays not of equal dimensions')

    if squared:
        return np.mean((true_targets - predicted_targets) ** 2) 
    else:
        return np.mean(true_targets - predicted_targets) 