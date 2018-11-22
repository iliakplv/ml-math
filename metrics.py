import math
import algebra


def cross_entropy(predictions, targets):
    if len(predictions) != len(targets):
        raise Exception('Different size vectors')

    predictions_log = [math.log(item) for item in predictions.vector]
    sum = algebra.list_sum([targets.vector[i] * predictions_log[i] for i in range(len(targets))])

    return -sum


def max_value_index(vector):
    max_index = 0
    max_value = vector.vector[0]
    for i in range(len(vector)):
        if vector.vector[i] > max_value:
            max_index = i
    return max_index


def accuracy(probability_vectors, labels):
    if len(probability_vectors) != len(labels):
        raise Exception('Different size lists')
    total = len(labels)
    total_correct = 0
    for i in range(total):
        if max_value_index(probability_vectors[i]) == labels[i]:
            total_correct += 1
    return total_correct / total


if __name__ == '__main__':
    targets = algebra.Vector([0.0, 1.0, 0.0])
    predictions = algebra.Vector([0.228, 0.619, 0.153])
    print(cross_entropy(predictions, targets))
    predictions = algebra.Vector([0.001, 0.998, 0.001])
    print(cross_entropy(predictions, targets))

    outputs = [
        algebra.Vector([0.8, 0.1, 0.1]),
        algebra.Vector([0.2, 0.6, 0.2]),
        algebra.Vector([0.1, 0.2, 0.7])
    ]
    labels = [0, 0, 2]
    print('Accuracy: {}'.format(accuracy(outputs, labels)))
    labels = [0, 1, 2]
    print('Accuracy: {}'.format(accuracy(outputs, labels)))
