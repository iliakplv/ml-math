import math
import algebra


def cross_entropy(predictions, targets):
    if len(predictions) != len(targets):
        raise Exception('Different size vectors')

    predictions_log = [math.log(item) for item in predictions.vector]
    sum = algebra.list_sum([targets.vector[i] * predictions_log[i] for i in range(len(targets))])

    return -sum


def non_zero_index(vector):
    vector = vector.vector
    for i, value in enumerate(vector):
        if value != 0:
            return i
    raise Exception('Value not found')


def max_value_index(vector):
    vector = vector.vector
    max_index = 0
    max_value = vector.vector[0]
    for i in range(len(vector)):
        if vector.vector[i] > max_value:
            max_index = i
    return max_index


def accuracy(y_hat_vectors, y_vectors):
    if len(y_hat_vectors) != len(y_vectors):
        raise Exception('Different size vectors')
    total = len(y_vectors)
    total_correct = 0
    for i in range(total):
        if max_value_index(y_hat_vectors[i]) == non_zero_index(y_vectors[i]):
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
