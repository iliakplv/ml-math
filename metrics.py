import math
import algebra


def cross_entropy(predictions, targets):
    if len(predictions) != len(targets):
        raise Exception('Different size vectors')

    predictions_log = [math.log(item) for item in predictions.vector]
    sum = algebra.list_sum([targets.vector[i] * predictions_log[i] for i in range(len(targets))])

    return -sum


if __name__ == '__main__':
    targets = algebra.Vector([0.0, 1.0, 0.0])
    predictions = algebra.Vector([0.228, 0.619, 0.153])
    print(cross_entropy(predictions, targets))
    predictions = algebra.Vector([0.001, 0.998, 0.001])
    print(cross_entropy(predictions, targets))
