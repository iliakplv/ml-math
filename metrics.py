import math
import algebra


def cross_entropy(y_hat, y):
    if len(y_hat) != len(y):
        raise Exception('Different size lists')

    predictions_log = [math.log(item) for item in y_hat]
    sum = algebra.list_sum([y[i] * predictions_log[i] for i in range(len(y))])

    return -sum


def cross_entropy_loss(y_hat_list, y_list):
    if len(y_hat_list) != len(y_list):
        raise Exception('Different size vectors')

    sum = 0
    total = len(y_list)

    for i in range(total):
        y_hat = y_hat_list[i]
        y = y_list[i]
        sum += cross_entropy(y_hat, y)

    return sum / total


def non_zero_index(l):
    for i, v in enumerate(l):
        if v != 0:
            return i
    raise Exception('Value not found')


def max_value_index(l):
    max_index = 0
    max_value = l[0]
    for i, v in enumerate(l):
        if v > max_value:
            max_index = i
    return max_index


def accuracy(y_hat_list, y_list):
    if len(y_hat_list) != len(y_list):
        raise Exception('Different size lists')
    total = len(y_list)
    total_correct = 0
    for i in range(total):
        if max_value_index(y_hat_list[i]) == non_zero_index(y_list[i]):
            total_correct += 1
    return total_correct / total


if __name__ == '__main__':
    targets = [0.0, 1.0, 0.0]
    predictions = [0.228, 0.619, 0.153]
    print(cross_entropy(predictions, targets))
    predictions = [0.001, 0.998, 0.001]
    print(cross_entropy(predictions, targets))
