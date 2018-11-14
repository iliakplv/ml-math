import math
import algebra


def relu(x):
    return algebra.Vector([item if item > 0 else 0 for item in x.vector])


def tanh(x):
    return algebra.Vector([math.tanh(item) for item in x.vector])


if __name__ == '__main__':
    relu(algebra.Vector([-0.2, -0.1, 0.0, 1.0, 2.0])).print()
    print()
    tanh(algebra.Vector([0.0, 0.5, 1.0, 5.0])).print()
