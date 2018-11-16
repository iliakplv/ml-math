import math
import algebra


def relu(x):
    return algebra.Vector([item if item > 0 else 0 for item in x.vector])


def tanh(x):
    return algebra.Vector([math.tanh(item) for item in x.vector])


def relu_derivative(x):
    return algebra.Vector([1.0 if item > 0 else 0 for item in x.vector])


def tanh_derivative(x):
    t = tanh(x)
    t_sqr_list = algebra.list_mul(t.vector, t.vector)
    return algebra.Vector([1.0 - item for item in t_sqr_list])


if __name__ == '__main__':
    v = algebra.Vector([-0.2, -0.1, 0.0, 1.0, 2.0, 8.0])
    relu(v).print()
    relu_derivative(v).print()

    print()

    v = algebra.Vector([0.0, 0.5, 1.0, 2.0, 8.0])
    tanh(v).print()
    tanh_derivative(v).print()
