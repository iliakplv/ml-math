import math
import algebra


def tanh(z):
    return algebra.Vector([math.tanh(item) for item in z.vector])


def tanh_derivative(z):
    ones = algebra.Vector([1.0 for _ in range(len(z))])
    t = tanh(z)
    return ones - t.mul_element_wise(t)


def tanh_back(dA, z):
    return dA.mul_element_wise(tanh_derivative(z))


if __name__ == '__main__':
    v = algebra.Vector([0.0, 0.5, 1.0, 2.0, 8.0])
    tanh(v).print()
    dA = algebra.Vector([0.0, 0.1, 0.2, 0.3, 0.4])
    tanh_back(dA, v).print()
