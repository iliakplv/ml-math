import math
import algebra


def tanh(z):
    return algebra.Vector([math.tanh(item) for item in z.vector])


def tanh_back(dA, z):
    ones = algebra.Vector([1.0 for _ in range(len(z))])
    t = tanh(z)
    dt = ones - t.mul_element_wise(t)  # dt/dx = 1 - tanh(x)^2
    return dA.mul_element_wise(dt)


if __name__ == '__main__':
    v = algebra.Vector([0.0, 0.5, 1.0, 2.0, 8.0])
    tanh(v).print()
    dA = algebra.Vector([0.0, 0.1, 0.2, 0.3, 0.4])
    tanh_back(dA, v).print()
