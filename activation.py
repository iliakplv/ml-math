import math
import algebra


def tanh(z):
    return algebra.Vector([math.tanh(item) for item in z.vector])


def tanh_back(dA, z):
    t = tanh(z)
    t_sqr_list = algebra.list_mul(t.vector, t.vector)
    return algebra.Vector([dA * (1.0 - item) for item in t_sqr_list])


if __name__ == '__main__':
    v = algebra.Vector([0.0, 0.5, 1.0, 2.0, 8.0])
    tanh(v).print()
