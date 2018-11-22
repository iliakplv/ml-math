import algebra
import activation


def layer_forward_prop(layer_input, weight_matrix, bias_vector, act_fun):
    z = weight_matrix * layer_input + bias_vector
    return z, act_fun(z)


def layer_back_prop(dA_curr, W_curr, Z_curr, A_prev, act_back, batch_size):
    dZ_curr = act_back(dA_curr, Z_curr)

    dW_curr = dZ_curr.mul_outer(A_prev) / batch_size
    db_curr = dZ_curr / batch_size
    dA_prev = W_curr.transpose() * dZ_curr

    return dW_curr, db_curr, dA_prev


if __name__ == '__main__':
    input = algebra.Vector([0.1, 0.2])
    weights = algebra.Matrix([[1, 2], [2, 1]])
    bias = algebra.Vector([1, 1])
    act_fun = activation.tanh

    z, a = layer_forward_prop(input, weights, bias, act_fun)
    z.print()
    a.print()

    # todo test layer_back_prop
