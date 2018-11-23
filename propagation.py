import algebra
import activation


def layer_forward_prop(layer_input, weight_matrix, bias_vector, act_fun):
    z = weight_matrix * layer_input + bias_vector
    return z, act_fun(z)


def net_forward_prop(layers, input, parameters, act_fun):
    layer_outputs = {}
    A_curr = input

    for l in range(layers):
        A_prev = A_curr

        W_curr = parameters['W{}'.format(l)]
        b_curr = parameters['b{}'.format(l)]
        Z_curr, A_curr = layer_forward_prop(A_prev, W_curr, b_curr, act_fun)

        layer_outputs['Z{}'.format(l)] = Z_curr
        layer_outputs['A{}'.format(l)] = A_curr

        return layer_outputs


def layer_back_prop(dA_curr, W_curr, Z_curr, A_prev, act_back, batch_size):
    dZ_curr = act_back(dA_curr, Z_curr)

    dW_curr = dZ_curr.mul_outer(A_prev) / batch_size
    db_curr = dZ_curr / batch_size
    dA_prev = W_curr.transpose() * dZ_curr

    return dW_curr, db_curr, dA_prev


if __name__ == '__main__':
    act_fun = activation.tanh
    act_back = activation.tanh_back

    A_prev = algebra.Vector([0.1, 0.2])
    W_curr = algebra.Matrix([
        [0.5, 0.6],
        [0.6, 0.7],
        [0.6, 0.5]])
    b_curr = algebra.Vector([0.4, 0.5, 0.5])

    print('\nForward prop (layer)')
    Z_curr, A_curr = layer_forward_prop(A_prev, W_curr, b_curr, act_fun)
    print('A_curr:')
    A_curr.print()

    print('\nBack prop (layer)')
    layer_gradient = algebra.Vector([0.1, 0.2, 0.1])

    dW_curr, db_curr, dA_prev = layer_back_prop(layer_gradient, W_curr, Z_curr, A_prev, act_back, 1)

    print('dW_curr:')
    dW_curr.print()
    print('db_curr:')
    db_curr.print()
    print('dA_prev:')
    dA_prev.print()
