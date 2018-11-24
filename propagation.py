import algebra
import activation


def layer_forward_prop(layer_input, weight_matrix, bias_vector, act_fun):
    z = weight_matrix * layer_input + bias_vector
    return z, act_fun(z)


def net_forward_prop(layers, net_input, params, act_fun):
    layer_outputs = {}
    A_curr = net_input

    for l in range(1, layers):
        A_prev = A_curr

        W_curr = params['W{}'.format(l)]
        b_curr = params['b{}'.format(l)]
        Z_curr, A_curr = layer_forward_prop(A_prev, W_curr, b_curr, act_fun)

        layer_outputs['A{}'.format(l - 1)] = A_prev
        layer_outputs['Z{}'.format(l)] = Z_curr

    return A_curr, layer_outputs


def layer_back_prop(dA_curr, W_curr, Z_curr, A_prev, act_back):
    m = len(A_prev)

    dZ_curr = act_back(dA_curr, Z_curr)

    dW_curr = dZ_curr.mul_outer(A_prev) / m
    db_curr = dZ_curr / m
    dA_prev = W_curr.transpose() * dZ_curr

    return dW_curr, db_curr, dA_prev


def net_back_prop(layers, layer_outputs, output_gradient, params, act_back):
    param_gradients = {}
    dA_prev = output_gradient

    for l in reversed(range(1, layers)):
        dA_curr = dA_prev

        W_curr = params['W{}'.format(l)]
        Z_curr = layer_outputs['Z{}'.format(l)]
        A_prev = layer_outputs['A{}'.format(l - 1)]

        dW_curr, db_curr, dA_prev = layer_back_prop(dA_curr, W_curr, Z_curr, A_prev, act_back)

        param_gradients['dW{}'.format(l)] = dW_curr
        param_gradients['db{}'.format(l)] = db_curr

    return param_gradients


def output_gradient(y_vector, y_hat_vector):
    gradient = []
    for i in range(len(y_vector)):
        y = y_vector[i]
        y_hat = y_hat_vector[i]
        g = -((y / y_hat) - ((1 - y) / (1 - y_hat)))
        gradient.append(g)
    return algebra.Vector(gradient)


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

    dW_curr, db_curr, dA_prev = layer_back_prop(layer_gradient, W_curr, Z_curr, A_prev, act_back)

    print('dW_curr:')
    dW_curr.print()
    print('db_curr:')
    db_curr.print()
    print('dA_prev:')
    dA_prev.print()
