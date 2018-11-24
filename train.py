import activation
import algebra
import metrics
import propagation
import random


def random_param():
    return random.uniform(0, 1.0)


def init_params(architecture):
    params = {}
    for i in range(1, len(architecture)):
        curr_size = architecture[i]
        prev_size = architecture[i - 1]

        # weight matrix (prev_size x curr_size)
        weights = algebra.Matrix([[random_param() for _ in range(prev_size)] for _ in range(curr_size)])
        params['W{}'.format(i)] = weights

        # bias vector (curr_size)
        biases = algebra.Vector([random_param() for _ in range(curr_size)])
        params['b{}'.format(i)] = biases

    return params


def update_params(layers, params, param_gradients, learning_rate):
    for l in range(1, layers):
        params['W{}'.format(l)] -= param_gradients['dW{}'.format(l)] * learning_rate
        params['b{}'.format(l)] -= param_gradients['db{}'.format(l)] * learning_rate


# TODO
def train(X, Y, nn_architecture, epochs, learning_rate):
    act_fun = activation.tanh
    act_fun_back = activation.tanh_back
    architecture = [3, 4, 3]
    layers = len(architecture)
    learning_rate = 0.0001

    params = init_params(architecture)

    for epoch in range(epochs):
        for example_idx in range(X):
            x = X[example_idx]
            y = Y[example_idx]

            y_hat, layer_outputs = propagation.net_forward_prop(layers, x, params, act_fun)

            # TODO metrics
            # TODO dimensions

            output_gradient = propagation.output_gradient(y, y_hat)

            param_gradients = propagation.net_back_prop(layers, layer_outputs, output_gradient, params, act_fun_back)

            update_params(layers, params, param_gradients, learning_rate)
