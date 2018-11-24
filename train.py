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


def train(X, Y):
    act_fun = activation.tanh
    act_fun_back = activation.tanh_back
    architecture = [2, 3, 2]
    layers = len(architecture)
    learning_rate = 0.01
    epochs = 1

    params = init_params(architecture)

    for epoch in range(epochs):
        for example_idx in range(len(X)):
            x = algebra.Vector(X[example_idx])
            y = algebra.Vector(Y[example_idx])

            # Forward prop
            y_hat, layer_outputs = propagation.net_forward_prop(layers, x, params, act_fun)

            # Metrics
            cross_entropy = metrics.cross_entropy(y_hat, y)
            y_hat_tests = []
            for test_idx in range(len(X)):
                x_test = algebra.Vector(X[test_idx])
                y_hat_test, _ = propagation.net_forward_prop(layers, x_test, params, act_fun)
                y_hat_tests.append(y_hat_test.vector)
            accuracy = metrics.accuracy(y_hat_tests, Y)
            print('L: {}\t\tA: {}'.format(cross_entropy, accuracy))

            # Backprop
            output_gradient = propagation.output_gradient(y, y_hat)
            param_gradients = propagation.net_back_prop(layers, layer_outputs, output_gradient, params, act_fun_back)

            # Weight update
            update_params(layers, params, param_gradients, learning_rate)
