import random

import algebra
import metrics
import propagation

memory = {}


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


def train(X, Y, act_fun, act_fun_back, architecture, loss_metric, learning_rate, epochs, metrics_period):
    layers = len(architecture)
    params = init_params(architecture)

    iterations = 0

    for epoch in range(epochs):
        for example_idx in range(len(X)):
            x = algebra.Vector(X[example_idx])
            y = algebra.Vector(Y[example_idx])

            y_hat, layer_outputs = propagation.net_forward_prop(layers, x, params, act_fun)

            output_gradient = propagation.output_gradient(y, y_hat)

            param_gradients = propagation.net_back_prop(layers, layer_outputs, output_gradient, params, act_fun_back)

            update_params(layers, params, param_gradients, learning_rate)

            iterations += 1

            # Metrics
            if iterations % metrics_period == 0:
                m_y_hat_list = []
                for m_idx in range(len(X)):
                    m_x = algebra.Vector(X[m_idx])
                    m_y_hat, _ = propagation.net_forward_prop(layers, m_x, params, act_fun)
                    m_y_hat_list.append(m_y_hat.vector)
                loss = metrics.loss_function(m_y_hat_list, Y, loss_metric)
                accuracy = metrics.accuracy(m_y_hat_list, Y)
                print('Epoch: {}\tIter: {}k\t\tLoss: {}\t\tAccuracy: {}'.format(
                    epoch + 1, iterations / 1000, loss, accuracy))

    memory['layers'] = layers
    memory['params'] = params
    memory['act_fun'] = act_fun


def predict(X, Y):
    if memory:
        layers = memory['layers']
        params = memory['params']
        act_fun = memory['act_fun']

        y_hats = []

        for example_idx in range(len(X)):
            x = algebra.Vector(X[example_idx])
            y = algebra.Vector(Y[example_idx])
            y_hat, _ = propagation.net_forward_prop(layers, x, params, act_fun)
            y_hats.append(y_hat.vector)
            print('\nExample #{}'.format(example_idx))
            y.print()
            y_hat.print()

        accuracy = metrics.accuracy(y_hats, Y)
        print('\nAccuracy: {}'.format(accuracy))


def print_params():
    layers = memory['layers']
    params = memory['params']

    for l in range(1, layers):
        w_index = 'W{}'.format(l)
        w = params[w_index]
        print('\n' + w_index)
        w.print()

        b_index = 'b{}'.format(l)
        b = params[b_index]
        print('\n' + b_index)
        b.print()
