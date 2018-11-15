import algebra
import activation


def layer_forward_prop(layer_input, weight_matrix, bias_vector, activation_function):
    z = weight_matrix * layer_input + bias_vector
    return activation_function(z)


# TODO back prob

if __name__ == '__main__':
    input = algebra.Vector([0.1, 0.2])
    weights = algebra.Matrix([[1, 2], [2, 1]])
    bias = algebra.Vector([1, 1])
    activation_function = activation.tanh

    output = layer_forward_prop(input, weights, bias, activation_function)
    output.print()
