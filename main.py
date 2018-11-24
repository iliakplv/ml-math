import random

import activation
import data
import metrics
import train

# input layer size must match the number of features
# output layer size must match the number of classes
network_architecture = [4, 5, 3]
activation_function = activation.tanh
activation_function_back = activation.tanh_back

loss_metric = metrics.mse
learning_rate = 0.000001
training_epochs = 10000

metrics_period = 10000  # calculate metrics every `metrics_period` iterations
test_examples = 5  # number of test examples for the trained network

if __name__ == '__main__':
    features, labels = data.get_training_data()

    train.train(features,
                labels,
                activation_function,
                activation_function_back,
                network_architecture,
                loss_metric,
                learning_rate,
                training_epochs,
                metrics_period)

    test_features = []
    test_labels = []
    for i in range(test_examples):
        r = random.randint(0, len(features))
        test_features.append(features[r])
        test_labels.append(labels[r])
    train.predict(test_features, test_labels)

    # train.print_params()
