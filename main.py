import train
import activation
import data

# Neural Network
activation_function = activation.tanh
activation_function_back = activation.tanh_back
network_architecture = [4, 5, 3]

# Hyperparameters
learning_rate = 0.0001
epochs = 1000

# Calculate metrics every `metrics_period` training iterations
metrics_period = 1000

if __name__ == '__main__':
    features, labels = data.get_training_data()
    train.train(features,
                labels,
                activation_function,
                activation_function_back,
                network_architecture,
                learning_rate,
                epochs,
                metrics_period)