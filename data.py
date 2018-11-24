import csv
import random


def encode_label(label):
    if label == 'Iris-setosa':
        return [1, 0, 0]
    elif label == 'Iris-versicolor':
        return [0, 1, 0]
    elif label == 'Iris-virginica':
        return [0, 0, 1]
    raise Exception('Unknown label')


def feature_column(features, column):
    return [feature[column] for feature in features]


def normalise_features(features):
    min0 = min(feature_column(features, 0))
    max0 = max(feature_column(features, 0))
    min1 = min(feature_column(features, 1))
    max1 = max(feature_column(features, 1))
    min2 = min(feature_column(features, 2))
    max2 = max(feature_column(features, 2))
    min3 = min(feature_column(features, 3))
    max3 = max(feature_column(features, 3))
    mins = [min0, min1, min2, min3]
    maxs = [max0, max1, max2, max3]

    for row in range(len(features)):
        for col in range(4):
            feature_max = maxs[col]
            feature_min = mins[col]
            value = features[row][col]
            norm = (value - feature_min) / (feature_max - feature_min)
            features[row][col] = norm


def get_training_data():
    """
    Read Iris dataset from 'data.csv'
    Encode labels (one-hot vectors)
    Shuffle dataset
    Normalise feature values
    :return: features, labels
    """
    features = []
    labels = []

    with open('data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        rows = [line for line in csv_reader]
        random.shuffle(rows)

        for vector in rows:
            feature_vector = [float(vector[i]) for i in range(4)]
            features.append(feature_vector)
            labels.append(encode_label(vector[4]))

    normalise_features(features)

    return features, labels


if __name__ == '__main__':
    features, labels = get_training_data()
    print(features)
    print(labels)
