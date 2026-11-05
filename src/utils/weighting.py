# Number of training reviews per score (ratings 1-6 mapped to [-1, 1])
train_class_counts = {
    -1.0: 379,
    -0.6: 2287,
    -0.2: 6004,
     0.2: 11304,
     0.6: 12614,
     1.0: 2161
}

# Inverse-frequency weights, so rarer scores count more in the loss
total_train_samples = sum(train_class_counts.values())
class_weights = {interval: total_train_samples / count for interval, count in train_class_counts.items()}
normalized_class_weights = {interval: weight / len(train_class_counts) for interval, weight in class_weights.items()}


def get_weight(scaled_score):
    rounded_score = round(scaled_score, 1)  # guard against floating point noise
    return normalized_class_weights[rounded_score]
