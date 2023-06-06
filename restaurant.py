import numpy as np
import pandas as pd
from collections import Counter
from math import log2


# Function to calculate entropy
def calculate_entropy(target_values):
    histogram = np.bincount(target_values)
    probabilities = histogram / len(target_values)
    return -np.sum([probability * log2(probability) for probability in probabilities if probability > 0])


# Function to calculate information gain
def calculate_information_gain(feature_values, target_values, split_value):
    target_values_less_than_split = target_values[feature_values < split_value]
    target_values_greater_than_split = target_values[feature_values >= split_value]
    return calculate_entropy(target_values) - \
        (len(target_values_less_than_split) * calculate_entropy(target_values_less_than_split) +
         len(target_values_greater_than_split) * calculate_entropy(target_values_greater_than_split)) / len(
            target_values)


# Function to find the best split
def find_best_split(feature_values, target_values):
    best_information_gain = -1
    best_split_value = -1
    for split_value in np.unique(feature_values):
        information_gain = calculate_information_gain(feature_values, target_values, split_value)
        if information_gain > best_information_gain:
            best_information_gain = information_gain
            best_split_value = split_value
    return best_split_value, best_information_gain


class TreeNode:
    def __init__(self, feature_name="", feature_value=None, true_branch=None, false_branch=None):
        self.feature_name = feature_name
        self.feature_value = feature_value
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_decision_tree(dataframe, target_values, unused_features):
    # Base case: if there are no more unused features, return a leaf node
    if len(unused_features) == 0:
        return TreeNode(feature_name="Leaf", feature_value=Counter(target_values))

    # Base case: if all the target values are the same, return a leaf node
    if len(set(target_values)) == 1:
        return TreeNode(feature_name="Leaf", feature_value=target_values.iloc[0])

    best_information_gain = -1
    best_split_value = -1
    best_feature = None

    # For each unused feature, calculate the information gain if we split on it
    for feature in unused_features:
        split_value, information_gain = find_best_split(dataframe[feature].values, target_values)
        if information_gain > best_information_gain:
            best_information_gain = information_gain
            best_split_value = split_value
            best_feature = feature

    # If we couldn't find a good split, return a leaf node
    if best_information_gain == -1:
        return TreeNode(feature_name="Leaf", feature_value=Counter(target_values))

    # Otherwise, split the dataset and recursively build the true and false branches
    unused_features.remove(best_feature)
    true_indices = dataframe[best_feature] < best_split_value
    false_indices = dataframe[best_feature] >= best_split_value
    true_branch = build_decision_tree(dataframe[true_indices], target_values[true_indices], unused_features.copy())
    false_branch = build_decision_tree(dataframe[false_indices], target_values[false_indices], unused_features.copy())

    return TreeNode(feature_name=best_feature, feature_value=best_split_value, true_branch=true_branch,
                    false_branch=false_branch)


def classify(tree, test_data):
    # Convert the test data to a list of dictionaries
    test_data_dicts = test_data.to_dict(orient='records')

    # Define a helper function for classifying a single instance
    def classify_instance(tree, instance):
        # If this is a leaf node, return its value
        if tree.feature_name == "Leaf":
            # If the feature_value is a Counter (which means multiple target values were present in this leaf),
            # return the most common target value
            if isinstance(tree.feature_value, Counter):
                return tree.feature_value.most_common(1)[0][0]
            # Otherwise, return the target value directly
            else:
                return tree.feature_value

        # If this is not a leaf node, move to the next level of the tree
        if instance[tree.feature_name] < tree.feature_value:
            return classify_instance(tree.true_branch, instance)
        else:
            return classify_instance(tree.false_branch, instance)

    # Use the classify_instance function to predict the class label for each instance in the test data
    predictions = [classify_instance(tree, instance) for instance in test_data_dicts]

    return predictions


def main():
    # Training Dataset
    restaurant_data = np.array([
        ['yes', 'no', 'no', 'yes', 'some', '$$$', 'no', 'yes', 'french', '0-10', 'yes'],
        ['yes', 'no', 'no', 'yes', 'full', '$', 'no', 'no', 'thai', '30-60', 'no'],
        ['no', 'yes', 'no', 'no', 'some', '$', 'no', 'no', 'burger', '0-10', 'yes'],
        ['yes', 'no', 'yes', 'yes', 'full', '$', 'yes', 'no', 'thai', '10-30', 'yes'],
        ['yes', 'no', 'yes', 'no', 'full', '$$$', 'no', 'yes', 'french', '>60', 'no'],
        ['no', 'yes', 'no', 'yes', 'some', '$$', 'yes', 'yes', 'italian', '0-10', 'yes'],
        ['no', 'yes', 'no', 'no', 'none', '$', 'yes', 'no', 'burger', '0-10', 'no'],
        ['no', 'no', 'no', 'yes', 'some', '$$', 'yes', 'yes', 'thai', '0-10', 'yes'],
        ['no', 'yes', 'yes', 'no', 'full', '$', 'yes', 'no', 'burger', '>60', 'no'],
        ['yes', 'yes', 'yes', 'yes', 'full', '$$$', 'no', 'yes', 'italian', '10-30', 'no'],
        ['no', 'no', 'no', 'no', 'none', '$', 'no', 'no', 'thai', '0-10', 'no'],
        ['yes', 'yes', 'yes', 'yes', 'full', '$', 'no', 'no', 'burger', '30-60', 'yes']
    ])

    # Column names
    column_names = ['alternate', 'bar', 'Fri/Sun', 'hungry', 'patrons', 'price', 'rain', 'res', 'type', 'wait-est',
                    'will-wait']

    # Create DataFrame
    df = pd.DataFrame(restaurant_data, columns=column_names)

    # Convert the 'will-wait' column to binary values
    df['will-wait'] = df['will-wait'].map({'yes': 1, 'no': 0})

    # Separate the features from the target
    features = df.drop('will-wait', axis=1)
    target_values = df['will-wait'].reset_index(drop=True)

    # Get the list of feature names
    feature_names = list(features.columns)

    # Build the decision tree
    decision_tree = build_decision_tree(features, target_values, feature_names)

    # Mock test data
    mock_data = pd.DataFrame([
        {'alternate': 'yes', 'bar': 'no', 'Fri/Sun': 'no', 'hungry': 'yes', 'patrons': 'some', 'price': '$$$',
         'rain': 'no',
         'res': 'yes', 'type': 'french', 'wait-est': '0-10'},
        {'alternate': 'yes', 'bar': 'no', 'Fri/Sun': 'no', 'hungry': 'yes', 'patrons': 'full', 'price': '$',
         'rain': 'no',
         'res': 'no', 'type': 'thai', 'wait-est': '30-60'},
        {'alternate': 'no', 'bar': 'yes', 'Fri/Sun': 'no', 'hungry': 'no', 'patrons': 'some', 'price': '$',
         'rain': 'no',
         'res': 'no', 'type': 'burger', 'wait-est': '0-10'},
        {'alternate': 'yes', 'bar': 'no', 'Fri/Sun': 'yes', 'hungry': 'yes', 'patrons': 'full', 'price': '$',
         'rain': 'yes',
         'res': 'no', 'type': 'thai', 'wait-est': '10-30'},
        {'alternate': 'yes', 'bar': 'no', 'Fri/Sun': 'yes', 'hungry': 'no', 'patrons': 'full', 'price': '$$$',
         'rain': 'no',
         'res': 'yes', 'type': 'french', 'wait-est': '>60'},
        {'alternate': 'no', 'bar': 'yes', 'Fri/Sun': 'no', 'hungry': 'yes', 'patrons': 'some', 'price': '$$',
         'rain': 'yes',
         'res': 'yes', 'type': 'italian', 'wait-est': '0-10'},
        {'alternate': 'no', 'bar': 'yes', 'Fri/Sun': 'no', 'hungry': 'no', 'patrons': 'none', 'price': '$',
         'rain': 'yes',
         'res': 'no', 'type': 'burger', 'wait-est': '0-10'},
        {'alternate': 'no', 'bar': 'no', 'Fri/Sun': 'no', 'hungry': 'yes', 'patrons': 'some', 'price': '$$',
         'rain': 'yes',
         'res': 'yes', 'type': 'thai', 'wait-est': '0-10'},
        {'alternate': 'no', 'bar': 'yes', 'Fri/Sun': 'yes', 'hungry': 'no', 'patrons': 'full', 'price': '$',
         'rain': 'yes',
         'res': 'no', 'type': 'burger', 'wait-est': '>60'},
        {'alternate': 'yes', 'bar': 'yes', 'Fri/Sun': 'yes', 'hungry': 'yes', 'patrons': 'full', 'price': '$$$',
         'rain': 'no', 'res': 'yes', 'type': 'italian', 'wait-est': '10-30'},
    ])

    predictions = classify(decision_tree, mock_data)

    # Create a list of lists with customer number and prediction
    formatted_predictions = [["Customer " + str(i + 1), "Will wait" if pred == 1 else "Will not wait"] for i, pred in
                             enumerate(predictions)]

    # Print the formatted predictions
    for pred in formatted_predictions:
        print(pred)


if __name__ == "__main__":
    main()
