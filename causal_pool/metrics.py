import numpy as np
from causal_pool.prompt_utils import letter_to_index


def calculate_per_question_accuracy(label, pred):
    """
    Computes the per-question accuracy for a given question.
    """
    return int(pred == label)

def calculate_per_option_accuracy(num_options, label, pred):
    """
    Computes the per-option accuracy for a given question.
    This is basically the hamming distance between the label and the prediction.
    Problems:
        1. Inflated by easy negatives (TN dominance)
        2. Conservative predictions are rewarded more than aggressive ones
    Examples:
        - num_options = 4, label = "AB", pred = "AC" -> 3/4
        - num_options = 4, label = "AB", pred = "ABC" -> 3/4
        - num_options = 4, label = "AB", pred = "CD" -> 0
        - num_options = 4, label = "AB", pred = "AB" -> 1
        - num_options = 4, label = "AB", pred = "A" -> 3/4
        - num_options = 4, label = "AB", pred = "ABCD" -> 2/4
        - num_options = 3, label = "AB", pred = "D" -> 0 (out of bounds)
    """
    pred_array = np.zeros(num_options)
    for choice in pred:
        index = letter_to_index(choice)
        # If index is out of bounds, treat as completely wrong (return 0.0)
        if index >= num_options:
            return 0.0
        if pred_array[index] == 1:
            return 0
        pred_array[index] = 1

    label_array = np.zeros(num_options)
    for choice in label:
        label_array[letter_to_index(choice)] = 1

    return np.sum(label_array == pred_array) / num_options
