"""
Evaluation utilities for satellite image segmentation.
"""

import numpy as np
from sklearn.metrics import f1_score


def calculate_f1_scores(true_masks, predicted_masks):
    """
    Calculate F1 scores (macro, micro, weighted) for segmentation.

    Args:
        true_masks: Array of ground truth masks
        predicted_masks: Array of predicted masks

    Returns:
        Dictionary with f1_macro, f1_micro, f1_weighted
    """
    # Flatten arrays
    true_flat = true_masks.flatten()
    pred_flat = predicted_masks.flatten()

    # Calculate F1 scores
    f1_macro = f1_score(true_flat, pred_flat, average='macro')
    f1_micro = f1_score(true_flat, pred_flat, average='micro')
    f1_weighted = f1_score(true_flat, pred_flat, average='weighted')

    return {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted
    }


def evaluate_model(model, test_images, test_masks):
    """
    Evaluate model on test data.

    Args:
        model: Trained Keras model
        test_images: Test images
        test_masks: Test masks

    Returns:
        Dictionary with loss, accuracy, predictions, and F1 scores
    """
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(test_images, test_masks)

    # Get predictions
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=-1)

    # Calculate F1 scores
    f1_scores = calculate_f1_scores(test_masks, predicted_labels)

    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'predictions': predicted_labels,
        **f1_scores
    }
