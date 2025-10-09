"""
Visualization utilities for satellite image segmentation.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history):
    """
    Plot training and validation loss and accuracy.

    Args:
        history: Keras History object from model.fit()
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot loss
    ax[0].plot(history.history['loss'], label='Training Loss')
    ax[0].plot(history.history['val_loss'], label='Validation Loss')
    ax[0].set_title('Model Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='upper left')
    ax[0].grid(True)

    # Plot accuracy
    if 'accuracy' in history.history:
        ax[1].plot(history.history['accuracy'], label='Training Accuracy')
        ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax[1].set_title('Model Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend(loc='upper left')
        ax[1].grid(True)

    plt.tight_layout()
    plt.show()


def display_sample(imgs, masks, predictions, index):
    """
    Display a sample image with its true mask and predicted mask.

    Args:
        imgs: Array of images
        masks: Array of true masks
        predictions: Array of predicted masks
        index: Index of the sample to display
    """
    plt.figure(figsize=(12, 4))

    # Image
    plt.subplot(1, 3, 1)
    plt.title('Image')
    plt.imshow(imgs[index])
    plt.axis('off')

    # True mask
    plt.subplot(1, 3, 2)
    plt.title('True Mask')
    plt.imshow(masks[index])
    plt.axis('off')

    # Predicted mask
    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask')
    plt.imshow(predictions[index])
    plt.axis('off')

    plt.show()


def visualize_augmentation(original_img, augmented_imgs, titles):
    """
    Visualize original and augmented images.

    Args:
        original_img: Original image
        augmented_imgs: List of augmented images
        titles: List of titles for each augmented image
    """
    n_images = len(augmented_imgs) + 1
    plt.figure(figsize=(15, 3))

    plt.subplot(1, n_images, 1)
    plt.title('Original')
    plt.imshow(original_img)
    plt.axis('off')

    for i, (img, title) in enumerate(zip(augmented_imgs, titles)):
        plt.subplot(1, n_images, i + 2)
        plt.title(title)
        plt.imshow(img)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
