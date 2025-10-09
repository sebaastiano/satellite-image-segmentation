"""
Data preprocessing module for satellite image segmentation.
"""

import numpy as np
import tensorflow as tf
import imgaug.augmenters as iaa


def select_data_subset(imgs, masks, percentage):
    """
    Select a random subset of data based on a given percentage.

    Args:
        imgs: Array of images
        masks: Array of masks
        percentage: Float between 0 and 1 representing the percentage to select

    Returns:
        Tuple of (imgs_subset, masks_subset)
    """
    if not (0 <= percentage <= 1):
        raise ValueError("Percentage must be between 0 and 1.")

    num_samples = int(len(imgs) * percentage)

    indices = np.arange(len(imgs))
    np.random.shuffle(indices)
    selected_indices = indices[:num_samples]

    imgs_subset = imgs[selected_indices]
    masks_subset = masks[selected_indices]

    return imgs_subset, masks_subset


def rotate(image):
    """Rotate image by 90 degrees."""
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.rot90(image)
    return image.numpy()


def flip_horizontally(image):
    """Flip image horizontally."""
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.flip_left_right(image)
    return image.numpy()


def flip_vertically(image):
    """Flip image vertically."""
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.flip_up_down(image)
    return image.numpy()


def vertical_shift(image, shift_percent):
    """Shift image vertically with wrap mode."""
    shift_augmenter = iaa.Affine(translate_percent={"y": shift_percent}, mode='wrap')
    shifted_image = shift_augmenter(image=image)
    return shifted_image


def horizontal_shift(image, shift_percent):
    """Shift image horizontally with wrap mode."""
    shift_augmenter = iaa.Affine(translate_percent={"x": shift_percent}, mode='wrap')
    shifted_image = shift_augmenter(image=image)
    return shifted_image


def augment_data(imgs_subset, masks_subset, shift_percent_h=0.25, shift_percent_v=0.25):
    """
    Apply data augmentation to images and masks.

    Operations include:
    - Rotation (90, 180, 270 degrees)
    - Horizontal and vertical flipping
    - Horizontal and vertical shifting

    Args:
        imgs_subset: Array of images
        masks_subset: Array of masks
        shift_percent_h: Horizontal shift percentage (default: 0.25)
        shift_percent_v: Vertical shift percentage (default: 0.25)

    Returns:
        Tuple of (imgs_augmented, masks_augmented) - concatenated and shuffled
    """
    num_images = imgs_subset.shape[0]

    # Initialize arrays for augmented data
    imgs_rotated_90 = np.zeros_like(imgs_subset)
    masks_rotated_90 = np.zeros_like(masks_subset)
    imgs_rotated_180 = np.zeros_like(imgs_subset)
    masks_rotated_180 = np.zeros_like(masks_subset)
    imgs_rotated_270 = np.zeros_like(imgs_subset)
    masks_rotated_270 = np.zeros_like(masks_subset)
    imgs_flipped_h = np.zeros_like(imgs_subset)
    masks_flipped_h = np.zeros_like(masks_subset)
    imgs_flipped_v = np.zeros_like(imgs_subset)
    masks_flipped_v = np.zeros_like(masks_subset)
    imgs_shifted_h = np.zeros_like(imgs_subset)
    masks_shifted_h = np.zeros_like(masks_subset)
    imgs_shifted_v = np.zeros_like(imgs_subset)
    masks_shifted_v = np.zeros_like(masks_subset)

    # Apply rotations
    for i in range(num_images):
        imgs_rotated_90[i] = rotate(imgs_subset[i])
        masks_rotated_90[i] = rotate(masks_subset[i])

    for i in range(num_images):
        imgs_rotated_180[i] = rotate(imgs_rotated_90[i])
        masks_rotated_180[i] = rotate(masks_rotated_90[i])

    for i in range(num_images):
        imgs_rotated_270[i] = rotate(imgs_rotated_180[i])
        masks_rotated_270[i] = rotate(masks_rotated_180[i])

    # Apply flipping
    for i in range(num_images):
        imgs_flipped_h[i] = flip_horizontally(imgs_subset[i])
        masks_flipped_h[i] = flip_horizontally(masks_subset[i])

    for i in range(num_images):
        imgs_flipped_v[i] = flip_vertically(imgs_subset[i])
        masks_flipped_v[i] = flip_vertically(masks_subset[i])

    # Apply shifting
    for i in range(num_images):
        imgs_shifted_h[i] = horizontal_shift(imgs_subset[i], shift_percent_h)
        masks_shifted_h[i] = horizontal_shift(masks_subset[i], shift_percent_h)

    for i in range(num_images):
        imgs_shifted_v[i] = vertical_shift(imgs_subset[i], shift_percent_v)
        masks_shifted_v[i] = vertical_shift(masks_subset[i], shift_percent_v)

    # Concatenate all augmented data
    imgs_aug = np.concatenate((imgs_subset, imgs_rotated_90, imgs_rotated_180, imgs_rotated_270,
                               imgs_flipped_h, imgs_flipped_v, imgs_shifted_v, imgs_shifted_h), axis=0)
    masks_aug = np.concatenate((masks_subset, masks_rotated_90, masks_rotated_180, masks_rotated_270,
                                masks_flipped_h, masks_flipped_v, masks_shifted_v, masks_shifted_h), axis=0)

    # Shuffle consistently
    indices = np.arange(imgs_aug.shape[0])
    np.random.shuffle(indices)
    imgs_aug = imgs_aug[indices]
    masks_aug = masks_aug[indices]

    return imgs_aug, masks_aug
