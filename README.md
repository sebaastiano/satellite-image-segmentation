# Satellite Image Segmentation

Deep Learning project for satellite image segmentation using U-Net architecture developed for the final exam of the course.

## Project description

This project implements a U-Net convolutional neural network for semantic segmentation of satellite images. The model classifies each pixel into one of 6 categories to identify different land types and features in satellite imagery.

## Features

- **U-Net architecture**: custom implementation with encoder-decoder structure and skip connections
- **Data augmentation**: rotation, flipping, and shifting techniques to increase training data
- **Hyperparameter tuning**: automated tuning using Keras Tuner with RandomSearch
- **Evaluation**: F1 scores (macro, micro, weighted), accuracy, and loss metrics
- **Interactive visualization**: image and mask comparison with interactive widgets

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sebaleye/satellite-image-segmentation.git
cd satellite-image-segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Using the Jupyter notebook

The complete workflow is available in [notebooks/DL_exam.ipynb](notebooks/DL_exam.ipynb). Open it with Jupyter:

```bash
jupyter notebook notebooks/DL_exam.ipynb
```

### Using Python modules

You can also import and use the modules directly in Python scripts:

```python
from src.data import select_data_subset, augment_data
from src.models import build_model
from src.utils import evaluate_model, plot_training_history

# load data
# imgs, masks = load_your_data()

# preprocess
imgs_subset, masks_subset = select_data_subset(imgs, masks, 0.3)
imgs_aug, masks_aug = augment_data(imgs_subset, masks_subset)

# build model (requires keras_tuner.HyperParameters)
# model = build_model(hp)

# evaluate
# results = evaluate_model(model, test_images, test_masks)
```

## Model architecture

The U-Net model consists of:

- **Encoder**: 4 encoder blocks with Conv2D layers, dropout, and max pooling
- **Bottleneck**: convolutional block at the deepest level
- **Decoder**: 4 decoder blocks with Conv2DTranspose and skip connections
- **Output**: 1x1 convolution with softmax activation for 6-class segmentation

### Hyperparameters tuned

- Base filters: [8, 12, 16]
- Dropout rate: [0.1 - 0.5]
- L2 regularization: [1e-5 - 1e-2]
- Batch size: [16, 32, 64]

## Data augmentation

To improve model generalization, the following augmentation techniques are applied:

1. **Rotation**: 90°, 180°, 270°
2. **Flipping**: horizontal and vertical
3. **Shifting**: horizontal and vertical (25% with wrap mode)

## Results

The model achieves the following performance metrics on the test set:

- **Accuracy**: ~63.5%
- **F1 macro**: ~0.28
- **F1 micro**: ~0.64
- **F1 weighted**: ~0.53

Note: F1 macro is lower due to class imbalance (background dominates most images).

## Performance considerations

The current implementation uses 30% of the original dataset for computational efficiency. Performance can be improved by:

- using the full dataset
- expanding hyperparameter search space
- training for more epochs

## Data

The project expects satellite image data in the following format:
- **Images**: RGB satellite images (256x256x3)
- **Masks**: segmentation masks with 6 classes (256x256x1)

**Note**: unfortunately, the original dataset used for this project is no longer publicly available. However, the code is designed to work with any similar satellite imagery dataset that meets the above specifications. You will need to provide your own data and modify the data loading section in the notebook accordingly.

## License

This project is part of an academic examination. It is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

U-Net architecture: Ronneberger et al. (2015)