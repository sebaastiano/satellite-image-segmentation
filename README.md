# Satellite Image Segmentation

Deep Learning project for satellite image segmentation using U-Net architecture developed for the final exam of the course.

## Project Description

This project implements a U-Net convolutional neural network for semantic segmentation of satellite images. The model classifies each pixel into one of 6 categories to identify different land types and features in satellite imagery.

## Features

- **U-Net Architecture**: custom implementation with encoder-decoder structure and skip connections
- **Data Augmentation**: rotation, flipping, and shifting techniques to increase training data
- **Hyperparameter Tuning**: automated tuning using Keras Tuner with RandomSearch
- **Evaluation**: F1 scores (macro, micro, weighted), accuracy, and loss metrics
- **Interactive Visualization**: image and mask comparison with interactive widgets

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

### Using the Jupyter Notebook

The complete workflow is available in [notebooks/DL_exam.ipynb](notebooks/DL_exam.ipynb). Open it with Jupyter:

```bash
jupyter notebook notebooks/DL_exam.ipynb
```

### Using Python Modules

You can also import and use the modules directly in your Python scripts:

```python
from src.data import select_data_subset, augment_data
from src.models import build_model
from src.utils import evaluate_model, plot_training_history

# Load your data
# imgs, masks = load_your_data()

# Preprocess
imgs_subset, masks_subset = select_data_subset(imgs, masks, 0.3)
imgs_aug, masks_aug = augment_data(imgs_subset, masks_subset)

# Build model (requires keras_tuner.HyperParameters)
# model = build_model(hp)

# Evaluate
# results = evaluate_model(model, test_images, test_masks)
```

## Model Architecture

The U-Net model consists of:

- **Encoder**: 4 encoder blocks with Conv2D layers, dropout, and max pooling
- **Bottleneck**: Convolutional block at the deepest level
- **Decoder**: 4 decoder blocks with Conv2DTranspose and skip connections
- **Output**: 1x1 convolution with softmax activation for 6-class segmentation

### Hyperparameters Tuned

- Base filters: [8, 12, 16]
- Dropout rate: [0.1 - 0.5]
- L2 regularization: [1e-5 - 1e-2]
- Batch size: [16, 32, 64]

## Data Augmentation

To improve model generalization, the following augmentation techniques are applied:

1. **Rotation**: 90°, 180°, 270°
2. **Flipping**: Horizontal and vertical
3. **Shifting**: Horizontal and vertical (25% with wrap mode)

## Results

The model achieves the following performance metrics on the test set:

- **Test Accuracy**: ~63.5%
- **F1 Macro**: ~0.28
- **F1 Micro**: ~0.64
- **F1 Weighted**: ~0.53

Note: F1 macro is lower due to class imbalance (background dominates most images).

## Performance Considerations

The current implementation uses 30% of the original dataset for computational efficiency. Performance can be improved by:

- using the full dataset
- expanding hyperparameter search space
- training for more epochs

## Data

The project expects satellite image data in the following format:
- **Images**: RGB satellite images (256x256x3)
- **Masks**: Segmentation masks with 6 classes (256x256x1)

**Note**: Unfortunately, the original dataset used for this project is no longer publicly available. However, the code is designed to work with any similar satellite imagery dataset that meets the above specifications. You will need to provide your own data and modify the data loading section in the notebook accordingly.

## License

This project is part of an academic examination. It is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

U-Net architecture: Ronneberger et al. (2015)