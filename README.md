# Satellite Image Segmentation

Deep Learning project for satellite image segmentation using U-Net architecture.

**Author:** Sebastiano Pietrasanta (513054)
**Course:** Deep Learning Exam - July 16th 2024

## Project Description

This project implements a U-Net convolutional neural network for semantic segmentation of satellite images. The model classifies each pixel into one of 6 categories to identify different land types and features in satellite imagery.

## Features

- **U-Net Architecture**: Custom implementation with encoder-decoder structure and skip connections
- **Data Augmentation**: Rotation, flipping, and shifting techniques to increase training data
- **Hyperparameter Tuning**: Automated tuning using Keras Tuner with RandomSearch
- **Comprehensive Evaluation**: F1 scores (macro, micro, weighted), accuracy, and loss metrics
- **Interactive Visualization**: Image and mask comparison with interactive widgets

## Project Structure

```
satellite_image_segmentation/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py       # Data loading, subset selection, and augmentation
│   ├── models/
│   │   ├── __init__.py
│   │   └── unet.py                # U-Net model architecture
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── evaluation.py          # Model evaluation metrics
│   │   └── visualization.py       # Plotting and visualization tools
│   └── __init__.py
├── notebooks/
│   └── DL_exam.ipynb              # Original Jupyter notebook
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd satellite_image_segmentation
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

- Using the full dataset
- Increasing model complexity
- Expanding hyperparameter search space
- Training for more epochs
- Using advanced regularization techniques

## Data

The project uses satellite image data downloaded from GitHub. The data contains:
- **Images**: RGB satellite images (256x256x3)
- **Masks**: Segmentation masks with 6 classes (256x256x1)

Data is automatically downloaded when running the notebook.

## License

This project is part of an academic examination.

## Acknowledgments

- Deep Learning course instructors
- U-Net architecture: Ronneberger et al. (2015)
- Keras and TensorFlow teams
