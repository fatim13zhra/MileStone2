# Medical Image Segmentation with Encoder-Decoder Architecture

This repository contains a deep learning-based medical image segmentation system developed at the [Budapest University of Technology and Economics](https://www.bme.hu/). The system utilizes an encoder-decoder architecture, enhanced with Atrous Spatial Pyramid Pooling (ASPP) and Batch Normalization, to improve segmentation of cardiac MRI images and other medical imaging datasets.

## Model Overview

The core of this model is an encoder-decoder architecture designed to process medical images and generate segmentation maps. The encoder extracts features through convolutional layers, while the decoder upscales the feature maps to produce segmentations at the original image resolution.

### Key Features:
- **Encoder-Decoder Architecture**: The encoder down-samples the input to capture hierarchical features, while the decoder upsamples to generate segmentation outputs.
- **Atrous Spatial Pyramid Pooling (ASPP)**: A multi-scale context aggregation technique that uses dilated convolutions to capture features at different scales.
- **Batch Normalization**: Helps stabilize training and improves convergence by normalizing the outputs of each layer.

### Model Components:
- **Encoder**: A series of convolutional layers that reduce spatial dimensions, extracting high-level features.
- **Decoder**: Uses upsampling layers to restore the input image's dimensions, predicting the segmentation map.
- **ASPP Module**: Applied in the encoder to capture long-range contextual information through dilated convolutions.
- **Batch Normalization**: Applied to all layers to normalize activations, which helps in faster convergence during training.

## Installation

### Prerequisites

You need the following dependencies installed to run this project:

- Python 3.6+
- PyTorch 1.7.0+
- NumPy
- OpenCV
- Matplotlib
- Other dependencies can be found in the `requirements.txt` file.

To install the necessary Python dependencies, run:

```bash
pip install -r requirements.txt
```

### Setting Up the Environment

1. **Download Dataset**: The dataset used for training the model (e.g., cardiac MRI images) must be obtained and organized into the appropriate directory structure. Please follow the dataset guidelines provided.

2. **Install Required Packages**:
   - For the required system dependencies, run:
     ```bash
     sudo apt-get install libopencv-dev
     ```

## Directory Structure

```
.
├── data
│   └── train
│       └── images
│       └── masks
│   └── val
│       └── images
│       └── masks
├── src
│   ├── model.py                  # Model definition (Encoder-Decoder + ASPP)
│   ├── train.py                  # Script to train the model
│   ├── evaluate.py               # Script to evaluate the model
│   ├── utils.py                  # Utility functions for data processing
├── config
│   ├── settings.cfg              # Configuration settings for training and evaluation
└── requirements.txt              # Python dependencies
```

## Usage

### Training the Model

To train the model on your dataset, run:

```bash
python src/train.py --config config/settings.cfg
```

This will start the training process. Make sure the `settings.cfg` file contains paths to your dataset and training parameters like learning rate, batch size, and epochs.

### Evaluating the Model

After training the model, you can evaluate it on the test set using the following command:

```bash
python src/evaluate.py --config config/settings.cfg
```

This will generate evaluation metrics like accuracy, IoU (Intersection over Union), and Dice coefficient, which will help in assessing the model's performance.

### Synthesizing Segmentation Results

To generate segmentation results for new images, use the following script:

```bash
python src/synthesize.py --input_image <path_to_image> --output_dir <path_to_output_directory>
```

This will produce the segmented output for the given image and save it to the specified output directory.

## Training Parameters

- **Learning Rate**: `0.001`
- **Batch Size**: `16`
- **Epochs**: `50`
- **Loss Function**: Cross-entropy loss
- **Optimizer**: Adam

## Results

This section describes the performance of the trained model on the test set, including evaluation metrics such as:
- **Accuracy**: Measures the overall correctness of the segmentation.
- **IoU (Intersection over Union)**: Measures the overlap between predicted and true segmentation regions.
- **Dice Coefficient**: A similarity measure between predicted and ground truth segmentation.

## Next Steps

1. **Fine-Tuning**: You can fine-tune the model on specific subsets of your dataset to improve performance on niche medical images.
2. **Enhancing Model Performance**: Experiment with advanced techniques such as attention mechanisms, GAN-based segmentation, or hybrid architectures.
3. **Deployment**: The model can be further optimized and deployed in a clinical setting for automatic medical image analysis.

## Contribution

Feel free to fork this repository, raise issues, and submit pull requests. Contributions to improve the performance, documentation, and functionality are welcome!

## Citation

If you use this work in your research or application, please cite it as follows:

- Ezzahra F.A. (2024). "Medical Image Segmentation using Encoder-Decoder Architecture and ASPP". [Accepted to Medical Image Processing Conference 2024].

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, suggestions, or issues, please feel free to create an issue on GitHub or contact the authors directly.

---

This project is developed and maintained at the [Budapest University of Technology and Economics](https://www.bme.hu/), Faculty of Electrical Engineering and Informatics.
```
