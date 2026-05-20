# Face Mask Detection using CNN and VGG16

This project implements a deep learning-based face mask detection system using **Convolutional Neural Networks (CNN)** and **VGG16 transfer learning**. The model classifies face images into multiple mask-wearing categories and evaluates performance using accuracy, loss curves, confusion matrix, ROC curve, precision-recall curve, and classification metrics.

## Project Overview

Face mask detection is an important computer vision application used in public safety, healthcare environments, educational institutions, and crowded public places. This project trains and compares two deep learning approaches:

1. A custom CNN model built from scratch
2. A transfer learning model using pre-trained VGG16

The goal is to classify images into three categories:

- With mask
- Without mask
- Incorrect mask

## Dataset

The notebook uses a face mask image dataset organized in directory format. The dataset path used in the notebook is:

```python
/kaggle/input/datasets/shiekhburhan/face-mask-dataset/FMD_DATASET
```

The dataset is loaded using `ImageDataGenerator` from TensorFlow/Keras. Images are resized to **128 × 128 pixels**, normalized by rescaling pixel values between 0 and 1, and divided into training and validation subsets.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Kaggle Notebook Environment

## Project Workflow

The project follows these main steps:

1. Import required libraries
2. Load and preprocess the image dataset
3. Resize images to 128 × 128 pixels
4. Normalize image pixel values
5. Split the data into training and validation sets
6. Build a custom CNN model
7. Train the CNN model
8. Plot training and validation accuracy/loss
9. Generate predictions on the test data
10. Evaluate the model using:
   - Confusion matrix
   - Classification report
   - Precision-recall curve
   - ROC curve
11. Build and train a simple CNN model
12. Build and train a VGG16 transfer learning model
13. Compare custom CNN and VGG16 performance
14. Compare training time of both models

## Model Architecture

### Custom CNN Model

The main CNN model contains:

- Conv2D layer with 32 filters
- MaxPooling2D layer
- Conv2D layer with 64 filters
- MaxPooling2D layer
- Conv2D layer with 128 filters
- MaxPooling2D layer
- Flatten layer
- Dense layer with 128 neurons
- Dropout layer with 0.5 rate
- Output layer with 3 neurons and softmax activation

The model is compiled using:

```python
optimizer = "adam"
loss = "categorical_crossentropy"
metrics = ["accuracy"]
```

### Simple CNN Model

A smaller baseline CNN model is also trained. It includes:

- One Conv2D layer
- One MaxPooling2D layer
- Flatten layer
- Softmax output layer

This model is used as a lightweight comparison model.

### VGG16 Transfer Learning Model

The complex model uses **VGG16** pre-trained on ImageNet. The top classification layers are removed and replaced with custom layers:

- VGG16 base model
- GlobalAveragePooling2D
- Dense layer with 128 neurons
- Dropout layer
- Softmax output layer with 3 classes

The VGG16 base layers are frozen during training.

## Evaluation Metrics

The notebook evaluates the model using multiple performance metrics:

- Accuracy
- Validation accuracy
- Training loss
- Validation loss
- Confusion matrix
- Precision
- Recall
- F1-score
- ROC-AUC curve
- Precision-recall curve
- Training time comparison

These metrics help analyze both classification performance and model efficiency.

## Results and Analysis

The notebook compares the performance of a simple CNN and a VGG16-based transfer learning model. The CNN model is useful for learning features directly from the face mask dataset, while VGG16 uses pre-trained ImageNet features to improve feature extraction.

The training and validation accuracy/loss plots help identify whether the model is learning properly or showing signs of overfitting. The confusion matrix and classification report provide class-wise performance for mask classification categories.

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/face-mask-detection-cnn-vgg16.git
cd face-mask-detection-cnn-vgg16
```

### 2. Install Required Libraries

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

### 3. Prepare the Dataset

Make sure your dataset is arranged in the following format:

```text
FMD_DATASET/
│
├── with_mask/
├── without_mask/
└── incorrect_mask/
```

Update the dataset path in the notebook if needed:

```python
dataset_dir = "path/to/FMD_DATASET"
```

### 4. Run the Notebook

Open the notebook in Jupyter Notebook, Google Colab, or Kaggle and run the cells step by step.

```bash
jupyter notebook
```

## Important Note

One earlier data-loading cell in the notebook contains a small typing error:

```python
b-atch_size=32
```

It should be:

```python
batch_size=32
```

The corrected version is included in the next cell of the notebook, so use the corrected data-loading cell when running the project.

## Repository Structure

```text
Face-Mask-Detection-CNN-VGG16/
│
├── Face Mask Detection using CNN and VGG16.ipynb
├── README.md
└── dataset/
    ├── with_mask/
    ├── without_mask/
    └── incorrect_mask/
```

## Future Improvements

- Add data augmentation such as rotation, zoom, and horizontal flipping
- Fine-tune selected VGG16 layers for better performance
- Use more advanced architectures such as MobileNetV2 or EfficientNet
- Save the trained model in `.h5` or `.keras` format
- Deploy the model using Streamlit or Flask
- Add real-time webcam-based face mask detection using OpenCV

## Conclusion

This project demonstrates how CNN and transfer learning can be used for face mask detection. The comparison between a custom CNN and VGG16 shows how different deep learning architectures can be applied to image classification tasks. The project is useful for understanding image preprocessing, CNN model design, transfer learning, and model evaluation in computer vision.
