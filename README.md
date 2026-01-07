# Diabetic Retinopathy Detection

## Overview

This project implements a **Diabetic Retinopathy (DR) Detection system** using retinal fundus images.
Diabetic Retinopathy is a diabetes-related eye disease that can lead to vision loss if not detected early.

The objective of this project is to **classify retinal images into two categories**:

* **DR** (Diabetic Retinopathy present)
* **No_DR** (No Diabetic Retinopathy)

Early detection of DR helps in **timely medical intervention** and reduces the risk of blindness.


## Technologies Used

* **Python**
* **TensorFlow / Keras**
* **NumPy**
* **OpenCV**
* **Matplotlib**
* **Scikit-learn**


## Features

* Image-based disease classification
* Binary classification using **Convolutional Neural Network (CNN)**
* Automatic feature extraction using convolution layers
* Image preprocessing and normalization
* Model training and validation
* Performance evaluation using:

  * Confusion Matrix
  * Accuracy
  * F1 Score
* Visualization of training and validation accuracy


## Modifications & Contributions

The project was adapted and enhanced to meet internship requirements:

* Organized dataset into **train and test directories**
* Implemented a **CNN model from scratch** for binary classification
* Applied **image normalization** for improved convergence
* Evaluated model using **confusion matrix and F1 score**
* Visualized training and validation accuracy curves
* Added sample prediction visualization for better interpretability

> These modifications focus on understanding CNN-based image classification and practical implementation rather than achieving maximum accuracy.


## Dataset Structure

```
dataset/
 ├── train/
 │    ├── DR/
 │    └── No_DR/
 │
 └── test/
      ├── DR/
      └── No_DR/
```

Each folder contains retinal fundus images belonging to the respective class.

## Model Used

A **Convolutional Neural Network (CNN)** consisting of:

* Input rescaling layer
* Convolutional layers with ReLU activation
* MaxPooling layers
* Fully connected (Dense) layers
* Sigmoid activation for binary classification


## How to Run the Project

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Load the dataset**:

```python
import tensorflow as tf

train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=(224, 224),
    batch_size=32,
    label_mode="binary"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(224, 224),
    batch_size=32,
    label_mode="binary",
    shuffle=False
)
```

3. **Train the model**:

```python
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=5
)
```

4. **Evaluate performance**:

```python
from sklearn.metrics import confusion_matrix, f1_score
```


## Output

* Model training and validation accuracy
* Confusion Matrix
* F1 Score
* Sample prediction visualization

| Metric   | Value                |
| -------- | -------------------- |
| Accuracy | ~80%                 |
| F1 Score | Computed on test set |

> **Observation:** The CNN model successfully learned discriminative features from retinal images and produced reliable classification results.


## Business & Medical Application

* Assist ophthalmologists in **early screening** of diabetic patients
* Reduce manual workload in retinal image analysis
* Enable **faster and scalable DR detection**
* Support preventive healthcare decision-making

## Limitations

* Limited dataset size
* Binary classification only
* Performance can be improved using larger datasets and transfer learning models


## Conclusion

This project demonstrates the effective use of **Convolutional Neural Networks** for medical image classification.
The implemented system provides a foundational approach for automated diabetic retinopathy detection and can be extended for real-world clinical applications.


## Author

**Supriya Kulkarni**
(**Intern ID:** SMI82128)
