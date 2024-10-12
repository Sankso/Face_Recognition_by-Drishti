# Drishti - Facial Recognition Project
Drishti is an advanced facial recognition project developed using a Siamese model. The project aims to accurately identify and verify individuals based on their unique facial features, leveraging deep learning techniques to enhance recognition performance.

### Positive example-
https://github.com/user-attachments/assets/f4f30009-b22f-4bb4-9a98-6ddcc48501b5

### Negative example-
https://github.com/user-attachments/assets/19ff746f-1c45-4a64-815e-c65354fb6417

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Model Training](#model-training)
- [Evaluation Metrics](#evaluation-metrics)
- [Future Improvements](#future-improvements)

## Introduction
Facial recognition technology is widely used in various applications, from security systems to personalized user experiences. Drishti implements a Siamese network, which is particularly effective in comparing and verifying facial images by measuring their similarity.

## Features
- **Siamese Network Architecture:** Utilizes a Siamese model to compute the similarity between face embeddings, allowing for robust face verification.
- **Face Detection and Recognition:** Capable of detecting and recognizing multiple faces in images and videos with high accuracy.
- **User-Friendly Interface:** Provides a straightforward interface for inputting images and obtaining recognition results.
- **Real-Time Processing:** Efficiently processes images and video streams for instant recognition.

## Technologies Used
- **TensorFlow/Keras:** Framework for building and training the Siamese model.
- **OpenCV:** Library for image processing and face detection tasks.
- **NumPy:** For numerical operations and data handling.
- **Matplotlib:** For visualizing training results and model performance.

## Future Improvements
- **Data Augmentation:** Implement various data augmentation techniques, such as rotation, scaling, and flipping, to enhance the diversity of the training dataset and improve the model's robustness against variations in facial expressions and orientations.
- **Additional Features:** Explore incorporating features such as emotion detection, age estimation, or gender classification to provide a more comprehensive analysis of faces.
- **Model Optimization:** Investigate model compression techniques, like quantization or pruning, to reduce the model size and improve inference speed, making it suitable for deployment on mobile devices and edge computing platforms.
- **Transfer Learning:** Consider using pre-trained models for feature extraction to leverage existing knowledge and potentially improve recognition accuracy, especially with limited training data.
- **Improved User Interface:** Enhance the user interface to provide more intuitive controls and visualizations of recognition results, making it easier for users to interact with the application.

## Evaluation Metrics
To assess the performance of the Drishti model, the following metrics are utilized:

- **Accuracy:** The proportion of correctly identified faces compared to the total number of faces in the dataset. It provides a general measure of model performance.

- **Precision:** The ratio of true positive predictions to the total predicted positives. It indicates the model's ability to identify relevant instances.
  
- **Recall (Sensitivity):** The ratio of true positive predictions to the actual positives. It measures the model's ability to find all relevant instances.

- **F1 Score:** The harmonic mean of precision and recall, providing a single score that balances both metrics. It is especially useful in scenarios with imbalanced classes.

## Model Training
The Siamese model is trained using a dataset of labeled facial images. The training process involves several key steps:

1. **Data Preparation:** Images are preprocessed to ensure consistency in size and format, including resizing and normalization.
  
2. **Loss Function:** The model employs contrastive loss, which encourages the model to minimize the distance between similar pairs while maximizing the distance between dissimilar pairs. This loss function is crucial for teaching the model to distinguish between faces effectively.

3. **Training Loop:** The model is trained over multiple epochs, where each epoch consists of:
   - Forward pass: Generating embeddings for input images.
   - Loss calculation: Computing the contrastive loss based on the embeddings.
   - Backpropagation: Updating the model weights using the calculated gradients to minimize the loss.

4. **Monitoring Performance:** During training, metrics such as accuracy, loss, and F1 score are monitored to assess the model's performance and make adjustments as needed.

5. **Validation:** A separate validation dataset is used to evaluate the model's performance after each epoch to prevent overfitting and ensure generalization.


