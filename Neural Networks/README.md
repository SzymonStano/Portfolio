# MNIST Handwritten Digit Classification with Neural Networks

This project involves classifying handwritten digits from the **MNIST** dataset using a neural network model built with **TensorFlow** and **Keras**. The MNIST dataset consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9), each 28x28 pixels in grayscale.

## Project Overview

The objective of this project is to develop a deep learning model that can classify the handwritten digits from the MNIST dataset. The project follows these key steps:

1. **Data Preparation**: The dataset was loaded, normalized (pixel values scaled between 0 and 1), and visualized to understand the distribution of digit classes.
2. **Model Architecture**: A **Feedforward Neural Network (FNN)** was built using **Keras**. The model consists of:
   - Flattening layer to reshape the 28x28 images into 1D vectors
   - Two **Dense** layers with ReLU activation for feature learning
   - A **Dropout** layer to reduce overfitting
   - A **softmax output layer** for classification into 10 categories (0-9)
3. **Model Training**: The model was trained on the training data with **early stopping** to prevent overfitting. The training process was monitored with validation accuracy.
4. **Model Evaluation**: After training, the model was evaluated on the test dataset, achieving an accuracy of ~97.9%. Additionally, performance metrics like **confusion matrix**, **precision**, **recall**, and **f1-score** were computed.

## Key Findings
- The model achieved an accuracy of approximately **97.9%** on the test dataset, demonstrating strong performance in classifying handwritten digits.
- The model’s **confusion matrix** indicated that it performed well across most digits, with some minor misclassifications occurring between visually similar digits (e.g., 3 and 5, or 8 and 9).
- The model's evaluation showed high **precision**, **recall**, and **f1-score** across most digit classes, highlighting its robustness.

## Tools and Libraries
- Python was used for all computations, model building, and evaluation.
- Key libraries and frameworks include:
  - `TensorFlow` and `Keras` for neural network modeling and training
  - `NumPy` for data manipulation
  - `Matplotlib` and `Seaborn` for data visualization and plotting
  - `sklearn` for metrics like confusion matrix and classification report

## Future Work & Suggestions for Improvement
### Hyperparameter Tuning
- While reasonable values were selected for the layers, hyperparameter tuning could further optimize the model. Experimenting with different optimizers, learning rates, and more layers could improve performance.

### Model Complexity
- Given the simplicity of the MNIST dataset, it would be beneficial to explore more complex models like **Convolutional Neural Networks (CNNs)**. CNNs are designed specifically for image classification tasks and typically outperform feedforward neural networks on this type of data.

### Model Evaluation
- Adding additional evaluation metrics, such as **ROC curves** and **AUC scores**, could provide a more comprehensive understanding of the model’s performance across different thresholds.
- Implementing **cross-validation** would help validate the model’s robustness and generalizability to unseen data.


## Conclusion
This project demonstrates the use of a simple **Feedforward Neural Network** to classify handwritten digits from the MNIST dataset. The model performed well, achieving nearly 98% accuracy on the test set. This project serves as a great introduction to deep learning techniques for image classification and the **MNIST** dataset.

