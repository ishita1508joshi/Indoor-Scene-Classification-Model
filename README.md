# Indoor Scene Classification Model
This project is a Convolutional Neural Network (CNN)-based image classification model using the MIT Indoor Scenes Dataset. For simplicity, only 10 categories are considered, including scenes like clothing stores, dining rooms, libraries, etc. This project uses TensorFlow, Keras, and OpenCV, along with data augmentation techniques to improve model accuracy.
### Project Overview
The model classifies indoor scenes into 10 selected categories:
- Clothing store
- Dining room
- Grocery store
- Kitchen
- Library
- Living room
- Mall
- Movie theater
- Museum
- Restaurant
### Dataset
MIT Indoor Scenes Dataset from Kaggle : https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019

Divided into two datasets:
- indoor_dataset
- test_dataset
### Features
- Data Augmentation: Random rotation, noise addition, and horizontal flipping to improve generalization.
- Model Architecture: Multi-layer CNN with Conv2D, MaxPooling, Dense, Dropout, and Softmax layers.
- Evaluation: Precision, recall, F1-score, accuracy, and confusion matrix for performance metrics.
### Installation
1. Mount Google Drive in Colab.
2. Install all necessary libraries:
   - tensorflow
   - numpy
   - skimage
   - opencv-python
   - keras
   - drive from google.colab
### Training the Model
Training model using indoor_dataset
- create_training_data function - generates and augments the dataset
- model.fit function - used for training
### Testing the Model
Testing model using test_dataset and metrics like:
- precision
- recall
- f1-score
- confusion matrix
### Results
Achieved 90% accuracy on test_dataset which demonstrates the model's strong performance in distinguishing between indoor scene categories.
