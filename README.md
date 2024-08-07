# Fashion-MNIST-Classifier
Description
The Fashion MNIST Classifier project is an implementation of a neural network model using TensorFlow and Keras to classify images from the Fashion MNIST dataset. The dataset consists of 60,000 grayscale images of 10 different fashion categories, each 28x28 pixels in size, and 10,000 images in the test set. The project involves:

Data Preparation: Normalizing pixel values to a range of 0 to 1 for both training and test images.
Model Design: Building a neural network with a flatten layer to reshape the input, a dense layer with 128 neurons using ReLU activation, and a final dense layer with 10 neurons using softmax activation to classify the images into one of 10 categories.
Model Training: Training the model using the training dataset for 5 epochs with Adam optimizer and sparse categorical cross-entropy loss.
Evaluation: Assessing the model’s performance on the test dataset to determine its accuracy.
Prediction and Visualization: Predicting the class of test images and displaying each image with its actual and predicted class labels.
The aim of this project is to develop a basic image classification model and evaluate its performance in classifying fashion items.
