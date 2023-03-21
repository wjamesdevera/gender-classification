# Introduction to Data Science and Analytics: Gender Classification with Deep Learning

Is an Introduction to Data Science and Analytics Project that implements a deep learning model for classifying whether a person is male or female. The model uses a convolutional neural network (CNN) to learn features from images of faces, and a fully connected neural network to make the binary classification.

## Data Set

The dataset used for training and testing the model is images from google search, which contains over 700 images of faces. Each image is labeled with various attributes, including gender.

For this project, we used a subset of the dataset that only contains images of male and female faces. The dataset was preprocessed to ensure that all images have the same size and are grayscale.
## Model Architecture

The model architecture consists of a CNN followed by a fully connected neural network. The CNN consists of 3 convolutional layers with ReLU activation and max pooling. The output of the CNN is fed into the fully connected neural network, which consists of 2 layers flatten and dense layer, followed by a sigmoid output layer for binary classification.

## Training

The model was trained using the Adam optimizer with a binary cross-entropy loss function. The batch size was set to 32 and the model was trained for 50 epochs.

## Evalutation

The model achieved an accuracy of 100% on the test set. We also evaluated the model using precision, recall, and accuracy score, which were all above 0.98.

## Usage
To use the model, simply load the trained weights and use the predict method to classify new images. The input image should be resized to 256x256.

``` Python
from keras.models import load_model
import cv2

model = load_model('gender_classification_model.h5')

def classify_gender(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    prediction = model.predict(np.expand_dims(image/255, 0));
    if prediction < 0.5:
        return 'female'
    else:
        return 'male'
```

## Conclusion

In this project, we implemented a deep learning model for classifying whether a person is male or female. The model achieved high accuracy on the test set and can be easily used for classification of new images.