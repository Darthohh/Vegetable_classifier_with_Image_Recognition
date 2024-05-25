# Machine Learning 
## Project Proposal Report
### Vegetable classifier with Image Recognition
Omar AlHaj Hasan – 20200624

Faris Ayyash - 20190276

# Introduction
Tools that streamline user engagement with daily chores, including cooking, are much needed in the fast-changing interface between technology and daily living. We are tackling the issue of people finding it difficult to recognize different vegetables, which can be difficult to cook and invent or do new recipes. Our approach involves the use of image recognition technology to identify Vegetables from photos. For those who are learning to cook or who are unfamiliar with various ingredient kinds, this is essential. Through its ability to help users comprehend and make use of the ingredients in their kitchens, the program may be used for both instructional reasons and meal planning assistance.

#	Algorithm Definition
We propose to use Transfer Learning with Convolutional Neural Network (CNN), specifically tailored for image classification tasks. This model will be trained to identify various vegetables from images.
MobileNetV2 is the main algorithmic backbone used in the proposed project. It is a highly efficient convolutional neural network (CNN) architecture designed specifically for mobile and edge devices. It builds on the ideas from the original MobileNet, enhancing the architecture with inverted residuals and linear bottlenecks.
Then we compared the results with another Transfer learning CNN which is VGG-16 (Visual Geometry Group Networks) which is a network architecture that is composed of multiple convolutional layers with small receptive fields (3x3 filters) followed by max-pooling layers.
Finally, we used MLP (Multilayer Perceptron) to compare results with CNN, mainly the MLP model consists of tf.keras.Sequential with three hidden dense layers with ReLU activation functions and an output layer with softmax activation for classification.


#	Dataset
We  utilize the Vegetable Image Recognition dataset from Kaggle, available at Kaggle Dataset Link. This dataset is specifically labeled and includes diverse images of vegetables, which are ideal for training our model. Each image is pre-labeled, which simplifies the process of training our CNN.



# General Steps
To reproduce the results of the vegetable classifier project using the TensorFlow and Keras libraries with a MobileNetV2/VGG-16 architecture we need to:
1.	Setup the Environment; we used Visual Studio Code with Python 3.10, also we used tensorflow version 2.15 to avoid any errors.
2.	Download the Dataset.
3.	Define Paths: Change the paths for train_dir, test_dir, and val_dir in the script to match where you've stored the corresponding datasets on your system.
4.	Load Image Paths and Generate DataFrames (MobileNetV2 / VGG-16).
5.	Configure the Image Data Generators
6.	Load the pretrained MobileNetV2/VGG-16 model.
7.	Build and Compile the Model.
8.	Train the Model.
9.	Evaluate Model Performance using the graphs.

To reproduce the results of the vegetable classifier project using the TensorFlow and Keras libraries with a MLP tf.keras.Sequential architecture we need to:
1.	Setup the Environment; we used Visual Studio Code with Python 3.10, also we used tensorflow version 2.15 to avoid any errors.
2.	Download the Dataset.
3.	Define Paths: Change the paths for train_dir, test_dir, and val_dir in the script to match where you've stored the corresponding datasets on your system.
4.	Load Image Paths and Generate DataFrames.
5.	Configure the Image Data Generators.
6.	Define the MLP model.
7.	Compile the Model: Construct the MLP model using tf.keras.Sequential with the specified architecture and compile it with suitable parameters.
8.	Train the Model.
9.	Evaluate Model Performance using the graphs.

