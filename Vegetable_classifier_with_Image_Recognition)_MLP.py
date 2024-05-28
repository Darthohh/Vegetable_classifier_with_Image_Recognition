"""
// ***************************************************************************
//  File Name      : Vegetable_classifier_with_Image_Recognition)_CNN_(VGG-16).py
//  Version        : 1.0
//  Description    : Vegetable classifier with Image Recognition
//  Authors        : Omar, Faris
//  IDE            : Vscode
//  Last Updated   : 08 May 2024
//  Libraries Used : numpy, pandas, pathlib, tensorflow, matplotlib.pyplot
// ***************************************************************************
"""

import numpy as np # Data Manipulation
import pandas as pd # Data Manipulation
from pathlib import Path # For File Paths
import os.path # For operating system path manipulation
import matplotlib.pyplot as plt # For Plotting Images
import tensorflow as tf # for Building and training CNN model

# Training Dir
train_dir = Path('D:/OneDrive/University Documents/Subjects/(4) 2nd Semester 2024/Artificial Intelligence and Machine Learning - 22570/Project/2024-05-29 Project Report and Presentation/Data_Set/train')
train_filepaths = list(train_dir.glob(r'**/*.jpg'))

# Test Dir
test_dir = Path('D:/OneDrive/University Documents/Subjects/(4) 2nd Semester 2024/Artificial Intelligence and Machine Learning - 22570/Project/2024-05-29 Project Report and Presentation/Data_Set/test')
test_filepaths = list(test_dir.glob(r'**/*.jpg'))

# Val Dir
val_dir = Path('D:/OneDrive/University Documents/Subjects/(4) 2nd Semester 2024/Artificial Intelligence and Machine Learning - 22570/Project/2024-05-29 Project Report and Presentation/Data_Set/validation')
val_filepaths = list(val_dir.glob(r'**/*.jpg'))

# Extracts labels from file paths.
# Creates a DataFrame with file paths and corresponding labels.
# Shuffles the DataFrame.
def proc_img(filepath):
    # Create a DataFrame with the filepath and the labels of the pictures
    labels = [os.path.split(os.path.split(str(fp))[0])[1] for fp in filepath]

    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels
    df = pd.concat([filepath, labels], axis=1)

    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

# Processing images
train_p = proc_img(train_filepaths)
test_p = proc_img(test_filepaths)
val_p = proc_img(val_filepaths)

print('~~~~ Training set ~~~~\n')
print(f'Number of pictures: {train_p.shape[0]}\n')
print(f'Number of different labels: {len(train_p.Label.unique())}\n')
print(f'Labels: {train_p.Label.unique()}')

# The DataFrame with the filepaths in one column and the labels in the other one
train_p.head(5)

# Create a DataFrame with one Label of each category
df_unique = train_p.copy().drop_duplicates(subset=["Label"]).reset_index()

# Display some pictures of the dataset
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 7),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(df_unique.Filepath[i]))
    ax.set_title(df_unique.Label[i], fontsize = 12)
plt.tight_layout(pad=0.5)
plt.show()

# Generator
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

# Data Augmentation
train_images = train_generator.flow_from_dataframe(
    dataframe = train_p,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=0,
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_images = train_generator.flow_from_dataframe(
    dataframe = val_p,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=0,
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_images = test_generator.flow_from_dataframe(
    dataframe = test_p,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# Define the MLP model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),  # Flatten the 2D image to 1D, Accepts the input features.
    tf.keras.layers.Dense(512, activation='relu'), # introduce non-linearity into the model
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # Assuming 5 classes, provides probability distributions for classification.
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_images,  
    validation_data=val_images, 
    batch_size=32,
    epochs=5,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )
    ]
)

# Plot the Accuracy Graph
pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot()
plt.title("Accuracy")
plt.show()

# Plot the Loss Graph
pd.DataFrame(history.history)[['loss','val_loss']].plot()
plt.title("Loss")
plt.show()
