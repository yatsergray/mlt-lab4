# CNN-Based Image Classification Project

## Overview
This project explores various convolutional neural network (CNN) architectures and transfer learning techniques for image classification. The goal is to analyze model performance on a binary classification task using different approaches, including fully connected networks, convolutional networks, and pre-trained models (VGG19, ResNet50).

## Project Structure
- `classes/` - Directory with helper classes for data processing and visualization.
- `data/` - Contains training, validation, and test datasets.

## Implementation

### Importing Required Libraries
```python
from keras import Sequential
from keras.applications import VGG19, ResNet50
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
```

### Data Preparation
```python
train_data_directory = "data/train"
valid_data_directory = "data/val"
test_data_directory = "data/test"

train_data_generator = ImageDataGenerator(rescale=1. / 255)
valid_data_generator = ImageDataGenerator(rescale=1. / 255)
test_data_generator = ImageDataGenerator(rescale=1. / 255)
```
```python
train_generator = train_data_generator.flow_from_directory(
   train_data_directory,
   target_size=(150, 150),
   batch_size=32,
   class_mode='binary'
)

valid_generator = valid_data_generator.flow_from_directory(
   valid_data_directory,
   target_size=(150, 150),
   batch_size=32,
   class_mode='binary'
)
```
![Sample Images](/images/data.png)

### Model Architectures

#### Fully Connected Neural Network
```python
fc_nn = Sequential([
   Flatten(input_shape=(150, 150, 3)),
   Dense(512, activation='relu'),
   Dense(256, activation='relu'),
   Dense(128, activation='relu'),
   Dense(1, activation='sigmoid')
])

fc_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
fc_nn_history = fc_nn.fit(train_generator, epochs=5, validation_data=valid_generator)
```
![Training Progress](images/train1.png)

#### Convolutional Neural Network (CNN)
```python
c_nn = Sequential([
   Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
   MaxPooling2D((2, 2)),
   Conv2D(64, (3, 3), activation='relu'),
   MaxPooling2D((2, 2)),
   Flatten(),
   Dense(64, activation='relu'),
   Dense(1, activation='sigmoid')
])

c_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
c_nn_history = c_nn.fit(train_generator, epochs=5, validation_data=valid_generator)
```
![Training Progress](images/train2.png)

### Transfer Learning (VGG19 & ResNet50)
```python
vgg19_base = VGG19(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
for layer in vgg19_base.layers:
   layer.trainable = False
vgg19 = Sequential([vgg19_base, Flatten(), Dense(512, activation='relu'), Dense(1, activation='sigmoid')])
```
```python
resnet50_base = ResNet50(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
for layer in resnet50_base.layers:
   layer.trainable = False
resnet50 = Sequential([resnet50_base, Flatten(), Dense(512, activation='relu'), Dense(1, activation='sigmoid')])
```
![Training Progress](images/train3.png)
![Training Progress](images/train4.png)

### Model Performance Evaluation
```python
fc_nn_accuracy = fc_nn.evaluate(test_generator)[1]
c_nn_accuracy = c_nn.evaluate(test_generator)[1]
vgg19_accuracy = vgg19.evaluate(test_generator)[1]
resnet50_accuracy = resnet50.evaluate(test_generator)[1]
```
```python
print(f"Fully Connected NN: {fc_nn_accuracy}")
print(f"Convolutional NN: {c_nn_accuracy}")
print(f"VGG19: {vgg19_accuracy}")
print(f"ResNet50: {resnet50_accuracy}")
```
![Performance Comparison](images/evaluate.png)

Model Training After increasing epochs:
```python
fc_nn_retrain_history = fc_nn.fit(
    train_generator,
    epochs=10,
    validation_data=valid_generator
)
c_nn_retrain_history = c_nn.fit(
    train_generator,
    epochs=10,
    validation_data=valid_generator
)
```
![After increasing epochs](images/train11.png)
![After increasing epochs](images/train22.png)

Model Performance Evaluation After increasing epochs:
```python
fc_nn_accuracy_extended = fc_nn.evaluate(test_generator)[1]
c_nn_accuracy_extended = c_nn.evaluate(test_generator)[1]

print(f"Fully Connected NN (Extended): {fc_nn_accuracy_extended}")
print(f"Convolutional NN (Extended): {c_nn_accuracy_extended}")
```
![After increasing epochs](images/evaluate1.png)

### Learning Curves Visualization
```python
_, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(fc_nn_history.history['accuracy'], label='Training Accuracy')
axes[0].plot(fc_nn_history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_title('Fully Connected NN')
axes[0].legend()
```
![Learning Curves](images/graph1.png)
![Learning Curves](images/graph2.png)

## Key Takeaways
- **Fully Connected NN:** Struggled with accuracy (~58%), confirming the limitations of dense-only architectures for image classification.
- **CNN:** Significantly improved classification performance (79%) by leveraging convolutional layers.
- **VGG19:** Achieved the highest accuracy (88%), demonstrating the power of pre-trained deep architectures.
- **ResNet50:** Had lower accuracy (71%) than expected, possibly due to limited dataset size or inadequate fine-tuning.
- **Overfitting:** Training performance was strong, but validation performance degraded over time, indicating potential overfitting.

### Possible Improvements
- **Data Augmentation:** Introduce transformations like rotation, flipping, and zooming to increase dataset variability.
- **Regularization Techniques:** Apply dropout layers, batch normalization, and L2 regularization to prevent overfitting.
- **Fine-tuning Pre-trained Models:** Unfreeze deeper layers in VGG19 and ResNet50 for better feature extraction.
- **Exploring Other Architectures:** Try EfficientNet, MobileNet, or custom CNN architectures for better trade-offs between performance and efficiency.

---
This project was created as part of my exploration of deep learning for image classification. 🚀
