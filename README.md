# Fashion MNIST Clothing Classification

## About

The `Fashion MNIST` dataset comprises of `70,000` `28x28` pixel grayscales images of `10` types of clothing items divided into `60,000` training and `10,000` testing samples. 

Each training and test example is assigned to one of the following labels:

- `0 - T-shirt/top`
- `1 - Trouser`
- `2 - Pullover`
- `3 - Dress`
- `4 - Coat`
- `5 - Sandal`
- `6 - Shirt`
- `7 - Sneaker`
- `8 - Bag`
- `9 - Ankle boot`


A few samples of how our dataset images look like:

![dataview](https://user-images.githubusercontent.com/53920732/129562722-9c793bae-c272-4f3b-b795-9b2dedb33874.png)

## Model

Broadly, first we pass our input data after numerically encoding them to our neural network. Our neural network is going to learn the representations (i.e. the patterns/ features/ weights). At the beginning, our neural network is going to initialize itself with random weights. It does this using the parameter kernel_initializer. We're going to show it different examples of the data we're trying to learn. And our neural network is going to update its representation outputs (weights and biases) based on these training examples. And its going to slowly adjust these patterns to better suit the data as best as it can. Ideally we aim for a case where it outputs all the correct values.

A weights matrix is has one value per data point. Whereas a bias matrix has one value per hidden layer. Every neuron has a bias vector. Each of these is paired with a weights matrix.

The bias vector also gets initialized. It is done using the parameter bias_initializer. It gets initialized to zeros, atleast in the cas of a TensorFlow Dense Layer.

The bias vector dictates how much the patterns within the corresponding weights matrix should influence the next layer.

In each layer in a deep learning model, the previous layer is its inputs. It is called a deep learning model because its deep (i.e. it has multiple layers). A deep learning model is just a neural network model with many layer. Each layer does it work to find patterns in the data and then feeds it to the next layer.

Steps in modelling:

- Creating a model
- Compiling the model
  - Defining a loss function (how wrong our models predictions are)
  - Setting up an optimizer (how your model should update its internal patterns to better its predictions)
 - Creating evaluation metrics (human interpretable values for how well our model is doing)
- Fitting a model (getting it to find patterns in our data)

We tested a few models to try and test a few things. Every next model has an improvement over the previous one.

### Model 1
```Python
#Experiment Model 1
#Set random seed
tf.random.set_seed(42)

#Build a Model

# 1. Create a Model
model_1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), #Flatten our data to (None, 784)                           
    tf.keras.layers.Dense(4, activation = 'relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(10, activation = tf.keras.activations.softmax)                               
])

# 2. Compile the Model
model_1.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ['accuracy']
)

# 3. Fit the Model
non_norm_history = model_1.fit(
    train_data,
    train_labels, 
    epochs=10,
    validation_data=(test_data, test_labels))
```
![his1](https://user-images.githubusercontent.com/53920732/129564247-600b8d84-25ed-482f-a2a7-9c19716f7849.png)

```Accuracy: 34%```

## Normalization

Since it is a dataset of grayscale images, there is only 1 color channel. The pixel values range from `0 to 255`, representing black and white. And the value in between different shades of grey. Neural Networks prefer data to be scaled. To get better results, we need to normalize our data i.e turn it into range of (0,1). We can normalize our data by dividing it by the max value i.e. dividing it by 255.

```Python
#Normalize the data
train_data_norm = train_data / 255.0
test_data_norm = test_data / 255.0
```

### Model 2

```Python
#Experiment Model 2 (Change from previous - data has been normalized)
#Set random seed
tf.random.set_seed(42)

#Build a Model

# 1. Create a Model
model_2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), #Flatten our data to (None, 784)                           
    tf.keras.layers.Dense(4, activation = 'relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(10, activation = tf.keras.activations.softmax)                               
])

# 2. Compile the Model
model_2.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ['accuracy']
)

# 3. Fit the Model
norm_history = model_2.fit(
    train_data_norm,
    train_labels, 
    epochs=10,
    validation_data=(test_data_norm, test_labels))
```

![his2](https://user-images.githubusercontent.com/53920732/129564264-150bdfd7-b620-4f6b-a319-1cee341262a3.png)


```Accuracy: 80%```

### Model 3 - Finding Ideal Learning Rate

```Python
#Experiment Model 3 (Change from previous - finding the ideal learning rate)

# Set random seed
tf.random.set_seed(42)

# Build the Model

# 1. Create a Model

model_3 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)                               
])
# 2. Compile the Model
model_3.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ['accuracy']
)

# Create a learning rate callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))

# 3. Fit the Model
history = model_3.fit(
      train_data_norm,
      train_labels,
      epochs=40,
      validation_data=(test_data_norm, test_labels),
      callbacks=[lr_scheduler]
)
```
![his3](https://user-images.githubusercontent.com/53920732/129564279-ebddc86a-02e8-43b2-ba7b-9f57788e6821.png)

![ideallearningrate](https://user-images.githubusercontent.com/53920732/129564303-e9995b11-e875-4c64-874d-c89aa3ad659b.png)



The ideal learning rate seems to be `0.001`, which is the default learning rate of Adam optimizer.

### Model 4

```Python
#Experiment Model 4 (Change from previous - using the ideal learning rate previously found)

# Set random seed
tf.random.set_seed(42)

#Build the Model

# 1. Create a Model
model_4 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
# 2. Compile the Model
model_4.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
# 3. Fit the Model
history = model_4.fit(
    train_data_norm,
    train_labels,
    epochs=30,
    validation_data=(test_data_norm,test_labels)
)

```

![his4](https://user-images.githubusercontent.com/53920732/129564323-6ce9f0f2-e073-4007-9042-a3fd96683509.png)


```
Accuracy: 80%
```

### Model 5

```Python
#Experiment Model 5 (Change from previous - adding more layers and hidden units and training for more epochs)

# Set random seed
tf.random.set_seed(42)

#Build the Model

# 1. Create a Model
model_5 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
# 2. Compile the Model
model_5.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
# 3. Fit the Model
history = model_5.fit(
    train_data_norm,
    train_labels,
    epochs=50,
    validation_data=(test_data_norm,test_labels)
)

```

![his5](https://user-images.githubusercontent.com/53920732/129564330-6e5a0ded-d62a-49df-8df9-60fbf9f9987b.png)


```
Accuracy: 87%
```

### Model 6

```Python
#Experiment Model 6 (Change from previous - adding more layers)

# Set random seed
tf.random.set_seed(42)

#Build the Model

# 1. Create a Model
model_6 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
# 2. Compile the Model
model_6.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
# 3. Fit the Model
history = model_6.fit(
    train_data_norm,
    train_labels,
    epochs=50,
    validation_data=(test_data_norm,test_labels)
)

```
![his6](https://user-images.githubusercontent.com/53920732/129564338-f3b76870-5c66-433d-8686-a7002646f7c9.png)


```
Accuracy: 87%
```

## Evaluation

### Confusion Matrix

![confusion matrix](https://user-images.githubusercontent.com/53920732/129564401-5bc69e25-279d-40bd-821c-79aae0b9cc3d.png)

### Visualize Predictions
Green - `Predicted Label == True Label`

Red - `Predicted Label != True Label`

![visualize predictions](https://user-images.githubusercontent.com/53920732/129564488-e2245b4c-dbf3-4767-b044-0fcb99176c2b.png)


## Model Summary

![model_summary](https://user-images.githubusercontent.com/53920732/129564532-d20a0e91-ca64-4458-9755-164f679864c0.png)


Since the model is only a simple feed forward neural network, accuracy doesnt seem to cross 90%. We can increase our accuracy by using a CNN Model.
