# MNIST Classfication
This repo classifies hand-written digits from the MNIST dataset using a feed-forward Multilayer Perceptron Network. MLPs are not the preferred way to process image data, but this serves as a good start to Deep Learning journey. The MNIST hand-written digit dataset is included in Tensorflow and can easily be imported and loaded, as we will see below. Using this dataset and a simple feed-forward network, we will demonstrate one approach for how to work with image data and build a network that classifies digits [0,9].
![keras-mnist-digits-classification](https://github.com/galax19ksh/MNIST/assets/112553872/566802a5-d6f8-4201-9aee-f9ed63edd14c)


## Main tools and libraries
Tensorflow, Keras

## 1. Load and Split the MNIST Dataset
The MNIST dataset contains 70,000 images partitioned into 60,000 for training and 10,000 for test. We further carve out 10,000 samples from the training data to use for validation.

## 2. Dataset Preprocessing
* **Feature Transformation and Normalization (Input):** We flatten the 2D array of 28x28 input image into 1D array of 784 features. We also normalize the pixel intensities to be in the range [0, 1].
*  **Label Encoding (output):** Instead of Integer encoding, we use one-hot encoding `to_categorical()` to convert each label into a binary vector where the length of the vector is equal to the number of classes [0 to 9].

## 3. Model Implementation
**Deep Neural Network Architecture:** The network architecture contains An input layer, two hidden layers, and an output layer.

**Input Data:** The image input data is pre-processed (flattened) from a 2-Dimensional array [28x28] to 1-Dimensional vector of length [784x1] where the elements in this input vector are the normalized pixel intensities. The input to the network is sometimes referred to as the input "layer", but it's not technically a layer in the network because there are no trainable parameters associated with it.

**Hidden Layers:** We have two hidden layers that contain some number of neurons (that we need to specify). Each of the neurons in these layers has a non-linear activation function (e.g., ReLU, Sigmoid, etc...). Notice that the first hidden layer has an input shape of [784,1] since the 28x28 image is flattened to a vector of length 784. The neurons in each of the hidden layers have activation functions called "ReLU" which stands for Rectified Linear Unit.

**Output Layer:** We now have 10 neurons in the output layer to represent the ten different classes (digits: 0 to 9), where the output of each represents the probability that the input image corresponds to the class associated with that neuron.

**Dense Layers:** All the layers in the network are fully connected, meaning that each neuron in a given layer is fully connected (or dense) to each of the neurons in the previous layer. The weights associated with each layer are represented in bold to indicate that these are matrices that contain each of the weights for all the connections between adjacent layers in the network.

**Softmax Function:** The values from each of the neurons in the output layer are passed through a softmax function which transform (normalizes) the raw output and produces a probability score as described above.

**Network Output:** The network output ( y′ ), is a vector of length ten, that contains the probabilities of each output neuron. Predicting the class label simply requires passing ( y′ ) through the argmax function to determine the index of the predicted label.

**Loss Function:** The loss function used is Cross Entropy Loss, which is generally the preferred loss function for classification problems. It is computed from the ground truth labels ( y ) and the output probabilities of the network ( y′ ). Note that  y  and  y′  are both vectors whose length is equal to the number of classes.

Although the diagram looks quite a bit different from the single-layer perceptron in the linear regression example, it is fundamentally very similar in terms of the processing that takes place during training and prediction. We still compute a loss based on the predicted output of the network and the ground truth label of the inputs. Backpropagation is used to compute the gradient of the loss with respect to the weights in the network. An optimizer (which implements gradient descent) is used to update the weights in the neural network.
![keras-mnist-mlp-network](https://github.com/galax19ksh/MNIST/assets/112553872/de479d0d-80e6-4b76-9b38-10b61ce75cf8)

1. **Build Model:** We build the model by adding required layers using `model.add()` function, specifying the "Dense" layers, no of neurons in each layer and suitable activation functions (relu, softmax etc).
2. **Compile Model:** This step defines the optimizer (RMSProp optimizer in Keras) and the loss function (`categorical_crossentropy` as we used one-hot encoding) that will be used in the training loop. This is also where we can specify any additional metrics like `accuracy` to track.
3. **Train Model:** To train the model we call the `fit()` method in Keras by explicitly specifying epochs, batch_size and the validation dataset using  `validation_data=(X_valid, y_valid))`. Afterwards, we can plot the training results.

## Model Evaluation

* We call the `predict()` method to retrieve all the predictions, and then we select a specific index from the test set and print out the predicted scores for each class.
* We can plot confusion matrix as a heatmap to examine the prediction accuracy.








