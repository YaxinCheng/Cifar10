# Cifar10
Convolution Neuron Network for [Cifar 10](https://www.cs.toronto.edu/~kriz/cifar.html)<br>
Different combinations are tried with various optimization techniques<br>
The main goal of this is to learn and practise __Deep Learning__:computer:<br>

## Phase 1 - Accuracy 49%
After a struggling of running the network, the first milestone is reached.<br>
It is not at a state of art, not even a good state. However it works now.<br>
### Structure
* 3x3+2(S) Conv, feature map channels: 32,  activation fn: elu
* 2x2+2(S) Max pool
* 3x3+2(S) Conv, feature map channels: 32,  activation fn: elu
* 3x3+2(S) Conv, feature map channels: 64,  activation fn: elu
* 3x3+2(S) Conv, feature map channels: 128, activation fn: elu
* Fully Connected with 512 neurons, activation fn: elu
* Fully Connected with 256 neurons, activation fn: elu
* Fully Connected with 128 neurons, activation fn: elu
* Fully Connected with 64  neurons, activation fn: elu
* Fully Connected with 10  neurons, activation fn: softmax 
### Training
* Learning rate: 0.005
* Learning algo: stochastic gradient descent
* Mini-batch: 128
* Optimizer: Adam
<<<<<<< HEAD
### Problem analysis:
* Wrong initialization of weights
* Wrong penality from drop out
=======
* Cost function: Cross Entropy
>>>>>>> 97711c8f2208b3d1c93228f89a141a35789c8c27
## Phase 2 - Accuracy 64%
Dropout was ditched completely, and the new batch normalization was added to the fully connected layers.<br>
More convolution and fully connected layers were added to the network which led to the accuracy of 64%.<br>
Simply inserting new layers will only increase the cost of computation, and more strategies should be applied to the network.<br>
### Structure
* 1x1+1(S) Conv, feature map channels: 32, activation fn: elu
* 3x3+1(S) Conv, feature map channels: 32, actiavtion fn: elu
* 3x3+2(S) Conv, feature map channels: 64, activation fn: elu
* 2x2+2(S) Max pool
* 1x1+1(S) Conv, feature map channels: 64, activation fn: elu
* 1x1+2(S) Conv, feature map channels: 128, activation fn: elu
* 3x3+1(S) Conv, feature map channels: 256, activation fn: elu
* 3x3+1(S) Conv, feature map channels: 256, activation fn: elu
* Fully Connected with 512 neurons, activation fn: elu, with batch normalization
* Fully Connected with 256 neurons, activation fn: elu, with batch normalization
* Fully Connected with 128 neurons, activation fn: elu, with batch normalization
* Fully Connected with 64  neurons, activation fn: elu, with batch normalization
* Fully Connected with 32  neurons, activation fn: elu, with batch normalization
* Fully Connected with 10  neurons, activation fn: softmax, with batch normalization
### Training
* Learning rate: 0.004
* Learning algo: stochastic gradient descent
* Mini-batch: 128
* Optimizer: Adam
### Problem analysis:
* Overfitting after a few batches of normalization.
* Cost function: Cross Entropy
