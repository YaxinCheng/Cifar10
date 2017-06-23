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
