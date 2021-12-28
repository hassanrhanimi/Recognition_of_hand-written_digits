# ---------------------- Recognition_of_hand-written_digits -----------------------------

As the project name implies, the goal is to recognise hand-witten digits (from 0 to 9). 
Nowadays, Automated handwritten digit recognition is widely used from recognizing zip 
codes (postal codes) on mail envelopes to recognizing amounts written on bank checks.

In this project, we'll train two machine learning algorithms :
- `Multi-Class Classification (One-Vs_all) Logistic regression`
- `Neural Network` 

We'll use tow langages :
- Matlab/octave (to really code the algorithms from scratch)
- Python using the librairie scikit-learn.

The given dataset contains 5000 training examples.

## 1 - Matlab/octave --------------------------------------------------------------------

the Training dataset of hand-written digits is given in the `data.mat` file 

### 1st: One-vs-all logistic regression algorithm----------------------------------------

We'll implement a logistic regression one-vs-all classification. 
The provided script solution.m will help you step through this part.

Files included in this part:
	`solution.m`        - Octave/MATLAB script that steps you through part 1
	`displayData.m`     - Function to help visualize the dataset
	`fmincg.m`          - Function minimization routine (similar to fminunc)
	`sigmoid.m`         - Sigmoid function
	`lrCostFunction.m`  - Logistic regression cost function
	`oneVsAll.m`        - Train a one-vs-all multi-class classifier
	`predictOneVsAll.m` - Predict using a one-vs-all multi-class classifier
        

 
### 2nd: Neural Networks learning algorithm-----------------------------------------------

In second part, we'll implement a neural network using the same training set as before. 
The neural network will be able to represent complex models that form non-linear hypotheses. 
The goal is to implement the forward propagation and the backpropagation algorithms for 
learning the neural network parameters.


Throughout this part, the script solution.m set up the dataset for the problem
and make calls to functions.


Files included in this part:
	`weights.mat`                - Neural network parameters for first verification of feedforward prop
	`displayData.m`              - Function to help visualize the dataset
	`fmincg.m`                   - Function minimization routine (similar to fminunc)
	`sigmoid.m`                  - Sigmoid function
	`computeNumericalGradient.m` - Numerically compute gradients
	`checkNNGradients.m`         - Function to help check your gradients
	`debugInitializeWeights.m`   - Function for initializing weights
	`predict.m`                  - Neural network prediction function
	`sigmoidGradient.m`          - Compute the gradient of the sigmoid function
	`randInitializeWeights.m`    - Randomly initialize weights
	`nnCostFunction.m`           - Neural network cost function
        

## 2 - Python - Scikit-Learn --------------------------------------------------------------------

