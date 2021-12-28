%% Part 1: One-vs-all

%% =========== Initialization
clear ; close all; clc

%% Setup the parameters to use for this part
% 20x20 Input Images of Digits
input_layer_size  = 400;  
% 10 labels, from 1 to 10 (note that we'll map "0" to label 10)
num_labels = 10;          

%% =========== Part 1: Loading and Visualizing Data =============
%  We start by first loading and visualizing a sample of the dataset.
%  The dataset contains handwritten digits.

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

% training data stored in arrays X, y
load('data.mat'); 
% You will have X, y in your environment
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Part 2: One-vs-All Training =========================
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Predict for One-Vs-All ===================

pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

