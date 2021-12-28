function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% =============================================================
% cost function J
  g = sigmoid(X * theta);
  J = sum(((y .* log(g) + (1 - y) .* log(1-g))/ (-m))) + lambda/(2 * m) .* (theta' * theta - theta(1,1)^2); 
 
% Grad of the J with respect to theta
  A = (1/m) * X' * (g-y);
  grad = A + (lambda/m) .* theta;
  grad(1,1)= A(1,1);
% =============================================================

grad = grad(:);

end
