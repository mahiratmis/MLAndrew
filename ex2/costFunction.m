function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
% You need to return the following variables correctly 
% X m by n+1, theta n+1 by 1 thus hteta is m by 1
hteta = sigmoid(X*theta) ;
J = -1/m* ( y' *log(hteta) + (1-y)' *log(1-hteta) )  

% vector of size m
grad = 1/m * (X' * (hteta-y));

% =============================================================

end
