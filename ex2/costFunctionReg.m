function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% You need to return the following variables correctly 
% X m by n+1, theta n+1 by 1 thus hteta is m by 1
hteta = sigmoid(X*theta) ;
J = -1/m* ( y' *log(hteta) + (1-y)' *log(1-hteta) )  + lambda/(2*m)*(theta(2:n)'*theta(2:n));
grad= (1/m * (X' * (hteta-y))) + lambda/m*theta;
grad(1) =1/m * (X(:,1)' * (hteta-y));

% =============================================================

end
