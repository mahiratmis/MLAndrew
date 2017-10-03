function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    hTheta = X*theta-y; % vector of size m
<<<<<<< HEAD
    
    theta0 = theta(1) - alpha* (1 / m) * (hTheta' * X(:,1))
    theta1 = theta(2) - alpha* (1 / m) * (hTheta' * X(:,2))
    theta = [theta0;theta1]
    %theta0 = theta(0) - alpha* (1 / m) * ((X*theta-y).*diag(X)








=======
    theta = theta - alpha* (1/m) * (X' * hTheta); 
>>>>>>> master

    % ============================================================

    % Save the cost J in every iteration    
<<<<<<< HEAD
    J_history(iter) = computeCostMulti(X, y, theta);

=======
    cost = computeCostMulti(X, y, theta);
    J_history(iter) = cost;
>>>>>>> master
end

end
