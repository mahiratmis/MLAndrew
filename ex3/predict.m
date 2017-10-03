function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.

% s2 is the number of hidden units in layer 2
% X size is (m,n), Theta1 size is (s2,n+1)
% add bias to X
a1 = [ones(m,1) X];
%weighted sum of the second layer
z2 = a1*Theta1';  % size is (m,s2)
%output of second layer, inputs for next layer
a2 = sigmoid(z2); % size is (m,s2)
a2 = [ones(m,1) a2]; % add bias size is (m,s2+1)

%s3 is number of hidden units in layer 3
%Theta2 size is (s3,s2+1)
z3 = a2*Theta2'; % size is (m,s3)
a3 = sigmoid(z3);
[val,p] = max(a3,[],2);

% =========================================================================


end
