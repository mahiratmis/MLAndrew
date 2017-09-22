function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly

[m,n] = size(X); 
mu = zeros(1, n);

for i=1:n,
  mu(i) = sum(X(:,i)) / m;
end

X = X - mu; % substract mean of the feature from each feature

sigma = zeros(1, n);
for i=1:n,
  sigma(i) = sqrt(sum(X(:,i).^2) / m-1);
end

X = X ./ sigma % divide each feature by it's standard deviation
X_norm = X;
% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       









% ============================================================

end
