function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

hThetaX = sigmoid(X * theta);
sumElems = -y .* log(hThetaX) - (1 - y) .* log(1 - hThetaX);
J = sum(sumElems) / m;

hDiff = hThetaX - y;
prod = X' * hDiff;
grad = prod / m;

end
