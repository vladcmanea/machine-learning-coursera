function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

thetaNoZeros = theta(1:end);
thetaNoZeros(1) = 0;

Xtheta = sigmoid(X * theta);
sumAll = (-y .* log(Xtheta)) - ((1 - y) .* log(1 - Xtheta));
J = (sum(sumAll) + lambda * (thetaNoZeros' * thetaNoZeros) / 2) / m;

% You need to return the following variables correctly 
diff = Xtheta - y;
grad = ((X' * diff) + lambda * thetaNoZeros) ./ m;
% grad = grad(:);

end
