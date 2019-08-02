function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

hThetaX = sigmoid(X * theta);
sumElems = -y .* log(hThetaX) - (1 - y) .* log(1 - hThetaX);

thetaNoZero = theta(1:end);
thetaNoZero(1) = 0;

J = sum(sumElems) / m + lambda * (thetaNoZero' * thetaNoZero) / m / 2;


hDiff = hThetaX - y;
prod = X' * hDiff;
grad = (prod + lambda * thetaNoZero) / m;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
