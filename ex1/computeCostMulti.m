function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

hsum = theta' * X';
diff = hsum - y';
sqrs = diff * diff';
sum = sum(sqrs);
J = sum / 2 / m;

end
