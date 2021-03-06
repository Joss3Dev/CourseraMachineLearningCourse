function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
X_col = X';  # 2x12
y_col = y';  # 1x12
h_x = theta' * X_col; # 1x12

h_minus_y = h_x - y_col; # 1x12
temp_theta = theta(2:end);  # 1x9
J = 1 / (2 * m) * sum(h_minus_y .^ 2) + lambda / (2 * m) * sum(temp_theta .^ 2);

temp_theta = theta;
temp_theta(1) = 0;

grad = 1 / m * sum(h_minus_y .* X_col, 2) + lambda / m * temp_theta; 

% =========================================================================

grad = grad(:);

end
