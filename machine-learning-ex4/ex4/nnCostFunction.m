function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% First the frontforward propagation
a_1 = [ones(m,1), X];   % 5000 x 401
z_2 = a_1 * Theta1';    % (5000x401) x (401x25) = 5000x25

a_2 = sigmoid(z_2);     % 5000 x 25
a_2 = [ones(m,1) a_2];   % 5000 x 26
z_3 = a_2 * Theta2';    % (5000x26) x (26x10) = 5000x10

h_x = a_3 = sigmoid(z_3);  % 5000 x 10

% Convert y to a matrix
y_matrix = zeros(m,num_labels);
for i = 1:m,
  y_matrix(i, y(i))=1;
endfor;

% Then we compute the costs of every class
cost = -y_matrix .* log(h_x) - (1 - y_matrix) .* log(1 - h_x);   % (5000x10)*(5000x10) - (5000x10)*(5000x10)
J = 1 / m * sum(sum(cost));

% Now we compute the regularization term
size_theta1 = size(Theta1);
size_theta2 = size(Theta2);
theta1_reg = [zeros(size_theta1(1),1), Theta1(:,2:end)];
theta2_reg = [zeros(size_theta2(1),1), Theta2(:,2:end)];
reg = lambda / (2 * m) * (sum(sum(theta1_reg.^2)) + sum(sum(theta2_reg.^2)));

% Add the regularization term to the cost
J = J + reg;



% Next we do the Backpropagation
% Initialize Deltas
Delta_1 = zeros(hidden_layer_size, input_layer_size+1); % 25x401
Delta_2 = zeros(num_labels, hidden_layer_size+1);       % 10x26
y_matrix = y_matrix'; % 10x5000

for t=1:m,
  
  % First the feedforward
  a_1 = [1; X(t, :)'];   % 401x1
  z_2 = Theta1 * a_1;    % (25x401) * (401x1) = 25x1
  a_2 = g_z2 = sigmoid(z_2);    % 25x1
  a_2 = [1; a_2];   % 26x1
  z_3 = Theta2 * a_2;    % (10x26) * (26x1) = 10x1
  h_x = a_3 = sigmoid(z_3);  % 10 x 1
  
  % Then we compute the deltas of every layer
  delta_3 = a_3 - y_matrix(:,t);  % (10x1) - (10x1) = 10x1
  delta_2 = Theta2(:, 2:end)' * delta_3 .* sigmoidGradient(z_2);  % (25x10) * (10x1) .* (25x1) = 25x1
  
  Delta_2 = Delta_2 + delta_3 * a_2'; % (10x26) + (10x1) * (1x26) = 10x26
  Delta_1 = Delta_1 + delta_2 * a_1'; % (25x401) + (25x1) * (1x401) = 25x401
  
endfor

temp_theta1 = Theta1; % 25x401
temp_theta1(:,1) = 0;
temp_theta2 = Theta2; % 10x26
temp_theta2(:,1) = 0;

Theta1_grad = 1 / m * Delta_1 + lambda / m * temp_theta1; 
Theta2_grad = 1 / m * Delta_2 + lambda / m * temp_theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
