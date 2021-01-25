function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
len_values = size(values)(2);
error = zeros(len_values,len_values)
for C_test = 1:len_values
  for sigma_test = 1:len_values
    model = svmTrain(X, y, values(C_test), @(x1, x2) gaussianKernel(x1, x2, values(sigma_test)));
    predictions = svmPredict(model, Xval);
    error(C_test,sigma_test) = mean(double(predictions ~= yval));
  endfor
endfor

[val, min_rows] = min(error);
[val, min_col] = min(min(error));
C = values(min_rows(min_col));
sigma = values(min_col);








% =========================================================================

end