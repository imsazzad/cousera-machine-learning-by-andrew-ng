function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
%fprintf('sz X =  %f\n', size(X));
%fprintf('sz theta =  %f\n', size(theta));
hypothesis =  X * theta;
single_err_arr = hypothesis - y;
single_err_arr = single_err_arr .* single_err_arr;
error = sum(single_err_arr);
error = error / 2 / m;
J = error;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

end
