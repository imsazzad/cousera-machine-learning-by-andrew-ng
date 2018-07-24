function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    %fprintf('sz X =  %f\n', size(X));
    
    hypothesis =  X * theta;
    single_err_arr = hypothesis - y;
    %fprintf('single err arr =  %f\n', single_err_arr);
    single_err_arr = single_err_arr .* X;
    gradient = sum(single_err_arr, 1);
    %fprintf('grad  =  %f\n', gradient);
    temp = alpha * gradient/ m;
    
    %fprintf('sz theta =  %f\n', size(theta));
    theta = theta - temp';

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
