function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
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
    %       of the cost function (computeCost) and gradient here.
    %
    %fprintf('sz theta =  %f ', size(theta));
    
    h = X * theta;
    %fprintf('sz hypothesis =  %f\n', size(h));
    single_error_arr = (h - y);
    %fprintf('sz X =  %f', size(X));
    %printf('sz single error array =  %f\n', size(single_error_arr));
    temp = single_error_arr .* X; %' * X;
    gradient = sum(temp, 1);
    fprintf('sz gradient =  %f\n', size(gradient));
    temp = alpha * gradient/m;
    fprintf('sz old theta =  %f\n', size(theta));
    theta = theta - temp';
    fprintf('sz theta =  %f\n', size(theta));

    % ============================================================

    % Save the cost J in every iteration
    %disp(size(X));
    %disp(size(y));
    %disp(size(theta));
    cost = computeCost(X, y, theta);
    %disp('cpost -');
    %disp(cost);
    J_history(iter) = cost;

end

end
