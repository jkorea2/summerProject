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

    summation0 = 0;
    summation1 = 0;

    for i = 1:m,
#     hypo(i) = theta(1) + theta(2) * X(i, 2);
      hypo = X * theta;

      cost(i) = hypo(i) - y(i);

#      times0(i) = cost(i) * X(i, 1);
#      times1(i) = cost(i) * X(i, 2);

      times = cost(i) * X;

#      summation0 = summation0 + times0(i);
#      summation1 = summation1 + times1(i);

      summation0 = summation0 + times(i, 1);
      summation1 = summation1 + times(i, 2);

    end

    theta(1) = theta(1) - alpha * summation0 / m;
    theta(2) = theta(2) - alpha * summation1 / m;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
