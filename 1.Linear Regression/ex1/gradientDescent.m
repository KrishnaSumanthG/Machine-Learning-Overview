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
%     h=(theta.')*(X.');
%     a=(h.')-y;
%     s1=sum(a);
%     b=(a.')*X(:,2);
%     s2=sum(b);
%     theta(1)=theta(1)-((alpha/m)*s1);
%     theta(2)=theta(2)-((alpha/m)*s2);
%     theta=[theta(1);theta(2)];
%     h=(theta.')*(X.');
%     a=(h.')-y;
%     b=a.'*X;
%     theta=theta-(alpha/m)*b;
    theta=theta-((alpha/m)*(transpose((transpose((theta.')*(X.'))-y)))*X).';




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
