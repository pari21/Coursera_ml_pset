function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1); % no. of rows of X i.e 5000
n = size(X, 2); % no. of cols of X i.e 400(1 training example is an img of 20 * 20 pixel that 400 values are written in a row into matrix X therefore 400).

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1); %size 10 * 401

% Add ones to the X data matrix
X = [ones(m, 1) X]; % 5000 * 401(adding col of 1)

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%





options = optimset('GradObj', 'on', 'MaxIter', 50);
initial_theta = zeros(n + 1,1);


% fminunc is advance optimization technique to find best parameters of theta. Here fmincg is defined which is more effecient.
for c = 1:num_labels
  [all_theta(c,:)] =  fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)),initial_theta, options);  % Usage: [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
                                                                                                     % it returns multiple values therefore [all_theta(c,:)] fun is written inside [].
end                                                                                                  % to see the values of theta enter all_theta in command window.
  

  
  











% =========================================================================


end
