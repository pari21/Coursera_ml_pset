function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%activation = sigmoid(X*Theta1');
%h = sigmoid(activation*Theta2');
%[val p] = max(h, [], 2); 
                                                                                
a1 = [ones(size(X, 1), 1) X]; %to add x0 intercept term. ones(2,3) means 2 rows n 3 cols. Here ones(size(X,1),1) X] size(X,1) gives no. of rows of X,
                                                                                                          % then 1 suggests 1 col X as it is.
z2 = a1 * Theta1'; %size of a1 is m*n (5000*401) n including bias unit. size of Theta1 is (h*n) (25*401) where h is hidden units. 
a2 = sigmoid(z2); %size of a2 m*h.
a2 = [ones(size(a2, 1), 1) a2]; %size of a2 m*(h+1) as we add col of 1 as bias unit.
a3 = sigmoid(a2*Theta2'); %size of Theta2 k*(h+1) where k is no. of labels. so a3 m*k.

[val p] = max(a3, [], 2);                                                                                                          









% =========================================================================


end
