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
y_t = zeros(m, num_labels);
for i = 1:m
  y_t(i,y(i)) = 1;
endfor


X = [ones(m,1),X];% Size(X) = 5000 401 

z2 = X *Theta1';% Size(z2) = 5000 25
a2 = sigmoid(z2); % size(a2) = 5000 25
a2 = [ones(size(X, 1),1), a2]; % size(a2) = 5000 26

z3 = a2 * Theta2'; %size(z3) = 5000 10
a3 = sigmoid(z3); %size(a3) = 5000 10


J =1/m*sum(sum(-y_t .* log(a3)- (1-y_t).*log(1-a3))); %size(J) = 1*1


Theta1_t = Theta1(:);  % size(Theta1) = 25 401  size(Theta1_t) = 10025
Theta1_t = Theta1_t(size(Theta1,1)+1:end,:); %size(Theta1_t) = 10000
Theta2_t = Theta2(:); %size(Theta2) = 10 26 size(Theta2_t) = 260 1
Theta2_t = Theta2_t(size(Theta2,1)+1:end,:); %size(Theta2_t) = 250


reg = lambda/(2*m) * (sum(Theta1_t(:).^2) + sum(Theta2_t(:).^2)); 
J = J + reg;


%back propagation
Delta1 = zeros(hidden_layer_size, input_layer_size+1); %size(Delta1) = 25*401
Delta2 = zeros(num_labels, hidden_layer_size+1); %size(Delta2) = 10*26
delta3 = zeros(num_labels,m); % size(delta3) = 10 *5000
delta2 = zeros(hidden_layer_size,m); % size(delta2) = 25 * 5000


delta3 = a3 -y_t; %size(delta3) = (5000*10) .- (5000*10) = 5000*10
delta2 = delta3 * Theta2(:,2:end).* sigmoidGradient(z2); %size(delta2) = (5000*10) * (10*25) .* (5000 * 25) = 5000 * 25
Delta2 = Delta2 .+ (delta3' * a2); %size(Delta2) = (10*26) .+ (10* 5000) * (5000*26) = 10*26
Delta1 = Delta1 .+ (delta2' * X);

%regularization
reg_2 = zeros(size(Delta2));
reg_1 = zeros(size(Delta1));
Delta2(:,1) = Delta2(:,1); 
Delta2(:,2:end) = Delta2(:,2:end) .+ lambda * Theta2(:,2:end); % size(:,2:end) = (10*25) .+ (10*25) = 10 *25
Delta1(:,1) = Delta1(:,1);
Delta1(:,2:end) = Delta1(:,2:end) .+ lambda * Theta1(:,2:end);
%size(reg_2) = 10 *26



Theta1_grad = (1/m.*Delta1);

Theta2_grad = (1/m.*Delta2);






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
