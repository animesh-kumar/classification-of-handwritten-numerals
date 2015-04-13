function [Weight_ij_Matrix,Weight_jk_Matrix] = train_nn(Input_Matrix,Target_Matrix)
% Append a column of ones at beginning of Input matrix to serve as bias
Input_Matrix=[ones(size(Input_Matrix,1),1) Input_Matrix];
% Loop over each row and run through the neural network model
%Initialize the weight matrix for ij to some random initial weight
Weight_ij_Matrix(1:513,1:513) = 0.001 + (0.002-0.001)*rand(513);
Weight_jk_Matrix(1:10,1:513) = 0.001 + (0.002-0.001)*rand(10,513);
learning_rate = 0.001 + (0.002-0.001)*rand();
disp (learning_rate);
inputCount = ceil(rand * size(Input_Matrix,1));
for iteration=1:1:25000
    % Each input has 513 features including a bias
    % Consider each feature as i, multiply each i * weight
    aj = Weight_ij_Matrix*transpose(Input_Matrix(inputCount,:));
    zj = tanh(aj);
    %yk_Matrix = Weight_jk_Matrix*Z_Matrix;
    yk_Matrix = Weight_jk_Matrix*zj;
    %y_calculated_matrix(inputCount,k) = yk_Matrix;
    
    % Now feed forward is completed let's do back propagation
    % we compute the ?’s for each output unit using
    % calculate ?k = yk ? tk
    derivative_k = yk_Matrix - transpose(Target_Matrix(inputCount,:));
    % Then we backpropagate these to obtain ?s for the hidden units using
    % ?j = (1 - zj^2 )Summation(1,k){ wkj*?k}
    % Find 1-zj^2
    derivative_j = (1 - zj.^2).*(transpose(Weight_jk_Matrix)*derivative_k);
    delta_ji = derivative_j * Input_Matrix(inputCount,:);
    Weight_ij_Matrix = Weight_ij_Matrix - learning_rate * delta_ji;
    delta_jk = derivative_k * transpose(zj);
    Weight_jk_Matrix = Weight_jk_Matrix - learning_rate * delta_jk;
    % Find the next random row to be fetched from input matrix
    inputCount = ceil(rand * size(Input_Matrix,1));
end