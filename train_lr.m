function [Weight_Matrix] = train_lr(Input_Matrix,Target_Matrix)
% Append a column of ones at beginning of Input matrix to serve as bias
Input_Matrix=[ones(size(Input_Matrix,1),1) Input_Matrix];
% Initialize the weight matrix with initial weights
Weight_Matrix(1:513,1:10) = 0.005 + (0.01-0.00005)*rand(513,10);
Ak_Matrix = Input_Matrix * Weight_Matrix;
% Run For loop on each row
num_rows_ak = size(Ak_Matrix,1);
y_calculated_matrix = zeros(size(Input_Matrix,1),10);
for i = 1:num_rows_ak
    % Fetch row i
    ak = Ak_Matrix(i,:);
    % y = exp(a)/sum(expof all classes)
    ak = exp(ak);
    sum_of_exponents = sum(ak);
    % Divide each element by summation of exponents
    y = ak/sum_of_exponents;
    y_calculated_matrix(i,:) = y;
end

% Learning the weights based on error (modify the weight as needed)
learning_rate = 0.001 + (0.002-0.001)*rand();
%learning_rate = 0.0013;
disp (learning_rate);
for iterations = 1:1:50
    % Check for each class
    for class = 1:1:10
        error = zeros(513,1);
        % Check each sample
        for sample = 1:1:19978
            error = transpose(Input_Matrix(sample,:))*(y_calculated_matrix(sample,class)-Target_Matrix(sample,class))+ error;
        end
        %Update the weight matrix for the given class
        Weight_Matrix(:,class) = Weight_Matrix(:,class)- (learning_rate*error);
    end
    Ak_Matrix = Input_Matrix * Weight_Matrix;
    y_calculated_matrix = zeros(size(Input_Matrix,1),10);
    for i = 1:num_rows_ak
        % Fetch row i
        ak = Ak_Matrix(i,:);
        % y = exp(a)/sum(expof all classes)
        ak = exp(ak);
        sum_of_exponents = sum(ak);
        % Divide each element by summation of exponents
        y = ak/sum_of_exponents;
        y_calculated_matrix(i,:) = y;
    end
   learning_rate = 0.001 + (0.002-0.001)*rand();
end