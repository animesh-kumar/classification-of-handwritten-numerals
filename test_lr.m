function [missclassification_rate] = test_lr(Test_Input_Matrix,Test_Target_Matrix,Weight_Matrix)
% Append a column of ones at beginning of Input matrix to serve as bias
Test_Input_Matrix=[ones(size(Test_Input_Matrix,1),1) Test_Input_Matrix];
Ak_Matrix = Test_Input_Matrix * Weight_Matrix;
% Class for saving the text file
lr_class = ones(1500,1);
% Run For loop on each row
num_rows_ak = size(Ak_Matrix,1);
y_calculated_matrix = zeros(size(Test_Input_Matrix,1),10);
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

% Handling misclassifications
% Check if the place of occurence of 1 is same as the maximum value in row
% of target matrix
missclassification = 0;
for i=1:1:1500
    % Occurence of 1 in Target matrix
    target_index = find(Test_Target_Matrix(i,:) == 1);
    % Occurent of max value in calculated matrix
    calculated_index = find(y_calculated_matrix(i,:) == max(y_calculated_matrix(i,:)));
    lr_class(i) = calculated_index -1;
    if(target_index~=calculated_index)
        missclassification = missclassification + 1 ;
    end
end

% Find the missclassification rate
missclassification_rate = (missclassification / 1500)*100;
disp (missclassification_rate);
fileID = fopen('classes_lr.txt','w');
for i = 1:1:1500
    %ip=strcat();
    fprintf(fileID,'%d\n',lr_class(i));
end
fclose(fileID);
end