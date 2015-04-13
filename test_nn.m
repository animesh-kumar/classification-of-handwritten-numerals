function [missclassification_rate] = test_nn(Test_Input_Matrix,Test_Target_Matrix,Weight_ij_Matrix,Weight_jk_Matrix)
% Append a column of ones at beginning of Input matrix to serve as bias
Test_Input_Matrix=[ones(size(Test_Input_Matrix,1),1) Test_Input_Matrix];
Y_Matrix = zeros(1500,10);
% Class for saving the text file
nn_class = ones(1500,1);
for inputCount=1:1:1500
    aj = Weight_ij_Matrix*transpose(Test_Input_Matrix(inputCount,:));
    zj = tanh(aj);
    yk_Vector = Weight_jk_Matrix*zj;
    %maxIndex = find(yk_Matrix == max(yk_Matrix));
    [val idx] = max(yk_Vector);
    yk_Vector = zeros(1,10);
    yk_Vector(idx) = 1;
    Y_Matrix(inputCount,:) = yk_Vector;
end

% Handling misclassifications
% Check if the place of occurence of 1 is same as the maximum value in row
% of target matrix
missclassification = 0;
for i=1:1:1500
    % Occurence of 1 in Target matrix
    target_index = find(Test_Target_Matrix(i,:) == 1);
    % Occurent of max value in calculated matrix
    calculated_index = find(Y_Matrix(i,:) == max(Y_Matrix(i,:)));
    nn_class(i) = calculated_index -1;
    if(target_index~=calculated_index)
        missclassification = missclassification + 1 ;    
    end
end

% Find the missclassification rate
missclassification_rate = (missclassification / 1500)*100;
fileID = fopen('classes_nn.txt','w');
for i = 1:1:1500
    fprintf(fileID,'%d\n',nn_class(i));
end
fclose(fileID);