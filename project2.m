load project2_data.mat
myubitname = 'animeshk';
mystudentnumber = 50134753;
fprintf('My ubit name is %s\n',myubitname);
fprintf('My student number is %d \n',mystudentnumber);
[Weight_Matrix] = train_lr(Input_Matrix,Target_Matrix);
[missclassification_rate] = test_lr(Test_Input_Matrix,Test_Target_Matrix,Weight_Matrix);
fprintf('Missclassification for Logistic Regression is %d\n', missclassification_rate);

[Weight_ij_Matrix,Weight_jk_Matrix] = train_nn(Input_Matrix,Target_Matrix);
[missclassification_rate] = test_nn(Test_Input_Matrix,Test_Target_Matrix,Weight_ij_Matrix,Weight_jk_Matrix);
fprintf('Missclassification for Neural Network is %d\n', missclassification_rate);