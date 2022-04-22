clear all
close all
clc

%load iris data
versicolor = load("iris_versicolor.txt");
virginica = load("iris_virginica.txt");
setosa = load("iris_setosa.txt");
% Put the two species into a data matrix.
% Just use two dimensions (3 and 4) for now.
X = [versicolor(:,1:2);virginica(:,1:2);setosa(:,1:2)];

% Add a third column that has the class labels.
% label 0 for versicolor, and 1 for virginica.
X(:,3) = [zeros(50,1);ones(50,1);2*ones(50,1)];

% Designate a point in the space to classify.
% Make the third element '99' to indicate
% missing class label.
x = [7.0, 4.0, 99];
% Create a matrix with the new observation
% as the first row and the remaining ones as the
% observed data.
Xn = [x ; X];

% First, get the pairwise distances.
D = squareform(pdist(Xn(:,1:2)));
% Then, find the ones closest to the new point.
% Do this by looking in the first row or column.
[pd, ind ] = sort(D(:,1));

% Find the frequency of classes within the 10
% nearest neighbors.
tabulate(Xn(ind(2:6),3));


Mdl = fitcknn(X(:,1:2),X(:,3),'NumNeighbors',3,'Standardize',1)
label = predict(Mdl,X(:,1:2));
% plotroc(X(:,3)', label')

perform=sum(X(:,3)==label)/length(label)*100


% Test Model
% k=5 feature=sepal length, sepal width
[X1,Y1]=TestModel(5,1,2);
% k=10 feature=sepal length, sepal width
[X2,Y2]=TestModel(10,1,2);
% k=5 feature=petal length, petal width
[X3,Y3]=TestModel(5,3,4);
% k=10 feature=petal length, petal width
[X4,Y4]=TestModel(10,3,4);

% Plot
figure(2)
hold on
plot(Y1{1,2},X1{1,2},'LineWidth',2);
plot(Y2{1,2},X2{1,2},'LineWidth',2);
plot(Y4{1,2},X4{1,2},'LineWidth',2);
plot(Y3{1,2},X3{1,2},'--','LineWidth',2);


% Define title, legend, label.
title('ROC of each model');
legend('k=5 feature=sepal length, sepal width','k=10 feature=sepal length, sepal width','k=5 feature=petal length, petal width','k=10 feature=patal length, petal width');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
hold off