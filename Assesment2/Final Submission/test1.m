clear all
close all
clc

%load iris data
versicolor = load("iris_versicolor.txt");
virginica = load("iris_virginica.txt");
setosa = load("iris_setosa.txt");
% Put the three species into a data matrix.
% Just use two dimensions (1 and 2) for now.
X = [versicolor(:,1:2);virginica(:,1:2);setosa(:,1:2)];

% Add a third column that has the class labels.
% label 0 for versicolor, and 1 for virginica. two for setosa
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

% Find the frequency of classes within the 3
% nearest neighbors.
tabulate(Xn(ind(2:6),3));


Mdl = fitcknn(X(:,1:2),X(:,3),'NumNeighbors',3,'Standardize',1)
label = predict(Mdl,X(:,1:2));
plotroc(X(:,3)', label')

perform=sum(X(:,3)==label)/length(label)*100