clear all
close all
clc

% load data, set k=5
setosa = load("iris_setosa.txt");
versicolor = load("iris_versicolor.txt");
virginica = load("iris_virginica.txt");
k=5;

% Put the three species into a data matrix.
% use sepal length and sepal width for classification.
X = [setosa(:,1:2);versicolor(:,1:2);virginica(:,1:2)];

% Add a third column that has the class labels.
% label -1 for setosa, 0 for versicolor, and 1 for virginica.
temp = zeros(50,1)-ones(50,1);
X(:,3) = [temp;zeros(50,1);ones(50,1)];

% Designate a point in the space to classify.
% Make the third element '99' to indicate
% missing class label.
x = [7,4,99];

% Create a matrix with the new observation
% as the first row and the remaining ones as the
% observed data.
Xn = [x ; X];

% First, get the pairwise distances.
D = squareform(pdist(Xn(:,1:2)));

% Then, find the ones closest to the new point.
% Do this by looking in the first row or column.
% Choose one 5 nearest neighbors
[pd, ind ] = sort(D(:,1));
distKNN = pd(2:k+1,:);

% Find the frequency of classes within the 5
% nearest neighbors.
tabulate(Xn(ind(2:6),3)); 

% Plot the scatter plot, the new sample, the radius of nearest neighbors
figure(1)
hold on
gscatter(X(:,1),X(:,2),X(:,3));
plot(x(1),x(2),'*');
theta = linspace(0,2*pi);
xi = distKNN(k,1)*cos(theta) + Xn(1,1);
yi = distKNN(k,1)*sin(theta) + Xn(1,2);
plot(xi,yi,'--k');

% Add legend, label, and title.
title('Iris Species Classification');
legend('setosa','versicolor','verginica','New Sample');
xlabel('sepal length');
ylabel('sepal width');
axis([4 8 2 5])
hold off

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
